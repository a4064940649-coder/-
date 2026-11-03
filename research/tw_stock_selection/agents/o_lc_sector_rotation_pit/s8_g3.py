#!/usr/bin/env python3
"""凍結後：(1) 驗證期安慰劑（1,000 組隨機產業排名，持續度＝挑選期實際訊號；扣成本前後）；
(2) G3：Hansen SPA_c（平穩 bootstrap L=6、B=10,000）與 Deflated Sharpe，族群 (i) 2,334 條＋本題 40 格、(ii) F_main（N=410）＋本題 40 格。
族群讀取唯讀 import o_g2_feasibility_stoprule/g2lib.py；窗＝研究中心族群窗（官方換股日），只取 2014-01 起（本題正式段；D1 未過）。先斷言 PREREG。"""
import sys, json, time, importlib.util
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lclib as X
import os
DRY = os.environ.get('LC_DRY') == '1'     # 凍結前的程式除錯：只用挑選期、輸出到 scratch，不碰 results.json
if DRY:
    _it = json.loads((HERE / 'iterations.json').read_text(encoding='utf-8'))
    PRE = dict(chosen=dict(cfg=_it['chosen']['cfg'], params=_it['versions'][_it['chosen']['version']]['params']), default=dict(cfg='COMP|k5|A30|M'),
               placebo_persistence=json.loads((HERE / 'explore.json').read_text(encoding='utf-8'))['persistence_jmed'], code_sha256={}, data_sha256={})
    RES = dict(PREREG=PRE)
    RESF = X.SCR / f'dry_{Path(__file__).stem}.json'
    if RESF.exists(): RES.update(json.loads(RESF.read_text(encoding='utf-8')))
    for f in ('dry_s6_oos.json',):
        if (X.SCR / f).exists(): RES.update({k: v for k, v in json.loads((X.SCR / f).read_text(encoding='utf-8')).items() if k == 'OOS'})
else:
    RES = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))
    PRE = RES['PREREG']
    RESF = HERE / 'results.json'
    assert X.canon(PRE) == RES['PREREG_sha256'], 'PREREG 雜湊不符'
    for f, h in PRE['code_sha256'].items(): assert X.sha(HERE / f) == h, f'{f} 與 PREREG 不符'
import lcrun as R
SEL = PRE['chosen']['cfg']; DEF = PRE['default']['cfg']; P = PRE['chosen']['params']
PERS = PRE['placebo_persistence']
def cfgt(nm):
    a, k, A, f = nm.split('|'); return (a, int(k[1:]), int(A[1:]) / 100, f)
R.init()
if __name__ == '__main__':
    t0 = time.time()
    out = {}
    # ---------- (1) 驗證期與正式全期安慰劑 ----------
    OOS = RES['OOS']['main']
    with Pool(4) as pool:
        for nm in (SEL, DEF):
            phi = R.phi_from_rank_ac(PERS[nm.split('|')[0]])
            for per, (m0, m1, me), key in ((('SEL', ('2014-01', '2016-12', '2017-01'), 'VAL'),) if DRY else (('VAL', ('2017-01', '2026-10', '2026-11'), 'VAL'),)):
                draws = pool.map(R.placebo_draw, [(5000 + i, cfgt(nm), phi, m0, m1, me, P) for i in range(20 if DRY else 1000)], chunksize=5 if DRY else 10)
                g = np.array([d[0] for d in draws]); n = np.array([d[1] for d in draws])
                ag = OOS[f'{nm}|zero|vsB|{key}']['jmed']['mean_pp']; an = OOS[f'{nm}|normal|vsB|{key}']['jmed']['mean_pp']
                out[f'placebo|{nm}|{per}'] = dict(phi=phi, actual_gross=ag, actual_net=an, p_gross=float((g >= ag).mean()), p_net=float((n >= an).mean()),
                                                  pct_gross=float((g < ag).mean() * 100), pct_net=float((n < an).mean() * 100),
                                                  placebo_gross_p50=float(np.median(g)), placebo_gross_p95=float(np.percentile(g, 95)),
                                                  placebo_net_p50=float(np.median(n)), placebo_net_p95=float(np.percentile(n, 95)))
                print(nm, per, out[f'placebo|{nm}|{per}'], flush=True)
    # ---------- (2) G3 ----------
    spec = importlib.util.spec_from_file_location('g2lib_ro', X.AG / 'o_g2_feasibility_stoprule' / 'g2lib.py')
    G = importlib.util.module_from_spec(spec); spec.loader.exec_module(G)
    FX, meta = G.family()
    navs = pd.read_parquet(X.SCR / 'navs_javg_formal.parquet')
    mine = pd.DataFrame({f'LCROT:{c}': G.win_ex(navs[c]) for c in navs.columns if c != 'B'})
    WIN = G.WIN
    m = (WIN >= pd.Timestamp('2014-01-01')) & mine.notna().all(axis=1).to_numpy()
    out['G3_windows'] = dict(n=int(m.sum()), first=str(WIN[m][0].date()), last=str(WIN[m][-1].date()))
    for fam, cols in (('all2334', list(FX.columns)), ('F_main', meta.member_id[meta.F_main.astype(bool)].tolist())):
        D = pd.concat([FX[cols], mine], axis=1)[m].fillna(0.0)
        Xf = D.to_numpy(); n, Nm = Xf.shape
        W = G.SP.sb_counts(n, 6, 10000, 20261103)
        Mb = G.SP.boot_means(W, Xf)
        om2 = G.SP.lrv(Xf, 'pr', 6); om = np.sqrt(om2); sn = np.sqrt(n)
        dbar = Xf.mean(0); tst = sn * dbar / om
        thr = -np.sqrt(om2 / n * 2 * np.log(np.log(n)))
        gc = np.where(dbar >= thr, dbar, 0.0)
        Z = (Mb - gc[None, :].astype(np.float32)) * (sn / om).astype(np.float32)[None, :]
        Tb = np.maximum(Z.max(axis=1), 0.0)
        sr_all = Xf.mean(0) / Xf.std(0, ddof=1); sr_all = sr_all[np.isfinite(sr_all)]
        et = G.SP.eff_trials(Xf[:, np.std(Xf, 0) > 0])
        r = dict(N=Nm, n_windows=n, Tb_q95=float(np.quantile(Tb, 0.95)), max_t_family=float(tst.max()), argmax=D.columns[int(np.argmax(tst))], M_eff_LiJi=et['M_eff_LiJi'])
        for nm in (SEL, DEF):
            k = list(D.columns).index(f'LCROT:{nm}')
            r[nm] = dict(t=float(tst[k]), p_SPA=float((Tb >= tst[k]).mean()) if tst[k] > 0 else 1.0,
                         DSR=G.SP.dsr(Xf[:, k], sr_all, Nm)['DSR'], DSR_LiJi=G.SP.dsr(Xf[:, k], sr_all, et['M_eff_LiJi'])['DSR'],
                         SR_m=float(Xf[:, k].mean() / Xf[:, k].std(ddof=1)))
        best = int(np.argmax([tst[list(D.columns).index(c)] for c in mine.columns]))
        r['best_of_40'] = dict(cfg=mine.columns[best], t=float(tst[list(D.columns).index(mine.columns[best])]))
        out[f'G3|{fam}'] = r
        print(fam, {k: v for k, v in r.items()}, flush=True)
    RES = json.loads(RESF.read_text(encoding='utf-8')) if RESF.exists() else {}
    RES['PLACEBO_G3'] = out
    RESF.write_text(X.dumps(RES), encoding='utf-8')
    print('time', round(time.time() - t0))
