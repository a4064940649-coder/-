#!/usr/bin/env python3
"""integrate_o18 正式執行：V7X 可執行（只跑一次）。名單＝gen_v7x_plans.py 重建的題 1 名單（sha 已凍結）；
cz_v7x.CSim（只讀 import）× K5T × GROW × {一般, 壓力} × {100萬, 300萬, 1000萬}；R4 設定＝orig_cfg＋f1, f2, f3=0.5, f4（與題 1、題 2 相同）。
引擎見 PREREG.v7x.engines（cons_main＝題 1 正式口徑，用來重現；v3_*＝題 2 cz3）。現行正式（orig×orig）直接讀 navs_oos_R4.parquet，用同一段統計程式計算。
python3 -I -B run_v7x.py（在 scratchpad 執行）"""
import sys, json, hashlib, time
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; AG = HERE.parent
def sha(f): return hashlib.sha256(Path(f).read_bytes()).hexdigest()
RJ = HERE / 'results.json'
R = json.loads(RJ.read_text(encoding='utf-8')); PR = R['PREREG']
assert hashlib.sha256(json.dumps(PR, ensure_ascii=False, sort_keys=True).encode()).hexdigest() == R['PREREG_sha256'], 'PREREG 被改'
assert 'RESULTS_V7X' not in R, '只跑一次'
for f, h in PR['file_sha256'].items(): assert sha(f if f.startswith('/') else HERE / f) == h, f'{f} 雜湊不符'
PF = Path(PR['v7x']['plans_file']); assert sha(PF) == PR['v7x']['plans_sha256']
sys.path.insert(0, str(AG / 'o_engine_delist_consideration')); import cz_v7x as CV, cz_engine as CE; sys.path.pop(0)
sys.path.insert(0, str(AG / 'o_delist_terms_dispmask_fix')); import cz3; sys.path.pop(0)
D = CV.D
sys.path.insert(0, str(CV.V7X)); import x_stats as S; sys.path.pop(0)
VR = json.loads((CV.V7X / 'results.json').read_text())
BASE = dict(VR['PREREG']['orig_cfg']); BASE.update(dict(f1=True, f2=True, f3=0.5, f4=True))
E2, E3 = cz3.load_E()
R2 = json.loads((AG / 'o_engine_delist_consideration/results.json').read_text(encoding='utf-8'))
dd = R2['PREREG']['events_v2']['summary']['d']['d_used']
kw = dict(TR=D.TR, IDX=D.IDX, Cn=D.C, CI=D.CI, On=D.O)
ST = {'cons_main': CE.build_settlement(E2, 'main', d=dd, **kw),
      'v3_main': cz3.build(E2, E3, 'main', **kw), 'v3_lo': cz3.build(E2, E3, 'lo', **kw), 'v3_hi': cz3.build(E2, E3, 'hi', **kw)}
TIERS = {'100萬': 1e6, '300萬': 3e6, '1000萬': 1e7}
CELLS = [(c, t) for c in ('normal', 'stress') for t in TIERS]
ENG = PR['v7x']['engines']
PL = pd.read_csv(PF, dtype={'picks': str, 'inserted': str}); PL = PL[PL.variant.isin(list(ENG))]
def to_plan(g): return sorted([(int(e), np.array([D.CI[c] for c in pk.split(',')], int), D.TPOS[pd.Timestamp(sg)]) for e, pk, sg in zip(g.exec_idx, g.picks, g.signal)], key=lambda x: x[0])
def job(a):
    v, j, s, pl = a
    out = {}; info = {}
    for c, t in CELLS:
        cfg = dict(tier=TIERS[t], mode='GROW', model='K5T', cost=c); cfg.update(BASE)
        for eg in ENG[v]:
            sm = CV.CSim(pl, dict(cfg), settle=ST[eg], cmode='consideration')
            out[(c, t, eg)] = sm.run()
            if (c, t) == ('normal', '100萬'):
                info[eg] = dict(n_settled=sm.acc['n_settled'], n_fallback=sm.acc['n_fallback'], ev=sorted({(e['code'], e['type']) for e in sm.log}))
    return v, j, s, out, info
if __name__ == '__main__':
    t0 = time.time()
    jobs = [(v, int(j), int(s), to_plan(g)) for (v, j, s), g in PL.groupby(['variant', 'j', 'seed'])]
    navs = {}; infos = {}
    with Pool(4) as P:
        for n, (v, j, s, out, info) in enumerate(P.imap_unordered(job, jobs, chunksize=1)):
            for ck, nv in out.items(): navs.setdefault((v, j) + ck, []).append((s, nv))
            infos[f'{v}|J{j}|s{s}'] = info
            if n % 40 == 0: print(n, len(jobs), round(time.time() - t0), flush=True)
    cols = {f'{v}|J{j}|{c}|{t}|{eg}': S.avg_nav([nv for s, nv in sorted(L, key=lambda x: x[0])]) for (v, j, c, t, eg), L in navs.items()}
    R4 = pd.read_parquet(CV.V7X / 'out' / 'navs_oos_R4.parquet')
    for j in range(13):
        for c, t in CELLS: cols[f'official|J{j}|{c}|{t}|orig'] = R4[f'B10|J{j}|K5T|{c}|{t}|GROW']
    df = pd.DataFrame(cols).sort_index(axis=1)
    df.to_parquet(Path(PR['scratch']) / 'navs_v7x_i18.parquet')
    def segx(nav, j, a, b):
        e = S.end_day('AB_V7_B10', j); nav = nav.loc[:e].dropna()
        x = nav[(nav.index >= a) & (nav.index <= b)]
        if len(x) < 20: return np.nan
        prev = nav[nav.index < x.index[0]]; s0 = prev.index[-1] if len(prev) else x.index[0]
        return S.cagr(nav, s0, x.index[-1]) - S.cagr(S.E50, s0, x.index[-1])
    ENG2 = dict(ENG); ENG2['official'] = ['orig']
    TAB = {}; YR = {}
    for v in ENG2:
        for eg in ENG2[v]:
            for c, t in CELLS:
                nv = [df[f'{v}|J{j}|{c}|{t}|{eg}'] for j in range(13)]
                T = S.jsumm([S.stats(nv[j], 'AB_V7_B10', j) for j in range(13)])
                for nm, (a, b) in {'S0612': ('2006-01-01', '2012-12-31'), 'S1318': ('2013-01-01', '2018-12-31'), 'S19on': ('2019-01-01', '2026-12-31')}.items():
                    xs = np.array([segx(nv[j], j, a, b) for j in range(13)])
                    T[f'exc_{nm}_med'] = float(np.nanmedian(xs)); T[f'exc_{nm}_p10'] = float(np.nanpercentile(xs, 10))
                TAB[f'{v}|{eg}|{c}|{t}'] = T
                if (c, t) == ('normal', '100萬'): YR[(v, eg)] = {j: pd.Series(S.yearly_exc(nv[j], 'AB_V7_B10', j)) for j in range(13)}
    def ydiff(a, b):
        Y = pd.DataFrame({j: YR[a][j] - YR[b][j] for j in range(13)}); m = Y.median(axis=1)
        return dict(med={int(y): float(x) for y, x in m.items()}, sum_med=float(m.sum()), sum_abs_med=float(m.abs().sum()))
    YD = {'REAL_opt_v3main_vs_REAL_opt_consmain': ydiff(('REAL_opt', 'v3_main'), ('REAL_opt', 'cons_main')),
          'REAL_opt_v3main_vs_official': ydiff(('REAL_opt', 'v3_main'), ('official', 'orig')),
          'REAL_opt_consmain_vs_official': ydiff(('REAL_opt', 'cons_main'), ('official', 'orig'))}
    # 相關（日報酬，j 中位數）：整合主口徑 vs 現行正式、vs 0050
    E50r = S.E50.pct_change()
    def cor(a, b, c='normal', t='100萬'):
        out = []
        for j in range(13):
            e = S.end_day('AB_V7_B10', j)
            x = df[f'{a[0]}|J{j}|{c}|{t}|{a[1]}'].loc[:e].pct_change()
            y = (df[f'{b[0]}|J{j}|{c}|{t}|{b[1]}'].loc[:e].pct_change() if b != '0050' else E50r.reindex(x.index))
            z = pd.concat([x, y], axis=1).dropna(); out.append(float(z.corr().iloc[0, 1]))
        return float(np.median(out))
    COR = {'REAL_opt_v3main~official': cor(('REAL_opt', 'v3_main'), ('official', 'orig')), 'REAL_opt_v3main~0050': cor(('REAL_opt', 'v3_main'), '0050'),
           'official~0050': cor(('official', 'orig'), '0050'), 'REAL_opt_v3main~REAL_opt_consmain': cor(('REAL_opt', 'v3_main'), ('REAL_opt', 'cons_main'))}
    # 重現題 1（REAL_opt × cons_main、orig×orig）
    T1 = json.loads((AG / 'o_delist_rev_backfill_allpit/results.json').read_text(encoding='utf-8'))['RESULTS_V7X']['table']
    F = ('exc_med', 'exc_p10', 't_0050_med', 'exc_Y19on_med', 'exc_S1318_med', 'exc_S0612_med', 'exc_S19on_med', 'maxDD_med')
    rep = {}
    for c, t in CELLS:
        rep[f'REAL_opt|cons_main|{c}|{t}'] = max(abs(TAB[f'REAL_opt|cons_main|{c}|{t}'][f] - T1[f'REAL_opt|cons_main|{c}|{t}'][f]) for f in F)
        rep[f'official|{c}|{t}'] = max(abs(TAB[f'official|orig|{c}|{t}'][f] - T1[f'orig|orig|{c}|{t}'][f]) for f in F)
    R = json.loads(RJ.read_text(encoding='utf-8')); assert 'RESULTS_V7X' not in R
    R['RESULTS_V7X'] = dict(utc=pd.Timestamp.now('UTC').isoformat(), runtime_s=round(time.time() - t0, 1), reproduce_T1_maxabs=rep, table=TAB, yearly=YD, corr_daily_jmed=COR,
                            settle_info_sample={k: v for k, v in sorted(infos.items()) if k.endswith('|s0')}, navs_file=str(Path(PR['scratch']) / 'navs_v7x_i18.parquet'))
    RJ.write_text(json.dumps(R, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    for k, T in TAB.items():
        print(k, {f: round(T[f], 3) for f in F})
    print('reproduce', max(rep.values()), round(time.time() - t0))
