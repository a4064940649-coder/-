#!/usr/bin/env python3
"""探索（只用挑選期 SEL：錨定月 2014-01～2016-12；D1 未過，挑選期只有 36 個月）。
看：產業 IC（P1）、分位價差、訊號相關、權重分布與追蹤誤差、周轉與成本、j 曲線（營收時效）、離散度分組、小產業是否主導。
不看 2017 年以後、也不看 2014 年以前的任何報酬。"""
import sys, json, time
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lclib as X, lcrun as R
from scipy.stats import spearmanr
M0, M1, ME = '2014-01', '2016-12', '2017-01'
SIGC = ['REV3', 'RS', 'RS12', 'RS6', 'RS3', 'H52', 'NH', 'COMP']
R.init()

def job(j):
    B = R.build(j, M0, M1, ME, keep_cs=True, grid=[('COMP', 5, 0.30, 'M'), ('REV3', 5, 0.30, 'M'), ('RS', 5, 0.30, 'M')])
    days = B['days']; ti = [X.IDX.get_loc(d) for d in days]
    rows = []; corr = []; ranks_prev = None; pers = {s: [] for s in SIGC}
    prev_ST = None
    for k, d in enumerate(days[:-1]):
        cs = B['rec'][d]['cs']; ST = cs['ST']
        sr, rb = R.sector_window_returns(cs, ti[k], ti[k + 1])
        ex = pd.Series(sr) - rb
        r = dict(d=str(d.date()), n_sec=len(ST), rbB=rb, disp_rev=float(ST['REV3'].std()) if ST['REV3'].notna().sum() > 3 else np.nan,
                 rev_cov_ok=float(ST['REV3'].notna().mean()))
        for s in SIGC:
            x = ST[s]; ok = x.notna() & ex.notna()
            r['IC_' + s] = float(spearmanr(x[ok], ex[ok])[0]) if ok.sum() >= 5 else np.nan
            if prev_ST is not None:
                c = ST.index.intersection(prev_ST.index); a, b = ST.loc[c, s], prev_ST.loc[c, s]; o = a.notna() & b.notna()
                if o.sum() >= 5: pers[s].append(float(spearmanr(a[o], b[o])[0]))
        top = ST['COMP'].sort_values(ascending=False)
        r['spread_COMP_top5_bot5'] = float(ex[top.index[:5]].mean() - ex[top.index[-5:]].mean())
        rv = ST['REV3'].dropna().sort_values(ascending=False)
        r['spread_REV3_top5_bot5'] = float(ex[rv.index[:5]].mean() - ex[rv.index[-5:]].mean()) if len(rv) >= 10 else np.nan
        # 小產業：前 5 名中 B 權重 < 2% 的家數
        Bs = ST['B'] / cs['mc'].sum()
        r['n_small_in_top5'] = int((Bs[top.index[:5]] < 0.02).sum())
        r['rotB'] = float(Bs.sum()); r['w2330'] = B['rec'][d]['w2330']; r['wfin'] = B['rec'][d]['wfin']
        cm = ST[['REV3', 'RS', 'H52', 'NH']].corr(method='spearman')
        corr.append(cm.to_numpy())
        rows.append(r); prev_ST = ST
    res = R.run_all(B)
    xb = {}
    for nm in B['plans']:
        for cost in ('normal', 'zero'):
            s, inf = res[(nm, cost)]; sb, _ = res[('B', cost)]
            xb[f'{nm}|{cost}'] = dict(**X.stats(s - sb), turnover_m=float(np.mean(inf['turnover'])), cost_pp_m=float(np.mean(inf['cost']) * 100))
    sB, iB = res[('B', 'normal')]
    xb['B|normal_vs0050'] = X.stats(sB - X.bench_win(days)); xb['B|turnover'] = float(np.mean(iB['turnover']))
    # 主動偏離（對 B）實際大小
    act = []
    for d in days[:-1]:
        pl = B['plans']['COMP|k5|A30|M'][d]; U, b = B['Bplan'][d]
        if pl is None: continue
        act.append(float(np.abs(pl[1] - b).sum() / 2))
    return j, rows, np.nanmean(np.array(corr), axis=0).tolist(), {s: float(np.mean(v)) if v else np.nan for s, v in pers.items()}, xb, float(np.mean(act))

if __name__ == '__main__':
    t0 = time.time()
    with Pool(4) as p: res = p.map(job, range(13))
    out = {'period': 'SEL 錨定月 2014-01～2016-12（36 個月；D1 未過，2006～2013 不在挑選期）', 'per_j': {}}
    IC = {s: [] for s in SIGC}; ICt = {s: [] for s in SIGC}
    allrows = []
    for j, rows, cm, pers, xb, act in res:
        D = pd.DataFrame(rows)
        allrows.append(D.assign(j=j))
        icj = {s: dict(mean=float(D['IC_' + s].mean()), t=X.tstat(D['IC_' + s])) for s in SIGC}
        for s in SIGC: IC[s].append(icj[s]['mean']); ICt[s].append(icj[s]['t'])
        hi = D.disp_rev >= D.disp_rev.quantile(2 / 3)
        out['per_j'][j] = dict(IC=icj, persistence=pers, corr_REV3_RS_H52_NH=np.round(cm, 3).tolist(), engine=xb, active_share=act,
                               spread_COMP=dict(mean=float(D.spread_COMP_top5_bot5.mean() * 100), t=X.tstat(D.spread_COMP_top5_bot5)),
                               spread_REV3=dict(mean=float(D.spread_REV3_top5_bot5.mean() * 100), t=X.tstat(D.spread_REV3_top5_bot5)),
                               disp_hi_spreadREV=float(D.spread_REV3_top5_bot5[hi].mean() * 100), disp_lo_spreadREV=float(D.spread_REV3_top5_bot5[~hi].mean() * 100),
                               n_small_in_top5=float(D.n_small_in_top5.mean()), rotB=float(D.rotB.mean()), w2330=float(D.w2330.mean()), wfin=float(D.wfin.mean()),
                               rev_cov_ok=float(D.rev_cov_ok.mean()), n_sec=float(D.n_sec.mean()))
    out['IC_jmed'] = {s: dict(mean=float(np.median(IC[s])), t=float(np.median(ICt[s]))) for s in SIGC}
    out['P1'] = dict(COMP_IC_mean_jmed=out['IC_jmed']['COMP']['mean'], COMP_IC_t_jmed=out['IC_jmed']['COMP']['t'],
                     verdict=('成立' if out['IC_jmed']['COMP']['mean'] > 0 and out['IC_jmed']['COMP']['t'] >= 2 else ('不成立（t<1：資訊假說不成立）' if out['IC_jmed']['COMP']['t'] < 1 else '不確定')))
    out['P3_jcurve_REV3_IC'] = {j: out['per_j'][j]['IC']['REV3']['mean'] for j in range(13)}
    out['P3_jcurve_RS_IC'] = {j: out['per_j'][j]['IC']['RS']['mean'] for j in range(13)}
    out['P3_REV3_j0_4_minus_j8_12'] = float(np.mean([out['P3_jcurve_REV3_IC'][j] for j in range(5)]) - np.mean([out['P3_jcurve_REV3_IC'][j] for j in range(8, 13)]))
    out['P3_dispersion'] = dict(hi=float(np.median([out['per_j'][j]['disp_hi_spreadREV'] for j in range(13)])), lo=float(np.median([out['per_j'][j]['disp_lo_spreadREV'] for j in range(13)])))
    keys = list(out['per_j'][0]['engine'])
    out['engine_jmed'] = {k: {m: float(np.median([out['per_j'][j]['engine'][k][m] if isinstance(out['per_j'][j]['engine'][k], dict) else out['per_j'][j]['engine'][k] for j in range(13)]))
                              for m in (out['per_j'][0]['engine'][k].keys() if isinstance(out['per_j'][0]['engine'][k], dict) else ['v'])} for k in keys}
    out['persistence_jmed'] = {s: float(np.median([out['per_j'][j]['persistence'][s] for j in range(13)])) for s in SIGC}
    out['active_share_jmed'] = float(np.median([out['per_j'][j]['active_share'] for j in range(13)]))
    out['struct_jmed'] = {k: float(np.median([out['per_j'][j][k] for j in range(13)])) for k in ('rotB', 'w2330', 'wfin', 'n_sec', 'n_small_in_top5', 'rev_cov_ok')}
    out['not_in_SEL'] = '2007-07 改制前後斷裂、2009／2020／2022 空頭後月份都不在挑選期（2014～2016），留到凍結後描述。'
    pd.concat(allrows).to_csv(X.SCR / 'explore_rows.csv', index=False)
    (X.HERE / 'explore.json').write_text(X.dumps(out), encoding='utf-8')
    print(json.dumps({k: out[k] for k in ('IC_jmed', 'P1', 'P3_REV3_j0_4_minus_j8_12', 'P3_dispersion', 'persistence_jmed', 'active_share_jmed', 'struct_jmed')}, ensure_ascii=False, indent=0, default=float))
    print(json.dumps(out['engine_jmed'], ensure_ascii=False, default=float))
    print('time', round(time.time() - t0))
