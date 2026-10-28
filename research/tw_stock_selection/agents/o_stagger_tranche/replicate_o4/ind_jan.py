import sys, pathlib, json, math; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.argv = ['x', 'anchor', 'next']
import ind_jscan as JS
C = JS.C; import numpy as np, pandas as pd
H = pathlib.Path(__file__).resolve().parent
out = {}
def P(k, v): out[k] = v; print(k, v, flush=True)
JJ = list(range(-8, 13))
def exs(costs, w):
    E, T, NV = {}, {}, {}
    for j in JJ:
        nav, lg, pl, sd, e_ = JS.run_j(j, C.AB(0), costs, w)
        ex = C.exw(nav, sd); am = JS.run_j.am
        E[j] = pd.Series(ex.values, index=pd.PeriodIndex([am[d] for d in ex.index], freq='M')); NV[j] = nav
    return pd.DataFrame(E), NV
EN, NVN = exs(C.NORMAL, 0.5); EG, _ = exs(C.GROSS, 0.5); EP, _ = exs(C.GROSS, 1.0); ER, _ = exs(C.GROSS, 0.0)
ERn, _ = exs(C.NORMAL, 0.0)
ts = np.array([C.tst(EN[j]) for j in JJ]); cg = np.array([C.cagr(NVN[j])[0] for j in JJ])
q = lambda v, x: float(np.percentile(v, x))
P('t_dist', dict(min=ts.min(), p10=q(ts, 10), p25=q(ts, 25), med=q(ts, 50), p75=q(ts, 75), p90=q(ts, 90), max=ts.max(), sd=float(ts.std(ddof=1))))
P('cagr_dist', dict(p10=q(cg, 10), med=q(cg, 50), p90=q(cg, 90)))
P('t_J0_J4', dict(med=float(np.median(ts[8:13])), lo=ts[8:13].min(), hi=ts[8:13].max()))
P('t_Jm8_Jm1_med', float(np.median(ts[:8])))
sel = EN[EN.index <= pd.Period('2018-12', 'M')]
tsel = np.array([C.tst(sel[j]) for j in JJ]); P('t_SEL_dist', dict(med=float(np.median(tsel)), lo=tsel.min(), hi=tsel.max()))
P('pct_of_value', {v: float((tsel < v).mean()) for v in (2.37, 2.27)})
P('pct_full', {v: float((ts < v).mean()) for v in (2.55, 2.46, 2.42)})
db10 = [C.tst(EN[j].drop(EN[j].dropna().sort_values(ascending=False).index[:10])) for j in JJ]
P('drop10_med', float(np.median(db10)))
# 新鮮度
def fresh(E, a, b, A=range(1, 6), B=range(-8, 0)):
    x = E.loc[(E.index >= pd.Period(a, 'M')) & (E.index <= pd.Period(b, 'M'))]
    d = (x[list(A)].mean(axis=1, skipna=True) - x[list(B)].mean(axis=1, skipna=True)).dropna()
    return round(float(d.mean() * 100), 3), round(C.tst(d), 2), len(d)
P('fresh', {'SEL_gross': fresh(EG, '2006-04', '2018-12'), 'SEL_normal': fresh(EN, '2006-04', '2018-12'), 'POST_normal': fresh(EN, '2019-01', '2026-10'),
            'S4_normal': fresh(EN, '2024-01', '2026-10'), 'ALL_normal': fresh(EN, '2006-04', '2026-10'),
            'post_hoc_J0_3_vs_J9_12_POST': fresh(EN, '2019-01', '2026-10', range(0, 4), range(9, 13)), 'post_hoc_SEL': fresh(EN, '2006-04', '2018-12', range(0, 4), range(9, 13)),
            'price_leg_SEL': fresh(EP, '2006-04', '2018-12'), 'rev_leg_SEL': fresh(ER, '2006-04', '2018-12')})
P('revleg_SEL_mean_by_j_gross', {j: round(float(ER[j][ER.index <= pd.Period('2018-12', 'M')].mean() * 100), 3) for j in JJ})
P('revleg_SEL_mean_by_j_normal', {j: round(float(ERn[j][ERn.index <= pd.Period('2018-12', 'M')].mean() * 100), 3) for j in JJ})
# 時點雜訊
cs = EN.std(axis=1, ddof=1) * 100
segs = {'SEL': ('2006-04', '2018-12'), 'S1': ('2006-01', '2013-12'), 'S2': ('2014-01', '2020-12'), '2008': ('2008-01', '2008-12'), 'S3': ('2021-03', '2026-10'), 'S4': ('2024-01', '2026-10')}
P('xsd', {k: round(float(cs[(cs.index >= pd.Period(a, 'M')) & (cs.index <= pd.Period(b, 'M'))].mean()), 2) for k, (a, b) in segs.items()})
cw = pd.DataFrame({j: C.exw(NVN[j], [d for d in C.OFFSD if d >= NVN[j].index[0]]) for j in (1, 6, 11)}).dropna()
P('corr_common', dict(J1_J6=float(cw[1].corr(cw[6])), J1_J11=float(cw[1].corr(cw[11]))))
tc = np.array([C.tst(C.exw(NVN[j], [d for d in C.OFFSD if d >= NVN[j].index[0]])) for j in JJ]); P('sd_common', float(tc.std(ddof=1)))
P('turn_range', [min(JS.run_j(j, C.AB(0))[1]['turnover'] for j in (0,)), None])
# J0 seeds
j0 = []
for s in range(5):
    nav, lg, pl, sd, e_ = JS.run_j(0, C.AB(s)); j0.append((C.cagr(nav)[0], C.tst(C.exw(nav, sd))))
P('J0_seeds', j0)
# 官方 AB seeds SEL t
import ind_detail as DT
oab = []
for s in range(5):
    nav, _ = DT.off_run(C.NORMAL, C.AB(s)); ex = C.exw(nav, C.OFFSD); m = DT.mask(ex.index, None, '2018-12')
    oab.append((C.cagr(nav)[0], C.tst(ex), C.tst(ex[m])))
P('OFF_AB_seeds', oab)
json.dump(out, open(H / 'ind_jan.json', 'w'), indent=1, default=float)
