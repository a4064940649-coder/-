import sys, pathlib, json, math; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import ind_tr as TRN
C = TRN.C; import numpy as np, pandas as pd
H = pathlib.Path(__file__).resolve().parent
OFFD = [(p, C.OFF[p]) for p in C.ANCH]
AM = {C.OFF[p]: p for p in C.ANCH}
def off_run(costs=C.NORMAL, ab=None):
    pl = C.plan(OFFD, ab); return C.tsim([pl], *costs, C.POS[OFFD[0][1]], C.TEND)
# EW 同池等權（官方日、無 AB、零成本）
ewpl = []
for p, s in OFFD:
    r = C.rank(p, s)
    if r['nel'] >= 50: ewpl.append(dict(e=r['e'], pk=np.where(r['el'])[0], p=p, s=s, t=r['t'], order=None))
EW, _ = C.tsim([ewpl], 0.0, 0.0, C.POS[OFFD[0][1]], C.TEND)
def wins(nav):
    w = C.wret(nav, C.OFFSD); return w
PER = {'SEL': (None, '2018-12'), 'VAL': ('2019-01', '2023-12'), 'HOLD': ('2024-01', '2026-10'), 'ALL': (None, None), 'POST': ('2019-01', None)}
def mask(idx, a, b):
    pr = pd.PeriodIndex([AM[d] for d in idx], freq='M'); m = np.ones(len(idx), bool)
    if a: m &= pr >= pd.Period(a, 'M')
    if b: m &= pr <= pd.Period(b, 'M')
    return m
def cagr_w(nav, a, b):
    w = wins(nav); m = mask(w.index, a, b); ii = np.where(m)[0]
    d0 = w.index[ii[0]]; k = C.OFFSD.index(w.index[ii[-1]]); d1 = C.OFFSD[k + 1]
    yrs = (d1 - d0).days / 365.25
    seg = nav[(nav.index >= d0) & (nav.index <= d1)]; eq = seg / seg.iloc[0]
    return float(eq.iloc[-1] ** (1 / yrs) - 1) * 100, float((eq / eq.cummax() - 1).min()) * 100
def full(nav, ref=None):
    ex = C.exw(nav, C.OFFSD); exe = C.exw(nav, C.OFFSD, EW)
    o = {}
    for k, (a, b) in PER.items():
        m = mask(ex.index, a, b); cg, mdd = cagr_w(nav, a, b)
        o[k] = dict(CAGR=cg, mdd=mdd, t=C.tst(ex[m]), mean=float(ex[m].mean() * 100), sd=float(ex[m].std(ddof=1) * 100), t_EW=C.tst(exe[m]),
                    E50=cagr_w(C.E50.reindex(nav.index), a, b)[0])
        if ref is not None:
            d = (wins(nav) - wins(ref))[m]; o[k]['pd'] = float(d.mean() * 100); o[k]['pd_t'] = C.tst(d)
    cm = C.calm(nav); o['t_cal_ALL'] = C.tst(cm); o['t_cal_SEL'] = C.tst(cm[cm.index <= pd.Period('2019-01', 'M')])
    o['TE_ALL'] = float(ex.std(ddof=1) * math.sqrt(12) * 100)
    # 端點敏感度
    x = ex.copy(); srt = x.sort_values(ascending=False)
    o['drop_last1'] = C.tst(x.iloc[:-1])
    for n in (1, 3, 5, 10): o[f'drop_best{n}'] = C.tst(x.drop(srt.index[:n]))
    yr = pd.Series(x.values, index=[d.year for d in x.index]).groupby(level=0).sum()
    by = int(yr.idxmax()); o['best_year'] = by; o['drop_best_year'] = C.tst(x[[d.year != by for d in x.index]])
    o['p1'] = 0.5 * (1 - math.erf(o['ALL']['t'] / math.sqrt(2)))
    return o
if __name__ == '__main__':
    OFFN, lo = off_run(); J1, l1 = TRN.run([1]); V8, l8 = TRN.run(*[TRN.VERS['V08'][0]], 0, 0, 1)
    R = {'OFF': full(OFFN), 'J+1': full(J1, OFFN), 'V08': full(V8, J1)}
    R['V08_vs_OFF'] = full(V8, OFFN)
    R['V08_mdd_full'] = C.cagr(V8); R['J1_full'] = C.cagr(J1); R['OFF_full'] = C.cagr(OFFN)
    R['turn'] = dict(OFF=lo['turnover'], J1=l1['turnover'], V08=l8['turnover'])
    for nm, cs in (('flat', C.FLAT), ('p03', C.P03), ('gross', C.GROSS)):
        a, _ = TRN.run([1], costs=cs); b, _ = TRN.run(TRN.VERS['V08'][0], 0, 0, 1, costs=cs)
        R[f'V08_{nm}'] = full(b, a)
    for k in ('OFF', 'J+1', 'V08', 'V08_vs_OFF', 'V08_flat', 'V08_p03', 'V08_gross'):
        print(k, json.dumps(R[k], default=float)[:1500])
    print({k: R[k] for k in ('V08_mdd_full', 'J1_full', 'OFF_full', 'turn')})
    json.dump(R, open(H / 'ind_detail.json', 'w'), indent=1, default=float)
    pd.DataFrame({'OFF': OFFN, 'J+1': J1, 'V08': V8, 'EW': EW}).to_parquet(H / 'ind_detail_navs.parquet')
