import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import time; t_=time.time()
import rcore as C, numpy as np, pandas as pd, json
print('load', time.time()-t_, 'anchors', len(C.ANCHORS), C.ANCHORS[0], C.ANCHORS[-1], 'npatch', C.NPATCH, 'xr', len(C.XR))
allm = pd.period_range('2006-04','2026-09',freq='M'); print('missing', [str(m) for m in allm if m not in C.ANCHORS])
days = [(m, C.OFFICIAL[m]) for m in C.ANCHORS]
pl = C.make_plan(days, 10, None)
t0 = C.POS[days[0][1]]
nav, lg = C.tsim([pl], *C.NORMAL, t0, C.TEND)
sd = [d for _, d in days] + [C.END]
ex = C.excess(nav, sd)
print('ALL', C.cagr_mdd(nav, '2006-04-01', '2026-10-12'), C.tstat(ex), len(ex), 'turn', lg['turnover'])
# compare with o3_core.simulate on same plan
nav2, _ = C.O3.simulate([(p[0], p[1]) for p in pl], *C.NORMAL, t0, C.TEND, xrules=C.XR)
print('maxreldiff vs o3', float((nav/nav2-1).abs().max()))
# compare picks with official V2 object
V=C.V; R=C.R
S = V.V2('gapfix', C.YL, C.Y5, exfin=True)
DS = R.Disp(R.load_list()); base = R.make_strategy('E1', DS)._mask
orig = S.cross_section
def cs(d, _o=orig, _m=base):
    old = C.K.disposal_mask; C.K.disposal_mask = _m
    try: return _o(d)
    finally: C.K.disposal_mask = old
S.cross_section = cs
plo = S.plan(10, 'official')
print('n plans', len(pl), len(plo), 'sigdays equal', [p[3] for p in pl]==[p[3] for p in plo])
diff=[(str(a[3].date()),) for a,b in zip(pl,plo) if a[0]!=b[0] or list(a[1])!=list(b[1])]
print('pick diffs', len(diff), diff[:5])
print(time.time()-t_)
