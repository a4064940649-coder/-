import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import rcore as C, numpy as np, pandas as pd
V=C.V; R=C.R
DS = R.Disp(R.load_list()); base = R.make_strategy('E1', DS)._mask
for seed in range(5):
    S = V.V2('gapfix', C.YL, C.Y5, exfin=True, unavail=V.make_unavail('mc', seed))
    orig = S.cross_section
    def cs(d, _o=orig, _m=base):
        old = C.K.disposal_mask; C.K.disposal_mask = _m
        try: return _o(d)
        finally: C.K.disposal_mask = old
    S.cross_section = cs
    plo = S.plan(10, 'official'); t0, t1, sd = S.bounds()
    nav, _ = C.O3.simulate([(p[0], p[1]) for p in plo], *C.NORMAL, t0, t1, xrules=C.XR)
    pl = C.make_plan([(m, C.OFFICIAL[m]) for m in C.ANCHORS], 10, C.AB(seed))
    nd = sum(1 for a, b in zip(pl, plo) if list(a[1]) != list(b[1]))
    print(seed, round(C.cagr_mdd(nav, nav.index[0], C.END)[0], 2), round(C.tstat(C.excess(nav, sd)), 3), 'pickdiff vs mine', nd)
