import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import rcore as C, numpy as np, pandas as pd
V=C.V
def per(m): return 'SEL' if m <= pd.Period('2018-12','M') else ('VAL' if m <= pd.Period('2023-12','M') else 'HOLD')
def run(days, ab):
    pl = C.make_plan(days, 10, ab); t0 = C.POS[pl[0][3]]
    nav, lg = C.tsim([pl], *C.NORMAL, t0, C.TEND)
    sd = [p[3] for p in pl] + [C.END]; ex = C.excess(nav, sd)
    am = {p[3]: p[2] for p in pl}; P = np.array([per(am[d]) for d in ex.index])
    return round(C.cagr_mdd(nav, nav.index[0], C.END)[0], 2), round(C.tstat(ex), 3), round(C.tstat(ex[P=='SEL']), 3), round(C.tstat(ex[P=='HOLD']), 3)
class ABV:
    def __init__(self, seed):
        self.U = V.make_unavail('mc', seed); self.uid = 1000 + seed
    def __call__(self, m, s): return self.U(s)
j0 = [(m, C.jday(m, 0)) for m in C.ANCHORS]
for s in range(5): print('J0 v2-style seed', s, run(j0, ABV(s)))
off = [(m, C.OFFICIAL[m]) for m in C.ANCHORS]
print('official noAB', run(off, None))
for s in range(5): print('official AB', s, run(off, C.AB(s)))
