import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import rcore as C, numpy as np, pandas as pd
def run(days, ab):
    pl = C.make_plan(days, 10, ab); t0 = C.POS[pl[0][3]]
    nav, lg = C.tsim([pl], *C.NORMAL, t0, C.TEND)
    sd = [p[3] for p in pl] + [C.END]; ex = C.excess(nav, sd)
    return round(C.cagr_mdd(nav, nav.index[0], C.END)[0], 2), round(C.tstat(ex), 3), len(pl)
off = [(m, C.OFFICIAL[m]) for m in C.ANCHORS]
n_eq = sum(C.OFFICIAL[m] == C.deadline(m) for m in C.ANCHORS); print('official==D', n_eq, 'official<D', sum(C.OFFICIAL[m] < C.deadline(m) for m in C.ANCHORS))
for s in range(5): print('official AB seed', s, run(off, C.AB(s)))
j0 = [(m, C.jday(m, 0)) for m in C.ANCHORS]
print('J0 noAB', run(j0, None))
for s in range(6): print('J0 seed', s, run(j0, C.AB(s)))
for j in (-6, -1, 1):
    dd = [(m, C.jday(m, j)) for m in C.ANCHORS]
    pl = C.make_plan(dd, 10, C.AB(0)); got = {p[2] for p in pl}
    print(j, 'skipped', [(str(m), str(C.jday(m, j).date()), C.jday(m,j) in C.POS) for m in C.ANCHORS if m not in got])
