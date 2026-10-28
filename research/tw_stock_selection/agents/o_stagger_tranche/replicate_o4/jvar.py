import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import rcore as C, numpy as np, pandas as pd
def per(m): return 'SEL' if m <= pd.Period('2018-12','M') else ('VAL' if m <= pd.Period('2023-12','M') else 'HOLD')
def run(j, abf=None):
    days = [(m, C.jday(m, j)) for m in C.ANCHORS]
    pl = C.make_plan(days, 10, abf or C.AB(0))
    wd = [(m, s) for m, s in days if s in C.POS]
    nx = C.jday(pd.Period('2026-10', 'M'), j); e_ = nx if (nx is not None and nx in C.POS) else C.IDX[-1]
    t0 = C.POS[wd[0][1]]; t1 = C.POS[e_]
    nav, lg = C.tsim([pl], *C.NORMAL, t0, t1)
    sd = [s for _, s in wd] + [e_]; ex = C.excess(nav, sd)
    am = {s: m for m, s in wd}; P = np.array([per(am[d]) for d in ex.index])
    return round(C.cagr_mdd(nav, nav.index[0], e_)[0], 3), round(C.tstat(ex), 4), round(C.tstat(ex[P=='VAL']), 4), len(pl)
tgt = {-8: (20.6766, 1.2336), -7: (18.1562, .8629), -1: (20.1981, 1.3218), 2: (22.2232, 1.871), 3: (24.3927, 2.1856), 12: (17.5995, .9134)}
for mode in ('month', 'signal', 'anchor'):
    C.REVRULE['mode'] = mode; C._CS.clear()
    for j in tgt: print(mode, j, tgt[j], run(j))
