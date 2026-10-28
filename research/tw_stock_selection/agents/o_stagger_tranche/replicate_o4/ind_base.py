import sys, pathlib, time; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
t_ = time.time()
import ind_core as C, numpy as np, pandas as pd
print('load', round(time.time() - t_, 1), 'anchors', len(C.ANCH), C.ANCH[0], C.ANCH[-1], 'npatch', C.NPATCH, 'X', len(C.XR))
# 官方日相對 D_m 的位置
pos = {}
for p in C.ANCH:
    d = C.OFF[p]; a, b = C.TX.get_loc(C.Dm(p)), C.TX.get_loc(d); pos[b - a] = pos.get(b - a, 0) + 1
print('official j counts', pos)
days = [(p, C.OFF[p]) for p in C.ANCH]
pl = C.plan(days, None)
t0 = C.POS[days[0][1]]
nav, lg = C.tsim([pl], *C.NORMAL, t0, C.TEND)
ex = C.exw(nav, C.OFFSD)
print('ALL', C.cagr(nav), 't', C.tst(ex), len(ex), 'turn', lg['turnover'])
nav2, _ = C.O3.simulate([(q['e'], q['pk']) for q in pl], *C.NORMAL, t0, C.TEND, xrules=C.XR)
print('max rel diff vs o3_core.simulate', float((nav / nav2 - 1).abs().max()), 'daily ret diff', float((nav.pct_change() - nav2.pct_change()).abs().max()))
for s in range(5):
    pa = C.plan(days, C.AB(s)); na, _ = C.tsim([pa], *C.NORMAL, t0, C.TEND)
    print('official AB seed', s, C.cagr(na)[0], C.tst(C.exw(na, C.OFFSD)))
print(time.time() - t_)
