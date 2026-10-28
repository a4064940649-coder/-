#!/usr/bin/env python3
import sys, json, time, pickle
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import r4_core as C
K = C.K
t_ = time.time()
S = C.make_S()
t0, t1, sd = S.bounds()
print('patch', C.NPATCH, 'xr', len(C.XR), 'sig', len(S.sig_days), 'load', round(time.time() - t_, 1), flush=True)

# 驗證排名＝V2.plan
pl_v2 = S.plan(10, 'official')
PL = {'BASE': C.plan_buffer(S, 10, None)}
assert len(pl_v2) == len(PL['BASE'])
for a, b in zip(pl_v2, PL['BASE']):
    assert a[0] == b['e'] and set(a[1]) == set(b['picks'])
print('planner(None)==V2.plan ok', flush=True)
for kb in (15, 20, 25, 30):
    PL[f'A{kb}'] = C.plan_buffer(S, 10, kb)

NORMAL = K.rates(0.001)
# 驗證引擎＝o3_core.simulate
nav_o3, lg = C.O3.simulate([(p[0], p[1]) for p in pl_v2], *NORMAL, t0, t1, xrules=C.XR)
nav_me, _ = C.sim([(p['e'], p['picks']) for p in PL['BASE']], *NORMAL, t0, t1)
print('engine max rel diff vs o3', float((nav_me / nav_o3 - 1).abs().max()), flush=True)
pickle.dump(dict(PL=PL, sd=sd, t0=t0, t1=t1), open(HERE / 'plans.pkl', 'wb'))
