import sys; sys.path.insert(0, "/home/user/-/research/tw_stock_selection/agents/o_disposal_rejudge/replicate_o3")
import json, pickle, time
import numpy as np, pandas as pd
import r_core as R
t = time.time()
VERS = ['E0', 'E1', 'E2', 'E3', 'E3@0.5', 'E3@0.7', 'E5']
PL, NAV = {}, {}
for n in (10, 30):
    for v in VERS:
        PL[(v, n)] = R.plan(v, n)
        NAV[(v, n, 'normal')] = R.simulate(PL[(v, n)])
    print('n', n, time.time() - t, flush=True)
for v in ('E1', 'E5', 'E0'):
    NAV[(v, 10, 'stressX')] = R.simulate(PL[(v, 10)], extra=0.005)
    NAV[(v, 10, 'v2stress')] = R.simulate(PL[(v, 10)], buy=0.005, sell=0.005)
pickle.dump(dict(PL=PL, NAV=NAV), open(R.HERE / 'my_runs.pkl', 'wb'))
print('done', time.time() - t)
