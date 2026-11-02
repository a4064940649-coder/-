"""Run all j for T1/T2 and write per-window series."""
import sys, time, pickle
sys.path.insert(0, "/home/user/-/research/tw_stock_selection/agents/o_0050_tilt_pit/replicate_o8")
import numpy as np, pandas as pd
from rcore import *

import os
W = World(fix4763=os.environ.get('FIX4763') == '1')
pre = W.build_pre()
pickle.dump(pre, open(OUT + '/pre_membership.pkl', 'wb'))
NP = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
rows = []
js = list(range(13)) + ['off']
for j in js:
    t0 = time.time()
    sd = W.official_days('2014-01', '2026-10') if j == 'off' else W.signal_days(j, '2014-01', '2026-10')
    pers = list(sd.keys()); dates = [sd[p] for p in pers]
    for ver in ('T1', 'T2'):
        WS = [W.weights(d, p, ver, perms=NP, seed=7000 + (j if j != 'off' else 99)) for d, p in zip(dates, pers)]
        res = {}
        for k, cost in (('T', (CB, CS)), ('B', (CB, CS)), ('T_st', (CST, CST)), ('B_st', (CST, CST))):
            res[k] = W.run([w[k[0]] for w in WS], dates, cost)
        for t, p in enumerate(pers[:len(res['T'])]):
            d0 = res['T'][t]['d']; d1 = res['T'][t]['d1']
            ws = WS[t]
            rows.append(dict(j=str(j), ver=ver, per=str(p), d=d0, d1=d1,
                             T=res['T'][t]['R'], B=res['B'][t]['R'], T_st=res['T_st'][t]['R'], B_st=res['B_st'][t]['R'],
                             turn=res['T'][t]['turn'], cost=res['T'][t]['cost'],
                             T_g=W.window_static(ws['T'], d0, d1), B_g=W.window_static(ws['B'], d0, d1),
                             P_g=W.window_static(ws['P'], d0, d1) if 'P' in ws else np.nan,
                             E50=W.r0050(d0, d1)))
    print(j, round(time.time() - t0, 1), flush=True)
    pd.DataFrame(rows).to_parquet(OUT + '/' + (sys.argv[2] if len(sys.argv) > 2 else 'series.parquet'))
pd.DataFrame(rows).to_parquet(OUT + '/' + (sys.argv[2] if len(sys.argv) > 2 else 'series.parquet'))
print('done')
