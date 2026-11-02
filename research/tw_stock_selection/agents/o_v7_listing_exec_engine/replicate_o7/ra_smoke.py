import sys, time; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
import r_env, numpy as np, pandas as pd
t=time.time(); L = r_env.load(); K, V = L.K, L.V
XR = L.setup_px(); YFX, Y5F = L.load_rfix(); GAP = L.gap_dict(); DMT = L.disp_mask_fn('trade')
print('load', time.time()-t)
S = L.V4(YFX, Y5F, DMT); t0, t1, sd = S.bounds(); pl = S.plan(10, 'official')
print('plan', time.time()-t, len(sd), K.IDX[t0], K.IDX[t1])
nav,_ = L.sim_plan(pl, t0, t1, 'normal', XR); print('sim', time.time()-t)
ew,_ = K.simulate([(p[0], p[2]) for p in pl], 0, 0, t0, t1)
print(L.seg(nav, ew, sd, '2006-04-01','2026-10-12'))
R = pd.read_parquet(Path(__file__).resolve().parent.parent/'navs_v7.parquet')
x = R['V7|normal'].dropna(); y = nav.reindex(x.index)
print('maxdiff vs researcher', float((x/x.iloc[0]-y/y.iloc[0]).abs().max()), x.index[0], nav.index[0])
