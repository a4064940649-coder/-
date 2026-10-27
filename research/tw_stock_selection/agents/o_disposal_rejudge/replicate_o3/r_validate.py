import sys; sys.path.insert(0, "/home/user/-/research/tw_stock_selection/agents/o_disposal_rejudge/replicate_o3")
import time; t=time.time()
import numpy as np, pandas as pd
import r_core as R
print('load', time.time()-t, len(R.SIG), R.SIG[0], R.END)
pl = R.plan('E1', 10, official_mask=True)
nav = R.simulate(pl)
ref = pd.read_parquet(R.MV2/'navs_gapfix.parquet')['EXFIN|n10|normal'].dropna()
c = nav.index.intersection(ref.index)
print('n', len(c), 'max rel diff', float((nav[c]/ref[c]-1).abs().max()), nav.iloc[-1], ref[c].iloc[-1])
print(R.cagr(nav,'all'))
print('time', time.time()-t)
