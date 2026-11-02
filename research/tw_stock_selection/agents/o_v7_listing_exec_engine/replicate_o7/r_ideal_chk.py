import sys; sys.path.insert(0, '/home/user/-/research/tw_stock_selection/agents/o_v7_listing_exec_engine/replicate_o7')
import pandas as pd, numpy as np
I = pd.read_parquet('../navs_ideal7.parquet'); print(sorted({c.split('|')[1] for c in I.columns})); print(sorted({c.split('|')[0] for c in I.columns}))
NB = pd.read_parquet('out/navs_jscan_b.parquet')
avg = lambda L: (1 + pd.concat([n.pct_change() for n in L], axis=1).mean(1).fillna(0)).cumprod()
for k, kk in [('AB', 'V7'), ('B10', 'V7_B10')]:
    for S in (20, 40):
        a = avg([NB[f'{kk}|j0|normal|s{s}'].dropna() for s in range(S)]); x = I[f'{k}|J0|base'].dropna()
        print(k, S, float((x / x.iloc[0] - a.reindex(x.index) / a.reindex(x.index).iloc[0]).abs().max()), x.index[0], x.index[-1], a.index[0])
