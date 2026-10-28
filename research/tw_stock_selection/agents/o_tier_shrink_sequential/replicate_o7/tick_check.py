import sys, time, json; sys.path.insert(0,'/home/user/-/research/tw_stock_selection/agents/o_tier_shrink_sequential/replicate_o7')
import numpy as np, pandas as pd
import ecore_r as E
D=E.Data(); spec=json.load(open(E.RAW/'model_spec.json')); M=E.ExecModel(spec)
CK=pd.read_parquet(E.RAW/'navs_main_check_slice.parquet')
t=time.time()
nav,_=E.simulate(D,0,tier=1e6,version='FIX',fee='A',model=M)
print('time',time.time()-t, 'diff', np.abs(nav.values-CK['100萬|FIX|A|s00'].reindex(nav.index).values).max())
t=time.time()
nav2,_=E.simulate(D,0,tier=1e6,version='FIX',fee='A',model=M,flags=dict(impact='tick'))
print('time',time.time()-t, nav.iloc[-1], nav2.iloc[-1])
