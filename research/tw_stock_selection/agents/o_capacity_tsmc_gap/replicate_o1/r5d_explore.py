import sys
import numpy as np, pandas as pd
sys.path.insert(0, '/home/user/-/research/tw_stock_selection/agents/o_capacity_tsmc_gap/replicate_o1')
import rcore as R
from rstat import PER
def pm(idx,p): return (idx>=pd.Timestamp(PER[p][2]))&(idx<=pd.Timestamp(PER[p][3]))
for n in (10,30):
  E=pd.read_pickle(R.HERE/f'events_n{n}.pkl')
  for m in ('oo','cc','oc'):
    for gcol in ('g4','g2'):
      for base in ('net','gross'):
        E['xg']=E[base]-E[f'{gcol}_{m}']
        M=E.groupby(['d','g'])['xg'].mean().unstack(); x=(M['A']-M['B']).dropna()
        print(n,m,gcol,base,{p:(round(x[pm(x.index,p)].mean()*100,3), round(R.tstat(x[pm(x.index,p)]),3)) for p in ('SEL','VAL','ALL')})
