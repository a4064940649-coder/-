import sys
import numpy as np, pandas as pd
sys.path.insert(0, '/home/user/-/research/tw_stock_selection/agents/o_capacity_tsmc_gap/replicate_o1')
import rcore as R
ADV20=R.V.rolling(20,min_periods=1).mean()
ds=[d for d in R.SIG if d.year==2026]
cr={d:R.Cross(d) for d in ds}
for nm,A in [('adv20',ADV20),('v60',R.V60)]:
  for sub in ('all2026','last','1012'):
    dd = ds if sub=='all2026' else ([ds[-2]] if sub=='last' else [ds[-1]])
    res={}
    for lo,hi in [(1,50),(101,200),(201,300)]:
        v=[]; vt=[]
        for d in dd:
            c=cr[d]; a=A.loc[d].to_numpy(float)
            g=np.where(c.el&(c.vrank>=lo)&(c.vrank<=hi))[0]; v+=list(2.8e9/(0.1*a[g]))
            t=np.where(np.isfinite(c.vrank)&(c.vrank>=lo)&(c.vrank<=hi))[0]; vt+=list(2.8e9/(0.1*a[t]))
        res[f'{lo}-{hi}']=(round(np.median(v),1), round(np.median(vt),1), round(2.8e9/(0.1*np.mean([x for x in [2.8e9/(0.1*y) for y in v]])),1) if False else round(2.8e9/(0.1*np.median(2.8e9/(0.1*np.array(v)))),1))
    print(nm, sub, [d.date() for d in dd][:1], res)
