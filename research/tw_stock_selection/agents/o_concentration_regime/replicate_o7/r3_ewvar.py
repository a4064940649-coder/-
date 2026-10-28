import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd, numpy as np
from rlib import *
P='/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel'
nav,e,plan,sig,ew=load()
val=pd.read_parquet(f'{P}/value.parquet'); cl=pd.read_parquet(f'{P}/close_raw.parquet'); adj=pd.read_parquet(f'{P}/adj.parquet'); op=pd.read_parquet(f'{P}/open.parquet')
cnt=cl.notna().sum(axis=1); good=cnt.index[cnt>=300]
avg=val.loc[good].rolling(60,min_periods=50).mean(); cg=cl.loc[good]
dd=pd.DatetimeIndex(plan.signal.tolist()+[pd.Timestamp('2026-10-12')])
ex=pd.DatetimeIndex(plan['exec'].tolist()+[pd.Timestamp('2026-10-13')])
adjf=adj.ffill()
# open adj = adj * open/close
oadj=adj*op/cl
res={}
for i in range(len(dd)-1):
    T,T2=dd[i],dd[i+1]; X,X2=ex[i],ex[i+1]
    a=avg.loc[T]; c=cg.loc[T]; pool=a[(a>=1e7)&(c>=10)].dropna().index
    o={}
    o['exec_close']=(adjf.loc[X2]/adj.loc[X]-1)[pool].dropna().mean() if X2 in adj.index else np.nan
    o['open_open']=(oadj.ffill().loc[X2]/oadj.loc[X]-1)[pool].dropna().mean() if X2 in adj.index else np.nan

    # daily-rebalanced equal weight within T..T2 (mean of daily returns compounded)
    rr=adj.loc[T:T2,pool].pct_change().iloc[1:]
    o['daily_rebal']=np.prod(1+rr.mean(axis=1))-1
    res[T]=o
x=pd.DataFrame(res).T
x=pd.concat([x,ew],axis=1,sort=True)
for a,b in [('2019-01-01','2022-12-31'),('2023-01-01','2026-09-30')]:
    m=(x.index>=a)&(x.index<=b)
    print(a,{c:round(ann(x[c][m]),4) for c in x.columns})
x.to_csv(f'{OUT}/ew_variants.csv')
