import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd, numpy as np
from rlib import *
P='/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel'
nav,e,plan,sig,ew=load()
val=pd.read_parquet(f'{P}/value.parquet'); cl=pd.read_parquet(f'{P}/close_raw.parquet'); adj=pd.read_parquet(f'{P}/adj.parquet'); op=pd.read_parquet(f'{P}/open.parquet')
cnt=cl.notna().sum(axis=1); good=cnt.index[cnt>=300]
avg=val.loc[good].rolling(60,min_periods=50).mean(); cg=cl.loc[good]
RAWD='/home/user/-/research/tw_stock_selection/agents/o_concentration_regime/raw'
sl=pd.read_csv(f'{RAWD}/stock_list.csv',dtype=str); dl=pd.read_csv(f'{RAWD}/delist_official.csv',dtype=str)
names=pd.concat([sl[['code','name']],dl[['code','name']]])
ky=set(names.code[names.name.str.contains('KY|DR',na=False)])|{c for c in val.columns if c.startswith('91')}
dd=pd.DatetimeIndex(plan.signal.tolist()+[pd.Timestamp('2026-10-12')])
ex=pd.DatetimeIndex(plan['exec'].tolist()+[pd.Timestamp('2026-10-13')])
adjf=adj.ffill(); oadj=(adj*op/cl)
res={}
for i in range(len(dd)-1):
    T,T2=dd[i],dd[i+1]; X,X2=ex[i],ex[i+1]
    a=avg.loc[T]; c=cg.loc[T]; pool=a[(a>=1e7)&(c>=10)].dropna().sort_values(ascending=False)
    t300k=pool.index[:300]; t300=[x for x in pool.index if x not in ky][:300]
    rc=(adjf.loc[T2]/adj.loc[T]-1)
    o={'t300_inclKY':rc[t300k].dropna().mean()}
    if X2 in adj.index:
        ro=(oadj.ffill().loc[X2]/oadj.loc[X]-1); rx=(adjf.loc[X2]/adj.loc[X]-1)
        o['t300_openopen']=ro[t300].dropna().mean(); o['t300_execclose']=rx[t300].dropna().mean()
        # T close -> ... no
    o['t300_noffill']=rc.where(adj.loc[T2].notna())[t300].dropna().mean()
    res[T]=o
x=pd.DataFrame(res).T
for a,b in [('2019-01-01','2022-12-31'),('2023-01-01','2026-09-30'),('2006-05-01','2018-12-31')]:
    m=(x.index>=a)&(x.index<=b)
    print(a,{c:round(ann(x[c][m]),4) for c in x.columns})
