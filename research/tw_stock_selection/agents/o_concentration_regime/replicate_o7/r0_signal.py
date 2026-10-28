# Independent replication: decision dates, top5 concentration, S2, EW pool returns
import pandas as pd, numpy as np, json, os
P='/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel'
RAW='/home/user/-/research/tw_stock_selection/agents/o_concentration_regime/raw'
OUT=os.path.dirname(os.path.abspath(__file__))
val=pd.read_parquet(f'{P}/value.parquet'); cl=pd.read_parquet(f'{P}/close_raw.parquet'); adj=pd.read_parquet(f'{P}/adj.parquet')
cnt=cl.notna().sum(axis=1)
good=cnt.index[cnt>=300]
print('bad days', list(cnt.index[cnt<300].date))
# decision dates: first GOOD day with day>=11 per month
g=pd.Series(good,index=good)
dd=g[g.dt.day>=11].groupby(g[g.dt.day>=11].dt.to_period('M')).min()
dd=pd.DatetimeIndex(dd.values)
plan=pd.read_csv(f'{RAW}/plan_V4.csv',parse_dates=['signal','exec'])
ps=set(plan.signal)
sub=dd[(dd>=plan.signal.min())&(dd<=plan.signal.max())]
print('n dd in plan range',len(sub),'plan',len(plan),'mismatch',sorted(set(sub)^ps))
# KY/DR
sl=pd.read_csv(f'{RAW}/stock_list.csv',dtype=str); dl=pd.read_csv(f'{RAW}/delist_official.csv',dtype=str)
names=pd.concat([sl[['code','name']],dl[['code','name']]])
ky=set(names.code[names.name.str.contains('KY',na=False)|names.name.str.contains('DR',na=False)])
ky|={c for c in val.columns if c.startswith('91')}
print('KY/DR in panel', len([c for c in val.columns if c in ky]))
vg=val.loc[good]; 
avg=vg.rolling(60,min_periods=50).mean()
cg=cl.loc[good]
keep=[c for c in val.columns if c not in ky]
rows=[]
for T in dd:
    a=avg.loc[T,keep]; c=cg.loc[T,keep]
    ok=a[(a>=1e7)&(c>=10)].dropna().sort_values(ascending=False)
    top=ok.iloc[:300]
    rows.append(dict(T=T,n_ok=len(ok),n=len(top),top5=top.iloc[:5].sum()/top.sum() if len(top) else np.nan))
s=pd.DataFrame(rows).set_index('T')
# also full liquid pool (incl KY) for EW
s['S2']=s.top5-s.top5.shift(6)
s.to_csv(f'{OUT}/signal.csv')
print(s.loc['2018-06':'2019-03']); print(s.tail(8))
