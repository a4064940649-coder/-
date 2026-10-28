# EW same-pool window returns (T close -> next T close), variants
import pandas as pd, numpy as np, os
P='/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel'
RAW='/home/user/-/research/tw_stock_selection/agents/o_concentration_regime/raw'
OUT=os.path.dirname(os.path.abspath(__file__))
val=pd.read_parquet(f'{P}/value.parquet'); cl=pd.read_parquet(f'{P}/close_raw.parquet'); adj=pd.read_parquet(f'{P}/adj.parquet')
cnt=cl.notna().sum(axis=1); good=cnt.index[cnt>=300]
sig=pd.read_csv(f'{OUT}/signal.csv',parse_dates=['T'],index_col='T')
dd=sig.index
avg=val.loc[good].rolling(60,min_periods=50).mean(); cg=cl.loc[good]
sl=pd.read_csv(f'{RAW}/stock_list.csv',dtype=str); dl=pd.read_csv(f'{RAW}/delist_official.csv',dtype=str)
names=pd.concat([sl[['code','name']],dl[['code','name']]])
ky=set(names.code[names.name.str.contains('KY|DR',na=False)])|{c for c in val.columns if c.startswith('91')}
keep=[c for c in val.columns if c not in ky]
adjf=adj.ffill()
res={}
for i in range(len(dd)-1):
    T,T2=dd[i],dd[i+1]
    a=avg.loc[T]; c=cg.loc[T]
    pool=a[(a>=1e7)&(c>=10)].dropna()
    r=(adjf.loc[T2]/adj.loc[T]-1)
    # delisted inside window: ffill gives last price -> approx
    out={}
    out['pool_all']=r[pool.index].dropna().mean()
    pk=pool[[x for x in pool.index if x in keep]]
    out['pool_exky']=r[pk.index].dropna().mean()
    out['top300']=r[pk.sort_values(ascending=False).index[:300]].dropna().mean()
    res[T]=out
ew=pd.DataFrame(res).T; ew.index.name='T'
ew.to_csv(f'{OUT}/ew.csv'); print(ew.tail())
