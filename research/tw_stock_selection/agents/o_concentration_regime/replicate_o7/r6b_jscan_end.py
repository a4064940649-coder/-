import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd, numpy as np
from rlib import *
P='/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel'
nav,e,plan,sig,ew=load()
# same-pool EW check from navs_v4 EW column
for c in ('AB_V4|EW','V4|EW'):
    s=nav[c]
    for a,b in (('2019-01-11','2023-01-11'),('2023-01-11','2026-10-12')):
        A=s.index[s.index>=a][0]; B=s.index[s.index<=b][-1]; y=(B-A).days/365.25
        print(c,a,'ann %.4f'%((s[B]/s[A])**(1/y)-1))
# ---- j-scan
cl=pd.read_parquet(f'{P}/close_raw.parquet'); cnt=cl.notna().sum(axis=1); good=pd.DatetimeIndex(cnt.index[cnt>=300])
top5=sig.top5  # top5 on base decision days only -> need top5 on any day: recompute
val=pd.read_parquet(f'{P}/value.parquet')
RAWD=RAW
sl=pd.read_csv(f'{RAWD}/stock_list.csv',dtype=str); dl=pd.read_csv(f'{RAWD}/delist_official.csv',dtype=str)
names=pd.concat([sl[['code','name']],dl[['code','name']]])
ky=set(names.code[names.name.str.contains('KY|DR',na=False)])|{c for c in val.columns if c.startswith('91')}
keep=[c for c in val.columns if c not in ky]
avg=val.loc[good,keep].rolling(60,min_periods=50).mean(); cg=cl.loc[good,keep]
def t5(T):
    a=avg.loc[T]; c=cg.loc[T]; ok=a[(a>=1e7)&(c>=10)].dropna().sort_values(ascending=False).iloc[:300]
    return ok.iloc[:5].sum()/ok.sum()
J=pd.read_parquet(f'{RAW}/navs_jscan_v6.parquet')
g=pd.Series(good,index=good); gpos={d:i for i,d in enumerate(good)}
base=g[g.dt.day>=10].groupby(g[g.dt.day>=10].dt.to_period('M')).min()
re_full=e.reindex(e.index.union(J.index)).ffill()
rows=[]
for j in range(13):
    nj=J[f'AB_V6|normal|J{j}'].dropna(); nj=nj[nj.index<=pd.Timestamp(os.environ.get('JEND','2026-12-31'))]
    dec=pd.DatetimeIndex([good[gpos[d]+j] for d in base.values if gpos[d]+j<len(good)])
    assert dec[(dec>=nj.index[0])][0]==nj.index[0], (j,dec[:3],nj.index[0])
    idx=nj.index
    dec=dec[(dec>=idx[0])&(dec<=idx[-1])]
    S2=pd.Series([t5(T) for T in dec.union(pd.DatetimeIndex(base.index.to_timestamp()).intersection([]))],index=dec)
    # need 6 lags earlier: compute top5 for decision dates before nav start too
    allde=pd.DatetimeIndex([good[gpos[d]+j] for d in base.values if gpos[d]+j<len(good)])
    allde=allde[allde<=idx[-1]]
    tt=pd.Series([t5(T) for T in allde],index=allde); s2=tt-tt.shift(6)
    w=(s2>0).astype(float).reindex(dec)
    ex=[idx[idx.get_loc(T)+1] if idx.get_loc(T)+1<len(idx) else None for T in dec]
    pl=pd.DataFrame({'signal':dec,'exec':ex}).dropna()
    rm=nj; re=re_full.reindex(idx)
    end=list(dec)+[idx[-1]] if dec[-1]<idx[-1] else list(dec)
    dd=pd.DatetimeIndex(end)
    navR,_=simulate(rm,re,pl,w.reindex(pl.signal),COST['normal'])
    R=window_returns(navR,dd); R0=window_returns(re,dd).reindex(R.index)
    m=(R.index>='2023-01-01')&(R.index<='2026-09-30')
    wb=w.reindex(R.index)[m].mean()
    navS,_=simulate(rm,re,pl,pd.Series(wb,index=pl.signal),COST['normal']); S=window_returns(navS,dd).reindex(R.index)
    navF,_=simulate(rm,re,pl,pd.Series(0.4792,index=pl.signal),COST['normal']); F=window_returns(navF,dd).reindex(R.index)
    ws=w.reindex(R.index)[m]
    rows.append(dict(j=j,n=int(m.sum()),first=str(R.index[m][0].date()),last_end=str(dd[-1].date()),vsS=(R-S)[m].mean()*100,vsF=(R-F)[m].mean()*100,vs0=(R-R0)[m].mean()*100,sw=int((ws.diff().abs()>0).sum())))
df=pd.DataFrame(rows); print(df.round(3).to_string())
print('median vsS %.3f (pos %d/13) vsF %.3f vs0050 %.3f (pos %d/13)'%(df.vsS.median(),(df.vsS>0).sum(),df.vsF.median(),df.vs0.median(),(df.vs0>0).sum()))
