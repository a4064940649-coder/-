import sys; sys.path.insert(0,"/home/user/-/research/tw_stock_selection/agents/o_turnover_fixedcost_diagnosis/replicate_o20")
from swaps import *
E=z['E50']; nav=pd.read_parquet(D+'navs_100n_avg20.parquet'); nav.index=pd.to_datetime(nav.index); e50=pd.Series(E,IDX)
c=pd.read_csv(D+'shapley_coalitions.csv.gz',keep_default_na=False)
def cg(x,a):
    x=x[x.index>=a]; y=(x.index[-1]-x.index[0]).days/365.25; return ((x.iloc[-1]/x.iloc[0])**(1/y)-1)*100
for var in ['REAL_opt','REAL_p90','REAL_carry']:
  rb=rebs(var); g19=c[(c.tier=='100萬')&(c.variant==var)&(c.coalition=='T')].set_index('j'); rr=[]
  for j,q in rb.groupby('j'):
    s=nav[f'{var}|{j}']; nv=s.reindex(IDX).values
    rp=np.array([nv[e2-1]/nv[e-1]-1 for e,e2 in zip(q.e,q.e2)]); rm=np.array([E[e2-1]/E[e-1]-1 for e,e2 in zip(q.e,q.e2)])
    ok=np.isfinite(rp); sel=ok&(pd.to_datetime(q.signal)<='2019-04-30').values
    b=np.polyfit(rm[sel],rp[sel],1)[0]; bf=np.polyfit(rm[ok],rp[ok],1)[0]
    s=s.dropna(); s=s[s.index<=JEND[j]]; r50=e50.reindex(s.index).pct_change().fillna(0)
    bn=(1+b*r50).cumprod(); a19=s[s.index>='2019-01-01'].index[0]
    rr.append(dict(b=b,bf=bf,net19=cg(s,a19)-cg(bn,a19),gross19=g19.loc[j,'CAGR_Y19on']-cg(bn,a19),netF=cg(s,s.index[0])-cg(bn,s.index[0]),grossF=g19.loc[j,'CAGR']-cg(bn,s.index[0])))
  print(var,pd.DataFrame(rr).median().round(3).to_dict())
# gross vs 0050
x=c[(c.tier=='100萬')&(c.variant=='REAL_opt')&(c.coalition=='T')]; print('gross19 median',x.CAGR_Y19on.median(),'-29.492=',x.CAGR_Y19on.median()-29.492)
# P6
sl=pd.read_csv(D+'slots_100n_REAL_opt_seed0.csv.gz',dtype={'code':str}); sl['signal']=pd.to_datetime(sl.signal)
inr=(sl.contrib>=-0.02)&(sl.contrib<=0.02); print('P6 share in band all',inr.mean().round(3))
y=sl[sl.signal>='2019-01-01']; iy=(y.contrib>=-0.02)&(y.contrib<=0.02); print(' Y19 share',iy.mean().round(3))
rows=[]
for j,q in y.groupby('j'):
    yr=years('Y19on',j); m=(q.contrib>=-0.02)&(q.contrib<=0.02)
    rows.append(dict(cost=q.cost_pct[m].sum()/yr*100,contrib=q.contrib[m].sum()/yr*100,contrib_raw=q.contrib_raw[m].sum()/yr*100,costall=q.cost_pct.sum()/yr*100))
print(pd.DataFrame(rows).median().round(2).to_dict())
