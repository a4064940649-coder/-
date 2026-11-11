import sys; sys.path.insert(0,"/home/user/-/research/tw_stock_selection/agents/o_turnover_fixedcost_diagnosis/replicate_o20")
from swaps import *
E=z['E50']
nav=pd.read_parquet(D+'navs_100n_avg20.parquet'); nav.index=pd.to_datetime(nav.index)
e50=pd.Series(E,IDX)
c=pd.read_csv(D+'shapley_coalitions.csv.gz',keep_default_na=False)
sl=pd.read_csv(D+'slots_100n_REAL_opt_seed0.csv.gz',dtype={'code':str})
for var in ['REAL_opt','REAL_p90','REAL_carry']:
  rb=rebs(var)
  rb['rp']=[np.mean([winret(cd,e,e2) for cd in p]) for p,e,e2 in zip(rb.picks,rb.e,rb.e2)]
  rb['ws']=IDX[rb.e.values]
  out=[]
  for mode in ['e','e-1']:
    rb['rm']=[E[e2]/E[e if mode=='e' else e-1]-1 for e,e2 in zip(rb.e,rb.e2)]
    bs=[];bf=[]
    for j,g in rb.groupby('j'):
      s=g[pd.to_datetime(g.signal)<= '2019-04-30']; bs.append(np.polyfit(s.rm,s.rp,1)[0]); bf.append(np.polyfit(g.rm,g.rp,1)[0])
    out.append((mode,round(np.median(bs),3),round(min(bs),3),round(max(bs),3),round(np.median(bf),3)))
    if mode=='e': BS=bs; BF=bf
  print(var,out)
  # beta-adjusted
  g19=c[(c.tier=='100萬')&(c.variant==var)&(c.coalition=='T')].set_index('j')
  gf=g19
  rr=[]
  for j in range(13):
    s=nav[f'{var}|{j}'].dropna(); s=s[s.index<=JEND[j]]
    r50=e50.reindex(s.index).pct_change()
    def cg(x,a):
      x=x[x.index>=a]; y=(x.index[-1]-x.index[0]).days/365.25; return ((x.iloc[-1]/x.iloc[0])**(1/y)-1)*100
    a19=s[s.index>='2019-01-01'].index[0]
    for bnm,bb in [('sel',BS[j]),('full',BF[j])]:
      bn=(1+bb*r50.fillna(0)).cumprod()
      rr.append(dict(j=j,b=bnm,net19=cg(s,a19)-cg(bn,a19),gross19=g19.loc[j,'CAGR_Y19on']-cg(bn,a19),netF=cg(s,s.index[0])-cg(bn,s.index[0]),grossF=g19.loc[j,'CAGR']-cg(bn,s.index[0])))
  d=pd.DataFrame(rr); print(d.groupby('b').median().round(2))
