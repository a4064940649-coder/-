import sys; sys.path.insert(0,"/home/user/-/research/tw_stock_selection/agents/o_turnover_fixedcost_diagnosis/replicate_o20")
from swaps import *
E=z['E50']; nav=pd.read_parquet(D+'navs_100n_avg20.parquet'); nav.index=pd.to_datetime(nav.index)
sl=pd.read_csv(D+'slots_100n_REAL_opt_seed0.csv.gz',dtype={'code':str})
rb=rebs('REAL_opt').set_index(['j','k'])
g=sl.groupby(['j','k']).agg(rmean=('r','mean'),csum=('contrib_raw','sum'),c2=('contrib','sum'),n=('r','size'),e=('e','first'),signal=('signal','first')).reset_index()
g['e2']=[rb.loc[(j,k),'e2'] for j,k in zip(g.j,g.k)]
print('slots Y19',(pd.to_datetime(g.signal)>='2019-01-01').sum(), sl[pd.to_datetime(sl.signal)>='2019-01-01'].shape)
for mode in ['e','e-1']:
  g['rm']=[E[e2]/E[e if mode=='e' else e-1]-1 for e,e2 in zip(g.e,g.e2)]
  for col in ['rmean','csum']:
    bs=[];bf=[]
    for j,q in g.groupby('j'):
      s=q[pd.to_datetime(q.signal)<='2019-04-30']; bs.append(np.polyfit(s.rm,s[col],1)[0]); bf.append(np.polyfit(q.rm,q[col],1)[0])
    print(mode,col,round(np.median(bs),3),round(min(bs),3),round(max(bs),3),'full',round(np.median(bf),3))
  # nav windows (20-seed avg) close e-1 -> e2-1 ?
  for sh in [0,1]:
    bs=[]; bf=[]
    for j,q in g.groupby('j'):
      s=nav[f'REAL_opt|{j}']; nv=s.reindex(IDX).values
      rp=np.array([nv[e2-sh]/nv[e-sh]-1 if np.isfinite(nv[e-sh]) else np.nan for e,e2 in zip(q.e,q.e2)])
      rm=np.array([E[e2-sh]/E[e-sh]-1 for e,e2 in zip(q.e,q.e2)])
      ok=np.isfinite(rp); sel=ok&(pd.to_datetime(q.signal)<='2019-04-30').values
      bs.append(np.polyfit(rm[sel],rp[sel],1)[0]); bf.append(np.polyfit(rm[ok],rp[ok],1)[0])
    print('nav shift',sh,round(np.median(bs),3),round(min(bs),3),round(max(bs),3),'full',round(np.median(bf),3))
