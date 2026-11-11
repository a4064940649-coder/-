import sys; sys.path.insert(0,"/home/user/-/research/tw_stock_selection/agents/o_turnover_fixedcost_diagnosis/replicate_o20")
from lib import *
sl=pd.read_csv(D+'slots_100n_REAL_opt_seed0.csv.gz',dtype={'code':str}); sl['signal']=pd.to_datetime(sl.signal)
for band in ['contrib','contrib_raw']:
  m_all=(sl[band].abs()<=0.02); print(band,'share all',m_all.mean().round(3),'Y19',m_all[sl.signal>='2019-01-01'].mean().round(3))
  y=sl[sl.signal>='2019-01-01']
  rows=[]
  for j,q in y.groupby('j'):
    yr=years('Y19on',j); m=q[band].abs()<=0.02
    rows.append({k:q[k][m].sum()/yr*100 for k in ['cost_pct','contrib','contrib_raw']}|{'net_c':(q.contrib-q.cost_pct)[m].sum()/yr*100,'netraw':(q.contrib_raw-q.cost_pct)[m].sum()/yr*100})
  print(pd.DataFrame(rows).median().round(2).to_dict())
