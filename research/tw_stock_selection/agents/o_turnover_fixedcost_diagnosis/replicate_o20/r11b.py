import sys; sys.path.insert(0,"/home/user/-/research/tw_stock_selection/agents/o_turnover_fixedcost_diagnosis/replicate_o20")
from lib import *
sl=pd.read_csv(D+'slots_100n_REAL_opt_seed0.csv.gz',dtype={'code':str}); sl['signal']=pd.to_datetime(sl.signal)
y=sl[sl.signal>='2019-01-01']
for band in ['contrib','contrib_raw']:
  m=y[band].abs()<=0.02; ty=sum(years('Y19on',j) for j in range(13))
  print(band,'pooled cost',y.cost_pct[m].sum()/ty*100,'contrib_raw',y.contrib_raw[m].sum()/ty*100,'contrib',y.contrib[m].sum()/ty*100)
  rows=[(q.cost_pct[q[band].abs()<=0.02].sum()/years('Y19on',j)*100, q.contrib_raw[q[band].abs()<=0.02].sum()/years('Y19on',j)*100, q.contrib[q[band].abs()<=0.02].sum()/years('Y19on',j)*100) for j,q in y.groupby('j')]
  print(' mean of j',np.mean(rows,axis=0).round(2))
