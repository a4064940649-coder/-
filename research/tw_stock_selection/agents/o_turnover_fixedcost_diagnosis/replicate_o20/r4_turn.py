import sys; sys.path.insert(0,"/home/user/-/research/tw_stock_selection/agents/o_turnover_fixedcost_diagnosis/replicate_o20")
from lib import *
t=load_tl('100n'); t['nv']=t.notional  # use notional
for basis in ['notional','pxsh']:
  t['x']=t[basis]/t.avail
  x=t[t.date>='2019-01-01']
  rows=[]
  for j,g in x.groupby('j'):
    y=years('Y19on',j); s=g.groupby('cls').x.sum()/y
    rows.append(s.rename(j))
  d=pd.DataFrame(rows); d['sell']=d.exit+d.rw_sell; d['buy']=d.entry+d.rw_buy; d['both']=d.sell+d.buy; d['single']=d.both/2; d['rw_avg']=(d.rw_sell+d.rw_buy)/2; d['ex_avg']=(d.exit+d.entry)/2
  print(basis,d.median().round(3).to_dict())
r=pd.read_csv(D+'rebs_allcells_seed0.csv.gz'); r=r[(r.cell=='normal|100萬')&(r.variant=='REAL_opt')&(r.signal>='2019-01-01')]
print('n_exits mean',r.n_exits.mean(), 'median by j', r.groupby('j').n_exits.mean().median(), 'n_reb',len(r))
