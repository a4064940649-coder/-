import sys; sys.path.insert(0,"/home/user/-/research/tw_stock_selection/agents/o_turnover_fixedcost_diagnosis/replicate_o20")
from lib import *
z=np.load(B+'o_exec_engine_rewrite_v7x/data/x_cache.npz',allow_pickle=True); IDX=pd.DatetimeIndex(z['IDX'])
r=pd.read_csv(D+'rebs_allcells_seed0.csv.gz',dtype={'code':str})
print(r.cell.value_counts())
r['edate']=IDX[r.e.values]
# check vs tradelog sums per (j,k)
t=load_tl('100n'); g=t.groupby(['j','k'])[['cost','tax','fee','tick','imp','odd']].sum()
rr=r[(r.cell=='normal|100萬')&(r.variant=='REAL_opt')].set_index(['j','k'])
m=g.join(rr[['cost','tax','fee','tick_amt','imp_amt','odd_amt']],rsuffix='_r')
print('max abs diff cost',(m.cost-m.cost_r).abs().max(), (m.tick-m.tick_amt).abs().max(),(m.imp-m.imp_amt).abs().max())
# drag by cell by e-date approximation
res={}
for (cell,v),gg in r.groupby(['cell','variant']):
  for per in ['Y19on','POSTODD','FULL']:
    a,b=PER[per]; x=gg[(gg.edate>=a)&(gg.edate<=b)]
    d=x.groupby('j').apply(lambda q:(q.cost/q.avail).sum()/years(per,q.name)*100)
    res[(cell,v,per)]=round(d.median(),3)
s=pd.Series(res).unstack(1); print(s.to_string())
