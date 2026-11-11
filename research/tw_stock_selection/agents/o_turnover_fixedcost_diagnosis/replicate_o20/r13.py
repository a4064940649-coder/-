import sys; sys.path.insert(0,"/home/user/-/research/tw_stock_selection/agents/o_turnover_fixedcost_diagnosis/replicate_o20")
from lib import *
z=np.load(B+'o_exec_engine_rewrite_v7x/data/x_cache.npz',allow_pickle=True); IDX=pd.DatetimeIndex(z['IDX']); e50=pd.Series(z['E50'],IDX)
nav=pd.read_parquet(D+'navs_100n_avg20.parquet'); nav.index=pd.to_datetime(nav.index)
c=pd.read_csv(D+'shapley_coalitions.csv.gz',keep_default_na=False)
def cg(x,a):
    x=x[x.index>=a]; y=(x.index[-1]-x.index[0]).days/365.25; return ((x.iloc[-1]/x.iloc[0])**(1/y)-1)*100
a=c[(c.coalition=='TXFKIC')&(c.variant=='REAL_opt')]
s0=a[a.tier=='100萬'].set_index('j').CAGR_Y19on
s20=[];e19=[];ep=[]
for j in range(13):
    s=nav[f'REAL_opt|{j}'].dropna(); s=s[s.index<=JEND[j]]; s20.append(cg(s,s[s.index>='2019-01-01'].index[0]))
    e=e50[(e50.index>=s.index[0])&(e50.index<=JEND[j])]; e19.append(cg(e,e[e.index>='2019-01-01'].index[0])); ep.append(cg(e,e[e.index>='2020-10-26'].index[0]))
s20=np.array(s20); print('seed0 med',s0.median(),'20seed med',np.median(s20),'max |diff|',np.abs(s0.values-s20).max())
t30=a[a.tier=='30萬'].set_index('j')
print('30萬 seed0 Y19 excess (med diff)',np.median(t30.CAGR_Y19on.values-np.array(e19)),'diffmed',t30.CAGR_Y19on.median()-np.median(e19))
print('30萬 POSTODD excess',np.median(t30.CAGR_POSTODD.values-np.array(ep)),'diffmed',t30.CAGR_POSTODD.median()-np.median(ep))
