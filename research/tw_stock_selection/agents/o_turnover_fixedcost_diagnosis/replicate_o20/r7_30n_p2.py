import sys; sys.path.insert(0,"/home/user/-/research/tw_stock_selection/agents/o_turnover_fixedcost_diagnosis/replicate_o20")
from swaps import *
SP='/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/'
ex,en,t30,rb=build('30n')
p=ex[ex.signal>='2020-10-26']; print('30n POSTODD swaps',len(p),'with entry',p.n_en.notna().sum())
p=p[p.n_en.notna()]
bad=p.edge<p.rtc; a=bad[p.price<50].mean(); b=bad[p.price>=100].mean(); print('P3 30n',round(a*100,2),round(b*100,2),round((a-b)*100,2))
print('BE 30n <50',p[p.price<50].rtc.median(),'>=100',p[p.price>=100].rtc.median(),'edge med',p.edge.median())
def boot(df,stat,B_=2000,seed=20261112):
    df=df.copy(); df['m']=df.signal.dt.to_period('M'); months=df.m.unique(); grp={m:g for m,g in df.groupby('m')}
    rng=np.random.default_rng(seed); vals=[]
    for _ in range(B_):
        pick=rng.choice(len(months),len(months),replace=True); bb=pd.concat([grp[months[i]] for i in pick])
        bd=bb.edge<bb.rtc; vals.append(bd[bb.price<50].mean()-bd[bb.price>=100].mean())
    return (np.array(vals)<=0).mean()
print('P3 30n p',boot(p,None))
ex.to_pickle(SP+'ex30.pkl')
# P2
t100=load_tl('100n')
def p2(t,a,b):
    x=t[(t.date>=a)&(t.date<=b)&(t.shares>0)]
    sp=x.tick+x.imp+x.odd; return sp.sum()/x.notional.sum(), (sp/x.notional).median(), x.tick.sum()/sp.sum()
for per in ['POSTODD','S0618','Y19on']:
    a,b=PER[per]; r30=p2(t30,a,b); r100=p2(t100,a,b)
    print(per,'30n',np.round(r30,5),'100n',np.round(r100,5),'ratio agg',round(r30[0]/r100[0],3),'ratio med',round(r30[1]/r100[1],3))
r=pd.read_csv(D+'rebs_allcells_seed0.csv.gz')
z=np.load(B+'o_exec_engine_rewrite_v7x/data/x_cache.npz',allow_pickle=True); IDX=pd.DatetimeIndex(z['IDX'])
r['ed']=IDX[r.e.values]
for cell in ['normal|30萬','normal|100萬','normal|300萬','normal|1000萬']:
    for per in ['POSTODD','Y19on']:
        a,b=PER[per]; x=r[(r.cell==cell)&(r.variant=='REAL_opt')&(r.ed>=a)]
        print(cell,per,'tick share',round(x.tick_amt.sum()/(x.tick_amt+x.imp_amt+x.odd_amt).sum(),3))
# worst-cell ratio from rebs aggregates (notional = not_* sums)
for v in ['REAL_opt','REAL_p90','REAL_carry']:
    q={}
    for cell in ['normal|30萬','normal|100萬']:
        x=r[(r.cell==cell)&(r.variant==v)&(r.ed>='2020-10-26')]
        q[cell]=(x.tick_amt+x.imp_amt+x.odd_amt).sum()/(x.not_exit+x.not_rw_sell+x.not_entry+x.not_rw_buy).sum()
    print(v,{k:round(vv*100,4) for k,vv in q.items()},'ratio',round(q['normal|30萬']/q['normal|100萬'],3))
