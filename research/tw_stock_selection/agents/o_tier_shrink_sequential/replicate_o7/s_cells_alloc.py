import sys, json, pickle; sys.path.insert(0,'/home/user/-/research/tw_stock_selection/agents/o_tier_shrink_sequential/replicate_o7')
from rlib import *
OUT='/home/user/-/research/tw_stock_selection/agents/o_tier_shrink_sequential/replicate_o7/out/'
PP=pickle.load(open(OUT+'post.pkl','rb')); post=PP['post']
fw=pd.read_parquet(RAW+'o6shrink/focus_windows.parquet'); F=(fw.sub(fw.r0050,axis=0)*100)
r50=fw.r0050.values
A=pd.read_parquet(RAW+'navs/navs_abavg.parquet'); EB=pd.read_parquet(RAW+'exec/bench.parquet')
T=pd.read_parquet(OUT+'navs_tick_tickstress_main.parquet')
z=wret(EB['AB_V4|normal'])*100
# check my 'main' (sqrt) seed-avg reproduces navs_abavg
for t in ['100萬','1000萬']:
    for m in ['FIX','GROW']:
        print('repro main',t,m,np.abs(T[f'main|{t}|{m}'].values-A[f'main|{t}|{m}|A'].reindex(T.index).values).max())
cells={}
for t in ['30萬','100萬','300萬','1000萬']:
    for m in ['FIX','GROW']:
        cells[f'{t}|{m}|normal|KAPPA']=F.B10.values+(wret(A[f'main|{t}|{m}|A'])*100-z).values
        cells[f'{t}|{m}|normal|TICK']=F.B10.values+(wret(T[f'tick|{t}|{m}'])*100-z).values
        cells[f'{t}|{m}|stress|KAPPA']=F.B10_stress.values+(wret(A[f'stress|{t}|{m}|B'])*100-z).values
        cells[f'{t}|{m}|stress|TICK']=F.B10_stress.values+(wret(T[f'tickstress|{t}|{m}'])*100-z).values
dist={}; ser={}
for k,s in cells.items():
    js=PP['jB'] if '|normal|' in k else PP['jBs']
    dist[k]=mix_shift(*post['B10'],np.mean(s-F.B10.values)+js); ser[k]=s
dist['AB']=post['AB']; ser['AB']=F.AB.values
dist['B10']=post['B10']; ser['B10']=F.B10.values
dist['AB_J']=mix_shift(*post['AB'],PP['jAB']); ser['AB_J']=F.AB.values
dist['B10_J']=mix_shift(*post['B10'],PP['jB']); ser['B10_J']=F.B10.values
dist['B10_stress_J']=mix_shift(*post['B10'],(F.B10_stress-F.B10).mean()+PP['jBs']); ser['B10_stress_J']=F.B10_stress.values
summ_=({k:dict(summ2(*v),raw=float(np.mean(ser[k])),raw_cell_drag=float(np.mean(ser[k]-F.B10.values))) for k,v in dist.items()})
for k,v in summ_.items(): print(k,{a:round(b,3) for a,b in v.items()})
# ---------------- allocation
FEE=0.001425*0.6
C_M_SELL,C_M_BUY=FEE+0.001+0.003,FEE+0.001
C_E_SELL,C_E_BUY=FEE+0.001+0.0005,FEE+0.0005
def sb_index(n,H,L,B,rng):
    idx=np.empty((B,H),np.int32); idx[:,0]=rng.integers(0,n,B); p=1/L
    for t in range(1,H):
        new=rng.random(B)<p; idx[:,t]=np.where(new,rng.integers(0,n,B),(idx[:,t-1]+1)%n)
    return idx
def sim(r50p,ep,alpha,x):
    B,H=r50p.shape; rm=np.maximum(r50p+alpha[:,None]+ep,-0.99)
    wm=np.full(B,x); we=np.full(B,1-x); W50=np.ones(B)
    for t in range(H):
        wm=wm*(1+rm[:,t]); we=we*(1+r50p[:,t]); Wp=wm+we
        if 0<x<1:
            d=wm-x*Wp; cost=np.where(d>0,d*(C_M_SELL+C_E_BUY),-d*(C_E_SELL+C_M_BUY)); Wp=Wp-cost
            wm=x*Wp; we=(1-x)*Wp
        W50=W50*(1+r50p[:,t])
    return wm+we, W50
NB=int(sys.argv[1]) if len(sys.argv)>1 else 20000
rng=np.random.default_rng(20261101)
idx={H:sb_index(245,H,6,NB,rng) for H in (120,240)}
U=rng.random(NB)
xs=np.round(np.arange(0,1.0001,0.1),2)
alloc={}
for k,(v,w) in dist.items():
    c=np.cumsum(w); a=v[np.minimum(np.searchsorted(c,U*c[-1]),len(v)-1)]/100
    e=(ser[k]-ser[k].mean())/100
    alloc[k]={}
    for H in (120,240):
        r50p=r50[idx[H]]; ep=e[idx[H]]; rows={}
        for x in xs:
            Wp,W50=sim(r50p,ep,a,x); R=Wp/W50
            rows[f'{x:.1f}']=dict(Elog=float(np.log(Wp).mean()/(H/12)),P_win=float((R>1).mean()),P_lt075=float((R<0.75).mean()),p10=float(np.percentile(R,10)))
        xst=max(rows,key=lambda q: rows[q]['Elog'])
        alloc[k][H]=dict(rows=rows,xstar=float(xst))
    print(k,'x*10y',alloc[k][120]['xstar'],'x*20y',alloc[k][240]['xstar'],'Pwin10 x1',round(alloc[k][120]['rows']['1.0']['P_win'],3),'Plt075',round(alloc[k][120]['rows']['1.0']['P_lt075'],3),flush=True)
json.dump(dict(summ=summ_,alloc=alloc),open(OUT+'cells_alloc.json','w'),ensure_ascii=False,indent=1)
