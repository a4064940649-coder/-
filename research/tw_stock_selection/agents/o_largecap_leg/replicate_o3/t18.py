import sys; sys.path.insert(0, '/home/user/-/research/tw_stock_selection/agents/o_largecap_leg/replicate_o3')
import pickle; from rlib import *
SC='/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/'
D=pickle.load(open(SC+'rep_o3_data.pkl','rb')); PL=pickle.load(open(SC+'rep_o3_plans.pkl','rb')); ss=PL['sigs']; sigs=D.sigs
N=pd.read_parquet('navs_rep2.parquet'); B=bench0050(D.nidx); rb=mret(B,sigs)
out={}
for nm,pm in [('pool',PL['pool']),('poolx',PL['poolx'])]:
    tg={t:{c:1/len(pm[t]) for c in pm[t]} for t in ss}
    out[nm],_=D.simulate(tg,'zero')
for nm,a,b in [('L-pool',N['L|normal'],out['pool']),('Ls-pool',N['L|stress3'],out['pool']),('LX-poolx',N['LX|normal'],out['poolx']),('Lzero-pool',N['L|zero'],out['pool'])]:
    d=mret(a,sigs)-mret(b,sigs); print(nm,{s:(round(seg_mask(d,s).mean()*100,3),round(tstat(seg_mask(d,s)),2)) for s in ['SEL','VAL','VAL_A','HOLD','ALL']})
d=mret(out['pool'],sigs)-rb; print('pool-0050', round(seg_mask(d,'SEL').mean()*100,3), round(tstat(seg_mask(d,'SEL')),2))
tg={}
for t in ss:
    w={c:0.07 for c in PL['main'][t]}
    for c in PL['pool'][t]: w[c]=w.get(c,0)+0.3/len(PL['pool'][t])
    tg[t]=w
nv,_=D.simulate(tg,'normal'); r=mret(nv,sigs)-rb
print('placebo',{s:(round(seg_nav_cagr(nv,None,s)*100,2),round(tstat(seg_mask(r,s)),3),round(seg_mask(r,s).std()*12**.5*100,2)) for s in ['SEL','VAL','ALL']})
pickle.dump(out,open(SC+'rep_o3_poolfreeze.pkl','wb'))
