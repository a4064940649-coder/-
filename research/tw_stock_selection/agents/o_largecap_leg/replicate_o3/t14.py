import sys; sys.path.insert(0, '/home/user/-/research/tw_stock_selection/agents/o_largecap_leg/replicate_o3')
import pickle, json, math, copy; from rlib import *
SC='/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/'
D=pickle.load(open(SC+'rep_o3_data.pkl','rb')); N=pd.read_parquet('navs_rep2.parquet'); sigs=D.sigs
PL=pickle.load(open(SC+'rep_o3_plans.pkl','rb')); ss=PL['sigs']
B=bench0050(D.nidx); rb=mret(B,sigs)
out={}
# placebo 不擋漲停
def mix(*parts):
    tg={}
    for t in ss:
        w={}
        for wt,m in parts:
            v=m[t] if isinstance(m[t],list) else [m[t]]
            for c in v: w[c]=w.get(c,0)+wt/len(v)
        tg[t]=w
    return tg
nv,_=D.simulate(mix((0.7,PL['main']),(0.3,PL['pool'])),'normal')
r=mret(nv,sigs)-rb
out['placebo_noLU']={s:(round(seg_nav_cagr(nv,None,s)*100,2),round(tstat(seg_mask(r,s)),3)) for s in ['SEL','VAL','ALL']}
print(out)
# 2330 漂移
r23=D.adjf['2330'].pct_change()
mu={}
for s,(a,b) in [('SEL',('2006-04-12','2019-01-11')),('VAL',('2019-01-12','2026-10-12'))]:
    x=D.ret['2330'].loc[a:b].dropna(); mu[s]=float(x.mean())
print('drift bp', {k:round(v*1e4,2) for k,v in mu.items()})
D2=copy.copy(D); D2.adjf=D.adjf.copy(); D2.adjopen=D.adjopen.copy()
rr=D.ret['2330'].fillna(0.0).copy()
adjm=pd.Series(0.0,index=rr.index)
adjm[(rr.index>pd.Timestamp('2006-04-11'))&(rr.index<=pd.Timestamp('2019-01-11'))]=mu['SEL']
adjm[(rr.index>pd.Timestamp('2019-01-11'))&(rr.index<=END)]=mu['VAL']
adjm[D.ret['2330'].isna()]=0
fac=((1+rr-adjm)/(1+rr)).cumprod()
D2.adjf['2330']=D.adjf['2330']*fac; D2.adjopen['2330']=D.adjopen['2330']*fac
res={}
for k in ['COMBO','TSMC30','MAIN']:
    nv2,_=D2.simulate(PL['TG'][k],'normal'); r=mret(nv2,sigs)-rb
    res[k]={s:round(tstat(seg_mask(r,s)),2) for s in ['SEL','VAL']}
print('drift0 (0050 unchanged)',res)
json.dump({'placebo_noLU':out,'drift_bp':mu,'drift0':res},open('t14.json','w'))
