import sys; sys.path.insert(0, '/home/user/-/research/tw_stock_selection/agents/o_largecap_leg/replicate_o3')
import pickle, json, math, copy; from rlib import *
SC='/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/'
D=pickle.load(open(SC+'rep_o3_data.pkl','rb')); sigs=D.sigs
PL=pickle.load(open(SC+'rep_o3_plans.pkl','rb'))
B=bench0050(D.nidx); rb=mret(B,sigs); bd=B.pct_change()
a23=D.adjf['2330'].reindex(D.nidx); r23=a23.pct_change()
SEGD={'SEL':('2006-04-12','2019-01-11'),'VAL':('2019-01-12','2026-10-12')}
for s,(a,b) in SEGD.items():
    x=(r23-bd).loc[a:b]; lx=(np.log1p(r23)-np.log1p(bd)).loc[a:b]
    print(s,'arith ex bp',round(x.mean()*1e4,2),'log ex bp',round(lx.mean()*1e4,2))
import sys as _s
mode=_s.argv[1] if len(_s.argv)>1 else 'arith'
rr=D.ret['2330'].fillna(0.0)
adjm=pd.Series(0.0,index=rr.index)
for s,(a,b) in SEGD.items():
    m=(r23-bd).loc[a:b].mean() if mode=='arith' else (np.log1p(r23)-np.log1p(bd)).loc[a:b].mean()
    adjm[(rr.index>=pd.Timestamp(a))&(rr.index<=pd.Timestamp(b))]=m
adjm[D.ret['2330'].isna()]=0
fac=((1+rr-adjm)/(1+rr)).cumprod() if mode=='arith' else np.exp(-adjm.cumsum())
D2=copy.copy(D); D2.adjf=D.adjf.copy(); D2.adjopen=D.adjopen.copy()
D2.adjf['2330']=D.adjf['2330']*fac; D2.adjopen['2330']=D.adjopen['2330']*fac
res={}
for k in ['COMBO','TSMC30','MAIN']:
    nv2,_=D2.simulate(PL['TG'][k],'normal'); r=mret(nv2,sigs)-rb
    res[k]={s:round(tstat(seg_mask(r,s)),2) for s in ['SEL','VAL']}
print(mode,res)
