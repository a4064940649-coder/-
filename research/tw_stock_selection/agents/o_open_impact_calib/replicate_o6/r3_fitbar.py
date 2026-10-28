# 逐 K 棒最小平方（E|r_b| 是每根 K 棒的期望）；VAL 以股日平均比較
import pandas as pd, numpy as np, json
OUT='/home/user/-/research/tw_stock_selection/agents/o_open_impact_calib/replicate_o6/'
sd=pd.read_parquet(OUT+'sd.parquet')
bars=[f'{m:02d}' for m in range(5,60,5)]
Q=sd[['q'+b for b in bars]].values.astype(float)
AR=sd[['ar'+b for b in bars]].values
Vsh=(sd.V60_l/sd.prevc).values; s=sd.sig60_l.values; y=sd.y.values
cal=(sd.per=='CAL').values; val=(sd.per=='VAL').values
dgrid=np.round(np.arange(0.20,1.0001,0.01),2)
beta=np.array([-8.177181876611714,0.0967309549710343,0.27359738939073935,-0.9010704677890121,0.8119480394199919,-0.31581463032857726]); smear=1.5788456225921357
xm=np.c_[np.ones(len(sd)),np.log(sd.vrank_l.fillna(300)),np.log(sd.prevc),np.log(np.maximum(s,1e-4)),np.zeros(len(sd)),np.zeros(len(sd))]
OAh=smear*sd.V60_l.values*np.exp(xm@beta); OAsh=OAh/sd.prevc.values
res={}
def ols(Xb,msk):
    x=Xb[msk].ravel(); yy=AR[msk].ravel(); A=np.c_[np.ones(len(x)),x]
    c=np.linalg.lstsq(A,yy,rcond=None)[0]; return c, np.sum((yy-A@c)**2)
def grid(den,msk):
    best=None
    for d in dgrid:
        c,sse=ols(s[:,None]*(Q/den[:,None])**d,msk)
        if best is None or sse<best[0]: best=(sse,d,c[0],c[1])
    return best[1:]
def report(name,yhat):
    e=np.abs(yhat[val]-y[val]); md=np.median(e)*1e4
    t=pd.DataFrame({'yh':yhat[val],'y':y[val],'d':sd.dec.values[val]}).groupby('d').sum(); r=t.yh/t.y
    print(f'{name}: VAL MdAE {md:.3f}bp  ratios {r.min():.3f}~{r.max():.3f}')
    res[name]=dict(mdae_bp=md,rmin=r.min(),rmax=r.max())
d,a,b=grid(Vsh,cal); print('M3V',d,a,b,'imp/sig',b*1e-3**d)
res['M3V_par']=dict(delta=d,a=a,b=b,imp_sig=b*1e-3**d)
report('M3V',a+b*s*np.mean((Q/Vsh[:,None])**d,1))
c2,_=ols(s[:,None]*np.sqrt(Q/Vsh[:,None]),cal); print('M2',c2); res['M2_par']=c2.tolist()
report('M2',c2[0]+c2[1]*s*np.mean(np.sqrt(Q/Vsh[:,None]),1))
c1,_=ols(s[:,None]*np.sqrt(Q/OAsh[:,None]),cal); print('M1',c1); res['M1_par']=c1.tolist()
r_=np.sqrt(sd.V60_l.values/OAh)[cal]
print('M1 Yeq median',c1[1]*np.median(r_),'mean',c1[1]*r_.mean(),'geo',c1[1]*np.exp(np.log(r_).mean()))
report('M1',c1[0]+c1[1]*s*np.mean(np.sqrt(Q/OAsh[:,None]),1))
# M3 候選：δ 自由但無 σ（a+b(q/V)^δ）、或 σ 指數自由
best=None
for d3 in dgrid:
    c,sse=ols(np.ones_like(s)[:,None]*(Q/Vsh[:,None])**d3,cal)
    if best is None or sse<best[0]: best=(sse,d3,c)
print('M3 cand no-sigma',best[1:]); report('M3cand_nosig',best[2][0]+best[2][1]*np.mean((Q/Vsh[:,None])**best[1],1))
dv,av,bv=grid(Vsh,val); print('M3V_VAL',dv,av,bv); res['M3V_VAL']=dict(delta=dv,b=bv)
# implied impact/sigma
json.dump({k:(v if not isinstance(v,dict) else {kk:float(vv) for kk,vv in v.items()}) for k,v in res.items()},open(OUT+'r3_fitbar.json','w'),indent=1)
np.savez(OUT+'bar_arrays.npz',Q=Q,AR=AR,Vsh=Vsh,s=s,cal=cal,date=sd.date.values)
