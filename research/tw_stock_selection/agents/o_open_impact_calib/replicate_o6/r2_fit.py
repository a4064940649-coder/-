import pandas as pd, numpy as np, json
OUT='/home/user/-/research/tw_stock_selection/agents/o_open_impact_calib/replicate_o6/'
sd=pd.read_parquet(OUT+'sd.parquet')
bars=[f'{m:02d}' for m in range(5,60,5)]
Q=sd[['q'+b for b in bars]].values.astype(float)
Vsh=(sd.V60_l/sd.prevc).values
s=sd.sig60_l.values; y=sd.y.values
cal=(sd.per=='CAL').values; val=(sd.per=='VAL').values
dgrid=np.round(np.arange(0.20,1.0001,0.01),2)
def X(delta,den=Vsh): return s*np.mean((Q/den[:,None])**delta,axis=1)
def ols(x,yy):
    A=np.c_[np.ones_like(x),x]; c=np.linalg.lstsq(A,yy,rcond=None)[0]; return c
best=None
for dl in dgrid:
    x=X(dl); a,b=ols(x[cal],y[cal]); sse=np.sum((y[cal]-a-b*x[cal])**2)
    if best is None or sse<best[0]: best=(sse,dl,a,b)
_,dl,a,b=best
print('M3V delta',dl,'a',a,'b',b,'imp/sig',b*1e-3**dl)
res={'M3V':dict(delta=float(dl),a=float(a),b=float(b),imp_sig=float(b*1e-3**dl))}
def report(name,yhat):
    e=np.abs(yhat[val]-y[val]); md=np.median(e)*1e4
    rat=pd.DataFrame({'yh':yhat[val],'y':y[val],'d':sd.dec.values[val]}).groupby('d').sum()
    r=(rat.yh/rat.y)
    print(name,'VAL MdAE bp',round(md,3),'dec ratios',r.round(3).min(),r.round(3).max())
    res[name+'_val']=dict(mdae_bp=float(md),ratio_min=float(r.min()),ratio_max=float(r.max()),ratios=r.round(4).tolist())
report('M3V',a+b*X(dl))
# M2 sqrt with V60
x2=X(0.5); a2,Y=ols(x2[cal],y[cal]); print('M2 a',a2,'Y',Y); res['M2']=dict(a=float(a2),Y=float(Y))
report('M2',a2+Y*x2)
# M1 sqrt with OA_hat
beta=np.array([-8.177181876611714,0.0967309549710343,0.27359738939073935,-0.9010704677890121,0.8119480394199919,-0.31581463032857726]); smear=1.5788456225921357
# 校準樣本股日：pick/drop 不適用→0（非換股語境）
xm=np.c_[np.ones(len(sd)),np.log(sd.vrank_l.fillna(300)),np.log(sd.prevc),np.log(np.maximum(s,1e-4)),np.zeros(len(sd)),np.zeros(len(sd))]
OAh=smear*sd.V60_l.values*np.exp(xm@beta)   # NT$
OAsh=OAh/sd.prevc.values
x1=X(0.5,OAsh); a1,k1=ols(x1[cal],y[cal]); print('M1 a',a1,'kappa',k1); res['M1']=dict(a=float(a1),kappa=float(k1))
# equivalent full-day Y: kappa*sqrt(V60/OA) -> median? 
ratio=np.sqrt(sd.V60_l.values/OAh)
print('M1 equiv Y median',k1*np.median(ratio[cal]),'mean',k1*np.mean(ratio[cal]), 'via fit', ols(x2[cal],(a1+k1*x1)[cal])[1])
res['M1']['Yeq_median']=float(k1*np.median(ratio[cal])); res['M1']['Yeq_fit']=float(ols(x2[cal],(a1+k1*x1)[cal])[1])
report('M1',a1+k1*x1)
# M3: free delta on sigma too? try a + b*sigma^g*(q/V)^d grid
best3=None
for g in np.round(np.arange(0.5,1.51,0.05),2):
  for d3 in dgrid[::2]:
    x=(s**g)*np.mean((Q/Vsh[:,None])**d3,axis=1); a3,b3=ols(x[cal],y[cal]); sse=np.sum((y[cal]-a3-b3*x[cal])**2)
    if best3 is None or sse<best3[0]: best3=(sse,g,d3,a3,b3)
print('M3 guess (sigma^g)',best3[1:])
_,g,d3,a3,b3=best3; report('M3guess_sig',a3+b3*(s**g)*np.mean((Q/Vsh[:,None])**d3,axis=1))
# M3V on VAL
bv=None
for d_ in dgrid:
    x=X(d_); aa,bb=ols(x[val],y[val]); sse=np.sum((y[val]-aa-bb*x[val])**2)
    if bv is None or sse<bv[0]: bv=(sse,d_,aa,bb)
print('M3V_VAL',bv[1:]); res['M3V_VAL']=dict(delta=float(bv[1]),b=float(bv[3]))
np.save(OUT+'arrays.npy',np.c_[y,s,Vsh,OAsh,cal,val])
json.dump(res,open(OUT+'r2_fit.json','w'),indent=1,ensure_ascii=False)
