import pandas as pd, numpy as np
OUT='/home/user/-/research/tw_stock_selection/agents/o_open_impact_calib/replicate_o6/'
sd=pd.read_parquet(OUT+'sd.parquet'); z=np.load(OUT+'bar_arrays.npz')
Q,AR,Vsh,s,cal=z['Q'],z['AR'],z['Vsh'],z['s'],z['cal']; val=(sd.per=='VAL').values; y=sd.y.values
beta=np.array([-8.177181876611714,0.0967309549710343,0.27359738939073935,-0.9010704677890121,0.8119480394199919,-0.31581463032857726]); smear=1.5788456225921357
xm=np.c_[np.ones(len(sd)),np.log(sd.vrank_l.fillna(300)),np.log(sd.prevc),np.log(np.maximum(s,1e-4)),np.zeros(len(sd)),np.zeros(len(sd))]
OAh=smear*sd.V60_l.values*np.exp(xm@beta); OAsh=OAh/sd.prevc.values
def rep(name,yhat):
    e=np.abs(yhat[val]-y[val]); t=pd.DataFrame({'yh':yhat[val],'y':y[val],'d':sd.dec.values[val]}).groupby('d').sum(); r=t.yh/t.y
    print(f'  {name}: MdAE {np.median(e)*1e4:.3f}  ratio {r.min():.3f}~{r.max():.3f}')
for nm,den in [('M2',Vsh),('M1',OAsh)]:
    X=s[:,None]*np.sqrt(Q/den[:,None])
    x=X[cal].ravel(); yy=AR[cal].ravel()
    k0=(x@yy)/(x@x); print(nm,'no-intercept',k0); rep(nm+'_noint',k0*X.mean(1))
    k1=(x@(yy-0.001))/(x@x); print(nm,'fixed a=.001',k1); rep(nm+'_a001',0.001+k1*X.mean(1))
    # per-stockday aggregated no intercept
    xs=X.mean(1)[cal]; print(nm,'sd-level noint',(xs@y[cal])/(xs@xs))
# M3: free delta no intercept
best=None
for d in np.round(np.arange(0.2,1.0001,0.01),2):
    X=s[:,None]*(Q/Vsh[:,None])**d; x=X[cal].ravel(); yy=AR[cal].ravel(); k=(x@yy)/(x@x); sse=np.sum((yy-k*x)**2)
    if best is None or sse<best[0]: best=(sse,d,k)
print('M3 noint',best[1:]); d,k=best[1:]; rep('M3_noint',k*(s[:,None]*(Q/Vsh[:,None])**d).mean(1))
