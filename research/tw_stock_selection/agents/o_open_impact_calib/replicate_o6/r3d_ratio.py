# M1／M2：比值估計 Σy/Σx（股日層級、無截距）；VAL 誤差；M3 候選
import pandas as pd, numpy as np, json
OUT='/home/user/-/research/tw_stock_selection/agents/o_open_impact_calib/replicate_o6/'
sd=pd.read_parquet(OUT+'sd.parquet'); z=np.load(OUT+'bar_arrays.npz')
Q,AR,Vsh,s,cal=z['Q'],z['AR'],z['Vsh'],z['s'],z['cal']; y=sd.y.values; val=(sd.per=='VAL').values
beta=np.array([-8.177181876611714,0.0967309549710343,0.27359738939073935,-0.9010704677890121,0.8119480394199919,-0.31581463032857726]); smear=1.5788456225921357
xm=np.c_[np.ones(len(sd)),np.log(sd.vrank_l.fillna(300)),np.log(sd.prevc),np.log(np.maximum(s,1e-4)),np.zeros(len(sd)),np.zeros(len(sd))]
OAh=smear*sd.V60_l.values*np.exp(xm@beta); OAsh=OAh/sd.prevc.values
def rep(name,yhat):
    e=np.abs(yhat[val]-y[val]); t=pd.DataFrame({'yh':yhat[val],'y':y[val],'d':sd.dec.values[val]}).groupby('d').sum(); r=t.yh/t.y
    print(f'  {name}: MdAE {np.median(e)*1e4:.3f}  ratio {r.min():.3f}~{r.max():.3f}'); return np.median(e)*1e4
out={}
days=sd.date.values; cd=days[cal]; ud=np.unique(cd); idx=np.searchsorted(ud,cd)
rng=np.random.default_rng(12345)
for nm,den in [('M2',Vsh),('M1',OAsh)]:
    xs=(s[:,None]*np.sqrt(Q/den[:,None])).mean(1)
    k=y[cal].sum()/xs[cal].sum(); print(nm,k)
    md0=rep(nm+' noint',k*xs)
    sy=np.bincount(idx,y[cal]); sx=np.bincount(idx,xs[cal])
    bs=[]
    for _ in range(1000):
        w=np.bincount(rng.integers(0,len(ud),len(ud)),minlength=len(ud)); bs.append((w@sy)/(w@sx))
    lo,hi=np.percentile(bs,[5,95]); print('  90% CI',lo,hi)
    out[nm]=dict(k=k,lo=lo,hi=hi,mdae=md0)
# M1 equiv full-day Y: κ·sqrt(V60/OA_hat)
r_=np.sqrt(sd.V60_l.values/OAh)
print('M1 equiv Y: median',out['M1']['k']*np.median(r_[cal]),'ratio-weighted',out['M1']['k']*(( (s[:,None]*np.sqrt(Q/OAsh[:,None])).mean(1))[cal].sum()/((s[:,None]*np.sqrt(Q/Vsh[:,None])).mean(1))[cal].sum()))
print('M1 equiv Y: geo-mean', out['M1']['k']*np.exp(np.mean(np.log(r_[cal]))))
# M3 candidate: ratio estimator with δ grid, min SSE
best=None
for d in np.round(np.arange(0.2,1.0001,0.01),2):
    xs=(s[:,None]*(Q/Vsh[:,None])**d).mean(1); k=y[cal].sum()/xs[cal].sum(); sse=np.sum((y[cal]-k*xs[cal])**2)
    if best is None or sse<best[0]: best=(sse,d,k)
print('M3 ratio δ',best[1:]); xs=(s[:,None]*(Q/Vsh[:,None])**best[1]).mean(1); rep('M3 ratio',best[2]*xs)
json.dump(out,open(OUT+'r3d_ratio.json','w'),indent=1)
