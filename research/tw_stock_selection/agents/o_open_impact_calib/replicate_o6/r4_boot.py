# M3V κ 區間：CAL 交易日群聯合 bootstrap 1,000 次，每次在 δ 格點重估 a,b,δ
import numpy as np, json
OUT='/home/user/-/research/tw_stock_selection/agents/o_open_impact_calib/replicate_o6/'
z=np.load(OUT+'bar_arrays.npz',allow_pickle=True)
Q,AR,Vsh,s,cal,date=z['Q'],z['AR'],z['Vsh'],z['s'],z['cal'].astype(bool),z['date']
Q,AR,Vsh,s,date=Q[cal],AR[cal],Vsh[cal],s[cal],date[cal]
ud,idx=np.unique(date,return_inverse=True); D=len(ud)
dg=np.round(np.arange(0.20,1.0001,0.01),2)
n=np.bincount(idx,minlength=D)*11.0; Sy=np.bincount(idx,AR.sum(1)); Syy=np.bincount(idx,(AR**2).sum(1))
Sx=np.zeros((len(dg),D)); Sxx=Sx.copy(); Sxy=Sx.copy()
for i,d in enumerate(dg):
    X=s[:,None]*(Q/Vsh[:,None])**d
    Sx[i]=np.bincount(idx,X.sum(1)); Sxx[i]=np.bincount(idx,(X*X).sum(1)); Sxy[i]=np.bincount(idx,(X*AR).sum(1))
def est(w):
    N=w@n; sy=w@Sy; syy=w@Syy; sx=Sx@w; sxx=Sxx@w; sxy=Sxy@w
    b=(sxy-sx*sy/N)/(sxx-sx*sx/N); a=(sy-b*sx)/N
    sse=syy-2*a*sy-2*b*sxy+N*a*a+2*a*b*sx+b*b*sxx
    j=np.argmin(sse); return dg[j],a[j],b[j]
d0,a0,b0=est(np.ones(D)); print('point',d0,a0,b0,b0*1e-3**d0)
rng=np.random.default_rng(20261028)
B=[]
for _ in range(1000):
    w=np.bincount(rng.integers(0,D,D),minlength=D).astype(float); d,a,b=est(w); B.append((d,a,b,b*1e-3**d))
B=np.array(B); order=np.argsort(B[:,3])
out={'point':dict(delta=float(d0),a=float(a0),b=float(b0),imp_sig=float(b0*1e-3**d0))}
for p in [5,10,90,95]:
    k=order[int(round(p/100*1000))-1] if p<50 else order[int(round(p/100*1000))-1]
    out[f'p{p}']=dict(delta=float(B[k,0]),b=float(B[k,2]),imp_sig=float(B[k,3]),pct=float(np.percentile(B[:,3],p)))
    print(p,out[f'p{p}'])
print('delta dist',np.unique(B[:,0],return_counts=True))
json.dump(out,open(OUT+'r4_boot.json','w'),indent=1)
