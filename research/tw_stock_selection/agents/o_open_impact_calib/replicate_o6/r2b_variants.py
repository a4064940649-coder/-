import pandas as pd, numpy as np
OUT='/home/user/-/research/tw_stock_selection/agents/o_open_impact_calib/replicate_o6/'
sd=pd.read_parquet(OUT+'sd.parquet')
bars=[f'{m:02d}' for m in range(5,60,5)]
Q=sd[['q'+b for b in bars]].values.astype(float)
AR=sd[['ar'+b for b in bars]].values
Vsh=(sd.V60_l/sd.prevc).values; s=sd.sig60_l.values; y=sd.y.values
cal=(sd.per=='CAL').values
dgrid=np.round(np.arange(0.20,1.0001,0.01),2)
def fit(xf,yy,msk):
    best=None
    for dl in dgrid:
        x=xf(dl)[msk]; A=np.c_[np.ones(len(x)),x]; c,res,_,_=np.linalg.lstsq(A,yy[msk],rcond=None)
        sse=np.sum((yy[msk]-A@c)**2)
        if best is None or sse<best[0]: best=(sse,dl,c[0],c[1])
    return best[1:]
# variant A: per-bar rows
Sb=np.repeat(s[:,None],11,1); Vb=np.repeat(Vsh[:,None],11,1); calb=np.repeat(cal[:,None],11,1)
print('per-bar', fit(lambda d:(Sb*(Q/Vb)**d).ravel(), AR.ravel(), calb.ravel()))
# variant B: mean q
qm=Q.mean(1)
print('meanq', fit(lambda d:s*(qm/Vsh)**d, y, cal))
# variant C: total q 11 bars
print('sumq', fit(lambda d:s*(Q.sum(1)/Vsh)**d, y, cal))
# variant D: y / sigma target ratio
print('V in value not shares (q*prevc/V60)', fit(lambda d:s*np.mean((Q*sd.prevc.values[:,None]/sd.V60_l.values[:,None])**d,1), y, cal))
