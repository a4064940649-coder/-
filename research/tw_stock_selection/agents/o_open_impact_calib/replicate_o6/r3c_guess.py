import pandas as pd, numpy as np
OUT='/home/user/-/research/tw_stock_selection/agents/o_open_impact_calib/replicate_o6/'
sd=pd.read_parquet(OUT+'sd.parquet'); z=np.load(OUT+'bar_arrays.npz')
Q,AR,Vsh,s,cal=z['Q'],z['AR'],z['Vsh'],z['s'],z['cal']; y=sd.y.values
beta=np.array([-8.177181876611714,0.0967309549710343,0.27359738939073935,-0.9010704677890121,0.8119480394199919,-0.31581463032857726]); smear=1.5788456225921357
xm=np.c_[np.ones(len(sd)),np.log(sd.vrank_l.fillna(300)),np.log(sd.prevc),np.log(np.maximum(s,1e-4)),np.zeros(len(sd)),np.zeros(len(sd))]
OAh=smear*sd.V60_l.values*np.exp(xm@beta); OAsh=OAh/sd.prevc.values
for nm,den in [('M2',Vsh),('M1',OAsh)]:
    X=s[:,None]*np.sqrt(Q/den[:,None]); xs=X.mean(1)
    yc=y[cal]; xc=xs[cal]
    print(nm,'sum ratio',yc.sum()/xc.sum(),'median ratio',np.median(yc/xc),'(y-.001) sum ratio',(yc-0.001).sum()/xc.sum(),
          'excess over a: med((y-a)/x) a=.001',np.median((yc-0.001)/xc))
    # sqrt of mean q
    X2=s*np.sqrt(Q.mean(1)/den); print('  sqrt(mean q): ols', np.polyfit(X2[cal],yc,1), 'noint',(X2[cal]@yc)/(X2[cal]@X2[cal]))
    # per-bar with intercept but impact relative to |r| minus a ; abs return sigma-normalised
    xb=X[cal].ravel(); yb=AR[cal].ravel()
    print('  per-bar median ratio',np.median(yb[xb>0]/xb[xb>0]))
