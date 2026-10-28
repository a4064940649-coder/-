import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd, numpy as np
from rlib import *
nav,e,plan,sig,ew=load()
dd=pd.DatetimeIndex(plan.signal.tolist()+[pd.Timestamp('2026-10-12')])
re=e
# 0050 window returns for all decision dates (from 2004) using signal.csv dates
D=sig.index
ef=e.reindex(e.index.union(D)).ffill(); r0=window_returns(ef,D)
ewv=pd.read_csv(f'{OUT}/ew_variants.csv',index_col=0,parse_dates=True)
def nw(y,X,L):
    X=np.column_stack([np.ones(len(X)),X]); b=np.linalg.lstsq(X,y,rcond=None)[0]; u=y-X@b
    XtXi=np.linalg.inv(X.T@X); S=(X*u[:,None]).T@(X*u[:,None])
    for l in range(1,L+1):
        wl=1-l/(L+1); G=(X[l:]*u[l:,None]).T@(X[:-l]*u[:-l,None]); S+=wl*(G+G.T)
    V=XtXi@S@XtXi; return b[1], b[1]/np.sqrt(V[1,1])
for c in ewv.columns:
    d=(np.log1p(r0.reindex(ewv.index))-np.log1p(ewv[c])).dropna()
    for L in (6,12):
        S1=d.rolling(L).sum().shift(1)  # S1 at T = sum of windows ending at T
        df=pd.DataFrame({'y':d,'x':S1}).dropna()
        out=[]
        for a,b in [('2006-01-01','2018-12-31'),('2019-01-01','2022-12-31'),('2023-01-01','2026-09-30')]:
            q=df[(df.index>=a)&(df.index<=b)]
            for lag in (L,):
                bb,t=nw(q.y.values,q.x.values,lag); out.append('%+.3f(%.2f)n%d'%(bb,t,len(q)))
        print(c,L,out)
