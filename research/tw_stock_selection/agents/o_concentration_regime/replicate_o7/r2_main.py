import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd, numpy as np, json, sys
from rlib import *
nav,e,plan,sig,ew=load()
idx=nav.index
rm=nav['AB_V4|normal']; re=e.reindex(idx).ffill()
dd=pd.DatetimeIndex(plan.signal.tolist()+[pd.Timestamp('2026-10-12')])
S2=sig.S2
w_rot=(S2>0).astype(float).reindex(plan.signal)
navR,tr=simulate(rm,re,plan,w_rot,COST['normal'])
R=window_returns(navR,dd)
R0=window_returns(re,dd).reindex(R.index); RM=window_returns(rm,dd).reindex(R.index)
per=dict(sel=('2019-01-01','2022-12-31'),oos=('2023-01-01','2026-09-30'),pre=('2006-05-01','2018-12-31'),full=('2006-05-01','2026-09-30'))
w=w_rot.reindex(R.index)
for k,(a,b) in per.items():
    m=(R.index>=a)&(R.index<=b)
    wbar=w[m].mean()
    navS,_=simulate(rm,re,plan,pd.Series(wbar,index=plan.signal),COST['normal'])
    S=window_returns(navS,dd).reindex(R.index)
    navF,_=simulate(rm,re,plan,pd.Series(0.4792,index=plan.signal),COST['normal'])
    F=window_returns(navF,dd).reindex(R.index)
    d=(R-S)[m]; d0=(R-R0)[m]; dm=(R-RM)[m]; dF=(R-F)[m]
    # switches within period: state changes between consecutive windows in period
    ws=w[m]; sw=int((ws.diff().abs()>0).sum())
    print(k,'n',m.sum(),'wbar %.4f'%wbar,'ann R %.3f S %.3f F %.3f M %.3f 0050 %.3f'%(ann(R[m]),ann(S[m]),ann(F[m]),ann(RM[m]),ann(R0[m])),
          'EW',{c:round(ann(ew[c].reindex(R.index)[m]),3) for c in ew.columns})
    print('   vsS %.3fpp t %.2f IR %.2f | vsF %.3f t %.2f | vs0050 %.3f t %.2f | vsM %.3f t %.2f | sw %d'%(d.mean()*100,tstat(d),d.mean()/d.std()*np.sqrt(12),dF.mean()*100,tstat(dF),d0.mean()*100,tstat(d0),dm.mean()*100,tstat(dm),sw))
    a0,b0=R.index[m][0],dd[list(dd).index(R.index[m][-1])+1]
    print('   MDD R %.3f M %.3f 0050 %.3f S %.3f'%(mdd_daily(navR,a0,b0),mdd_daily(rm,a0,b0),mdd_daily(re,a0,b0),mdd_daily(navS,a0,b0)), 'sd diff %.4f'%d.std())
