import sys, json, pickle
import numpy as np, pandas as pd
sys.path.insert(0, '/home/user/-/research/tw_stock_selection/agents/o_capacity_tsmc_gap/replicate_o1')
import rcore as R
from r2_tsmc import crosses
from r5_attr import stock_win, summarize_month, Sl, PK, buy, sell, ewnav, ew_win
def poolrank(cr):
    v=R.V60.iloc[cr.i].to_numpy(float); x=np.where(cr.el, v, np.nan)
    return pd.Series(x).rank(ascending=False, method='first').to_numpy()
for n in (10,30):
  picks=PK[(n,0.0)]
  for rkdef in ('vrank','poolrank'):
    for cdef in ('always','turnover','gross'):
      dp={}; dg={}; g1={}; g1e={}
      for k,d in enumerate(Sl):
        cr=crosses[d]; pk=picks[d]; prev=set(picks[Sl[k-1]]) if k>0 else set(); nx=set(picks[Sl[k+1]]) if k+1<len(Sl) else set()
        rr=stock_win(d,'oo'); r=rr.copy()
        for j in pk:
            b = buy if (cdef=='always' or j not in prev) else 0
            s = sell if (cdef=='always' or j not in nx) else 0
            if cdef=='gross': b=s=0
            r[j]=(1+rr[j])*(1-s)/(1+b)-1
        rk = cr.vrank if rkdef=='vrank' else poolrank(cr)
        ew=np.nanmean(rr[cr.el])
        A=[j for j in pk if rk[j]>100]; B=[j for j in pk if rk[j]<=100]
        if A and B:
            dp[d]=np.mean(r[A])-np.mean(r[B])
            dg[d]=(np.mean(r[A])-np.nanmean(rr[cr.el&(rk>100)]))-(np.mean(r[B])-np.nanmean(rr[cr.el&(rk<=100)]))
        C=[j for j in pk if rk[j]<=50]
        if C: g1[d]=np.mean(r[C])-ew
        g1e[d]=np.nanmean(rr[cr.el&(rk<=50)])-ew
      a=summarize_month(dp); b=summarize_month(dg); c=summarize_month(g1); e=summarize_month(g1e)
      print(n, rkdef, cdef, 'pool', {p:(a[p]['pp'],a[p]['t'],a[p]['n']) for p in ('SEL','VAL','ALL')}, 'grpALL', (b['ALL']['pp'],b['ALL']['t']), 'g1-50 VAL', c['VAL']['pp'], e['VAL']['pp'], 'HOLD', c['HOLD']['pp'], e['HOLD']['pp'])
