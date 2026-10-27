import sys; sys.path.insert(0,'/home/user/-/research/tw_stock_selection/agents/o_capacity_tsmc_gap/replicate_o1')
import rcore as R, pandas as pd, numpy as np
import importlib
NAV=pd.read_parquet(R.HERE/'navs_tsmc.parquet'); S=R.SIG[R.SIG<=R.EVAL_END]; B=R.E50
nav0=NAV['n10|w0|normal'].dropna(); idx=nav0.index
r_s=nav0.pct_change(); r_b=B.reindex(idx).pct_change()
buy,sell=R.COST['normal']
def run(resets, cfun):
    wS,wB=0.7,0.3; val=[1.0]
    for k in range(1,len(idx)):
        d=idx[k]; wS*=1+r_s.iloc[k]; wB*=1+r_b.iloc[k]; v=wS+wB
        if d in resets:
            tS=0.7*v; dS=tS-wS
            c=cfun(dS)
            v2=v-c; wS,wB=0.7*v2,0.3*v2; v=v2
        val.append(v)
    return pd.Series(val,index=idx)
pb=R.win_returns(B,S)
def st(m):
    pr=R.win_returns(m,S); ex=pr-pb
    a=ex[(ex.index>='2019-04-01')&(ex.index<='2026-09-30')]
    return round(R.cagr(m,'2006-04-12','2026-10-12')*100,2), round(R.tstat(ex),3), round(R.cagr(m,'2019-04-12','2026-10-12')*100,2), round(R.tstat(a),3)
sig=set(S); exd=set(R.IDX[R.IDX.get_loc(d)+1] for d in S if d<R.EVAL_END)
for nm,cf in [('nocost',lambda x:0),('both_legs',lambda x: abs(x)*(buy+sell)),('one_leg',lambda x: abs(x)*(buy if x>0 else sell))]:
    for rn,rs in [('sig',sig),('exec',exd)]:
        print(nm, rn, st(run(rs,cf)))
