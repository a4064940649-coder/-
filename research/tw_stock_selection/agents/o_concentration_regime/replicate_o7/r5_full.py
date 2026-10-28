import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd, numpy as np, json
from rlib import *
P='/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel'
nav,e,plan,sig,ew=load()
idx=nav.index; END=pd.Timestamp('2026-10-12')
rm=nav['AB_V4|normal']; re=e.reindex(idx.union(e.index)).ffill().reindex(idx)
dd=pd.DatetimeIndex(plan.signal.tolist()+[END])
PER=dict(sel=('2019-01-01','2022-12-31'),oos=('2023-01-01','2026-09-30'),pre=('2006-05-01','2018-12-31'),full=('2006-05-01','2026-09-30'))
def annc(navd,a,b):  # calendar annualization between window endpoints
    yrs=(b-a).days/365.25; return (navd[b]/navd[a])**(1/yrs)-1
def win_end(T,dd): return dd[list(dd).index(T)+1]
def evaluate(rmain,reft,plan,dd,wt,cost,tag,frozen=0.4792,periods=('sel','oos'),verbose=True,extra=None):
    navR,_=simulate(rmain,reft,plan,wt,cost)
    R=window_returns(navR,dd); R0=window_returns(reft,dd).reindex(R.index); RM=window_returns(rmain,dd).reindex(R.index)
    w=wt.reindex(R.index); out={}
    navF,_=simulate(rmain,reft,plan,pd.Series(frozen,index=plan.signal),cost); F=window_returns(navF,dd).reindex(R.index)
    for k in periods:
        a,b=PER[k]; m=(R.index>=a)&(R.index<=b); wbar=w[m].mean()
        navS,_=simulate(rmain,reft,plan,pd.Series(wbar,index=plan.signal),cost); S=window_returns(navS,dd).reindex(R.index)
        A=R.index[m][0]; B=win_end(R.index[m][-1],dd)
        o=dict(n=int(m.sum()),wbar=wbar,annR=annc(navR,A,B),annS=annc(navS,A,B),annF=annc(navF,A,B),annM=annc(rmain,A,B),ann0=annc(reft,A,B),
               vsS=(R-S)[m].mean()*100,tS=tstat((R-S)[m]),IR=(R-S)[m].mean()/(R-S)[m].std()*np.sqrt(12),vsF=(R-F)[m].mean()*100,tF=tstat((R-F)[m]),
               vs0=(R-R0)[m].mean()*100,t0=tstat((R-R0)[m]),vsM=(R-RM)[m].mean()*100,tM=tstat((R-RM)[m]),sw=int((w[m].diff().abs()>0).sum()),
               mddR=mdd_daily(navR,A,B),mddM=mdd_daily(rmain,A,B),mdd0=mdd_daily(reft,A,B),mddS=mdd_daily(navS,A,B),sd=(R-S)[m].std()*100)
        o['_dS']=(R-S)[m]; o['_dF']=(R-F)[m]; o['_d0']=(R-R0)[m]; o['_dM']=(R-RM)[m]; o['_w']=w[m]; o['_R']=R[m]; o['_R0']=R0[m]; o['_RM']=RM[m]
        out[k]=o
        if verbose: print(tag,k,' '.join(f'{x}={v:.4f}' if isinstance(v,float) else f'{x}={v}' for x,v in o.items() if not x.startswith('_')))
    return out,navR
res={}
# ---------- main V5
wV5=(sig.S2>0).astype(float)
V5,navR=evaluate(rm,re,plan,dd,wV5.reindex(plan.signal),COST['normal'],'V5',periods=('sel','oos','pre','full'))
# extra benchmarks OOS/sel: MIX30, M_E50_30, AB_V4_B10, EW top300
navMIX,_=simulate(rm,re,plan,pd.Series(0.3,index=plan.signal),COST['normal'])
o6=pd.read_parquet(f'{RAW}/navs_integrate_o6.parquet')['M_E50_30|AB|normal']
for k in ('sel','oos'):
    a,b=PER[k]; Rk=V5[k]['_R']; A=Rk.index[0]; B=win_end(Rk.index[-1],dd)
    ewk=ew.top300.reindex(Rk.index); yrs=(B-A).days/365.25
    print(k,'MIX30 ann %.4f mdd %.4f'%(annc(navMIX,A,B),mdd_daily(navMIX,A,B)),'M_E50_30 %.4f'%annc(o6,A,B),'B10 %.4f'%annc(nav['AB_V4_B10|normal'],A,B),
          'EWtop300 win-ann %.4f cal-ann %.4f'%(ann(ewk),np.prod(1+ewk)**(1/yrs)-1))
# ---------- decomposition
for k in ('sel','oos'):
    o=V5[k]; w=o['_w']; wb=w.mean(); ewk=ew.top300.reindex(w.index)
    dpart=((w-wb)*(o['_R0']-ewk)).sum()*100; apart=(-(w-wb)*(o['_RM']-ewk)).sum()*100
    print(k,'decomp d-timing %.1fpp alpha-timing %.1fpp total %.1fpp sum(diffS) %.1fpp'%(dpart,apart,dpart+apart,o['_dS'].sum()*100))
# ---------- segments, block bootstrap
def segs(w): return (w.diff().abs()>0).cumsum()
rng=np.random.default_rng(7)
for k in ('sel','oos'):
    o=V5[k]; g=segs(o['_w']); print(k,'segments',g.nunique())
    for nm in ('_dS','_d0'):
        x=o[nm]; blocks=[x[g==s].values for s in g.unique()]; n=len(x); ms=[]
        for _ in range(20000):
            buf=[]
            while len(buf)<n: buf.extend(blocks[rng.integers(len(blocks))])
            ms.append(np.mean(buf[:n]))
        print('   block-boot',nm,'90%% [%.2f, %.2f]'%tuple(np.percentile(ms,[5,95])*100))
# MDE
o=V5['oos']; print('MDE80 %.2fpp sd %.2f'%((1.959964+0.841621)*o['sd']/np.sqrt(o['n']),o['sd']))
# ---------- endpoints (OOS)
dS=o['_dS']; dF=o['_dF']; d0=o['_d0']
for kk in (1,3):
    best=dS.sort_values(ascending=False).index[:kk]
    print('drop best',kk,[str(x.date()) for x in best],'vsS %.3f vsF %.3f vs0050 %.3f'%(dS.drop(best).mean()*100,dF.drop(best).mean()*100,d0.drop(best).mean()*100))
    bF=dF.sort_values(ascending=False).index[:kk]; b0=d0.sort_values(ascending=False).index[:kk]
    print('   own-best: vsF %.3f vs0050 %.3f'%(dF.drop(bF).mean()*100,d0.drop(b0).mean()*100))
x=pd.Timestamp('2023-12-11'); print('drop 2023-12-11->2024-02-15 window: vsS %.3f vsF %.3f vs0 %.3f'%(dS.drop(x).mean()*100,dF.drop(x).mean()*100,d0.drop(x).mean()*100))
# ---------- stress
rmS=nav['AB_V4|stress']
evaluate(rmS,re,plan,dd,wV5.reindex(plan.signal),COST['stress'],'V5stress',periods=('oos',))
# ---------- alt ETF legs, V6 leg, exec cells
for code in ('0052','006208'):
    s=etf_long(code); s=s.reindex(idx.union(s.index)).ffill().reindex(idx)
    evaluate(rm,s,plan,dd,wV5.reindex(plan.signal),COST['normal'],'V5_'+code,periods=('oos',))
v6=pd.read_parquet(f'{RAW}/navs_v6.parquet'); p6=pd.read_csv(f'{RAW}/plan_V6.csv',parse_dates=['signal','exec'])
p6=p6[p6.signal.isin(plan.signal)].reset_index(drop=True); print('plan_V6 same signals/exec as V4:', (p6.signal.values==plan.signal.values).all(), (p6['exec'].values==plan['exec'].values).all())
rm6=v6['AB_V6|normal'].reindex(idx)
evaluate(rm6,re,p6,dd,wV5.reindex(p6.signal),COST['normal'],'V5_V6leg',periods=('oos',))
ex=pd.read_parquet(f'{RAW}/navs_exec_tiers_abavg.parquet')
for tier in ('30萬','100萬','300萬','1000萬'):
    for gf in ('GROW','FIX'):
        for abc in ('A',):
            c=f'main|{tier}|{gf}|{abc}'
            evaluate(ex[c].reindex(idx),re,plan,dd,wV5.reindex(plan.signal),COST['normal'],'EXEC_'+c,periods=('oos',))
# ---------- other versions
lr0=np.log1p(window_returns(re.reindex(idx.union(e.index)).ffill() if False else e.reindex(e.index.union(sig.index)).ffill(),sig.index))
dlog=(lr0-np.log1p(ew.top300)).dropna()
S1={}
for L in (3,6,12): S1[f'L{L}']=dlog.rolling(L).sum().shift(1)
S1['12-1']=dlog.rolling(11).sum().shift(2)
def hyst(s,h):
    st=[];cur=np.nan
    for v in s.values:
        if np.isnan(v): st.append(np.nan); continue
        if np.isnan(cur): cur=1.0 if v>0 else 0.0
        if v>h: cur=1.0
        elif v<-h: cur=0.0
        st.append(cur)
    return pd.Series(st,s.index)
vers={}
for nm,key in (('V1','L3'),('V2','L6'),('V3','L12'),('V4','12-1')):
    vers[nm]=(S1[key]>0).astype(float).where(S1[key].notna())
    vers[nm+'neg']=(S1[key]<=0).astype(float).where(S1[key].notna())
vers['V6']=(sig.S2>0).astype(float)*0.7
vers['V7']=hyst(sig.S2,0.02)
vers['V8']=hyst(sig.S2,0.02)*0.7
vers['CORE50_SAT']=0.5+0.5*(sig.S2>0).astype(float)
for nm,wt in vers.items():
    evaluate(rm,re,plan,dd,wt.reindex(plan.signal),COST['normal'],nm,periods=('sel','oos'))
