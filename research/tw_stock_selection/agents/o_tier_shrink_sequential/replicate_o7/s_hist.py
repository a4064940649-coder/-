import sys; sys.path.insert(0,'/home/user/-/research/tw_stock_selection/agents/o_tier_shrink_sequential/replicate_o7')
from rlib import *
fw=pd.read_parquet(RAW+'o6shrink/focus_windows.parquet')
A=pd.read_parquet(RAW+'navs/navs_abavg.parquet'); EB=pd.read_parquet(RAW+'exec/bench.parquet')
z=wret(EB['AB_V4|normal'])
idx=fw.index; grid=wgrid()
P={'ALL':('2006-04-01','2026-10-12'),'SEL':('2006-04-01','2019-04-30'),'VAL':('2019-05-01','2023-12-31'),'HOLD':('2024-01-01','2026-10-12')}
def stats_(r,bench=None,per='ALL'):
    a,b=P[per]; m=(idx>=a)&(idx<=b); rr=np.asarray(r)[m]
    st=[g for g,mm in zip(grid[:-1],m) if mm][0]; en=grid[1:][np.where(m)[0][-1]]
    yrs=(en-st).days/365.25
    nav=np.cumprod(1+rr); cagr=nav[-1]**(1/yrs)-1
    mdd=(nav/np.maximum.accumulate(np.concatenate([[1],nav]))[1:]-1).min()
    out=dict(cagr=round(cagr*100,1),mdd=round(mdd*100,1))
    if bench is not None:
        d=rr-np.asarray(bench)[m]; out['t']=round(d.mean()/(d.std(ddof=1)/np.sqrt(len(d))),2)
    return out
def stats_alt(r,per='ALL'):
    a,b=P[per]; m=(idx>=a)&(idx<=b); rr=np.asarray(r)[m]
    return round(((np.prod(1+rr))**(12/len(rr))-1)*100,1)
r5=fw.r0050.values
FEE=0.001425*0.6
C_M_SELL,C_M_BUY=FEE+0.001+0.003,FEE+0.001
C_E_SELL,C_E_BUY=FEE+0.001+0.0005,FEE+0.0005
def mix(rm,x):
    out=[]
    for a,b in zip(rm,r5):
        wm=x*(1+a); we=(1-x)*(1+b); W=wm+we; d=wm-x*W
        c=d*(C_M_SELL+C_E_BUY) if d>0 else -d*(C_E_SELL+C_M_BUY); out.append(W-c-1)
    return np.array(out)
for per in P: print('0050',per,stats_(r5,None,per),stats_alt(r5,per))
for per in ['ALL','HOLD']: print('AB',per,stats_(fw.AB.values,r5,per))
exec100=fw.B10.values+(wret(A['main|100萬|FIX|A'])-z).values
print('100F x1',stats_(exec100,r5,'ALL'))
m90=mix(exec100,0.9)
print('100F x0.9',stats_(m90,r5,'ALL'),'HOLD',stats_(m90,r5,'HOLD'),'vs AB t',stats_(m90,fw.AB.values,'ALL'))
g1000=fw.B10.values+(wret(A['main|1000萬|GROW|A'])-z).values
print('1000G x1',stats_(g1000,r5,'ALL'),'HOLD',stats_(g1000,r5,'HOLD'))
print('EW',stats_(fw.EW.values,r5,'ALL'))
print('--- 12/n annualization')
print('AB',stats_alt(fw.AB.values),stats_alt(fw.AB.values,'HOLD'),'100F',stats_alt(exec100),'x.9',stats_alt(m90),stats_alt(m90,'HOLD'),'1000G',stats_alt(g1000),stats_alt(g1000,'HOLD'),'EW',stats_alt(fw.EW.values))
d=m90-exec100; print('t(x.9 - x1 same cell)',d.mean()/(d.std(ddof=1)/np.sqrt(len(d))))
