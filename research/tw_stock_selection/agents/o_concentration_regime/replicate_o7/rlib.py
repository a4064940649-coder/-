# Replication library: two-leg daily engine with switching at exec close
import pandas as pd, numpy as np, os
RAW='/home/user/-/research/tw_stock_selection/agents/o_concentration_regime/raw'
OUT=os.path.dirname(os.path.abspath(__file__))
FEE=0.001425*0.6
COST=dict(normal=dict(m_buy=FEE+0.001,m_sell=FEE+0.003+0.001,e_buy=FEE+0.0005,e_sell=FEE+0.001+0.0005),
          stress=dict(m_buy=0.005,m_sell=0.005,e_buy=FEE+0.0005+0.003,e_sell=FEE+0.001+0.0005+0.003))
def load():
    nav=pd.read_parquet(f'{RAW}/navs_v4.parquet')
    e=pd.read_csv(f'{RAW}/etf_0050.csv',parse_dates=['date'],index_col='date')['adj']
    plan=pd.read_csv(f'{RAW}/plan_V4.csv',parse_dates=['signal','exec'])
    sig=pd.read_csv(f'{OUT}/signal.csv',parse_dates=['T'],index_col='T')
    ew=pd.read_csv(f'{OUT}/ew.csv',parse_dates=['T'],index_col='T')
    return nav,e,plan,sig,ew
def etf_long(code):
    d=pd.read_csv(f'{RAW}/etf_long.csv.gz',dtype={'code':str},parse_dates=['date'])
    a=d[d.code==code].set_index('date')['adj'].copy()
    if code=='0052':  # 1:7 split on 2025-11-17 not adjusted in source
        a[a.index<pd.Timestamp('2025-11-17')]/=7.0
    return a
def simulate(rm, re, plan, wtarget, cost, start_w=None):
    """rm, re: daily NAV-level series (main, etf) on common index. plan: signal/exec.
    wtarget: Series indexed by signal date -> target 0050 weight, applied at exec close.
    returns daily NAV series of portfolio."""
    idx=rm.index
    rmr=rm.pct_change().fillna(0).values; rer=re.reindex(idx).ffill().pct_change().fillna(0).values
    ex=dict(zip(plan['exec'],plan['signal']))
    E=np.nan; M=np.nan; nav=np.full(len(idx),np.nan); trades=[]
    for k,d in enumerate(idx):
        if not np.isnan(E):
            E*=1+rer[k]; M*=1+rmr[k]
        if d in ex:
            s=ex[d]
            if s not in wtarget.index or np.isnan(wtarget[s]): 
                nav[k]=E+M; continue
            w=wtarget[s]
            if np.isnan(E):
                E=w; M=1-w; nav[k]=1.0; continue
            V=E+M; tE=w*V
            if tE>E+1e-12:   # sell main, buy etf
                x=tE-E   # target gross; solve with costs: sell y of M -> y*(1-ms)*(1-eb) into E
                # value after costs: choose y s.t. E+y(1-ms)(1-eb) = w*(E+M-y+y(1-ms)(1-eb))
                f=(1-cost['m_sell'])*(1-cost['e_buy'])
                y=(w*(E+M)-E)/(f-w*f+w)
                E+=y*f; M-=y; trades.append((d,'toE',y/V))
            elif tE<E-1e-12:
                f=(1-cost['e_sell'])*(1-cost['m_buy'])
                tM=(1-w)
                y=(tM*(E+M)-M)/(f-tM*f+tM)
                M+=y*f; E-=y; trades.append((d,'toM',y/V))
        nav[k]=E+M
    return pd.Series(nav,idx),trades
def window_returns(navd, dd):
    s=navd.reindex(dd)
    return (s.shift(-1)/s-1).dropna()
def ann(r): r=r.dropna(); return np.prod(1+r)**(12/len(r))-1
def mdd_daily(navd,a,b):
    s=navd[(navd.index>=a)&(navd.index<=b)].dropna(); s=s/s.iloc[0]; return (s/s.cummax()-1).min()
def tstat(x): x=pd.Series(x).dropna(); return x.mean()/x.std(ddof=1)*np.sqrt(len(x))
