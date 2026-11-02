# Independent replication (o9) of o_disposal_official_fullhist event layer.
import numpy as np, pandas as pd, sys, json
A='/home/user/-/research/tw_stock_selection/agents/'
z=np.load(A+'o_exec_engine_rewrite_v7x/data/x_cache.npz',allow_pickle=True)
GOOD=z['GOOD']
IDX=pd.DatetimeIndex(z['IDX'])[GOOD]
COLS=[str(c) for c in z['COLS']]
cix={c:i for i,c in enumerate(COLS)}
def g(k): return z[k][GOOD]
R=g('R'); C=g('C'); O=g('O'); V=g('V'); V60p=g('V60p'); PC=g('PC'); SIGp=g('SIGp')
T=len(IDX)
liq=(V60p>=1e7)&(PC>=10)
okR=~np.isnan(R)
ewmask=liq&okR
EW=np.where(ewmask,R,0).sum(1)/np.maximum(ewmask.sum(1),1)
EW[ewmask.sum(1)==0]=0
R0=np.nan_to_num(R)
LR=np.log1p(R0)       # stock cumulative log
LEW=np.log1p(EW)
CS=np.vstack([np.zeros(len(COLS)),np.cumsum(LR,0)])   # CS[t+1]=sum up to t
CE=np.concatenate([[0],np.cumsum(LEW)])
e50=pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/etf_0050.csv',parse_dates=['date']).set_index('date')['adj']
E50=e50.reindex(IDX.union(e50.index)).ffill().reindex(IDX).values
tw=pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/taiex_twii.csv',parse_dates=['date']).set_index('date')['close'].dropna()
bear=(tw<tw.rolling(200,min_periods=200).mean())
lastpx=np.array([np.where(~np.isnan(C[:,j]))[0].max() if (~np.isnan(C[:,j])).any() else -1 for j in range(len(COLS))])
lastR=np.array([np.where(okR[:,j])[0].max() if okR[:,j].any() else -1 for j in range(len(COLS))])

def lim(t): return 0.07 if IDX[t]<pd.Timestamp('2015-06-01') else 0.10

def load_events(sec_mode='cal', chain_invalid=True):
    d=pd.read_csv(A+'o_disposal_rejudge/disposal_hist_rejudge.csv',dtype={'code':str})
    d['ord']=np.arange(len(d))
    d=d[d.code.str.fullmatch(r'\d{4}')&~d.code.str.startswith('0')&~d.name.str.contains('DR')].copy()
    for c in ['announce','start','end']: d[c]=pd.to_datetime(d[c])
    si=IDX.searchsorted(d.start.values,'left'); ei=IDX.searchsorted(d.end.values,'right')-1
    d['si']=si; d['ei']=ei
    d['valid']=(si<T)&(ei>si)&d.code.isin(cix)
    d=d.sort_values(['code','start','announce','ord'])
    d['s_date']=[IDX[i] if i<T else pd.NaT for i in si] if False else None
    d['s_date']=[IDX[min(i,T-1)] if i<T else pd.Timestamp('2100-01-01') for i in d.si]
    d['e_date']=[IDX[i] if i>=0 else pd.NaT for i in d.ei]
    sec=[];adopt=[]
    for code,gg in d.groupby('code',sort=False):
        prev_e=[]; last_end=None
        for r in gg.itertuples():
            if sec_mode=='cal':
                lo=r.s_date-pd.Timedelta(days=30)
                sec.append(any((pe>=lo)&(pe<r.s_date) for pe in prev_e))
            else:
                sec.append(any((pe>=r.si-30)&(pe<r.si) for pe in prev_e_i) if False else False)
            prev_e.append(r.e_date)
            if last_end is not None and r.start<=last_end+pd.Timedelta(days=3):
                adopt.append(False)
            else:
                if chain_invalid or r.valid:
                    adopt.append(True); last_end=r.end
                else:
                    adopt.append(False)
    d['second']=sec; d['adopt']=adopt
    return d

def exit_idx(j,e):
    # R5
    if e>lastR[j]: return lastR[j], True
    t=e
    while t<T:
        if okR[t,j] and R[t,j]>-(lim(t)-0.01): return t,False
        if t>=lastR[j]: return lastR[j],True
        t+=1
    return None,False

def bear_before(t):
    p=tw.index[tw.index<IDX[t]]
    return bool(bear.loc[p[-1]])

def run(d, use_bear=True, thr_shift=0.0, liqv=1e7, entry_off=0, heldcheck=True):
    ev=d[d.valid&d.adopt].sort_values(['si','announce','ord'])
    held={}; out=[]
    for r in ev.itertuples():
        j=cix[r.code]; s=r.si; e=r.ei
        if heldcheck and r.code in held and s<=held[r.code]: continue
        if use_bear and bear_before(s): continue
        if not okR[s,j] or R[s,j]>=lim(s)-0.01+thr_shift: continue
        if not (V60p[s,j]>=liqv and PC[s,j]>=10): continue
        x,dl=exit_idx(j,e)
        if x is None or x<=s: continue
        held[r.code]=x
        ret=np.expm1(CS[x+1,j]-CS[s+1,j]); ew=np.expm1(CE[x+1]-CE[s+1])
        b50=E50[x]/E50[s]-1
        cost=0.012 if r.second else 0.008
        out.append(dict(code=r.code,ann=r.announce,start=r.start,end=r.end,s=IDX[s],e=IDX[e],x=IDX[x],si=s,ei=e,xi=x,
            second=r.second,kind=r.kind,regime=r.regime,mm=r.match_min,ret=ret,ew=ew,b50=b50,cost=cost,
            gross=ret-ew,net=ret-ew-cost,n50=ret-b50-cost,delist=dl))
    return pd.DataFrame(out)

def tm(df,col='net'):
    m=df.groupby(df.s.dt.to_period('M'))[col].mean()
    if len(m)<2: return np.nan,len(m)
    return m.mean()/(m.std(ddof=1)/np.sqrt(len(m))),len(m)

def summ(df,col='net'):
    if len(df)==0: return dict(n=0)
    t,nm=tm(df,col)
    return dict(n=len(df),mean=round(df[col].mean()*100,3),med=round(df[col].median()*100,3),win=round((df[col]>0).mean()*100,1),t=round(t,3),nm=nm)

def second_td_prev(d):
    d=d.sort_values(['code','start','announce','ord']).copy()
    out=[];anyv=[]
    for code,gg in d.groupby('code',sort=False):
        pe=[]
        for r in gg.itertuples():
            out.append(bool(pe) and (pe[-1]>=r.si-30) and (pe[-1]<r.si))
            anyv.append(any((x>=r.si-30)&(x<r.si) for x in pe))
            pe.append(r.ei)
    d['second_cal']=d['second']; d['second']=out; d['second_tdany']=anyv
    return d
