# -*- coding: utf-8 -*-
"""凍結後雜項檢查（由數支臨時腳本串接，各段獨立重載資料）：還原價獨有訊號的浮點平手、處置排除、漲停判定、月超額 Newey-West、門檻平手、拿掉量縮的分群穩健性、單一訊號路徑相依、2856／2018-01-13 幽靈日。python3 -I -B misc_checks.py"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# ---- t8
import rep_lib as L, numpy as np, pandas as pd
D=L.Data(); sig=L.signals(D); sigA=L.signals(D,mode='adj')
# (2) adj-only signals without F change in 251 bars
Fp=D.pack(D.F); ka=set(zip(sig.code,sig.di))
for r in sigA.itertuples():
    if (r.code,r.di) in ka: continue
    i=r.bar-1; w=Fp[max(i-250,0):i+1,r.j]
    if np.nanmax(w)/np.nanmin(w)-1<=1e-9:
        w2=Fp[max(i-300,0):i+1,r.j]
        print('adj-only no F change:',r.code,str(r.date)[:10],'d1 %.10f d2 %.10f d3 %.10f'%(r.d1,r.d2,r.d3),'close',r.close/D.F[r.di,r.j],'pivot',r.pivot/D.F[r.di,r.j],'prev_close',r.prev_close/D.F[r.di,r.j],'pivot_prev',r.pivot_prev/D.F[r.di,r.j],'low52x1.3',r.low52*1.3/D.F[r.di,r.j],'high52*.75',r.high52*.75/D.F[r.di,r.j],'stk-idx',r.stk_roc-r.idx_roc,'F chg in 300:',np.nanmax(w2)/np.nanmin(w2)-1>1e-9)
        # raw-mode values same bar
        sr=L.signals(D,ablate=('trend','rs','contract','voldry','cross','breakvol','minvol'),min_bars=r.bar) if False else None
# (3) contract-ablation untradable categories
s=L.signals(D,ablate=('contract',)); tr=L.tradable(D,s)
print('contract abl: n',len(s),'lim_up',int(tr.lim_up.sum()),'no_open',int(tr.no_open.sum()),'disp',int(tr.disp.sum()),'ok',int(tr.ok.sum()))
x=tr[tr.disp|tr.no_open]; print(x[['code','date','lim_up','no_open','disp']].to_string())
# borderline limit-ups: open exactly at limit, or flag!=0 at e
e=tr.ei.values; j=tr.j.values
tr['open_e']=D.O[e,j]; tr['uplim']=D.UPLIM[e,j]; tr['flag_e']=D.FLAG[e,j]; tr['ref']=D.REF[e,j]; tr['prevC']=[D.C[:ei,jj][~np.isnan(D.C[:ei,jj])][-1] for ei,jj in zip(e,j)]
lu=tr[tr.lim_up]; print('lim_up with flag_e!=0:'); print(lu[lu.flag_e!=0][['code','date','open_e','uplim','ref','prevC','flag_e']].to_string())
nl=tr[~tr.lim_up&(tr.flag_e!=0)]; nl=nl.assign(up_alt=[L.floor_tick((c/(1+r))*(1+l)) for c,r,l in zip(D.C[nl.ei.values,nl.j.values],D.R[nl.ei.values,nl.j.values],D.lim[nl.ei.values])])
print('not lim_up but flag_e!=0 and open>=alt limit (ref=close/(1+ret)):'); print(nl[nl.open_e>=nl.up_alt-1e-9][['code','date','open_e','uplim','up_alt','ref','prevC','flag_e']].to_string())
# default set: same check
tr0=L.tradable(D,sig); print('default lim_up',int(tr0.lim_up.sum()))
# (6) NW t on monthly excess, E3 cash/etf
etf=pd.Series(D.ETF,index=D.dates)
def nw(x,lags):
    x=np.asarray(x); n=len(x); e=x-x.mean(); v=(e**2).sum()
    for l in range(1,lags+1): v+=2*(1-l/(lags+1))*(e[l:]*e[:-l]).sum()
    return x.mean()/np.sqrt(v/n**2)
for idle in ('cash','etf'):
    nav=L.run_portfolio(D,sig,'E3',idle)['nav']; df=pd.DataFrame({'s':nav,'b':etf.reindex(nav.index)}); me=df.groupby([df.index.year,df.index.month]).tail(1); r=me.pct_change().dropna(); x=(r.s-r.b)
    print(idle,'t',x.mean()/(x.std(ddof=1)/np.sqrt(len(x))),'NW3',nw(x.values,3),'NW6',nw(x.values,6),'AC1',np.corrcoef(x[1:],x[:-1])[0,1], '2019+: t',x[x.index>='2019-01-01'].mean()/(x[x.index>='2019-01-01'].std(ddof=1)/np.sqrt((x.index>='2019-01-01').sum())),'NW3',nw(x[x.index>='2019-01-01'].values,3))
# ---- t9
import rep_lib as L, numpy as np, pandas as pd
D=L.Data(); sig=L.signals(D)
# float ties: 放寬/收緊等號
a=L.signals(D,dict(MaxLastDepth=10+1e-9,MinFirstDepth=10-1e-9)); b=L.signals(D,dict(MaxLastDepth=10-1e-9,MinFirstDepth=10+1e-9))
print('ties pass',len(a),'ties fail',len(b),'base',len(sig))
k=set(zip(sig.code,sig.date)); print('extra if ties pass:',[(c,str(d)[:10]) for c,d in zip(a.code,a.date) if (c,d) not in k])
# quarter-cluster t for no-voldry ablation
s=L.signals(D,ablate=('voldry',)); ev=L.event_study(D,s); ev['q']=ev.date.dt.to_period('Q').astype(str); ev['yr']=ev.date.dt.year
for h in (20,60):
    d=ev[ev['r%d'%h].notna()]
    for bn in ('pool','etf'):
        x=(d['r%d'%h]-d['%s%d'%(bn,h)]).values
        print('no-voldry',h,bn,'mean %.4f'%x.mean(),'t_month %.2f'%L.cluster_t(x,d.ym.values)['t'],'t_quarter %.2f'%L.cluster_t(x,d.q.values)['t'],'t_year %.2f'%L.cluster_t(x,d.yr.values)['t'],'median %.4f'%np.median(x))
    y=d.date.dt.year
    for nm,m in (('sel',(y>=2006)&(y<=2014)),('val',(y>=2015)&(y<=2020)),('hold',y>=2021),('2019+',y>=2019)):
        x=(d['r%d'%h]-d['pool%d'%h]).values[m]; x2=(d['r%d'%h]-d['etf%d'%h]).values[m]
        print('   ',nm,h,int(m.sum()),'pool %.4f (%.2f)'%(x.mean(),L.cluster_t(x,d.ym.values[m])['t']),'etf %.4f (%.2f)'%(x2.mean(),L.cluster_t(x2,d.ym.values[m])['t']))
# ---- t10
import rep_lib as L, numpy as np, pandas as pd
D=L.Data(); sig=L.signals(D)
def show(tag,s):
    for idle in ('cash','etf'):
        r=L.run_portfolio(D,s,'E3',idle); p=L.perf(D,r)
        print(tag,idle,'full %.2f%% ex %+.2f t %.2f | 2019 %.2f%% ex %+.2f t %.2f'%(p['full']['ann']*100,p['full']['ex']*100,p['full']['t'],p['2019']['ann']*100,p['2019']['ex']*100,p['2019']['t']), r['stats']['full'], r['stats']['cant_afford'])
show('base',sig)
s2=sig[~((sig.code=='3579')&(sig.date=='2018-01-15'))]; show('drop phantom-bar signal 3579',s2)
s3=sig[~(((sig.code=='3579')&(sig.date=='2018-01-15'))|((sig.code=='2856')&(sig.date=='2018-08-10')))]; show('drop 3579+2856',s3)
# leave-one-signal-out sensitivity on E3 (how much one signal moves the 2019+ number)
rng=np.random.RandomState(1); idx=rng.choice(len(sig),60,replace=False); out=[]
for i in idx:
    s=sig.drop(sig.index[i]); 
    pc=L.perf(D,L.run_portfolio(D,s,'E3','cash')); pe=L.perf(D,L.run_portfolio(D,s,'E3','etf'))
    out.append((pc['2019']['ex'],pe['2019']['ex'],pe['full']['ex'],pc['full']['ex']))
a=np.array(out)*100
print('leave-one-out (60 random): cash 2019 ex min %.2f max %.2f sd %.2f | etf 2019 ex min %.2f max %.2f sd %.2f | etf full min %.2f max %.2f | cash full min %.2f max %.2f'%(a[:,0].min(),a[:,0].max(),a[:,0].std(),a[:,1].min(),a[:,1].max(),a[:,1].std(),a[:,2].min(),a[:,2].max(),a[:,3].min(),a[:,3].max()))
# phantom date details
i=list(D.dates).index(pd.Timestamp('2018-01-13')); print('2018-01-13 weekday',pd.Timestamp('2018-01-13').day_name(),'bars',int(D.has[i].sum()),'src',pd.Series(D.SRC[i]).value_counts().to_dict())
# ---- t11
import rep_lib as L, numpy as np, pandas as pd
D=L.Data(); sig=L.signals(D)
tr=L.tradable(D,sig)
B='/home/user/-/research/tw_stock_selection/agents/o_ext_vcp_xs/'
t=pd.read_csv(B+'trades.csv.gz',dtype={'code':str},parse_dates=['signal_date','entry_date','exit_date'])
last=pd.Series([D.dates[np.nonzero(D.has[:,j])[0][-1]] for j in range(D.N)],index=D.codes)
for rule,mr in [('E1_hold20','E1'),('E2_hold60','E2'),('E3_stop8_ma50','E3'),('E4_stop8_tp20','E4'),('E5_close_below_pivot','E5'),('E6_trail10','E6')]:
    for idle in ('cash','etf'):
        r=L.run_portfolio(D,sig,mr,idle); mt=r['trades']
        bought=r['stats']['bought']
        # reconstruct never-closed: signals bought but not in trades -> emulate by tracking: rerun with log? simpler: entries = tradable signals minus those in trades... use nav decomposition
        print(mr,idle,'bought',bought,'closed',len(mt),'open at end',bought-len(mt))
    A=t[t.rule==rule]
    print('   researcher trades',len(A),'exit==last bar of stock:',int((A.exit_date.values==last.reindex(A.code).values).sum()))
# which positions are open at the end in E3 cash: find signals whose stock has last bar before panel end and entry near
j=list(D.codes).index('2856'); hb=np.nonzero(D.has[:,j])[0]
print('2856 last bar',D.dates[hb[-1]],'bars',len(hb)); sl=(D.dates>='2018-07-09')&(D.dates<='2018-07-16'); print(pd.DataFrame({'C':D.C[sl,j],'V':D.V[sl,j],'src':D.SRC[sl,j]},index=D.dates[sl]))
sl=(D.dates>='2018-08-08')&(D.dates<='2018-10-05'); x=pd.DataFrame({'O':D.O[sl,j],'C':D.C[sl,j],'A':D.A[sl,j],'ma50':D.AMA50[sl,j]},index=D.dates[sl]); print(x.dropna().tail(12))
print(sig[sig.code=='2856'][['code','date','close','pivot','bar']])
# zombie check generally: for every rule, any signal whose stock's last bar < entry+hold limit and before panel end
for r in sig.itertuples():
    lb=last[r.code]
    if lb<D.dates[-1] and (lb-r.date).days<400: print('signal near delisting:',r.code,str(r.date)[:10],'last bar',str(lb)[:10])
