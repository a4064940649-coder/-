import sys; sys.path.insert(0,'/home/user/-/research/tw_stock_selection/agents/o_disposal_official_fullhist/replicate_o9')
from xexec import *
import itertools
d=second_td_prev(load_events())
def limprice(pc,t,up=True):
    L=lim(t); p=pc*(1+L) if up else pc*(1-L); tk=tick(p)
    return (np.floor(p/tk+1e-9)*tk) if up else (np.ceil(p/tk-1e-9)*tk)
def lvl_close(t,j): return CS[t+1,j]
def lvl_open(t,j): return CS[t+1,j]-np.log(C[t,j]/O[t,j])
def r5_from(j,t):
    last=lastR[j]
    while t<=last:
        if sellable(t,j): return t
        t+=1
    return last
def run9(ek,xk,stress=False):
    ev_=d[d.valid&d.adopt].sort_values(['si','announce','ord'])
    held={}; out=[]
    for r in ev_.itertuples():
        j=cix[r.code]; s=r.si; e=r.ei
        if r.code in held and s<=held[r.code]: continue
        if bear_before(s): continue
        if not (V60p[s,j]>=1e7 and PC[s,j]>=10): continue
        if ek=='C1':
            if not okR[s,j] or R[s,j]>=lim(s)-0.01: continue
            ent=s; L0=lvl_close(s,j); ew0=CE[s+1]; open_ent=False
        else:
            if s+1>=T: continue
            t1=s+1
            if ek=='C2':
                if not okR[t1,j] or R[t1,j]>=lim(t1)-0.01: continue
                ent=t1; L0=lvl_close(t1,j); ew0=CE[t1+1]; open_ent=False
            else:
                if not (okR[t1,j] and np.isfinite(O[t1,j]) and O[t1,j]>0): continue
                if O[t1,j]>=limprice(C[s,j],t1,True)-1e-9: continue
                ent=t1; L0=lvl_open(t1,j); ew0=CE[t1]; open_ent=True   # EW from close s
        # exit plan
        open_exit=False
        if xk=='CL': x=r5_from(j,e) if e<=lastR[j] else lastR[j]
        elif xk=='CL1': x=r5_from(j,e-1) if e-1<=lastR[j] else lastR[j]
        else:
            t2=e+1
            if t2<=lastR[j] and okR[t2,j] and np.isfinite(O[t2,j]) and O[t2,j]>limprice(C[e,j] if np.isfinite(C[e,j]) else PC[t2,j],t2,False)+1e-9:
                x=t2; open_exit=True
            else: x=r5_from(j,min(t2,lastR[j]))
        if x<ent or (x==ent and not (open_ent and not open_exit)): continue
        if x==ent and open_ent and open_exit: continue
        held[r.code]=x
        evd=dict(code=r.code,si=s,ei=e,regime=r.regime,mm=r.match_min)
        N0=100000.0*cpif(IDX[ent].year)/cpif(2026)
        Q=qref(ent,j,evd); q=min(N0,0.2*Q)
        if q<=0: continue
        cb=side_cost(ent,j,q,Q,False,stress)
        rem=1.0; proc=0.0; bench=0.0; t=x; first=True; last=lastR[j]
        while rem>1e-12:
            if first and open_exit:
                L1=lvl_open(t,j); ew1=CE[t]  # EW to close e
                Qs=0.25*np.nan_to_num(V[t,j])
            else:
                if t>last: t=last
                if t<last and not sellable(t,j): t+=1; continue
                L1=lvl_close(t,j); ew1=CE[t+1]; Qs=qref(t,j,evd)
            val=rem*q*np.exp(L1-L0)
            sold=val if (t>=last and not (first and open_exit)) else min(val,0.2*Qs)
            if sold<=0: first=False; t+=1; continue
            frac=sold/val*rem
            proc+=sold*(1-side_cost(t,j,sold,max(Qs,1e-9),True,stress)); bench+=frac*np.expm1(ew1-ew0)
            rem-=frac
            if first and open_exit: first=False   # remainder to close of same day onward
            else: t+=1
        net=proc/(q*(1+cb))-1-bench
        out.append(dict(s=IDX[s],code=r.code,net_x=net))
    return pd.DataFrame(out)
rows=[]
for ek,xk in itertools.product(['C1','O2','C2'],['CL','CL1','OA']):
    for st in [False,True]:
        df=run9(ek,xk,st); f=df[(df.s>='2006-04-01')&(df.s<='2026-10-22')]
        s=summ(f,'net_x'); rows.append(dict(ek=ek,xk=xk,stress=st,**s)); print(ek,xk,st,s)
R9=pd.DataFrame(rows); R9.to_csv('g1_9.csv',index=False)
for st in [False,True]:
    m=R9[R9.stress==st]['mean']; print('stress' if st else 'normal','median',m.median(),'p10',np.percentile(m,10))
