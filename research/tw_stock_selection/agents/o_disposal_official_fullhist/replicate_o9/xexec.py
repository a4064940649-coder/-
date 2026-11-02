import sys,json; sys.path.insert(0,'/home/user/-/research/tw_stock_selection/agents/o_disposal_official_fullhist/replicate_o9')
from ev import *
cpi=json.load(open(A+'o_exec_capacity_tiers/raw/cpi_tw.json'))['index_2006_100']
def cpif(y): return cpi.get(str(max(2006,min(2026,y))))
def tick(p):
    return 0.01 if p<10 else 0.05 if p<50 else 0.1 if p<100 else 0.5 if p<500 else 1.0 if p<1000 else 5.0
def feerate(t,stress):
    return 0.001425*(0.6 if (stress or IDX[t]<pd.Timestamp('2015-01-01')) else 0.28)
def side_cost(t,j,q,Q,sell,stress):
    c=max(20.0,feerate(t,stress)*q)/q + max(tick(C[t,j])/C[t,j],0.001) + 0.5*np.nan_to_num(SIGp[t,j])*np.sqrt(q/Q)
    if sell: c+=0.003
    if stress: c+=0.004
    return c
def qref(t,j,ev):
    v=np.nan_to_num(V[t,j])
    if ev['si']<=t<=ev['ei']:
        if ev['regime']=='prepay_only': return 0.01*v
        mm=ev['mm'];  mm=20.0 if (mm is None or not np.isfinite(mm) or mm<=0) else mm
        return v/(270.0/mm)
    return 0.25*v
def sellable(t,j): return okR[t,j] and R[t,j]>-(lim(t)-0.01)
def exec_trade(ev, ent, ext, slot=100000.0, stress=False, cap=0.2):
    """ent: entry index (close). ext: planned exit index (close). Returns dict or None."""
    j=cix[ev['code']]
    N0=slot*cpif(IDX[ent].year)/cpif(2026)
    Q=qref(ent,j,ev)
    q=min(N0,cap*Q)
    if q<=0: return None
    cb=side_cost(ent,j,q,Q,False,stress)
    pos=q  # current market value fraction tracking via growth
    rem=1.0; proceeds=0.0; bench=0.0; t=ext
    last=lastR[j]
    while rem>1e-12:
        if t>last: t=last
        grow=np.expm1(CS[t+1,j]-CS[ent+1,j]); ewg=np.expm1(CE[t+1]-CE[ent+1])
        val=rem*q*(1+grow)
        if t==last:
            Qs=max(qref(t,j,ev),1e-9); sold=val
        else:
            if not sellable(t,j): t+=1; continue
            Qs=qref(t,j,ev); sold=min(val,cap*Qs)
            if sold<=0: t+=1; continue
        frac=sold/val*rem
        cs=side_cost(t,j,sold,max(Qs,1e-9),True,stress)
        proceeds+=sold*(1-cs); bench+=frac*ewg
        rem-=frac; t+=1
        if t>last and rem>1e-12: t=last
    ret=proceeds/(q*(1+cb))-1
    return dict(fill=q/N0, ret_x=ret, net_x=ret-bench, q=q)
