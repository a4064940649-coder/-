import sys,json; sys.path.insert(0,'/home/user/-/research/tw_stock_selection/agents/o_disposal_official_fullhist/replicate_o9')
from xexec import *
tr=run(second_td_prev(load_events()),heldcheck=False).rename(columns={'mm':'mm'})
tr=tr[(tr.s>='2006-04-01')]
d=second_td_prev(load_events())[['code','start','ord']]
tr=tr.merge(d,on=['code','start'],how='left').sort_values(['si','ann','ord']).reset_index(drop=True)
B50=0.000855; S50=0.000855+0.001
g50=np.r_[1.0,E50[1:]/E50[:-1]]
byday={}
for r in tr.itertuples(): byday.setdefault(r.si,[]).append(r)
def sleeve(tier=1e6, mode='FIX', stress=False, slots=10, rescale=True, start='2006-04-01', sell_first=False):
    t0=IDX.searchsorted(pd.Timestamp(start))
    K=lambda t: tier*cpif(IDX[t].year)/cpif(2026)
    z=K(t0); pos=[]; base=z; mret={}; daily=[]
    for t in range(t0,T):
        if t>t0: z*=g50[t]
        # sales
        keep=[]
        for p in pos:
            j=p['j']; last=lastR[j]
            if t>=p['x'] and (t>=last or sellable(t,j)):
                val=p['rem']*p['q']*np.exp(CS[t+1,j]-p['L0'])
                Qs=qref(t,j,p['ev']); sold=val if t>=last else min(val,0.2*Qs)
                if sold>0:
                    z+=sold*(1-side_cost(t,j,sold,max(Qs,1e-9),True,stress))*(1-B50)
                    p['rem']*=(1-sold/val)
            if p['rem']>1e-12: keep.append(p)
        nsold_today=len(pos)-len(keep)
        pos=keep
        posval=lambda: sum(p['q']*p['rem']*np.exp(CS[t+1,p['j']]-p['L0']) for p in pos)
        for r in byday.get(t,[]):
            nocc=len(pos) if sell_first else len(pos)+nsold_today
            if nocc>=slots: continue
            if any(p['j']==cix[r.code] for p in pos): continue
            j=cix[r.code]; ev=dict(code=r.code,si=r.si,ei=r.ei,regime=r.regime,mm=r.mm)
            tgt=K(t)/slots if mode=='FIX' else (z+posval())/slots
            Q=qref(t,j,ev); q=min(tgt,0.2*Q)
            if q<=0: continue
            need=q*(1+side_cost(t,j,q,Q,False,stress))
            z-=need/(1-S50)*g50[t]
            pos.append(dict(j=j,q=q,L0=CS[t+1,j],rem=1.0,ev=ev,x=r.xi))
        eq=z+posval(); daily.append((IDX[t],eq,len(pos)))
        if t+1>=T or IDX[t+1].to_period('M')!=IDX[t].to_period('M'):
            mret[IDX[t].to_period('M')]=eq/base-1
            if t+1<T:
                if mode=='FIX' and rescale:
                    f=K(t+1)/eq; z*=f
                    for p in pos: p['q']*=f
                    base=K(t+1)
                else: base=eq
    return pd.Series(mret), pd.DataFrame(daily,columns=['d','eq','npos'])
e50m=pd.Series(E50,index=IDX).resample('ME').last().pct_change()
e50m.index=e50m.index.to_period('M')
def stats(mr):
    b=e50m.reindex(mr.index); ex=mr-b
    def tt(x): return x.mean()/(x.std(ddof=1)/np.sqrt(len(x)))
    nav=(1+mr).cumprod(); yrs=len(mr)/12
    out=dict(cagr=round((nav.iloc[-1]**(1/yrs)-1)*100,2), cagr50=round(((1+b).prod()**(1/yrs)-1)*100,2),
        mdd=round(((nav/nav.cummax())-1).min()*100,1), ex=round(ex.mean()*100,3), t=round(tt(ex),3),
        t19=round(tt(ex[ex.index>=pd.Period('2019-01','M')]),3))
    for nm,a,b_ in [('seg1','2006-01','2012-12'),('seg2','2013-01','2018-12'),('seg3','2019-01','2026-12')]:
        x=ex[(ex.index>=pd.Period(a,'M'))&(ex.index<=pd.Period(b_,'M'))]; out[nm]=round(x.mean()*100,3)
    yr=ex.groupby(ex.index.year).sum(); best=yr.sort_values().index[-3:].tolist()
    x=ex[~ex.index.year.isin(best)]; out['best3']=best; out['t_drop_best3']=round(tt(x),3)
    # Newey-West t lag ~ 
    def nw(x,L=6):
        x=np.asarray(x)-np.mean(x); n=len(x); g0=(x@x)/n; s=g0
        for l in range(1,L+1): s+=2*(1-l/(L+1))*(x[l:]@x[:-l])/n
        return np.mean(np.asarray(ex.loc[x_index])) if False else None
    return out, ex
if __name__=='__main__':
    res={}
    for tier in [1e6,3e6,1e7]:
        for mode in ['FIX','GROW']:
            for st in [False,True]:
                if mode=='FIX' and tier!=1e6 and st: continue
                mr,dd=sleeve(tier,mode,st); s,ex=stats(mr)
                key=f'{mode}_{int(tier/1e4)}w_{"stress" if st else "normal"}'; res[key]=s; print(key,s)
                if key=='FIX_100w_normal': ex.to_csv('focus_ex.csv'); mr.to_csv('focus_mret.csv')
    json.dump(res,open('sleeve.json','w'),indent=1,default=str)
