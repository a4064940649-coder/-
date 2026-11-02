import sys,json; sys.path.insert(0,'.')
from ev import *
d=second_td_prev(load_events())
o=pd.read_csv('my_oos_feat.csv.gz',dtype={'code':str},parse_dates=['s'])
p=pd.read_csv(A+'o_disposal_official_fullhist/placebo_events.csv.gz',dtype={'code':str},parse_dates=['date'])
p=p[p.seg=='OOS']
RUN6=np.full(R.shape,np.nan); RUN6[7:]=np.expm1(CS[7:-1]-CS[1:-7])
ADJ=g('ADJ'); CF=g('CF')
occ={}
for r in d[d.code.isin(cix)&(d.si<T)].itertuples(): occ.setdefault(cix[r.code],[]).append((r.si,max(r.ei,r.si)))
dv=d[d.valid]
def go(RR,cuts):
    q=lambda v: np.searchsorted(cuts,v,side='right'); rows=[]
    for r in o.itertuples():
        s=r.si; x=r.xi; jt=cix[r.code]; r6=RR[s,jt]
        elig=(V60p[s]>=1e7)&(PC[s]>=10)&okR[s]&(np.nan_to_num(R[s],nan=9)<lim(s)-0.01)&~np.isnan(RR[s]); elig[jt]=False
        cand=np.array([j for j in np.where(elig)[0] if not any(a<=x and b>=s-30 for a,b in occ.get(j,[]))],dtype=int)
        same=cand[q(RR[s,cand])==q(r6)]; m='q'
        if len(same)==0: same=cand[np.abs(RR[s,cand]-r6)<=0.05]; m='pm5'
        if len(same)==0: rows.append(dict(code=r.code,s=r.s,dif=np.nan,m='none',n=0,b=q(r6))); continue
        rows.append(dict(code=r.code,s=r.s,dif=r.ret-np.mean(np.expm1(CS[x+1,same]-CS[s+1,same])),m=m,n=len(same),b=q(r6)))
    return pd.DataFrame(rows)
for nm,RR in [('rawC',None),('CF',None),('adjC',None)]:
    X={'rawC':C,'CF':CF,'adjC':C*ADJ}[nm]
    RR=np.full(R.shape,np.nan); RR[7:]=X[6:-1]/X[:-7]-1
    cuts=np.nanpercentile([RR[r.si,cix[r.code]] for r in dv.itertuples() if r.si>=7],[20,40,60,80])
    res=go(RR,cuts); pp=res[res.m!='none']
    mm=res.merge(p,left_on=['code','s'],right_on=['code','date'])
    print(nm,cuts,len(pp)/len(res),summ(pp,'dif'),'bucket mism',(mm.b!=mm.bucket).sum(),'n mism',(mm.n!=mm.pl_n).sum())
RR=np.full(R.shape,np.nan); RR[7:]=C[6:-1]/C[:-7]-1
cuts=np.nanpercentile([RR[r.si,cix[r.code]] for r in dv.itertuples() if r.si>=7],[20,40,60,80])
res=go(RR,cuts); mm=res.merge(p,left_on=['code','s'],right_on=['code','date'])
mm['dn']=mm.n-mm.pl_n
print(mm.dn.describe()); print(mm[mm.dn!=0][['code','s','b','bucket','n','pl_n','dif','ev_minus_pl']].head(15))
