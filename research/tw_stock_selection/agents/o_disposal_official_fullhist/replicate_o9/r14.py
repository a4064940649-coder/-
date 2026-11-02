import sys,json; sys.path.insert(0,'.')
from ev import *
d=second_td_prev(load_events())
o=pd.read_csv('my_oos_feat.csv.gz',dtype={'code':str},parse_dates=['s'])
RUN6=np.full(R.shape,np.nan); RUN6[7:]=np.expm1(CS[7:-1]-CS[1:-7])
dv=d[d.valid]
c_valid=np.nanpercentile([RUN6[r.si,cix[r.code]] for r in dv.itertuples()],[20,40,60,80])
da=d[d.code.isin(cix)&(d.si<T)&(d.si>=7)]
c_all=np.nanpercentile([RUN6[r.si,cix[r.code]] for r in da.itertuples()],[20,40,60,80])
print(c_valid,c_all)
full=pd.read_csv(A+'o_disposal_rejudge/disposal_hist_rejudge.csv',dtype={'code':str})
full['start']=pd.to_datetime(full.start); full['end']=pd.to_datetime(full.end)
full=full[full.code.isin(cix)]
full['si']=IDX.searchsorted(full.start.values,'left'); full['ei']=IDX.searchsorted(full.end.values,'right')-1
def mkocc(df):
    occ={}
    for r in df.itertuples(): occ.setdefault(cix[r.code],[]).append((r.si,max(r.ei,r.si)))
    return occ
def go(cuts,occ,mode_ex='overlap',win30=30,liq_x=False):
    q=lambda v: np.searchsorted(cuts,v,side='right')
    rows=[]
    for r in o.itertuples():
        s=r.si; x=r.xi; jt=cix[r.code]
        elig=(V60p[s]>=1e7)&(PC[s]>=10)&okR[s]&(np.nan_to_num(R[s],nan=9)<lim(s)-0.01)&~np.isnan(RUN6[s])
        elig[jt]=False
        if mode_ex=='overlap': f=lambda j: any(a<=x and b>=s-win30 for a,b in occ.get(j,[]))
        else: f=lambda j: any(s-win30<=a<=x for a,b in occ.get(j,[]))
        cand=np.array([j for j in np.where(elig)[0] if not f(j)],dtype=int)
        same=cand[q(RUN6[s,cand])==q(r.run6)] if len(cand) else cand; m='q'
        if len(same)==0: same=cand[np.abs(RUN6[s,cand]-r.run6)<=0.05]; m='pm5'
        if len(same)==0: rows.append(dict(s=r.s,dif=np.nan,m='none')); continue
        rows.append(dict(s=r.s,dif=r.ret-np.mean(np.expm1(CS[x+1,same]-CS[s+1,same])),m=m))
    p=pd.DataFrame(rows); pp=p[p.m!='none']
    return len(pp)/len(p), summ(pp,'dif')
oc=mkocc(d[d.code.isin(cix)&(d.si<T)]); of=mkocc(full)
for nm,cu,occ,me in [('valid,common,ovl',c_valid,oc,'overlap'),('all,common,ovl',c_all,oc,'overlap'),('valid,full,ovl',c_valid,of,'overlap'),('valid,common,start',c_valid,oc,'start'),('all,full,ovl',c_all,of,'overlap')]:
    print(nm, go(cu,occ,me))
