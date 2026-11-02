import sys,json; sys.path.insert(0,'.')
from ev import *
d=second_td_prev(load_events())
o=pd.read_csv('my_oos_feat.csv.gz',dtype={'code':str},parse_dates=['s'])
RUN6=np.full(R.shape,np.nan); RUN6[7:]=np.expm1(CS[7:-1]-CS[1:-7])
occ={}
for r in d[d.code.isin(cix)&(d.si<T)].itertuples(): occ.setdefault(cix[r.code],[]).append((r.si,max(r.ei,r.si)))
for code,dt in [('2025','2006-04-25'),('2458','2007-04-02')]:
    r=o[(o.code==code)&(o.s==dt)].iloc[0]; s=r.si; x=r.xi
    base=(V60p[s]>=1e7)&(PC[s]>=10)&okR[s]&(np.nan_to_num(R[s],nan=9)<lim(s)-0.01)&~np.isnan(RUN6[s]); base[cix[code]]=False
    nod=np.array([not any(a<=x and b>=s-30 for a,b in occ.get(j,[])) for j in range(len(COLS))])
    v=np.sort(RUN6[s][base&nod]); print(code, 'n',len(v), 'run6 values around rank 307:', v[300:312], 'below 0.0321:', (v<0.032107).sum(), ' between .0321,.3059', ((v>=0.0321)&(v<0.30593)).sum())
# alternative run6 definitions for cuts
dv=d[d.valid]
alt={}
alt['valid_si']=np.nanpercentile([RUN6[r.si,cix[r.code]] for r in dv.itertuples()],[20,40,60,80])
# raw close based
RC=np.full(R.shape,np.nan); RC[7:]=C[6:-1]/C[:-7]-1
alt['rawC']=np.nanpercentile([RC[r.si,cix[r.code]] for r in dv.itertuples()],[20,40,60,80])
alt['adopt']=np.nanpercentile([RUN6[r.si,cix[r.code]] for r in dv[dv.adopt].itertuples()],[20,40,60,80])
print(alt)
