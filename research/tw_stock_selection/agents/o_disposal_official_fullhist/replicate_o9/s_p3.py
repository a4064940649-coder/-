import sys,json; sys.path.insert(0,'.')
from ev import *
d=second_td_prev(load_events())
o=pd.read_csv('my_oos_feat.csv.gz',dtype={'code':str},parse_dates=['s'])
cuts=np.array(json.load(open('ptests.json'))['run6_cuts'])
RUN6=np.full(R.shape,np.nan); RUN6[7:]=np.expm1(CS[7:-1]-CS[1:-7])  # RUN6[t]= cum R over t-6..t-1
assert abs(RUN6[o.si.iloc[0],cix[o.code.iloc[0]]]-o.run6.iloc[0])<1e-12
# disposal occupancy per stock: intervals [si,ei] for all records with code in panel
dd=d[d.code.isin(cix)&(d.si<T)]
occ={}
for r in dd.itertuples(): occ.setdefault(cix[r.code],[]).append((r.si,max(r.ei,r.si)))
def has_disp(j,a,b): return any(x<=b and y>=a for x,y in occ.get(j,[]))
q=lambda v: np.searchsorted(cuts,v,side='right')
rows=[]
nst=len(COLS)
for r in o.itertuples():
    s=r.si; x=r.xi; jt=cix[r.code]
    elig=(V60p[s]>=1e7)&(PC[s]>=10)&okR[s]&(np.nan_to_num(R[s],nan=9)<lim(s)-0.01)&~np.isnan(RUN6[s])
    elig[jt]=False
    cand=[j for j in np.where(elig)[0] if not has_disp(j,s-30,x)]
    cand=np.array(cand,dtype=int)
    myq=q(r.run6)
    same=cand[q(RUN6[s,cand])==myq] if len(cand) else cand
    mode='q'
    if len(same)==0:
        same=cand[np.abs(RUN6[s,cand]-r.run6)<=0.05]; mode='pm5'
    if len(same)==0: mode='none'; pl=np.nan
    else: pl=np.mean(np.expm1(CS[x+1,same]-CS[s+1,same]))
    rows.append(dict(code=r.code,s=r.s,ret=r.ret,pl=pl,mode=mode,nc=len(same)))
p=pd.DataFrame(rows); p['dif']=p.ret-p.pl
print(p['mode'].value_counts(normalize=True))
pp=p[p['mode']!='none']
print(summ(pp,'dif'), 'n',len(pp))
p.to_csv('my_p3.csv.gz',index=False)
