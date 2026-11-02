import sys,json; sys.path.insert(0,'.')
from ev import *
d=second_td_prev(load_events())
o=pd.read_csv('my_oos_feat.csv.gz',dtype={'code':str},parse_dates=['s'])
cuts=np.array(json.load(open('ptests.json'))['run6_cuts'])
RUN6=np.full(R.shape,np.nan); RUN6[7:]=np.expm1(CS[7:-1]-CS[1:-7])
occ={}
for r in d[d.code.isin(cix)&(d.si<T)].itertuples(): occ.setdefault(cix[r.code],[]).append((r.si,max(r.ei,r.si)))
q=lambda v: np.searchsorted(cuts,v,side='right')
for code,dt,target in [('2025','2006-04-25',106),('2509','2006-05-17',57),('2458','2007-04-02',307),('2418','2007-04-18',403)]:
    r=o[(o.code==code)&(o.s==dt)].iloc[0]; s=r.si; x=r.xi
    base=(V60p[s]>=1e7)&(PC[s]>=10)&okR[s]&(np.nan_to_num(R[s],nan=9)<lim(s)-0.01)&~np.isnan(RUN6[s]); base[cix[code]]=False
    nod=np.array([not any(a<=x and b>=s-30 for a,b in occ.get(j,[])) for j in range(len(COLS))])
    inq=q(np.nan_to_num(RUN6[s],nan=-9))==q(r.run6)
    v={}
    v['base']=(base&nod&inq).sum()
    v['okx']=(base&nod&inq&okR[x]).sum()
    v['allok']=(base&nod&inq&okR[s:x+1].all(0)).sum()
    v['C_s10raw']=(base&nod&inq&(C[s]>=10)).sum()
    v['R>-thr']=(base&nod&inq&(np.nan_to_num(R[s])>-(lim(s)-0.01))).sum()
    v['nod60']=(base&inq&np.array([not any(a<=x and b>=s-60 for a,b in occ.get(j,[])) for j in range(len(COLS))])).sum()
    v['V60p_s1']=(base&nod&inq&(V60p[s+1]>=1e7)).sum()
    print(code,dt,target,v)
