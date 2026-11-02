import sys, json; sys.path.insert(0,'.')
from ev import *
d=second_td_prev(load_events()); tr=run(d)
tr.to_csv('my_trades.csv.gz',index=False)
P=lambda a,b: tr[(tr.s>=a)&(tr.s<=b)]
o=P('2006-04-01','2021-09-30')
res={}
def S(df,col='net'):
    s=summ(df,col); return s
res['OOS']=S(o); res['OOS_gross']=S(o,'gross'); res['OOS_n50']=S(o,'n50')
k=int(np.floor(0.05*len(o))); res['OOS_extop5']=round(o.net.sort_values().iloc[:len(o)-k].mean()*100,3)
for c in [0.012,0.02]:
    res['cost%.3f'%c]=S(o.assign(nc=o.gross-c),'nc')
# spec-literal second sensitivity
dd=d.set_index(['code','start'])
for col in ['second_tdany','second_cal']:
    sc=np.array([bool(np.atleast_1d(dd.loc[(a,b),col])[0]) for a,b in zip(o.code,o.start)])
    res['OOS_'+col]=S(o.assign(nc=o.gross-np.where(sc,0.012,0.008)),'nc')
# worst case: events blocked only because no price at s
ev_=d[d.valid|~d.code.isin(cix)]
ev_=d[d.adopt].sort_values(['si','announce','ord'])
ev_=ev_[(ev_.s_date>='2006-04-01')&(ev_.s_date<='2021-09-30')]
held={r.code:r.xi for r in tr.itertuples()}  # approximate: check overlap with actual trades
miss=[]
for r in ev_.itertuples():
    if not r.valid and r.code in cix: continue
    if r.code in cix:
        j=cix[r.code]
        if okR[r.si,j]: continue
    # held?
    h=tr[(tr.code==r.code)&(tr.si<r.si)]
    if len(h) and r.si<=h.xi.max(): continue
    if bear_before(r.si): continue
    miss.append((r.code,str(r.s_date.date()),r.second))
res['missing_events']=miss
w=np.concatenate([o.net.values,[-0.30-0.0]*len(miss)])
mo=pd.concat([o[['s','net']],pd.DataFrame({'s':[pd.Timestamp(x[1]) for x in miss],'net':-0.30})])
res['worst']=S(mo)
# seen/unseen via proxy events: overlap of [s,x] with proxy [start,end]
px=pd.read_csv(A+'o_disposal_rejudge/proxy_events.csv',dtype={'code':str},parse_dates=['start','end'])
def seen(r):
    g=px[px.code==r.code]; return bool(((g.start<=r.x)&(g.end>=r.s)).any())
o=o.assign(seen=[seen(r) for r in o.itertuples()])
res['unseen']=S(o[~o.seen]); res['seen']=S(o[o.seen])
for nm,a,b in [('p06_15','2006-04-01','2015-12-31'),('p16_19','2016-01-01','2019-12-31'),('p20_21','2020-01-01','2021-09-30'),
   ('dt06_13','2006-04-01','2013-12-31'),('dt14_17','2014-01-01','2017-04-27'),('dt17_21','2017-04-28','2021-09-30'),
   ('pre04_06','2004-01-01','2006-03-31'),('sel','2021-10-01','2023-12-31'),('desc','2024-01-01','2026-10-22'),('new','2026-08-10','2026-10-22'),('full','2006-04-01','2026-10-22')]:
    res[nm]=S(P(a,b))
json.dump(res,open('event_results.json','w'),indent=1,default=str)
for k,v in res.items(): print(k,v)
