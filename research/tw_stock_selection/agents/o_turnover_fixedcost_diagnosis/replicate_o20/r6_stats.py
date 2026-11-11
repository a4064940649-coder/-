import sys; sys.path.insert(0,"/home/user/-/research/tw_stock_selection/agents/o_turnover_fixedcost_diagnosis/replicate_o20")
from swaps import *
SP='/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/'
ex=pd.read_pickle(SP+'ex100.pkl'); en=pd.read_pickle(SP+'en100.pkl'); t=load_tl('100n'); rb=rebs()
rbk=rb.set_index(['j','k'])
y=ex[ex.signal>='2019-01-01'].copy(); s=ex[ex.signal<'2019-01-01'].copy()
# turnover split by exit bucket (notional/avail share)
av=t.groupby(['j','k']).avail.first()
for nm,df in [('Y19',y)]:
    df['tw']=df.notional/av.reindex(list(zip(df.j,df.k))).values
    sh=df.groupby('rb').tw.sum()/df.tw.sum()*100; print('exit turnover share by rank',sh.round(1).to_dict())
    sh2=df.groupby('rb').notional.sum()/df.notional.sum()*100; print(' (raw notional)',sh2.round(1).to_dict())
# entry source: rank at previous signal
rk=pd.read_csv(D+'ranks_seed0.csv.gz',dtype={'code':str}); rk=rk[rk.variant=='REAL_opt'].set_index(['j','signal','code'])
en=en.copy(); en['signal']=pd.to_datetime(en.signal)
ey=en[en.signal>='2019-01-01'].copy()
src=[]
for j,k,c in zip(ey.j,ey.k,ey.code):
    if k==0: src.append('first'); continue
    ps=rbk.loc[(j,k-1),'signal']
    try: q=rk.loc[(j,ps,c)]; src.append(bucket_rank(q['rank'],q['why']))
    except KeyError: src.append('not60')
ey['src']=src
print('entry src share (count)',(ey.src.value_counts(normalize=True)*100).round(1).to_dict())
print('entry src share (notional)',(ey.groupby('src').notional.sum()/ey.notional.sum()*100).round(1).to_dict())
# P3 trade-level
ty=t[(t.date>='2019-01-01')&(t.shares>0)]
lo=ty.open<50
print('P3 x (count share <50)',lo.mean().round(4),'cost share',(ty.cost[lo].sum()/ty.cost.sum()).round(4))
ts=t[(t.date<'2019-01-01')&(t.shares>0)]; lo2=ts.open<50; print('S0618 x',lo2.mean().round(4),'cost share',(ts.cost[lo2].sum()/ts.cost.sum()).round(4), 'ratio',(ts.cost[lo2].sum()/ts.cost.sum())/lo2.mean())
# also via signal-date filter
def p3(df):
    bad=df.edge<df.rtc; a=bad[df.price<50].mean(); b=bad[df.price>=100].mean(); return a,b,a-b
def p4(df):
    out={}
    for g,q in df.groupby('rb'):
        out[g]=dict(n=len(q),rate=round((q.edge<q.rtc).mean()*100,1),edge=round(q.edge.median()*100,2),rtc=round(q.rtc.median()*100,2))
    bad=df[df.edge<df.rtc]; out['share1115_in_bad']=round((bad.rb=='11~15').mean()*100,1)
    d=(df.edge-df.rtc); out['diff']=d[df.rb=='>50'].median()-d[df.rb=='11~15'].median()
    return out
for nm,df in [('Y19',y),('S0618',s)]:
    print(nm,'P3',[round(x*100,2) for x in p3(df)]); print(nm,'P4',p4(df))
    print(nm,'price buckets',{g:(len(q),round((q.edge<q.rtc).mean()*100,1)) for g,q in df.groupby('pb')})
# bootstrap
def boot(df,stat,B_=2000,seed=20261112):
    df=df.copy(); df['m']=df.signal.dt.to_period('M'); months=df.m.unique(); grp={m:g for m,g in df.groupby('m')}
    rng=np.random.default_rng(seed); vals=[]
    for b in range(B_):
        pick=rng.choice(len(months),len(months),replace=True)
        bb=pd.concat([grp[months[i]] for i in pick]); vals.append(stat(bb))
    vals=np.array(vals); return (vals<=0).mean()
p3s=lambda d:p3(d)[2]
def p4s(d):
    x=d.edge-d.rtc; return x[d.rb=='>50'].median()-x[d.rb=='11~15'].median()
for nm,df in [('Y19',y),('S0618',s)]:
    print(nm,'boot p P3',boot(df,p3s),'P4',boot(df,p4s))
# P5
yr=rb[pd.to_datetime(rb.signal)>='2019-01-01']; ne=ex.groupby(['j','k']).size()
yr=yr.assign(nx=ne.reindex(list(zip(yr.j,yr.k))).fillna(0).values)
print('P5 Y19 n reb',len(yr),'>=5',(yr.nx>=5).sum(),'<=2',(yr.nx<=2).sum())
sr=rb[(pd.to_datetime(rb.signal)<'2019-01-01')]; sr=sr.assign(nx=ne.reindex(list(zip(sr.j,sr.k))).fillna(0).values)
s2=s.join(ne.rename('nx'),on=['j','k'])
print('S0618 <=2 days',((sr.nx<=2)&(sr.nx>0)).sum(),'incl zero',(sr.nx<=2).sum(),'edge med <=2',s2[s2.nx<=2].edge.median(),' >=5',s2[s2.nx>=5].edge.median())
# breakeven
print('BE Y19 <50',y[y.price<50].rtc.median(),'>=100',y[y.price>=100].rtc.median(),'edge med',y.edge.median())
# cost paid on bad swaps pp/yr
ec=en.groupby(['j','k']).cost.sum(); nx=ex.groupby(['j','k']).size()
y['swcost']=y.cost+ec.reindex(list(zip(y.j,y.k))).values/nx.reindex(list(zip(y.j,y.k))).values
y['avail']=av.reindex(list(zip(y.j,y.k))).values
def dragsig(df):
    return np.median([ (g.swcost/g.avail).sum()/years('Y19on',j)*100 for j,g in df.groupby('j')])
print('drag bad',dragsig(y[y.edge<y.rtc]),'all',dragsig(y))
# AUC
from itertools import combinations
def auc(score,lab):
    r=pd.Series(score).rank().values; n1=lab.sum(); n0=len(lab)-n1
    a=(r[lab].sum()-n1*(n1+1)/2)/(n1*n0); return max(a,1-a)
sc=rk['score']
for nm,df in [('Y19',y),('S0618',s)]:
    lab=(df.edge<df.rtc).values
    feats={'rtc':df.rtc.values,'nx':nx.reindex(list(zip(df.j,df.k))).values,'price':df.price.values,'rank':df['rank'].fillna(999).values}
    print(nm,'AUC',{k:round(auc(v,lab),3) for k,v in feats.items()})
y.to_pickle(SP+'y100.pkl'); s.to_pickle(SP+'s100.pkl')
