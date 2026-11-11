# 獨立複製員 o20：只依規格文字（SOP、ER2_SPEC、a_register 讀法）重建 C0 2006～2013 名單，與 lists_C0 比對
import json, sys, numpy as np, pandas as pd
P='/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel/'
A='/home/user/-/research/tw_stock_selection/agents/'
OUT=A+'o_g6_allpit_list_trace_stage2/replicate_o20/'
OPT=dict(pool_order=sys.argv[1] if len(sys.argv)>1 else 'top300_then_filter')
c=pd.read_parquet(P+'close_raw.parquet'); adj=pd.read_parquet(P+'adj.parquet'); val=pd.read_parquet(P+'value.parquet'); op=pd.read_parquet(P+'open.parquet')
cols=list(c.columns); adj=adj[cols]; val=val[cols]; op=op[cols]
good=c.notna().sum(1)>=300
cg=c[good]; ag=adj[good]; vg=val[good]
V60=vg.rolling(60,min_periods=60).mean()
S52=ag/ag.rolling(252,min_periods=252).max()
# 訊號日：每月第一個 ≥11 日的 GOOD 日
gi=cg.index; sd=[]
for (y,m),g in pd.Series(gi,index=gi).groupby([gi.year,gi.month]):
    g=g[g.index.day>=11]
    if len(g): sd.append(g.index[0])
sd=[d for d in sd if pd.Timestamp('2006-04-01')<=d<=pd.Timestamp('2013-12-31')]
# 金融、外國
sl=pd.read_csv('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/stock_list.csv',dtype=str).fillna('')
fin=set(sl.code[sl.industry=='金融保險業'])|{x for x in cols if x.startswith('28')}|{'5876','5880'}
rv=pd.read_csv('/home/user/niklavsd/bottleneck-signals/data/processed/tw_revenue.csv',dtype={'code':str})
names={}
for df in (sl[['code','name']], rv[['code','name']].drop_duplicates()):
    for k,n in zip(df.code,df.name): names.setdefault(k,set()).add(str(n))
def foreign(k):
    if k.startswith('91'): return True
    return any(('KY' in n.upper()) or ('DR' in n.upper()) or n.startswith('F-') for n in names.get(k,()))
FOR={k for k in cols if foreign(k)}
rv['month']=pd.to_datetime(rv.month).dt.to_period('M')
rv=rv.sort_values('market',ascending=False).drop_duplicates(['month','code'],keep='first')  # sii 優先
R=rv.pivot(index='month',columns='code',values='revenue_k_twd')
R=R.reindex(pd.period_range(R.index.min(),R.index.max(),freq='M'))
base=R.shift(12); yoy=R/base-1; yoy[(base<=0)|base.isna()|R.isna()]=np.nan
def y3(d):
    m=pd.Period(d,'M'); ms=[m-1,m-2,m-3]
    x=yoy.reindex(ms); return x.mean(skipna=False)   # 三個月都需有值
def tick(p):
    return np.where(p<10,0.01,np.where(p<50,0.05,np.where(p<100,0.1,np.where(p<500,0.5,np.where(p<1000,1,5))))) if pd.Timestamp('2005')<pd.Timestamp('2099') else None
def limit_up(prev,d):
    pct=0.07 if d<pd.Timestamp('2015-06-01') else 0.10
    raw=prev*(1+pct); t=tick(raw); return np.floor(raw/t+1e-9)*t
allidx=c.index
lists={}; held=set(); log=[]
for d in sd:
    v=V60.loc[d]; cl=c.loc[d]
    nd=allidx[allidx.get_loc(d)+1]
    if OPT['pool_order']=='top300_then_filter':
        ok=v.dropna().sort_values(ascending=False,kind='mergesort')
        ok=ok[cl.reindex(ok.index).notna()]
        top=ok.index[:300]; pool=[k for k in top if v[k]>=1e7 and cl[k]>=10]
    elif OPT['pool_order']=='filter_then_top300':
        ok=v[(v>=1e7)&(cl>=10)].dropna().sort_values(ascending=False,kind='mergesort'); pool=list(ok.index[:300])
    elif OPT['pool_order']=='filter_exfin_then_top300':
        ok=v[(v>=1e7)&(cl>=10)].dropna(); ok=ok[[k for k in ok.index if k not in fin and k not in FOR]].sort_values(ascending=False,kind='mergesort'); pool=list(ok.index[:300])
    elif OPT['pool_order']=='filter_exall_then_top300':
        ok=v[(v>=1e7)&(cl>=10)].dropna(); ok=ok[[k for k in ok.index if k not in fin and k not in FOR and k[0]!='0']].sort_values(ascending=False,kind='mergesort'); pool=list(ok.index[:300])
    s=S52.loc[d]; yy=y3(d)
    el=[k for k in pool if pd.notna(s.get(k)) and pd.notna(op.loc[nd,k]) and k not in fin and k not in FOR and pd.notna(yy.get(k,np.nan))]
    df=pd.DataFrame({'s':s[el].values,'y':yy.reindex(el).values,'ord':[cols.index(k) for k in el]},index=el)
    df['sc']=0.5*df.s.rank(pct=True,method='average')+0.5*df.y.rank(pct=True,method='average')
    df=df.sort_values(['sc','y','ord'],ascending=[False,False,True])
    pick=[]
    for k in df.index:
        if len(pick)==10: break
        if k not in held:
            pc=c.loc[d,k]; o=op.loc[nd,k]
            if pd.notna(pc) and pd.notna(o) and o>=limit_up(pc,nd)-1e-9:
                log.append((str(d.date()),k,'limitup_skip')); continue
        pick.append(k)
    lists[str(d.date())]=pick; held=set(pick)
ref=json.load(open(A+'o_g8_protocol_fin_erratum/ERRATUM.json'))['lists_C0']
ref={k:v for k,v in ref.items() if k<='2013-12-31'}
inter=uni=0; diff=[]
print('dates mine',len(lists),'ref',len(ref),'date set equal',set(lists)==set(ref))
for dd in sorted(set(ref)|set(lists)):
    a=set(lists.get(dd,[])); b=set(ref.get(dd,[])); inter+=len(a&b); uni+=len(a|b)
    if a!=b: diff.append((dd,sorted(a-b),sorted(b-a)))
match=sum(len(set(lists.get(dd,[]))&set(ref[dd])) for dd in ref)
print('slot match',match,'/',sum(len(v) for v in ref.values()),'Jaccard(pooled)',inter/uni)
for x in diff: print(x)
json.dump(dict(opt=OPT,lists=lists,match=match,jacc=inter/uni,diff=diff,limitup=log),open(OUT+f"r3_c0_{OPT['pool_order']}.json",'w'),ensure_ascii=False,indent=0)
