import sys; sys.path.insert(0,"/home/user/-/research/tw_stock_selection/agents/o_turnover_fixedcost_diagnosis/replicate_o20")
from swaps import *
E=z['E50']; nav=pd.read_parquet(D+'navs_100n_avg20.parquet'); nav.index=pd.to_datetime(nav.index); e50=pd.Series(E,IDX)
c=pd.read_csv(D+'shapley_coalitions.csv.gz',keep_default_na=False)
def cg(x,a):
    x=x[x.index>=a]; y=(x.index[-1]-x.index[0]).days/365.25; return ((x.iloc[-1]/x.iloc[0])**(1/y)-1)*100
var='REAL_opt'; rb=rebs(var); g=c[(c.tier=='100萬')&(c.variant==var)&(c.coalition=='T')].set_index('j'); A=[];Bn=[];Bs=[];G=[]
for j,q in rb.groupby('j'):
    s=nav[f'{var}|{j}']; nv=s.reindex(IDX).values
    rp=np.array([nv[e2-1]/nv[e-1]-1 for e,e2 in zip(q.e,q.e2)]); rm=np.array([E[e2-1]/E[e-1]-1 for e,e2 in zip(q.e,q.e2)])
    ok=np.isfinite(rp); sel=ok&(pd.to_datetime(q.signal)<='2019-04-30').values
    b=np.polyfit(rm[sel],rp[sel],1)[0]; bf=np.polyfit(rm[ok],rp[ok],1)[0]
    s=s.dropna(); s=s[s.index<=JEND[j]]; r50=e50.reindex(s.index).pct_change().fillna(0)
    A.append(cg(s,s.index[0])); Bs.append(cg((1+b*r50).cumprod(),s.index[0])); Bn.append(cg((1+bf*r50).cumprod(),s.index[0])); G.append(g.loc[j,'CAGR'])
A,Bs,Bn,G=map(np.array,(A,Bs,Bn,G))
print('full: med diff sel',np.median(A-Bs),'full-beta',np.median(A-Bn),'diff of med sel',np.median(A)-np.median(Bs),'fullbeta',np.median(A)-np.median(Bn))
print('gross full: sel',np.median(G-Bs),'fullbeta',np.median(G-Bn),'diffmed',np.median(G)-np.median(Bs),np.median(G)-np.median(Bn))
# bad swap cost variants
SP='/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/'
y=pd.read_pickle(SP+'y100.pkl')
tl=load_tl('100n'); av=tl.groupby(['j','k']).avail.first()
y['c2']=y.rtc*y.notional
def dr(df,col):
    return np.median([(q[col]/q.avail).sum()/years('Y19on',j)*100 for j,q in df.groupby('j')])
bad=y.edge<y.rtc
print('rtc*notional: bad',dr(y[bad],'c2'),'all',dr(y,'c2'))
print('exit only: bad',dr(y[bad],'cost'),'all',dr(y,'cost'))
