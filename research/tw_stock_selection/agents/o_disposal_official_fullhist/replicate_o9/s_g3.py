import sys,json; sys.path.insert(0,'/home/user/-/research/tw_stock_selection/agents/o_disposal_official_fullhist/replicate_o9')
from s_sleeve import *
from scipy import stats as sst
def daily_nav(tier=1e6,mode='FIX',st=False):
    mr,dd=sleeve(tier,mode,st)
    K=lambda dt: tier*cpif(dt.year)/cpif(2026)
    nav=[];v=1.0;prev=None;pm=None
    for r in dd.itertuples():
        m=r.d.to_period('M')
        if prev is None: base=K(r.d)
        elif m!=pm: base=K(r.d) if mode=='FIX' else prev
        else: base=prev
        v*=r.eq/base; nav.append(v); prev=r.eq; pm=m
    return pd.Series(nav,index=dd.d), mr
nav,mr=daily_nav()
print('daily MDD', ((nav/nav.cummax())-1).min())
# P6
mn=pd.read_parquet(A+'o_exec_engine_rewrite_v7x/out/navs_oos_R4.parquet')
cols=[c for c in mn.columns if c.startswith('B10|J') and c.endswith('|K5T|normal|100萬|FIX')]
print(len(cols))
mm=mn[cols].resample('ME').last().pct_change(); mm.index=mm.index.to_period('M')
main=mm.mean(1)
b=e50m
se=(mr-b.reindex(mr.index)); me=(main-b.reindex(main.index))
j=pd.concat([se,me],axis=1).dropna()
print('P6 corr',j.corr().iloc[0,1], 'n',len(j), '2019+', j[j.index>=pd.Period('2019-01','M')].corr().iloc[0,1])
# G3
f=pd.read_parquet('/home/user/-/research/tw_stock_selection/agents/o_main_spa_breaks/data/family_excess.parquet')
bm=pd.read_parquet('/home/user/-/research/tw_stock_selection/agents/o_main_spa_breaks/data/bench.parquet')
m=pd.read_csv('/home/user/-/research/tw_stock_selection/agents/o_main_spa_breaks/data/family_members.csv')
F=f[m[m.F_long].member_id]
idx=f.index
e50s=pd.Series(E50,index=IDX)
def win(series):
    out=[]
    for i,d0 in enumerate(idx):
        d1=idx[i+1] if i+1<len(idx) else IDX[-1]
        a=series.asof(pd.Timestamp(d0)); b_=series.asof(pd.Timestamp(d1)); out.append(b_/a-1)
    return pd.Series(out,index=idx)
w50=win(e50s); print('0050 window check maxdiff',(w50-bm.r0050).abs().max())
foc=win(nav)-w50
X=F.copy(); X['FOCUS']=foc
X=X.dropna(how='any'); n,k=X.shape; print('W windows',n,'members',k)
D=X.values; dbar=D.mean(0)
rng=np.random.default_rng(20261102); B=10000; L=6
def sb_idx(n,L,rng):
    p=1.0/L; ix=np.empty(n,dtype=int); ix[0]=rng.integers(n)
    for t in range(1,n):
        ix[t]=rng.integers(n) if rng.random()<p else (ix[t-1]+1)%n
    return ix
means=np.empty((B,k))
for b_ in range(B):
    ii=sb_idx(n,L,rng); means[b_]=D[ii].mean(0)
omega=np.sqrt(n)*means.std(0,ddof=0)
Tk=np.sqrt(n)*dbar/omega
thr=-np.sqrt(2*np.log(np.log(n)))
mu_c=np.where(Tk>=thr,dbar,0.0)
Tb=np.sqrt(n)*(means-dbar+ (dbar-mu_c)*0 - mu_c*0)  # placeholder
Zc=np.sqrt(n)*(means-dbar+(dbar-mu_c)*0)/omega  # centered
# Hansen SPA_c: T*_b = max(max_k sqrt(n)(dbar*_k - g_c(dbar_k))/omega_k, 0) with g_c = dbar*1{...}
Zc=np.sqrt(n)*(means-mu_c)/omega
Tstar=np.maximum(Zc.max(1),0)
fi=k-1
p_focus=(Tstar>=Tk[fi]).mean(); p_family=(Tstar>=max(Tk.max(),0)).mean()
rank=int((Tk>Tk[fi]).sum())+1
print('focus T',Tk[fi],'p_focus',p_focus,'rank',rank,'of',k,'family p',p_family)
# DSR with Li-Ji effective number
C=np.corrcoef(D.T); ev=np.linalg.eigvalsh(C); ev=np.clip(ev,0,None)
neff=np.sum((ev>=1).astype(float)+(ev-np.floor(ev)))
x=D[:,fi]; sr=x.mean()/x.std(ddof=1); sk=sst.skew(x); ku=sst.kurtosis(x,fisher=False)
srs=dbar/D.std(0,ddof=1); vsr=srs.var(ddof=1)
g=0.5772156649
sr0=np.sqrt(vsr)*((1-g)*sst.norm.ppf(1-1/neff)+g*sst.norm.ppf(1-1/(neff*np.e)))
dsr=sst.norm.cdf((sr-sr0)*np.sqrt(n-1)/np.sqrt(1-sk*sr+(ku-1)/4*sr**2))
print('neff',neff,'SR',sr,'SR0',sr0,'DSR',dsr)
json.dump(dict(mdd_daily=float(((nav/nav.cummax())-1).min()),p6=float(j.corr().iloc[0,1]),p6_19=float(j[j.index>=pd.Period('2019-01','M')].corr().iloc[0,1]),
   spa_focus_p=float(p_focus),rank=rank,k=k,n=n,neff=float(neff),dsr=float(dsr)),open('g3.json','w'),indent=1)
