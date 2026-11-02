import sys, json; sys.path.insert(0,'.')
from ev import *
d=second_td_prev(load_events())
tr=pd.read_csv('my_trades.csv.gz',dtype={'code':str},parse_dates=['s','x','e','start','end','ann'])
o=tr[(tr.s>='2006-04-01')&(tr.s<='2021-09-30')].copy()
def clt(y,x,g):
    X=np.column_stack([np.ones(len(y)),x]); b=np.linalg.lstsq(X,y,rcond=None)[0]; u=y-X@b
    XtXi=np.linalg.inv(X.T@X); M=np.zeros((2,2)); G=0
    for gg in np.unique(g):
        m=g==gg; s=X[m].T@u[m]; M+=np.outer(s,s); G+=1
    n,k=X.shape; c=G/(G-1)*(n-1)/(n-k)
    V=c*XtXi@M@XtXi; return b[1]*100, b[1]/np.sqrt(V[1,1]), G
def diff(df,mask,col='gross'):
    g=df.s.dt.to_period('M').astype(str).values
    return dict(diff=round(clt(df[col].values,mask.astype(float).values,g)[0],3),t=round(clt(df[col].values,mask.astype(float).values,g)[1],3),
                n1=int(mask.sum()),n0=int((~mask).sum()),m1=round(df[col][mask].mean()*100,3),m0=round(df[col][~mask].mean()*100,3))
res={}
for col in ['gross','net']:
    res['P2a_'+col]=diff(o,o.regime=='periodic_match',col)
    ov=o[(o.s>='2009-01-01')&(o.s<='2012-12-31')]
    res['P2a_overlap_'+col]=diff(ov,ov.regime=='periodic_match',col)
    q=o[(o.s>='2010-12-01')&o.mm.isin([5,20])]
    res['P2b_'+col]=diff(q,q.mm==20,col)
# vdrop, run6
j_=[cix[c] for c in o.code]
def vdrop(r):
    j=cix[r.code]; a=V[r.si:r.ei+1,j]; b=V[r.si-20:r.si,j]
    return np.nanmean(a)/np.nanmean(b)-1
def run6(si,j): 
    return np.expm1(CS[si,j]-CS[si-6,j])   # s-1 vs s-7 : cum R over s-6..s-1
o['vdrop']=[vdrop(r) for r in o.itertuples()]
o['run6']=[run6(r.si,cix[r.code]) for r in o.itertuples()]
# run6 cuts over all common-stock disposal records
dv=d[d.valid]
r6all=np.array([run6(r.si,cix[r.code]) if r.si>=7 else np.nan for r in dv.itertuples()])
cuts=np.nanpercentile(r6all,[20,40,60,80])
res['run6_cuts']=cuts.tolist()
for col in ['gross','net']:
    res['P4_vdrop_'+col]=diff(o,o.vdrop<=o.vdrop.median(),col)
    res['P4_limit10_'+col]=diff(o,o.s>='2015-06-01',col)
    res['P4_second_'+col]=diff(o,o.second.astype(bool),col)
    res['P4_run6top_'+col]=diff(o,o.run6>=cuts[3],col)
# above 200MA without bear switch
nb=run(d,use_bear=False); nb=nb[(nb.s>='2006-04-01')&(nb.s<='2021-09-30')]
nb['bull']=[not bear_before(si) for si in nb.si]
for col in ['gross','net']: res['P4_bull_'+col]=diff(nb,nb.bull,col)
# P5
el=[]
for r in o.itertuples():
    j=cix[r.code]; N=r.ei-r.si; h=r.si+N//2
    early=np.expm1(CS[h+1,j]-CS[r.si+1,j])-np.expm1(CE[h+1]-CE[r.si+1])
    late=np.expm1(CS[r.xi+1,j]-CS[h+1,j])-np.expm1(CE[r.xi+1]-CE[h+1])
    el.append((early,late))
o['early'],o['late']=zip(*el); o['lme']=o.late-o.early
res['P5']=summ(o,'lme'); res['P5_early']=round(o.early.mean()*100,3); res['P5_late']=round(o.late.mean()*100,3)
o.to_csv('my_oos_feat.csv.gz',index=False)
json.dump(res,open('ptests.json','w'),indent=1,default=float)
for k,v in res.items(): print(k,v)
