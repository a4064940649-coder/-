import numpy as np, pandas as pd
RAW='/home/user/-/research/tw_stock_selection/agents/o_tier_shrink_sequential/raw/'
def lrv(x, L=6):
    x=np.asarray(x,float); x=x-x.mean(); T=len(x)
    v=x@x/T
    for l in range(1,L+1):
        w=1-l/(L+1); v+=2*w*(x[l:]@x[:-l])/T
    return v
def hacvar_mean(x,L=6): return lrv(x,L)/len(x)
def load_family():
    fe=pd.read_parquet(RAW+'spa/family_excess.parquet')
    cl=pd.read_csv(RAW+'o6shrink/clusters_k161.csv')
    X=fe[cl.member_id].fillna(0.0)*100
    return X, cl

from scipy.optimize import minimize
from scipy import stats
GRID=np.round(np.arange(-8,4+1e-9,0.02),10)

def ncs_basis(x, knots):
    # natural cubic spline basis (ESL 5.4): 1, x, N_k
    K=len(knots); x=np.asarray(x,float)
    def d(k):
        return (np.maximum(x-knots[k],0)**3-np.maximum(x-knots[K-1],0)**3)/(knots[K-1]-knots[k])
    cols=[np.ones_like(x),x]+[d(k)-d(K-2) for k in range(K-2)]
    return np.column_stack(cols)

def g2_Q(grid=GRID, df=5):
    knots=np.quantile(grid, np.linspace(0,1,df+1))
    B=ncs_basis(grid,knots)[:,1:]
    B=B-B.mean(0)
    U,S,Vt=np.linalg.svd(B,full_matrices=False)
    U=U[:,S>1e-8*S[0]]
    return U/np.sqrt((U**2).sum(0))

def g2_fit(xh, se, grid=GRID, df=5, c0=0.1):
    Q=g2_Q(grid,df)
    P=stats.norm.pdf((xh[:,None]-grid[None,:])/se[:,None])/se[:,None]
    def g_of(a):
        z=Q@a; z=z-z.max(); e=np.exp(z); return e/e.sum()
    def obj(a):
        g=g_of(a); f=P@g
        ll=np.log(f).sum()
        pen=c0*np.sqrt(a@a)
        # gradient
        W=(P*g[None,:])/f[:,None]  # n x m posterior weights
        gl=Q.T@(W.sum(0)-len(xh)*g)
        na=np.sqrt(a@a)
        gp=c0*a/na if na>0 else 0*a
        return -(ll-pen), -(gl-gp)
    a0=np.zeros(Q.shape[1])+1e-3
    r=minimize(obj,a0,jac=True,method='BFGS',options={'gtol':1e-10,'maxiter':5000})
    return g_of(r.x), r

def post_grid(g, x, s, grid=GRID):
    w=g*stats.norm.pdf((x-grid)/s); return w/w.sum()

def conv_normal(p, v, grid=GRID):
    # convolve grid density with N(0,v); output on the same grid (extended internally then cropped w/ renorm)
    h=grid[1]-grid[0]
    k=int(np.ceil(8*np.sqrt(v)/h))
    off=np.arange(-k,k+1)*h
    ker=stats.norm.pdf(off/np.sqrt(v)); ker/=ker.sum()
    q=np.convolve(p,ker)  # length len(p)+2k
    g2=np.concatenate([grid[0]+np.arange(-k,0)*h, grid, grid[-1]+np.arange(1,k+1)*h])
    return g2, q/q.sum()

def summ(gr,p):
    m=(gr*p).sum(); sd=np.sqrt(((gr-m)**2*p).sum()); c=np.cumsum(p)
    q=lambda a: np.interp(a,c,gr)
    return dict(mean=m,sd=sd,p_pos=p[gr>0].sum(),q10=q(.1),q50=q(.5),q90=q(.9))

def setup_g2():
    X,cl=load_family()
    meds=cl[cl.is_medoid].member_id.tolist()
    M=X[meds]; P=M.mean(axis=1)
    vP=hacvar_mean(P.values)
    xh=M.mean().values; se=np.sqrt([hacvar_mean((M[c]-P).values) for c in meds])
    g,r=g2_fit(xh,se)
    return dict(X=X,cl=cl,M=M,P=P,vP=vP,g=g,xh=xh,se=se,meds=meds)

def sigma_w(X,cl):
    num=den=0
    for c,gq in cl.groupby('cluster'):
        m=len(gq)
        if m<3: continue
        D=X[gq.member_id].values; Db=D.mean(1)
        v=np.var(D.mean(0),ddof=1)
        h=np.mean([hacvar_mean(D[:,k]-Db) for k in range(m)])*m/(m-1)
        num+=(m-1)*(v-h); den+=m-1
    return np.sqrt(num/den)

def v5_post(S, series, cluster=114, sw=None, grid=GRID):
    """series: pp per window focus; returns (grid, posterior pmf) for alpha"""
    cl=S['cl']; X=S['X']
    med=cl[(cl.cluster==cluster)&cl.is_medoid].member_id.iloc[0]
    xm=X[med].mean(); sm=np.sqrt(hacvar_mean((X[med]-S['P']).values))
    pm=post_grid(S['g'],xm,sm)                      # medoid G2 posterior (grid)
    D=np.asarray(series)-X[med].values
    d=D.mean(); sd=np.sqrt(hacvar_mean(D))
    sc=np.sqrt(2)*sw/np.sqrt(3)
    h=grid[1]-grid[0]
    k=int(np.ceil(12/h)); dg=np.arange(-k,k+1)*h
    pd_=stats.t.pdf(dg/sc,3)*stats.norm.pdf((d-dg)/sd); pd_/=pd_.sum()
    q=np.convolve(pm,pd_)
    gr=np.concatenate([grid[0]+np.arange(-k,0)*h,grid,grid[-1]+np.arange(1,k+1)*h])
    # add back common error vP
    kk=int(np.ceil(8*np.sqrt(S['vP'])/h)); off=np.arange(-kk,kk+1)*h
    ker=stats.norm.pdf(off/np.sqrt(S['vP'])); ker/=ker.sum()
    q2=np.convolve(q,ker)
    gr2=np.concatenate([gr[0]+np.arange(-kk,0)*h,gr,gr[-1]+np.arange(1,kk+1)*h])
    return gr2, q2/q2.sum(), dict(xm=xm,d=d,sd=sd)

def wgrid():
    b=pd.read_parquet(RAW+'spa/bench.parquet'); return list(b.index)+[pd.Timestamp('2026-10-12')]
def wret(s, grid=None):
    grid=grid or wgrid()
    s=s.dropna(); idx=pd.DatetimeIndex(grid)
    a=s.asof(idx).values
    # windows not covered (before first valid) -> NaN
    a=np.where(idx<s.index[0], np.nan, a)
    return pd.Series(a[1:]/a[:-1]-1,index=grid[:-1])
def r0050():
    e=pd.read_csv(RAW+'navs/etf_0050.csv',parse_dates=['date']).set_index('date').adj
    return wret(e)

def jshifts(base='AB_V6_B10', cost='normal'):
    j=pd.read_parquet(RAW+'navs/navs_jscan_v6.parquet'); v6=pd.read_parquet(RAW+'navs/navs_v6.parquet')
    off=wret(v6[f'{base}|{cost}'])
    return np.array([((wret(j[f'{base}|{cost}|J{k}'])-off)*100).fillna(0).mean() for k in range(13)])

def mix_shift(gr,p,shifts):
    vals=np.concatenate([gr+s for s in shifts]); w=np.concatenate([p/len(shifts)]*len(shifts))
    o=np.argsort(vals,kind='stable'); return vals[o], w[o]

def summ2(v,w):
    m=(v*w).sum(); c=np.cumsum(w); q=lambda a: v[np.searchsorted(c,a)]
    return dict(mean=float(m), p_pos=float(w[v>0].sum()+0.5*w[v==0].sum()), q10=float(q(.1)), q90=float(q(.9)))
