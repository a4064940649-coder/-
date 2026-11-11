import sys; sys.path.insert(0,"/home/user/-/research/tw_stock_selection/agents/o_turnover_fixedcost_diagnosis/replicate_o20")
from lib import *
z=np.load(B+'o_exec_engine_rewrite_v7x/data/x_cache.npz',allow_pickle=True)
IDX=pd.DatetimeIndex(z['IDX']); COLS=list(z['COLS']); CI={c:i for i,c in enumerate(COLS)}
O=z['O']; CF=z['CF']; LADJ=np.log(np.where(np.isfinite(z['ADJ']),z['ADJ'],1.0)); CL=np.vstack([np.zeros((1,LADJ.shape[1])),np.cumsum(LADJ,axis=0)])
def winret(code,e,e2):
    c=CI[code]; p0=O[e,c]
    if not np.isfinite(p0) or p0<=0: p0=CF[e-1,c]
    p1=O[e2,c]
    if not np.isfinite(p1) or p1<=0: p1=CF[e2,c]
    g=CL[e2+1,c]-CL[e+1,c]   # sum logADJ over e+1..e2
    return p1*np.exp(g)/p0-1
def rebs(variant='REAL_opt',seed=0):
    p=pd.read_csv(D+'plans_real_allseeds.csv.gz',dtype=str)
    p=p[(p.variant==variant)&(p.seed==str(seed))].copy()
    p['j']=p.j.astype(int); p['e']=p.exec_idx.astype(int)
    p=p.sort_values(['j','e']); p['k']=p.groupby('j').cumcount()
    jend_idx={j:int(np.searchsorted(IDX,JEND[j],side='right')-1) for j in JEND}
    p['e2']=p.groupby('j').e.shift(-1)
    p['e2']=[int(x) if pd.notna(x) else jend_idx[j] for x,j in zip(p.e2,p.j)]
    p['picks']=p.picks.str.split(',')
    return p[['j','k','signal','e','e2','picks']]
def bucket_rank(rank,why):
    if why!='el' or pd.isna(why): return 'F_'+(why if isinstance(why,str) else 'na')
    if rank<=10: return '<=10'
    if rank<=15: return '11~15'
    if rank<=30: return '16~30'
    if rank<=50: return '31~50'
    return '>50'
def bucket_px(p):
    return '<20' if p<20 else '20~50' if p<50 else '50~100' if p<100 else '100~500' if p<500 else '>=500'
def build(tier='100n',variant='REAL_opt'):
    t=load_tl(tier); rb=rebs(variant); rb.to_pickle  # noqa
    rk=pd.read_csv(D+'ranks_seed0.csv.gz',dtype={'code':str}); rk=rk[rk.variant==variant]
    rk=rk.set_index(['j','signal','code'])
    rbk=rb.set_index(['j','k'])
    agg=t.groupby(['j','k','code','cls']).agg(cost=('cost','sum'),notional=('notional','sum'),ch05=('cost_h05','sum'),ch10=('cost_h10','sum')).reset_index()
    agg['rate']=agg.cost/agg.notional; agg['rate_h05']=agg.ch05/agg.notional; agg['rate_h10']=agg.ch10/agg.notional
    ex=agg[agg.cls=='exit'].copy(); en=agg[agg.cls=='entry'].copy()
    for df in (ex,en):
        df['signal']=[rbk.loc[(j,k),'signal'] for j,k in zip(df.j,df.k)]
        df['e']=[rbk.loc[(j,k),'e'] for j,k in zip(df.j,df.k)]
        df['e2']=[rbk.loc[(j,k),'e2'] for j,k in zip(df.j,df.k)]
        df['r']=[winret(c,e,e2) for c,e,e2 in zip(df.code,df.e,df.e2)]
        df['price']=[O[e,CI[c]] for c,e in zip(df.code,df.e)]
    ent=en.groupby(['j','k']).agg(r_en=('r','mean'),rate_en=('rate','mean'),rate_en_h05=('rate_h05','mean'),rate_en_h10=('rate_h10','mean'),n_en=('r','size'),pe=('price','mean'))
    ex=ex.join(ent,on=['j','k'])
    ex['edge']=ex.r_en-ex.r; ex['rtc']=ex.rate+ex.rate_en; ex['rtc_h05']=ex.rate_h05+ex.rate_en_h05; ex['rtc_h10']=ex.rate_h10+ex.rate_en_h10
    rr=[];ww=[]
    for j,s,c in zip(ex.j,ex.signal,ex.code):
        try: q=rk.loc[(j,s,c)]; rr.append(q['rank']); ww.append(q['why'])
        except KeyError: rr.append(np.nan); ww.append('na')
    ex['rank']=rr; ex['why']=ww
    ex['rb']=[bucket_rank(a,b) for a,b in zip(ex['rank'],ex.why)]
    ex['pb']=[bucket_px(p) for p in ex.price]
    ex['signal']=pd.to_datetime(ex.signal)
    return ex,en,t,rb
