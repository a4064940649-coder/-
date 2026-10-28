import sys, json; sys.path.insert(0,'/home/user/-/research/tw_stock_selection/agents/o_tier_shrink_sequential/replicate_o7')
import numpy as np, pandas as pd
from multiprocessing import Pool
import ecore_r as E
D=None
def init():
    global D; D=E.Data()
def job(a):
    kind,tier,ver,seed=a
    spec=json.load(open(E.RAW/'model_spec.json'))
    fee='A'
    if kind=='tickstress': spec['odd_stress']=True; fee='B'
    imp='tick' if kind.startswith('tick') else 'sqrt'
    nav,_=E.simulate(D,seed,tier=E.TIERS[tier],version=ver,fee=fee,model=E.ExecModel(spec),flags=dict(impact=imp))
    return a, nav.pct_change().fillna(0).values, nav.index
if __name__=='__main__':
    kinds=sys.argv[1].split(',')
    nseed=int(sys.argv[2]) if len(sys.argv)>2 else 40
    jobs=[(k,t,v,s) for k in kinds for t in E.TIERS for v in ('FIX','GROW') for s in range(nseed)]
    acc={}
    with Pool(4,initializer=init) as p:
        for a,r,idx in p.imap_unordered(job,jobs,chunksize=2):
            key=f'{a[0]}|{a[1]}|{a[2]}'; acc.setdefault(key,[]).append(r)
    out=pd.DataFrame({k:np.cumprod(1+np.mean(v,axis=0)) for k,v in acc.items()},index=idx)
    out.to_parquet(f'/home/user/-/research/tw_stock_selection/agents/o_tier_shrink_sequential/replicate_o7/out/navs_{"_".join(kinds)}.parquet')
    print(out.iloc[-1])
