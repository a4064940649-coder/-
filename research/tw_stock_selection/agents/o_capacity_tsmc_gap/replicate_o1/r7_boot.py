#!/usr/bin/env python3
"""步驟 7：max-t bootstrap（挑選期；去均值後整月重抽，B=10000）。"""
import sys, json
import numpy as np, pandas as pd
sys.path.insert(0, '/home/user/-/research/tw_stock_selection/agents/o_capacity_tsmc_gap/replicate_o1')
import rcore as R
from rstat import PER, S, pb
NAV = pd.read_parquet(R.HERE/'navs_tsmc.parquet'); NP = pd.read_parquet(R.HERE/'navs_pools.parquet')
def sel(s): return s[(s.index>=PER['SEL'][2])&(s.index<=PER['SEL'][3])]
p0 = R.win_returns(NAV['n10|w0|normal'], S)
D_pool = pd.concat([sel(R.win_returns(NP[k], S)-p0) for k in ('51-300','101-300')], axis=1).to_numpy()
D_w = pd.concat([sel(R.win_returns(NAV[f'n10|w{w}|normal'], S)-pb) for w in (0,10,20,30)], axis=1).to_numpy()
def tcols(X): return X.mean(0)/(X.std(0,ddof=1)/np.sqrt(len(X)))
out={}
for nm, X in (('pool', D_pool), ('tsmc_w', D_w)):
    tobs=tcols(X); mobs=tobs.max(); Xc=X-X.mean(0)
    for B, seed in ((10000,0),(10000,1),(100000,0)):
        rng=np.random.default_rng(seed); n=len(X); hit=0
        for _ in range(B):
            idx=rng.integers(0,n,n); hit += tcols(Xc[idx]).max() >= mobs
        out[f'{nm}_B{B}_s{seed}']={'t_obs':[round(float(t),3) for t in tobs],'p':hit/B}
        print(nm,B,seed,out[f'{nm}_B{B}_s{seed}'])
json.dump(out, open(R.HERE/'res_boot.json','w'), indent=1)
