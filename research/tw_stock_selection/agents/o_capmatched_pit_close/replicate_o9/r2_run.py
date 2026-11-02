#!/usr/bin/env python3
"""主跑：O9_CAR=<car> python3 -I -B r2_run.py <PERIOD> [nrand] [cost]
每個日期集合（OFF＋J0..J12）跑實際＋nrand 次隨機（default_rng(10000+s)），存窗報酬。"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from multiprocessing import Pool
import rcore as R

PER = sys.argv[1]; NR = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
COST = sys.argv[3] if len(sys.argv) > 3 else 'normal'
B, S = R.L.rates(COST)

def job(name):
    days, end = R.dateset(name, PER)
    W = R.Windows(days, end)
    na = R.sim(W, R.plan_for(days, lambda cs: None), B, S)
    rr = np.empty((NR, len(days)))
    for s in range(NR):
        rg = np.random.default_rng(10000 + s)
        n = R.sim(W, R.plan_for(days, R.rand_score(rg)), B, S); rr[s] = n[1:] / n[:-1] - 1
    e50 = R.L.E50.reindex(list(days) + [end]).to_numpy()
    return name, dict(days=np.array([str(d.date()) for d in days]), end=str(end.date()), act=na[1:] / na[:-1] - 1, rnd=rr, e50=e50[1:] / e50[:-1] - 1)

if __name__ == '__main__':
    t0 = time.time()
    with Pool(4) as p:
        res = dict(p.map(job, R.SETS, chunksize=1))
    out = {}
    for k, v in res.items():
        for kk, vv in v.items(): out[f'{k}|{kk}'] = vv
    tag = f"{os.environ.get('O9_CAR')}_{PER}_{COST}_{NR}"
    np.savez_compressed(R.HERE / f'out_{tag}.npz', **out)
    for k in R.SETS:
        v = res[k]; x = v['act'] - v['rnd'].mean(axis=0)
        print(k, len(x), v['days'][0], v['end'], f"{x.mean()*100:.3f}pp t {R.tstat(x):.3f}", flush=True)
    print('sec', time.time() - t0)
