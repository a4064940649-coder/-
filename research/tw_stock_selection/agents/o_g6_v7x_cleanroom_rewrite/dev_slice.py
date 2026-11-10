# R1 開發切片（A 端）：只寫淨值到 scratchpad，不輸出報酬
import numpy as np, pandas as pd, sys, pickle, json
import xdata, xengine, xlists, xstats, xrun
ver = sys.argv[1]
L = xlists.Lists(); W = xengine.World()
endd = xstats.last_le(L.dates, '2008-12-31')
jobs = []
for j in range(3):
    days = [x for x in L.jdays(j) if x[1] < endd]
    for s in range(3):
        plan = L.plan(days, seed=s)
        jobs.append((('dev', j, s), plan, endd, dict(tier=1_000_000, cost='normal', merger='orig')))
res = xrun.run_grid(jobs, procs=3)
out = {}
for j in range(3):
    navs = [res[('dev', j, s)][0] for s in range(3)]
    out[j] = xstats.seed_avg(navs)
pickle.dump({'dates': L.dates, 'navs': out, 'end': endd}, open(xdata.SCR + f'dev_{ver}.pkl', 'wb'))
print('written', ver)
