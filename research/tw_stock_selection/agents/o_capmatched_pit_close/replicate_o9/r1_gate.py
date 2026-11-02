#!/usr/bin/env python3
"""重現閘門：o7 原輸入、官方日 2019-01～2026-10-12、50 次隨機（seed 5000+）；並核對自寫引擎 vs simulate_w。
O9_CAR=car_o7 python3 -I -B r1_gate.py"""
import time, json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import rcore as R
L, T = R.L, R.T
t0 = time.time()
days, end = R.dateset('OFF', 'GATE')
W = R.Windows(days, end)
print('gate windows', len(days), days[0].date(), end.date(), flush=True)
pa = R.plan_for(days, lambda cs: None)
na = R.sim(W, pa)
# 核對：simulate_w
plan = {e: p for e, p in zip(W.ex, pa)}
nav_ref, _ = L.simulate_w(plan, R.BUY, R.SELL, W.t0, W.t1, G=R.G)
ref = nav_ref.reindex(list(days) + [end]).to_numpy()
print('actual engine max rel diff', float(np.max(np.abs(na / ref - 1))), flush=True)
rs = []; md = 0.0
for s in range(50):
    rg = np.random.default_rng(5000 + s)
    p = R.plan_for(days, R.rand_score(rg))
    n = R.sim(W, p); rs.append(n[1:] / n[:-1] - 1)
    if s < 3:
        nr, _ = L.simulate_w({e: q for e, q in zip(W.ex, p)}, R.BUY, R.SELL, W.t0, W.t1, G=R.G)
        r1 = nr.reindex(list(days) + [end]).to_numpy(); r1 = r1[1:] / r1[:-1] - 1
        md = max(md, float(np.max(np.abs(r1 - rs[-1]))))
ra = na[1:] / na[:-1] - 1
x = ra - np.mean(rs, axis=0)
out = dict(n=len(x), mean_pp=float(x.mean() * 100), t=R.tstat(x), max_win_diff_vs_simulate_w=md, sec=time.time() - t0)
print(json.dumps(out), flush=True)
json.dump(out, open(R.HERE / 'out_gate.json', 'w'), indent=1)
