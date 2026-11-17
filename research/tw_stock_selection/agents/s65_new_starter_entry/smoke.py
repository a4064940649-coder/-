#!/usr/bin/env python3
"""S65 事前檢查（PREREG 之前允許做的事）：現行重現、副本逐位相同、起始月清單、單次耗時。不算任何進場方式的報酬。→ data_check.json"""
import sys, json, time, resource
from pathlib import Path
import numpy as np, pandas as pd
H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
import s65lib as L
from s65lib import s10lib, stlib
t0 = time.time()
P, U, RT = L.load()
S = L.base_signal(P, U, RT, 0)
nav, lg, st = s10lib.simulate_s10(P, S, U, capital=1e6, sat_w=0.7, sizing='W10', lock=False, cost=L.COSTS['strict'])
D19 = P.IDX[P.IDX < pd.Timestamp('2019-01-01')][-1]; DE = P.IDX[-1]
c19 = stlib.cagr(nav, D19, DE); n = int((lg['why'] != 'open_end').sum())
t1 = time.time(); n2, _, _ = L.sim(P, S, U); dt = time.time() - t1
info = S['info']
ex = [(x['month'], str(P.IDX[x['k'] + 1].date()) if x['k'] + 1 < P.T else None, x['n_tgt']) for x in info]
starts = [e for e in ex if e[1] and '2016-02-01' <= e[1] <= '2025-10-31']
out = dict(since2019=c19, n_closed=n, bitwise_main=bool(c19 == L.ACC['since2019'] and n == 381), copy_bitwise=bool(np.array_equal(nav.to_numpy(), n2.to_numpy())),
           n_info=len(info), first_exec=ex[0], last_exec=ex[-1], n_starts=len(starts), starts_first=starts[0], starts_last=starts[-1],
           n_tgt_zero_months=[e[0] for e in starts if e[2] == 0], full_sim_secs=round(dt, 2), src_sha=L.SRC_SHA,
           load_secs=round(t1 - t0), peak_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))
json.dump(out, open(H / 'data_check.json', 'w'), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
assert out['bitwise_main'] and out['copy_bitwise']
