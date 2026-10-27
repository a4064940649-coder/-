#!/usr/bin/env python3
"""步驟 1：重現 main_v2 排除金融 n10 一般成本淨值，逐日比對 navs_gapfix。"""
import sys, pickle
import numpy as np, pandas as pd
sys.path.insert(0, '/home/user/-/research/tw_stock_selection/agents/o_capacity_tsmc_gap/replicate_o1')
import rcore as R

REF = pd.read_parquet('/home/user/-/research/tw_stock_selection/agents/main_v2/navs_gapfix.parquet')

crosses = {}
for d in R.SIG:
    if R.POS[d] + 1 >= len(R.IDX) or d >= R.EVAL_END: continue
    crosses[d] = R.Cross(d, exfin=True)
pickle.dump({d: (c.el, c.score, c.y3, c.s52, c.vrank, c.order) for d, c in crosses.items()}, open(R.HERE / 'cross_exfin.pkl', 'wb'))
held = set(); ev = {}; picks = {}; poolsz = {}
for d, cr in crosses.items():
    poolsz[d] = int(cr.el.sum())
    if cr.el.sum() < 50: continue
    gu = R.open_limit_up(cr.ex)
    pk = R.pick(cr, 10, held, gu)
    ev[cr.ex] = {j: 1.0 for j in pk}; picks[d] = [R.COLS[j] for j in pk]
    held = set(pk)
nav = R.simulate(ev, 'normal')
ref = REF['EXFIN|n10|normal'].dropna()
nav = nav.reindex(ref.index)
diff = (nav / ref - 1)
print('pool sizes min', min(poolsz.values()), 'months skipped', [str(d.date()) for d, s in poolsz.items() if s < 50])
print('final', nav.iloc[-1], ref.iloc[-1], 'max abs rel diff', diff.abs().max())
bad = diff[diff.abs() > 1e-9]
print('first diff', bad.index[:5].tolist() if len(bad) else None)
print('CAGR', R.cagr(nav, '2006-04-12', R.EVAL_END))
pickle.dump({'picks': picks, 'ev': ev}, open(R.HERE / 'picks_w0.pkl', 'wb'))
