#!/usr/bin/env python3
"""步驟 4：股池（51～300、101～300、前 100）n10 一般成本。"""
import sys, json, pickle
import numpy as np, pandas as pd
sys.path.insert(0, '/home/user/-/research/tw_stock_selection/agents/o_capacity_tsmc_gap/replicate_o1')
import rcore as R
from r2_tsmc import build
from rstat import seg_stats, PER, S, pb

NAV = pd.read_parquet(R.HERE / 'navs_tsmc.parquet')
p_orig = R.win_returns(NAV['n10|w0|normal'], S)
out = {}; navs = {}
R.COST['zero'] = (0.0, 0.0)
for lo, hi in [(51, 300), (101, 300), (1, 100)]:
    cr = {d: R.Cross(d, exfin=True, rank_lo=lo, rank_hi=hi) for d in R.SIG if d < R.EVAL_END}
    ev, pk = build(10, 0.0, cr)
    nav = R.simulate(ev, 'normal')
    ew = R.simulate({c.ex: {j: 1.0 for j in np.where(c.el)[0]} for c in cr.values()}, 'zero')
    k = f'{lo}-{hi}'; navs[k] = nav; navs[k + '|EW'] = ew
    o = {p: seg_stats(nav, p) for p in PER}
    pr = R.win_returns(nav, S); pe = R.win_returns(ew, S)
    for p in ('SEL', 'VAL', 'ALL', 'HOLD'):
        ma, mb = PER[p][2], PER[p][3]; m = (pr.index >= ma) & (pr.index <= mb)
        d = (pr - p_orig)[m]; e = (pr - pe)[m]
        o[p]['pair_vs_orig_pp'] = round(d.mean() * 100, 3); o[p]['pair_vs_orig_t'] = round(R.tstat(d), 3)
        o[p]['vs_ownEW_pp'] = round(e.mean() * 100, 3); o[p]['vs_ownEW_t'] = round(R.tstat(e), 3)
    o['poolsize_median'] = float(np.median([c.el.sum() for c in cr.values()]))
    out[k] = o
    print(k, json.dumps(o, ensure_ascii=False))
pd.DataFrame(navs).to_parquet(R.HERE / 'navs_pools.parquet')
json.dump(out, open(R.HERE / 'res_pools.json', 'w'), ensure_ascii=False, indent=1)
