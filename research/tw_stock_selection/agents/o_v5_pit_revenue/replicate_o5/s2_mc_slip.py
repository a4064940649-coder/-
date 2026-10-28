#!/usr/bin/env python3
"""首次值界限（22 版 × AB 40 seed）與成本損益兩平（滑價 0.10%～0.40%）。"""
import sys, json
sys.path.insert(0, '/home/user/-/research/tw_stock_selection/agents/o_v5_pit_revenue/replicate_o5')
import numpy as np, pandas as pd
from multiprocessing import Pool
import rcore5 as R
HERE = '/home/user/-/research/tw_stock_selection/agents/o_v5_pit_revenue/replicate_o5/'
D = R.official_dates()
VERS = [f'fv_mc_s{s}' for s in range(20)] + ['fv_mc_neg', 'fv_mc_pos']
_REV = {}
def one(args):
    ver, seed = args
    if ver not in _REV: _REV[ver] = R.Rev('fv', mc=ver)
    pl = R.build_plan(D, _REV[ver], U=R.unavail_mc(seed))
    return R.run(pl, *R.rates())
REVFV = R.Rev('fv')
def slip(args):
    seed, slips = args
    pl = R.build_plan(D, REVFV, U=R.unavail_mc(seed))
    return [R.run(pl, *R.rates('normal', s)) for s in slips]
if __name__ == '__main__':
    pl0 = R.build_plan(D, REVFV); W = R.wins(pl0); d0 = W[0][0]
    out = {'mc': {}}
    with Pool(4) as P:
        for v in VERS:
            navs = P.map(one, [(v, s) for s in range(40)])
            nav = R.avg_nav(navs)
            out['mc'][v] = dict(t=R.tstat(R.excess(nav, W)), cagr=R.cagr(nav, d0) * 100)
            print(v, out['mc'][v], flush=True)
        SL = [round(0.001 + 0.0002 * i, 6) for i in range(16)]
        res = P.map(slip, [(s, SL) for s in range(40)])
    ts = [R.tstat(R.excess(R.avg_nav([res[s][i] for s in range(40)]), W)) for i in range(len(SL))]
    out['slip_grid'] = dict(zip([f'{x*100:.2f}%' for x in SL], ts))
    def cross(target):
        for i in range(len(SL) - 1):
            if (ts[i] - target) * (ts[i + 1] - target) <= 0:
                return (SL[i] + (target - ts[i]) * (SL[i + 1] - SL[i]) / (ts[i + 1] - ts[i])) * 100
        return None
    out['breakeven_pct'] = {'2.3': cross(2.3), '2.0': cross(2.0)}
    t = [x['t'] for x in out['mc'].values()]; c = [x['cagr'] for x in out['mc'].values()]
    tm = [out['mc'][f'fv_mc_s{s}']['t'] for s in range(20)]
    out['bounds'] = dict(t_min=min(t), t_max=max(t), cagr_min=min(c), cagr_max=max(c), mc_t_median=float(np.median(tm)))
    json.dump(out, open(HERE + 's2_mc_slip.json', 'w'), indent=1)
    print(json.dumps({k: out[k] for k in ['bounds', 'breakeven_pct', 'slip_grid']}, indent=1))
