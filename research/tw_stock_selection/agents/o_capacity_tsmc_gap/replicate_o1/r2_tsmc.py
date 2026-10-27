#!/usr/bin/env python3
"""步驟 2：台積電固定權重 w0/10/20/30、一般與壓力成本；各期統計；0050 混合對照；逐年。"""
import sys, pickle, json
import numpy as np, pandas as pd
sys.path.insert(0, '/home/user/-/research/tw_stock_selection/agents/o_capacity_tsmc_gap/replicate_o1')
import rcore as R

J2330 = R.CI['2330']
crosses = {d: R.Cross(d, exfin=True) for d in R.SIG if d < R.EVAL_END}


def build(n, w, crosses=crosses):
    held = set(); ev = {}; picks = {}
    for d, cr in crosses.items():
        if cr.el.sum() < 50: continue
        gu = R.open_limit_up(cr.ex)
        hs = held | ({J2330} if w > 0 and picks else set())
        pk = R.pick(cr, n, hs, gu)
        wt = {j: (1 - w) / len(pk) for j in pk}
        if w > 0: wt[J2330] = wt.get(J2330, 0.0) + w
        ev[cr.ex] = wt; picks[d] = pk; held = set(pk)
    return ev, picks


if __name__ == '__main__':
    navs = {}; P = {}
    for n in (10, 30):
        for w in (0.0, 0.1, 0.2, 0.3):
            if n == 30 and w > 0: continue
            ev, pk = build(n, w); P[(n, w)] = pk
            for c in ('normal', 'stress'):
                navs[f'n{n}|w{int(w*100)}|{c}'] = R.simulate(ev, c)
    # EW 同池（零成本、凍結持有）
    ev_ew = {cr.ex: {j: 1.0 for j in np.where(cr.el)[0]} for cr in crosses.values()}
    R.COST['zero'] = (0.0, 0.0)
    navs['EW'] = R.simulate(ev_ew, 'zero')
    NAV = pd.DataFrame(navs)
    NAV.to_parquet(R.HERE / 'navs_tsmc.parquet')
    pickle.dump(P, open(R.HERE / 'picks_main.pkl', 'wb'))
    REF = pd.read_parquet('/home/user/-/research/tw_stock_selection/agents/main_v2/navs_gapfix.parquet')
    for a, b in [('n10|w0|normal', 'EXFIN|n10|normal'), ('n10|w0|stress', 'EXFIN|n10|stress'), ('n30|w0|normal', 'EXFIN|n30|normal'), ('n30|w0|stress', 'EXFIN|n30|stress'), ('EW', 'EXFIN|EW')]:
        r = REF[b].dropna(); x = NAV[a].reindex(r.index)
        print('check', a, b, float((x / r - 1).abs().max()))
