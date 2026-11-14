# -*- coding: utf-8 -*-
"""安慰劑（含殭屍修正）：同日期同筆數隨機買股，E3＋空位 0050，下市股最後一根強制出場。python3 -I -B review_a2_placebo_fixed.py"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import rep_lib as L
HERE = os.path.dirname(os.path.abspath(__file__)) + '/'
D = L.Data(); sig = L.signals(D)
pool = (D.V / 1000.0 >= 500) & (D.barno >= 260) & D.has
ts = L.signals(D, ablate=('contract', 'voldry', 'cross', 'breakvol', 'minvol'))
tmask = np.zeros_like(pool); tmask[ts.di.values, ts.j.values] = True; tmask &= pool; del ts
def placebo(base, mask, seed):
    rng = np.random.RandomState(seed); rows = []
    for di_, k in base.groupby('di').size().items():
        cand = np.nonzero(mask[di_])[0]
        if len(cand) == 0: continue
        for j in rng.choice(cand, min(k, len(cand)), replace=False):
            rows.append((D.codes[j], j, di_, D.dates[di_], rng.rand(), np.nan))
    return pd.DataFrame(rows, columns=['code', 'j', 'di', 'date', 'vbreak', 'pivot_adj'])
out = {}
g = pd.read_csv(HERE + 'my_grid81_delistfix.csv')
for tag, mask in (('pool', pool), ('trendRS', tmask)):
    res = []
    for sd in range(200):
        p = L.perf(D, L.run_portfolio(D, placebo(sig, mask, 1000 + sd), 'E3', 'etf', delist_exit=True))
        res.append((p['full']['ex'], p['2019']['ex'], p['full']['t']))
    a = np.array(res)
    out[tag] = dict(n=200, full_ex_med=float(np.median(a[:, 0])), full_ex_mean=float(a[:, 0].mean()), p05=float(np.quantile(a[:, 0], .05)), p95=float(np.quantile(a[:, 0], .95)), share_pos=float((a[:, 0] > 0).mean()),
                    y19_ex_med=float(np.median(a[:, 1])), y19_share_pos=float((a[:, 1] > 0).mean()), t_med=float(np.median(a[:, 2])),
                    default_fixed_pctile=float((a[:, 0] < 0.00889).mean()), grid_median_fixed_pctile=float((a[:, 0] < float(g.etf_full_ex_fixed.median())).mean()),
                    grid_max_fixed_pctile=float((a[:, 0] < float(g.etf_full_ex_fixed.max())).mean()),
                    default_2019_fixed_pctile=float((a[:, 1] < -0.0364).mean()))
    print(tag, {k: round(v, 4) for k, v in out[tag].items()}, flush=True)
json.dump(out, open(HERE + 'review_a2_placebo_fixed.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
