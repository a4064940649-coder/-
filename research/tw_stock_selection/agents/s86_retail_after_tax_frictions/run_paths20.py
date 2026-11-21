#!/usr/bin/env python3
"""S86 補充：標準路徑對「零股／整張取整」很敏感（扣一點錢就可能改變某筆買幾股），單一路徑的差值有雜訊。
這裡用正式的 20 條路徑（60／80／100／150／200 萬 × 位移 0～3）算每一項摩擦的平均拖累。輸出 results_paths20.json。"""
import json, sys, itertools
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import s86lib as L
from s86lib import stlib, COSTS
HERE = Path(__file__).resolve().parent
P, U, SG = L.setup((0, 1, 2, 3))
DY, DYM, NQ, dy50, e50ev, dinfo = L.div_arrays(P)
D0, DE = P.IDX[P.t0], P.IDX[-1]; D19 = P.IDX[P.IDX < pd.Timestamp('2019-01-01')][-1]
BASE = dict(sat_w=0.7, sizing='W10', lock=False, cost=COSTS['strict'])
PATHS = list(itertools.product((6e5, 8e5, 1e6, 1.5e6, 2e6), (0, 1, 2, 3)))
ITEMS = {'base': {}, 'tax-3.5': dict(div_tax=-0.035), 'tax5': dict(div_tax=0.05), 'tax12': dict(div_tax=0.12), 'tax20': dict(div_tax=0.20), 'tax28': dict(div_tax=0.28),
         'nhi': dict(nhi_rate=0.0211, nhi_thr=20000.0), 'delay20': dict(delay=20), 'fee6': dict(fee_rate=L.FEE_FULL * 0.6), 'fee10': dict(fee_rate=L.FEE_FULL),
         'minfee0': dict(minfee_v=0.0)}
out = {}
for k, kw in ITEMS.items():
    v19, vf, yl = [], [], []
    for cap, s in PATHS:
        n, _, st = L.simulate_s86(P, SG[s], U, capital=cap, DY=DY, dy50=dy50, **BASE, **kw)
        v19.append(float(stlib.cagr(n, D19, DE))); vf.append(float(stlib.cagr(n, D0, DE)))
        if k == 'base':
            d = st['_dv']; ks = (np.arange(P.T) > P.t0) & (P.IDX.year.to_numpy() >= 2019)
            yl.append(float(((d['stk_gross'] + d['e50_gross'])[ks] / d['nav_prev'][ks]).sum() / len(set(P.IDX.year[ks]))))
    out[k] = dict(c19=v19, full=vf)
    if yl: out[k]['div_pct_nav_per_year_2019on'] = yl
b = out['base']
summ = {k: dict(median_c19=float(np.median(v['c19'])), median_full=float(np.median(v['full'])),
                mean_drag_c19_pp=float(np.mean(np.array(v['c19']) - np.array(b['c19'])) * 100), mean_drag_full_pp=float(np.mean(np.array(v['full']) - np.array(b['full'])) * 100),
                sd_drag_c19_pp=float(np.std(np.array(v['c19']) - np.array(b['c19'])) * 100), n_worse=int((np.array(v['c19']) < np.array(b['c19'])).sum())) for k, v in out.items()}
summ['base']['div_pct_nav_per_year_2019on_mean'] = float(np.mean(b['div_pct_nav_per_year_2019on']))
e = {}
bb = L.bench50(P, dy50)[0]
for k, kw in ITEMS.items():
    if k in ('fee6', 'fee10', 'minfee0'): continue
    n = L.bench50(P, dy50, **kw)[0]
    e[k] = dict(c19=float(stlib.cagr(n, D19, DE)), full=float(stlib.cagr(n, D0, DE)), drag_c19_pp=float((stlib.cagr(n, D19, DE) - stlib.cagr(bb, D19, DE)) * 100), drag_full_pp=float((stlib.cagr(n, D0, DE) - stlib.cagr(bb, D0, DE)) * 100))
json.dump(dict(paths=out, summary=summ, e50=e), open(HERE / 'results_paths20.json', 'w'), ensure_ascii=False, indent=1)
for k in ITEMS:
    print(k, {a: round(c, 3) for a, c in summ[k].items()}, '0050', {a: round(c, 4) for a, c in e.get(k, {}).items()})
