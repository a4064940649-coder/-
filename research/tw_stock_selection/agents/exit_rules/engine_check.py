#!/usr/bin/env python3
"""引擎斷言（PREREG 引擎條款）：無出場＝rlib BASE；停損＝rlib R3；D 三條規則無出場＝原資料夾淨值。結果寫到 results.json ENGINE_CHECK。"""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.dont_write_bytecode = True
import xlib as X
R, L = X.R, X.L
T = time.time()
b, s = L.cost_rates(0.001)
out = {}
for n in (10, 30):
    items, pl = X.main_items(n); T0 = items[0]['t']; T1 = len(X.IDX) - 1
    for kw, rule in [({}, None), (dict(stop=0.15, stop_mode='cash'), dict(sl=0.15)), (dict(stop=0.20, stop_mode='next'), dict(sl=0.20))]:
        a, ia = R.simulate(pl, b, s, T0, T1, n=n, **kw)
        c, ic = X.simulate(items, 1, b, s, T0, T1, rule=rule, mode=kw.get('stop_mode', 'cash'))
        md = float((a / c - 1).abs().max()); out[f'n{n}_{kw}'] = md
        print(n, kw, md, ia.get('n_stops'), len(ic['sell_t']), round(time.time() - T, 1), flush=True)
        assert md < 1e-12, md
        if not kw: assert ia['turnover_per_rebalance'] == ic['turnover_per_rebalance']
AG = X.AG
for name, fn, col, path in [('weekly_reversal', X.weekly_reversal_items, 'main_normal', 'w3_weekly_reversal/navs.parquet'),
                            ('high_volume', X.high_volume_items, 'main_GKM_weekly_20d_normal', 'w3_high_volume_premium/navs.parquet'),
                            ('surge', X.surge_items, 'S_main_0.001', 'w3_revenue_sue/nav.parquet')]:
    r = fn(X.VALUE_ORIG); items, K = r[0], r[1]
    T0 = items[0]['t']; T1 = len(X.IDX) - 1
    nav, inf = X.simulate(items, K, b, s, T0, T1)
    ref = pd.read_parquet(AG / path)[col].dropna()
    ra = nav.pct_change(); rb = ref.pct_change(); ix = ra.dropna().index.intersection(rb.dropna().index)
    md = float((ra[ix] - rb[ix]).abs().max()); out[name] = {'max_abs_daily_ret_diff': md, 'first': str(nav.index[0].date()), 'ref_first': str(ref.index[0].date()), 'n_days': len(ix)}
    print(name, out[name], round(time.time() - T, 1), flush=True)
    assert md < 1e-9, md
res = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))
res['ENGINE_CHECK'] = out
(HERE / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
print('engine ok')
