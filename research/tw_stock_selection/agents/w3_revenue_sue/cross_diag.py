#!/usr/bin/env python3
"""cross_run.py 之後的診斷：子期間配對 t 與逐年報酬（寫入 results_cross.json 的 DIAG_sub）。python3 -I cross_diag.py"""
import json, sys
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1])); from btlib import tstat  # noqa: E402
n = pd.read_parquet(HERE / 'cross_nav.parquet'); idx = n.index; s = pd.Series(idx, index=idx)
ds = [g[g.dt.day >= 11].iloc[0] for _, g in s.groupby([idx.year, idx.month]) if (g.dt.day >= 11).any()]
m = n.reindex(ds); pr = (m.shift(-1) / m - 1).iloc[:-1]
out = {'paired_sub': {}, 'yearly_pct': {}}
for k in [c for c in n.columns if c[0] in 'ab']:
    ref = 'main_n30' if '30' in k else 'main_n10'; ex = pr[k] - pr[ref]; o = {}
    for a, b in (('2014-01-01', '2016-01-31'), ('2016-02-01', '2020-12-31'), ('2021-03-01', '2023-12-31'), ('2024-01-01', '2026-10-31')):
        e = ex[(ex.index >= a) & (ex.index <= b)].dropna(); o[f'{a[:7]}_{b[:7]}'] = {'n': len(e), 'excess_pct_m': round(float(e.mean() * 100), 2), 't': tstat(e)}
    out['paired_sub'][k] = o
r = n.pct_change().dropna(how='all')
out['yearly_pct'] = {c: {int(y): round(float((1 + g).prod() - 1) * 100, 1) for y, g in r[c].groupby(r.index.year)} for c in n.columns}
f = HERE / 'results_cross.json'; res = json.loads(f.read_text(encoding='utf-8')); res['DIAG_sub'] = out
f.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8'); print('ok')
