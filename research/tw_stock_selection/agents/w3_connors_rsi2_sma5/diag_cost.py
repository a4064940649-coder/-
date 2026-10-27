#!/usr/bin/env python3
"""事後診斷（不影響判定）：PRIMARY 事件在教學原文「來回 0.5%」與損益兩平成本下的淨異常。python3 -I diag_cost.py"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from btlib import tstat
df = pd.read_csv(HERE / 'primary_events.csv.gz', parse_dates=['sig', 'entry'])
W = {'select_2005-01_2019-04': ('2005-01-01', '2019-04-30'), 'valid_2019-05_2026-10': ('2019-05-01', '2026-10-31'), 'y2024-01_2025-12': ('2024-01-01', '2025-12-31')}
out = {}
for k, (a, b) in W.items():
    x = df[(df.sig >= a) & (df.sig <= b)]; em = x.entry.dt.to_period('M')
    o = {}
    for rt in (0.0, 0.003, 0.005, 0.0067):
        net = (1 + x.raw) * (1 - rt) - 1; ab = net - x.ew
        o[f'rt{rt}'] = {'mean_ab_pct': round(float(ab.mean() * 100), 3), 't_month': tstat(ab.groupby(em).mean())}
    o['breakeven_roundtrip_pct'] = round(float(((1 + x.raw) - (1 + x.ew)).mean() / (1 + x.raw).mean() * 100), 3)
    out[k] = o
print(json.dumps(out, ensure_ascii=False, indent=1))
d = json.loads((HERE / 'results.json').read_text(encoding='utf-8')); d['diag_cost_post_hoc_not_for_verdict'] = out
(HERE / 'results.json').write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding='utf-8')
