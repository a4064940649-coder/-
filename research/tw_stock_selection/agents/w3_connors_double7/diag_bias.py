#!/usr/bin/env python3
"""事後診斷（不改判定）：PRIMARY 事件依訊號日 SMA200 乖離分層，挑選期 vs 驗證期淨異常與月分群 t。python3 -I diag_bias.py"""
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
df = pd.read_csv(HERE / 'primary_events.csv.gz', parse_dates=['sig', 'entry'])
b, s = 0.001 + 0.001425 * 0.6, 0.001 + 0.001425 * 0.6 + 0.003
df['net'] = (1 + df.raw) * (1 - s) / (1 + b) - 1; df['ab'] = df.net - df.ew
for nm, (a, z) in {'select': ('2005-01-01', '2019-04-30'), 'valid': ('2019-05-01', '2026-10-31')}.items():
    x = df[(df.sig >= a) & (df.sig <= z)]
    for k, g in x.groupby(pd.cut(x.bias, [0, 0.05, 0.1, 0.2, 0.4, 10]), observed=True):
        gm = g.ab.groupby(g.entry.dt.to_period('M')).mean()
        print(nm, k, len(g), 'ab_net%', round(g.ab.mean() * 100, 3), 'month_mean%', round(gm.mean() * 100, 3),
              't_month', round(gm.mean() / gm.std() * np.sqrt(len(gm)), 2), 'ab_gross%', round((g.raw - g.ew).mean() * 100, 3))
