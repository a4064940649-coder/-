#!/usr/bin/env python3
"""分段年化的起點口徑比較：(a) v4lib.seg（段起點前一日收盤）；(b) 段內第一個訊號日收盤 → 段內最後一窗終點。"""
import os, sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
import numpy as np, pandas as pd
PANEL = os.environ.get('O6R_PANEL', 'compat'); OUT = HERE / f'out_{PANEL}'
NV = pd.read_parquet(OUT / 'navs.parquet'); sd = pd.DatetimeIndex(json.load(open(OUT / 'plans.json'))['sig_days'])
E50 = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/etf_0050.csv', parse_dates=['date']).set_index('date')['adj'].sort_index()
E50 = E50.reindex(NV.index.union(E50.index)).ffill().reindex(NV.index)
PER = {'S1': ('2006-04-01', '2013-12-31'), 'S2': ('2014-01-01', '2020-12-31'), 'S3': ('2021-03-01', '2026-10-12'),
       'S4': ('2024-01-01', '2026-10-12'), 'SEL': ('2006-04-01', '2018-12-31'), 'VAL': ('2019-01-01', '2023-12-31')}
out = {}
for k, (a, b) in PER.items():
    s = [d for d in sd if pd.Timestamp(a) <= d <= pd.Timestamp(b)]
    i0 = list(sd).index(s[0]); i1 = list(sd).index(s[-1]) + 1
    d0, d1 = sd[i0], sd[min(i1, len(sd) - 1)]
    yrs = (d1 - d0).days / 365.25
    f = lambda x: round(((x.loc[d1] / x.loc[d0]) ** (1 / yrs) - 1) * 100, 2)
    out[k] = {'from': str(d0.date()), 'to': str(d1.date()), 'AB_V6': f(NV['AB_V6|normal']), '0050': f(E50), 'AB_V6_B10': f(NV['AB_V6_B10|normal'])}
json.dump(out, open(OUT / 'seg_firstsig.json', 'w'), indent=1); print(json.dumps(out, indent=1))
