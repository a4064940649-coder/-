#!/usr/bin/env python3
"""挑選期配對差另算：訊號日 2006-04～2019-03（窗終點 ≤ 2019-04 月）。只用 r_navs_*.parquet 與面板日曆。"""
import sys, math, json
import pandas as pd, numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent
def t(x): x = x.dropna(); return float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x))))
out = {}
C = pd.read_parquet('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel/close_raw.parquet', columns=['2330'])
IDX = C.index
seen = set(); SD = []
for d in IDX:
    if d.day >= 11 and (d.year, d.month) not in seen: seen.add((d.year, d.month)); SD.append(d)
for e in ('v2', 'v3'):
    N = pd.read_parquet(HERE / f'r_navs_{e}.parquet')
    D = pd.DatetimeIndex([d for d in SD if d in N.index])
    w = (N.reindex(D).shift(-1) / N.reindex(D) - 1).iloc[:-1]
    for nm in ('A5', 'PLAC'):
        for c in ('normal', 'stress'):
            x = (w[f'{nm}|{c}'] - w[f'orig|{c}'])
            x = x[(x.index >= '2006-04-01') & (x.index <= '2019-03-31')]
            out[f'{e}|{nm}|{c}|SEL_sig_to_2019-03'] = (round(x.mean() * 100, 3), round(t(x), 2), int(x.notna().sum()))
print(json.dumps(out, indent=0))
