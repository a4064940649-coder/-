#!/usr/bin/env python3
"""複製員淨值（rcore＋複製員面板）vs 研究員 navs_v7：最大相對差（以第一個共同日正規化）。輸出 rep_nav.json。"""
import json
from pathlib import Path
import numpy as np, pandas as pd
H = Path(__file__).resolve().parent; TOP = H.parent
REP = pd.read_parquet(H / 'out_rep.parquet'); NV = pd.read_parquet(TOP / 'navs_v7.parquet')
out = {}
for a, b in (('AB_V7', 'AB_V7|normal'), ('AB_V7_B10', 'AB_V7_B10|normal'), ('V7', 'V7|normal')):
    x = NV[b].dropna(); y = REP[a].reindex(x.index); y = y / y.iloc[0] * x.iloc[0]
    out[a] = {'max_rel_diff': float((y / x - 1).abs().max()), 'n': int(x.notna().sum())}
sd_rep = list(REP['V7_signal_days'].dropna().index.strftime('%Y-%m-%d')); sd_me = list(pd.read_csv(TOP / 'plan_V7.csv')['signal'])
out['signal_days_identical'] = sd_rep == sd_me
(H / 'rep_nav.json').write_text(json.dumps(out, ensure_ascii=False, indent=1)); print(out)
