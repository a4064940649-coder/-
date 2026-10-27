#!/usr/bin/env python3
"""抽樣核對：獨立重算 2023-06-30 的 SHORT／LONG 訊號，與 run.py 結果比對。"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run as R
d = pd.Timestamp('2023-06-30')
a = R.ADJ
i = a.index.get_loc(d)
mism = 0; n = 0
for c in R.cols:
    s = a[c].iloc[:i + 1]
    if s.iloc[-125:].isna().any(): continue
    m = {k: s.iloc[-k:].mean() for k in (5, 10, 20, 60, 120)}
    m5 = {k: s.iloc[-k - 5:-5].mean() for k in (5, 10, 20, 60, 120)}
    sh = m[5] > m[10] > m[20] > m[60] and all(m[k] > m5[k] for k in (5, 10, 20, 60)) and s.iloc[-1] > m[20]
    lo = m[20] > m[60] > m[120] and all(m[k] > m5[k] for k in (20, 60, 120)) and s.iloc[-1] > m[20]
    n += 1
    mism += (bool(R.S_SHORT.loc[d, c]) != sh) + (bool(R.S_LONG.loc[d, c]) != lo)
print('checked', n, 'mismatches', mism, 'SHORT count', int(R.S_SHORT.loc[d].sum()), 'LONG', int(R.S_LONG.loc[d].sum()))
ev, pools, nb, info = R.monthly_events(R.S_SHORT, 'ALL')
print(info.loc['2023-06-30'])
