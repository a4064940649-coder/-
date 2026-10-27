#!/usr/bin/env python3
"""抽查 PRIMARY 事件（手算 BIAS、進出場價），並列出極端值。"""
import pandas as pd, numpy as np
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
import run as R
df = pd.read_csv(R.HERE / 'primary_events.csv.gz', parse_dates=['sig', 'entry'], dtype={'code': str})
for seed in (1, 7):
    row = df.sample(1, random_state=seed).iloc[0]; c = row['code']; j = R.cols.index(c); t = R.IDX.get_loc(row['sig'])
    a = R.ADJ[c]; ma = a.rolling(20).mean()
    print(c, row['sig'].date(), 'BIAS20', round(a.iloc[t] / ma.iloc[t] - 1, 4), 'V60', round(R.V60[c].iloc[t] / 1e6, 1), 'close', R.CR[c].iloc[t])
    e = t + 1; x = e + int(row['hold'])
    path = [(str(R.IDX[s].date()), round(a.iloc[s] / ma.iloc[s] - 1, 4)) for s in range(e, x)]
    print(' entry open', R.AOn[e, j], 'exit', R.IDX[x].date(), R.PXo[x, j], 'raw', round(R.PXo[x, j] / R.AOn[e, j] - 1, 4), 'file raw', round(row['raw'], 4))
    print(' bias path', path)
print(df.nsmallest(5, 'raw')[['code', 'sig', 'hold', 'raw', 'delist']]); print(df.nlargest(5, 'raw')[['code', 'sig', 'hold', 'raw']])
print('months with most events'); print(df.groupby(df['entry'].dt.to_period('M')).size().sort_values().tail(6))
