#!/usr/bin/env python3
"""比對複製員面板與研究員面板（只讀研究員 data/ 的輸出檔）。用法：r3_cmp_panel.py <suffix>"""
import sys, json
import numpy as np, pandas as pd
from pathlib import Path
H = Path(__file__).resolve().parent
R = H.parent / 'data'; M = H / 'data'
suf = sys.argv[1] if len(sys.argv) > 1 else ''
out = {}
for k in ['close_raw', 'open', 'high', 'low', 'value', 'ret', 'flag', 'src', 'adj']:
    a = pd.read_parquet(R / f'{k}.parquet'); b = pd.read_parquet(M / f'{k}{suf}.parquet')
    assert a.index.equals(b.index) and a.columns.equals(b.columns), k
    x = a.to_numpy(); y = b.to_numpy()
    both = np.isfinite(x) & np.isfinite(y); na = np.isfinite(x) != np.isfinite(y)
    eq = both & (np.abs(x - y) <= 1e-9 * np.maximum(1, np.abs(x)))
    w = (a.index >= '2024-01-02') & (a.index <= '2025-09-23')
    o = dict(cells_R=int(np.isfinite(x).sum()), nan_mismatch=int(na.sum()), eq_pct=round(eq.sum() / max(both.sum(), 1) * 100, 4),
             neq_win=int((both & ~eq)[w].sum()), neq_out=int((both & ~eq)[~w].sum()), maxabs=float(np.nanmax(np.abs(np.where(both, x - y, 0)))))
    if k == 'adj':
        rx = a.pct_change(fill_method=None).to_numpy(); ry = b.pct_change(fill_method=None).to_numpy()
        bb = np.isfinite(rx) & np.isfinite(ry)
        o['dailyratio_eq_pct'] = round(float((np.abs(rx - ry)[bb] < 1e-9).mean() * 100), 5)
    out[k] = o
    print(k, o)
json.dump(out, open(H / f'cmp_panel{suf}.json', 'w'), indent=1)
