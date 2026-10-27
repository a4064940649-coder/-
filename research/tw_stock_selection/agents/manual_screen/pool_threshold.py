#!/usr/bin/env python3
"""流動性池第 300 名的 60 日均成交值（給散戶當『成交金額門檻』參考），輸出 results_pool.json。"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import ms_core as K
W = K.web_signals(); out = {}
for d in K.rebalance_days('2006-01-01'):
    t = K.IDX.get_loc(d); v = W['V60L'][t]; px = K.Cn[t]
    ok = np.isfinite(v) & np.isfinite(px) & (px >= 10) & (v >= 1e7)
    s = np.sort(v[ok])[::-1]
    if len(s) >= 300: out[str(d.date())] = round(float(s[299]) / 1e8, 3)
ser = pd.Series(out); ser.index = pd.to_datetime(ser.index)
y = ser.groupby(ser.index.year).median().round(2).to_dict()
res = {'th300_60d_avg_value_億_by_year_median': {int(k): v for k, v in y.items()}, 'last': {k: out[k] for k in list(out)[-6:]}}
print(res)
(K.HERE / 'results_pool.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
