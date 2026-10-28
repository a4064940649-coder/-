#!/usr/bin/env python3
"""步驟 0b：把執行模擬需要的面板欄位匯出到本題 data/（PREREG 雜湊涵蓋），之後 sim.py／replicate 只讀 data/。
來源：o_v4_revenue_patch_official/v4lib.py（只讀 import；setup_px() 套用 P 下市股價格，取得 X 出清規則）。
匯出（只取曾經入選的股票欄，日期＝面板全期）：
  data/px_open.parquet、px_close.parquet（原始價，P 已補）、px_grow.parquet（1＋ret，無成交＝1）、px_traded.parquet（0/1）
  data/feat_daily.parquet：長表 (date, code) 只在「曾入選股 × 全部交易日」中 V60/σ60/池內成交值名次 有值的列
     V60＝gapfix 60 GOOD 日平均成交值；sig60＝60 個 GOOD 日 ret 標準差（窗內 ≥ 40 筆）；vrank＝當日（V60≥1e7、收盤≥10）中 V60 由大到小名次
  data/prevclose.parquet：前一筆收盤（跨停牌）
  data/xrules.json：X 出清規則（factor, release_date, type, valid_until）
  data/bench.parquet：0050 含息（面板日曆）、navs_v4 的 AB_V4|normal、AB_V4|EW、V4|normal
用法：python3 -I -B build_exec_data.py
"""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
V4D = HERE.parent / 'o_v4_revenue_patch_official'
sys.path.insert(0, str(V4D))
import numpy as np, pandas as pd
import v4lib as L
sys.path.pop(0)
K, V = L.K, L.V
XR = L.setup_px()
IDX, COLS = K.IDX, K.COLS
p = pd.concat([pd.read_csv(HERE / 'data' / 'plans_ab.csv', dtype=str), pd.read_csv(HERE / 'data' / 'plan_v4.csv', dtype=str)])
codes = sorted({c for x in p.picks for c in x.split(',')})
J = np.array([K.CI[c] for c in codes])
def wide(a): return pd.DataFrame(a[:, J], index=IDX, columns=codes)
wide(K.On).to_parquet(HERE / 'data' / 'px_open.parquet')
wide(K.Cn).to_parquet(HERE / 'data' / 'px_close.parquet')
wide(K.GROW).to_parquet(HERE / 'data' / 'px_grow.parquet')
wide(K.TRADED.astype(np.int8)).to_parquet(HERE / 'data' / 'px_traded.parquet')
wide(K.PREVC).to_parquet(HERE / 'data' / 'prevclose.parquet')

# V60（gapfix）與 σ60（GOOD 日）
_, V60 = V.rolling_signals('gapfix')
V60 = np.asarray(V60)
G = np.asarray(K.GOOD)
R = pd.DataFrame(K.Rn[G], index=IDX[G], columns=COLS)
SIG = R.rolling(60, min_periods=40).std().reindex(IDX).ffill().to_numpy()
base = np.isfinite(V60) & (V60 >= 1e7) & np.isfinite(K.Cn) & (K.Cn >= 10)
vv = np.where(base, V60, -np.inf)
rk = np.full(V60.shape, np.nan)
order = np.argsort(-vv, axis=1, kind='stable')
ranks = np.empty_like(order); ar = np.arange(V60.shape[1])
for t in range(len(IDX)):
    ranks[t, order[t]] = ar
rk = np.where(base, ranks + 1.0, np.nan)
F = pd.DataFrame({'V60': V60[:, J].ravel(), 'sig60': SIG[:, J].ravel(), 'vrank': rk[:, J].ravel()},
                 index=pd.MultiIndex.from_product([IDX, codes], names=['date', 'code'])).reset_index()
F = F[np.isfinite(F.V60) | np.isfinite(F.sig60)]
F.to_parquet(HERE / 'data' / 'feat_daily.parquet', index=False)

xr = {COLS[j]: [float(v[0]), str(pd.Timestamp(v[1]).date()), v[2], (str(pd.Timestamp(v[3]).date()) if len(v) > 3 else None)] for j, v in XR.items() if COLS[j] in set(codes)}
json.dump(xr, open(HERE / 'data' / 'xrules.json', 'w'), ensure_ascii=False, indent=1)
NV = pd.read_parquet(V4D / 'navs_v4.parquet')
B = pd.DataFrame({'E50': K.E50}).reindex(IDX)
for c in ('AB_V4|normal', 'AB_V4|stress', 'AB_V4|EW', 'V4|normal'): B[c] = NV[c].reindex(IDX)
B.to_parquet(HERE / 'data' / 'bench.parquet')
print('codes', len(codes), 'xrules', len(xr), 'feat rows', len(F))
