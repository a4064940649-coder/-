#!/usr/bin/env python3
"""第 0 步 B：上市＋上櫃日線寬表（2015-01-05～2026-10-30），含息報酬由官方「漲跌價差」推算：
ret = close / (close − change) − 1（證交所／櫃買的漲跌以除權息參考價為基準，所以這就是含息報酬）。
另算漲停價（參考價×1.1 取跳動點下界）、60 日均成交值（跳過缺資料日）、σ60。輸出 data/panel.npz。"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from pcore import DATA, floor_tick

d = pd.read_parquet(DATA / 'daily_long.parquet')
d = d[d['mkt'].isin(['twse', 'tpex'])].copy()
d = d[~d['code'].str.startswith('0')]
d['ref'] = d['close'] - d['change']
d.loc[~(d['ref'] > 0), 'ref'] = np.nan
d['ret'] = d['close'] / d['ref'] - 1
piv = lambda c: d.pivot(index='date', columns='code', values=c).sort_index()
C = piv('close'); IDX = C.index; COLS = C.columns
O = piv('open').reindex(index=IDX, columns=COLS); H = piv('high').reindex(index=IDX, columns=COLS)
L = piv('low').reindex(index=IDX, columns=COLS); VAL = piv('amount').reindex(index=IDX, columns=COLS)
RET = piv('ret').reindex(index=IDX, columns=COLS); REF = piv('ref').reindex(index=IDX, columns=COLS)
SH = piv('shares').reindex(index=IDX, columns=COLS)
mk = d.sort_values('date').groupby('code')['mkt'].last().reindex(COLS)
MKT = np.where(mk.to_numpy() == 'twse', 1, 2)
# 每日檔數；少於前後 20 日中位數 80% 的日子視為缺資料日
n = C.notna().sum(axis=1)
med = n.rolling(41, center=True, min_periods=5).median()
GAP = (n < 0.8 * med).to_numpy()
# 異常報酬：|ret|>0.5 視為資料錯誤，改 NaN（記錄筆數）
Rn = RET.to_numpy().copy()
bad = np.abs(Rn) > 0.5
nbad = int(np.nansum(bad)); Rn[bad] = np.nan
Cn = C.to_numpy(); Vn = VAL.to_numpy(); On = O.to_numpy()
# 漲停價（2015-06-01 起 10%）
LIMUP = floor_tick(REF.to_numpy() * 1.10)
# 60 日均成交值：只用非缺資料日；要求 60 日內至少 50 個有值
g = ~GAP
Vg = pd.DataFrame(Vn[g])
v60g = Vg.rolling(60, min_periods=50).mean().to_numpy()
V60 = np.full(Vn.shape, np.nan); V60[g] = v60g
V60 = pd.DataFrame(V60).ffill(limit=5).to_numpy()
s60g = pd.DataFrame(Rn[g]).rolling(60, min_periods=40).std().to_numpy()
SIG = np.full(Vn.shape, np.nan); SIG[g] = s60g
SIG = pd.DataFrame(SIG).ffill(limit=5).to_numpy()
np.savez_compressed(DATA / 'panel.npz', IDX=IDX.values.astype('datetime64[ns]'), COLS=np.array(COLS, dtype=str), MKT=MKT,
                    O=On, H=H.to_numpy(), L=L.to_numpy(), C=Cn, V=Vn, R=Rn, LIMUP=LIMUP, SH=SH.to_numpy(), V60=V60, SIG=SIG, GAP=GAP)
info = {'days': int(len(IDX)), 'codes': int(len(COLS)), 'twse': int((MKT == 1).sum()), 'tpex': int((MKT == 2).sum()),
        'gap_days': [str(x.date()) for x in IDX[GAP]], 'bad_ret_cells': nbad,
        'stocks_per_day_median_by_year': {int(k): float(v) for k, v in n.groupby(IDX.year).median().items()}}
json.dump(info, open(DATA / 'panel_info.json', 'w'), ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in info.items() if k != 'gap_days'}, ensure_ascii=False), 'gap days', len(info['gap_days']), info['gap_days'][:20])
