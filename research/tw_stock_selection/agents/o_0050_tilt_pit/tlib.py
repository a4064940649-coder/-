#!/usr/bin/env python3
"""o_0050_tilt_pit 共用庫。只讀 raw/、本資料夾 data/，以及（唯讀、原地引用）V7 凍結面板與 o_infra_v3 下市股價。
不 import 其他資料夾的程式；成本、訊號與營收已公告規則照 o_capmatched_selection/cmlib.py 重寫（口徑相同）。"""
import json, math, hashlib
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
RAW = HERE / 'raw'; DATA = HERE / 'data'; DATA.mkdir(exist_ok=True)
AG = HERE.parent
V7D = AG / 'o_v7_listing_exec_engine' / 'data'
def sha(f): return hashlib.sha256(Path(f).read_bytes()).hexdigest()

# ---------------- 面板：V7 凍結面板 + o_infra_v3 補 2019-05-17～2023 下市股 ----------------
def _rp(n): return pd.read_parquet(V7D / f'{n}.parquet')
C = _rp('close_raw'); O = _rp('open'); V = _rp('value'); RET = _rp('ret'); FLAG = _rp('flag'); HI = _rp('high')
IDX = C.index; COLS = C.columns; N = len(COLS); CI = {c: j for j, c in enumerate(COLS)}
_dp = pd.read_parquet(AG / 'o_infra_v3_rebuild' / 'delist_prices.parquet')
_dp['date'] = pd.to_datetime(_dp['date']); _dp = _dp[_dp.code.isin(CI) & _dp.date.isin(IDX)]
INFRA_FILLED = 0
for col, M in [('close', C), ('open', O), ('value', V), ('ret', RET), ('high', HI)]:
    P = _dp.pivot_table(index='date', columns='code', values=col, aggfunc='last')
    for c in P.columns:
        s = P[c].dropna(); m = M.loc[s.index, c].isna()
        if col == 'close': INFRA_FILLED += int(m.sum())
        M.loc[s.index[m.to_numpy()], c] = s[m.to_numpy()].to_numpy()
Cn = C.to_numpy(); On = O.to_numpy(); Vn = V.to_numpy(); Rn = RET.to_numpy(); Fn = FLAG.to_numpy()
TRADED = np.isfinite(Cn)
GROW = np.where(TRADED, 1.0 + np.nan_to_num(Rn, nan=0.0), 1.0)
LAST_TRADE = np.where(TRADED.any(axis=0), len(IDX) - 1 - np.argmax(TRADED[::-1], axis=0), -1)
CFF = C.ffill().to_numpy()
PREVC = C.ffill().shift(1).to_numpy()
NOBS = TRADED.sum(axis=1); GOOD = NOBS >= 300

# 除權息日（flag 1：面板 ret 設 0）→ exfix：收盤/開盤 ×（1＋當日非除權息股開盤跳空中位數）−1（同 cmlib.grow_exfix，正式口徑）
def _grow_exfix():
    G = GROW.copy()
    with np.errstate(invalid='ignore', divide='ignore'):
        gap = On / PREVC - 1
        normal = TRADED & np.isfinite(On) & (np.nan_to_num(Fn, nan=0) == 0) & np.isfinite(gap)
        med = np.array([np.median(gap[t][normal[t]]) if normal[t].sum() > 50 else 0.0 for t in range(len(IDX))])
        fx = (Fn == 1) & TRADED & np.isfinite(On) & (On > 0)
        alt = Cn / On * (1 + med[:, None])
    G[fx] = alt[fx]
    return G
GX = _grow_exfix()

# ---------------- 基準 ----------------
_e = pd.read_csv(RAW / 'bench' / 'etf_0050.csv', parse_dates=['date']).set_index('date')['adj'].sort_index()
E50 = _e.reindex(IDX.union(_e.index)).ffill().reindex(IDX)
_tx = pd.read_csv(RAW / 'bench' / 'taiex_twii.csv', parse_dates=['date'])['date']
TCAL = pd.DatetimeIndex(sorted(set(_tx) | set(IDX)))

# ---------------- 日曆 ----------------
def official_days(start='2005-01-01', end=None):
    out, seen = [], set()
    ii = IDX[IDX >= pd.Timestamp(start)]
    if end is not None: ii = ii[ii <= pd.Timestamp(end)]
    for d in ii:
        if d.day >= 11 and (d.year, d.month) not in seen:
            seen.add((d.year, d.month)); out.append(d)
    return out
def deadline(m):
    p = pd.Period(m, 'M'); day = 15 if str(p) in ('2019-02', '2023-04') else 10
    return TCAL[TCAL >= pd.Timestamp(p.year, p.month, day)][0]
def jscan_days(j, start='2005-01-01', end=None):
    """J{j}：營收期限日 D_m 之後第 j 個面板交易日（j＝0 為 D_m 當日或其後第一個面板日）。"""
    out = []
    for p in pd.period_range(pd.Timestamp(start), pd.Timestamp(end) if end else IDX[-1], freq='M'):
        D = deadline(p); k = IDX.searchsorted(D)
        if k + j < len(IDX): out.append(IDX[k + j])
    return out

def dumps(o): return json.dumps(o, ensure_ascii=False, indent=1, default=lambda z: float(z) if isinstance(z, (np.floating, np.integer)) else (bool(z) if isinstance(z, np.bool_) else str(z)))
def tstat(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 3: return float('nan')
    s = x.std(ddof=1)
    return float(x.mean() / (s / math.sqrt(len(x)))) if s > 0 else float('nan')
