#!/usr/bin/env python3
"""o_pit_shares_industry_2006_2013 共用庫（資料題：只做股數／產業重建與閘門檢核，不算任何報酬）。

紀律：只讀原始收盤價 close_raw、開盤價 open、成交值 value、公司行動旗標 flag（排名、ADV、事件日）；
      **不讀 adj、ret**，不算任何策略、基準或組合報酬。審查員可用 grep 'adj\\|ret' 掃描本資料夾程式。
python3 -I -B 執行；不寫其他資料夾。
"""
import json, hashlib
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
RAW = HERE / 'raw'; DATA = HERE / 'data'
SCR = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/opit'); SCR.mkdir(parents=True, exist_ok=True)
V7D = AG / 'o_v7_listing_exec_engine' / 'data'
O8 = AG / 'o_0050_tilt_pit'
O12 = AG / 'o_lc_sector_rotation_pit'

def sha(f): return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def dumps(o): return json.dumps(o, ensure_ascii=False, indent=1, default=lambda z: float(z) if isinstance(z, (np.floating, np.integer)) else (bool(z) if isinstance(z, np.bool_) else (z.tolist() if isinstance(z, np.ndarray) else str(z))))

# ---------------- 面板（只讀 close_raw / open / value / flag） ----------------
def _rp(n): return pd.read_parquet(V7D / f'{n}.parquet')
C = _rp('close_raw'); O = _rp('open'); V = _rp('value'); FLAG = _rp('flag')
IDX = C.index; COLS = list(C.columns); N = len(COLS); CI = {c: j for j, c in enumerate(COLS)}
# o_infra_v3 下市股價（2019-05-17 起；只讀 close/open/value 三欄）
_dp = pd.read_parquet(AG / 'o_infra_v3_rebuild' / 'delist_prices.parquet', columns=['date', 'code', 'open', 'close', 'value'])
_dp['date'] = pd.to_datetime(_dp['date']); _dp = _dp[_dp.code.isin(CI) & _dp.date.isin(IDX)]
for col, M in [('close', C), ('open', O), ('value', V)]:
    P = _dp.pivot_table(index='date', columns='code', values=col, aggfunc='last')
    for c in P.columns:
        s = P[c].dropna(); m = M.loc[s.index, c].isna().to_numpy()
        M.loc[s.index[m], c] = s[m].to_numpy()
Cn = C.to_numpy(); On = O.to_numpy(); Vn = V.to_numpy(); Fn = np.nan_to_num(FLAG.to_numpy(), nan=0)
TRADED = np.isfinite(Cn)
CFF = C.ffill().to_numpy()
PREVC = C.ffill().shift(1).to_numpy()
NOBS = TRADED.sum(axis=1); GOOD = NOBS >= 300
DRMASK = np.array([c.startswith('91') or c.startswith('0') for c in COLS])
def ti(d): return int(IDX.searchsorted(pd.Timestamp(d), side='right') - 1)

def listed_mask():
    Lt = pd.read_parquet(V7D / 'listed_twse.parquet').reindex(columns=COLS).astype(float)
    Lt = Lt.reindex(IDX.union(Lt.index)).sort_index().ffill().reindex(IDX)
    return Lt.fillna(1.0).to_numpy() > 0.5
LST = listed_mask()

def alive_at(t):
    return TRADED[max(t - 5, 0):t + 1].any(axis=0) & np.isfinite(CFF[t]) & LST[t] & ~DRMASK

_ADV = {}
def adv120():
    if 'a' not in _ADV:
        Vd = pd.DataFrame(np.where(TRADED, Vn, np.nan))
        _ADV['a'] = Vd.rolling(120, min_periods=60).mean().to_numpy()
    return _ADV['a']

def v60_main():
    """主策略口徑（ia_core strict）：只用有值檔數 ≥ 300 的日子滾動 60 日平均成交值（min_periods 60）。"""
    g = np.where(GOOD)[0]
    v = pd.DataFrame(Vn[g]).rolling(60, min_periods=60).mean().to_numpy()
    out = np.full((len(IDX), N), np.nan); out[g] = v
    return out

def main_rebalance_days(start, end):
    """主策略換股日：每月第一個「日 ≥ 11」的面板交易日。"""
    out, seen = [], set()
    for d in IDX[(IDX >= pd.Timestamp(start)) & (IDX <= pd.Timestamp(end))]:
        if d.day >= 11 and (d.year, d.month) not in seen:
            seen.add((d.year, d.month)); out.append(d)
    return out

def month_last_days(start, end):
    s = pd.Series(IDX, index=IDX)
    s = s[(s.index >= pd.Timestamp(start)) & (s.index <= pd.Timestamp(end))]
    return list(s.groupby([s.index.year, s.index.month]).max())

def qend_day(q):
    """季末（含當日）最後一個面板交易日。"""
    return IDX[ti(q)]

# ---------------- 除權事件（不讀 ret） ----------------
def gap_median():
    with np.errstate(invalid='ignore', divide='ignore'):
        gap = On / PREVC - 1
    normal = TRADED & np.isfinite(On) & (Fn == 0) & np.isfinite(gap)
    return np.array([np.median(gap[t][normal[t]]) if normal[t].sum() > 50 else 0.0 for t in range(len(IDX))])

def open_factor():
    """事件日（flag 1/2/3）的價格調整因子估計 f＝前收×(1＋當日非事件股開盤跳空中位數)÷開盤（只用開盤與前收，不讀 ret）。"""
    med = gap_median()
    with np.errstate(invalid='ignore', divide='ignore'):
        f = PREVC * (1 + med[:, None]) / On
    m = np.isin(Fn, (1, 2, 3)) & TRADED & np.isfinite(On) & (On > 0)
    out = np.full((len(IDX), N), np.nan); out[m] = f[m]
    return out
