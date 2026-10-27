# -*- coding: utf-8 -*-
"""o_cb_board_runup 共用：面板（經 main_v2/v2_core → independent_audit/ia_core，只讀 import）、同池等權、事件報酬。"""
import os, sys, json, math, hashlib
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
sys.path.insert(0, str(AG / 'main_v2'))
import v2_core as V
sys.path.pop(0)
K = V.K
IDX, COLS, N, CI = K.IDX, K.COLS, K.N, K.CI
FINMASK = V.FINMASK
S52, V60 = V.rolling_signals('gapfix')
R = np.where(K.TRADED, np.clip(np.nan_to_num(K.Rn, nan=0.0), -0.5, 0.5), np.nan)
Cn, On = K.Cn, K.On
ADJ = K.MYADJ.to_numpy()

FEE, TAX = K.FEE, K.TAX
COST_RT = 0.0087      # 題目規定的一般來回成本
COST_RT_STRESS = 0.012

# ---------- 同池等權（每月換股日的前 300 流動性池；與主策略基礎池相同，不排除金融／處置） ----------
REB = K.rebalance_days('2005-01-01')
def pool_on(t):
    v = V60[t]; px = Cn[t]
    base = np.isfinite(v) & (v >= 1e7) & np.isfinite(px) & (px >= 10)
    vv = np.where(base, v, -np.inf)
    o = np.argsort(-vv, kind='stable')[:min(300, int(base.sum()))]
    m = np.zeros(N, bool); m[o] = True; m &= base
    return m

def _pool():
    pe = np.full(len(IDX), np.nan); pe_io = np.full(len(IDX), np.nan)
    inpool = np.zeros((len(IDX), N), bool)
    tr = [IDX.get_loc(d) for d in REB]
    io = np.where(np.isfinite(Cn) & np.isfinite(On) & (On > 0), Cn / On - 1, np.nan)   # 開盤→收盤（當日）
    io = np.clip(io, -0.5, 0.5)
    for k, t in enumerate(tr):
        t2 = tr[k + 1] if k + 1 < len(tr) else len(IDX) - 1
        m = pool_on(t)
        seg = slice(t + 1, t2 + 1)
        with np.errstate(all='ignore'):
            pe[seg] = np.nanmean(R[seg][:, m], axis=1)
            pe_io[seg] = np.nanmean(io[seg][:, m], axis=1)
        inpool[seg] = m
    return pe, pe_io, inpool, io
PEW, PEW_IO, INPOOL, IO = _pool()

def tpos_after(d):
    """嚴格晚於日曆日 d 的第一個交易日索引（公告日 d 的隔一交易日）。"""
    i = int(IDX.searchsorted(pd.Timestamp(d), side='right'))
    return i if i < len(IDX) else None

def tpos_on_before(d):
    i = int(IDX.searchsorted(pd.Timestamp(d), side='right')) - 1
    return i if i >= 0 else None

def tpos_on_after(d):
    i = int(IDX.searchsorted(pd.Timestamp(d), side='left'))
    return i if i < len(IDX) else None

def hold_ret(j, t1, t2):
    """t1 開盤買、t2 收盤賣的含息報酬；回傳 (個股, 同池等權, 有效)。
    個股：(C[t1]/O[t1]) × Π_{t1+1..t2}(1+ret)。停牌日 ret 視為 0；交易日 < 70% 視為無效。"""
    if t1 is None or t2 is None or t2 >= len(IDX) or t2 < t1: return np.nan, np.nan, False
    if not (np.isfinite(On[t1, j]) and On[t1, j] > 0 and np.isfinite(Cn[t1, j])): return np.nan, np.nan, False
    r = R[t1 + 1:t2 + 1, j]
    if len(r) and np.isfinite(r).mean() < 0.7: return np.nan, np.nan, False
    s = (1 + IO[t1, j]) * np.prod(1 + np.nan_to_num(r))
    b = (1 + np.nan_to_num(PEW_IO[t1])) * np.prod(1 + np.nan_to_num(PEW[t1 + 1:t2 + 1]))
    return float(s - 1), float(b - 1), True

def car(j, t0, a, b):
    """Σ(ret − 同池等權) over t0+a..t0+b（收盤到收盤）。"""
    lo, hi = t0 + a, t0 + b
    if lo < 1 or hi >= len(IDX): return np.nan
    r = R[lo:hi + 1, j]; p = PEW[lo:hi + 1]
    ok = np.isfinite(r)
    if ok.mean() < 0.7: return np.nan
    return float(np.nansum(np.where(ok, r - np.nan_to_num(p), 0.0)))

def tstat(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 3: return float('nan')
    s = x.std(ddof=1)
    return float(x.mean() / (s / math.sqrt(len(x)))) if s > 0 else float('nan')

def cluster_t(vals, keys):
    s = pd.Series(np.asarray(vals, float), index=pd.Index(keys)).dropna()
    if len(s) == 0: return dict(n=0)
    g = s.groupby(level=0).mean()
    return dict(n=int(len(s)), mean=float(s.mean()), median=float(s.median()), t_cl=tstat(g.to_numpy()),
                n_cl=int(len(g)), t_ev=tstat(s.to_numpy()), pos=float((s > 0).mean()))

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def conv(o):
    if isinstance(o, (np.floating,)): return None if not np.isfinite(o) else round(float(o), 6)
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.bool_,)): return bool(o)
    if isinstance(o, (pd.Timestamp,)): return str(o.date())
    if isinstance(o, float): return None if not math.isfinite(o) else round(o, 6)
    return str(o)

def dump(obj, name):
    (HERE / name).write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=conv), encoding='utf-8')
