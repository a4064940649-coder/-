#!/usr/bin/env python3
"""本題共用：路徑、面板載入、工具函式。只讀 data/（由 s0a_ingest.py 從 raw/ 轉出）與研究中心唯讀資料。"""
import json, hashlib
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / 'data'
AG = HERE.parent
RAW = Path('/home/user/-/research/tw_stock_selection/raw/jb7k6295r7-bot_tw-stock-data')
E50_CSV = '/home/user/jonatw/finlab-txd/data/raw/etf_0050.csv'
SEL_END_Q = (2019, 1)       # 挑選期：公告季 ≤ 2019Q1
VAL_START_Q = (2019, 2)     # 驗證期：2019Q2～2026Q2
FIN_IND = {'bd', 'basi', 'ins', 'fh', 'mim'}   # MOPS 金融保險業財報格式（排除）


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def tick(p):
    p = np.asarray(p, float)
    return np.select([p < 10, p < 50, p < 100, p < 500, p < 1000], [0.01, 0.05, 0.1, 0.5, 1.0], 5.0)


def floor_tick(x):
    x = np.asarray(x, float)
    t = tick(x)
    return np.floor(x / t + 1e-9) * t


def deadline(y, q):
    """IFRS（2013 起）一般公司法定期限：Q1 5/15、Q2 8/14、Q3 11/14、Q4（年報）次年 3/31。"""
    return {1: pd.Timestamp(y, 5, 15), 2: pd.Timestamp(y, 8, 14), 3: pd.Timestamp(y, 11, 14), 4: pd.Timestamp(y + 1, 3, 31)}[q]


def qend(y, q):
    return pd.Timestamp(y, 3 * q, 1) + pd.offsets.MonthEnd(0)


def load_panel():
    z = np.load(DATA / 'panel.npz', allow_pickle=False)
    return {k: z[k] for k in z.files}


def e50_on(idx):
    e = pd.read_csv(E50_CSV, parse_dates=['date']).set_index('date')['adj'].sort_index()
    return e.reindex(pd.DatetimeIndex(idx).union(e.index)).ffill().reindex(pd.DatetimeIndex(idx)).to_numpy()


def tstat(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 3 or x.std(ddof=1) == 0:
        return np.nan
    return x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))


def nw_t(x, lags=None):
    """Newey-West t（重疊報酬用）。"""
    x = np.asarray(x, float); x = x[np.isfinite(x)]; n = len(x)
    if n < 5:
        return np.nan
    if lags is None:
        lags = int(np.floor(4 * (n / 100) ** (2 / 9)))
    m = x.mean(); e = x - m
    s = (e @ e) / n
    for L in range(1, lags + 1):
        w = 1 - L / (lags + 1)
        s += 2 * w * (e[L:] @ e[:-L]) / n
    return m / np.sqrt(s / n) if s > 0 else np.nan
