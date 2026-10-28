"""replicate_o5 共用：時間格點、窗報酬、統計工具（獨立實作）。"""
import math
from pathlib import Path
import numpy as np, pandas as pd
AG = Path('/home/user/-/research/tw_stock_selection/agents')
H = AG / 'o_main_spa_breaks' / 'replicate_o5'
OUT = H / 'out'; OUT.mkdir(exist_ok=True)
PANEL = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel')
END = pd.Timestamp('2026-10-12')

def grid():
    p = pd.read_csv(AG / 'o_v4_revenue_patch_official/plan_V4.csv')
    g = list(pd.to_datetime(p['signal'])) + [END]
    return pd.DatetimeIndex(g)
G = grid(); NW = len(G) - 1
WSTART = {'W_long': 0, 'W_19': int((G[:-1] >= '2019-05-01').argmax()), 'W_21': int((G[:-1] >= '2021-03-01').argmax())}
SEL = slice(0, 157); VAL = slice(157, 213); HOLD = slice(213, 245)

def win_ret(nav, stop_days=7):
    """日頻淨值 Series → 245 窗簡單報酬；格點取 asof；淨值結束超過 7 天或還沒開始 → NaN。"""
    s = nav.dropna().sort_index()
    s = s[~s.index.duplicated(keep='last')]
    if len(s) < 2: return np.full(NW, np.nan)
    idx = s.index
    pos = idx.searchsorted(G, side='right') - 1
    v = np.where(pos >= 0, s.to_numpy()[np.clip(pos, 0, None)], np.nan).astype(float)
    last = idx[np.clip(pos, 0, None)]
    stale = (pos < 0) | ((G - last).days > stop_days)
    v[stale] = np.nan
    v[v <= 0] = np.nan
    return v[1:] / v[:-1] - 1

def e0050():
    d = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/etf_0050.csv', parse_dates=['date']).set_index('date')['adj']
    return d

def tstat(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x))))

def pr_omega2(d, L):
    """Politis-Romano 平穩 bootstrap 長期變異數（Hansen 2005 用的 kappa）。d: (n,) 或 (n,k)。"""
    d = np.asarray(d, float)
    if d.ndim == 1: d = d[:, None]
    n = d.shape[0]; p = 1.0 / L
    x = d - d.mean(0)
    g0 = (x * x).mean(0)
    om = g0.copy()
    for i in range(1, n):
        k = ((n - i) / n) * (1 - p) ** i + (i / n) * (1 - p) ** (n - i)
        gi = (x[i:] * x[:-i]).sum(0) / n
        om += 2 * k * gi
    return om

def sb_indices(n, L, B, seed):
    """平穩 bootstrap 索引 (B, n)。"""
    rng = np.random.default_rng(seed)
    p = 1.0 / L
    idx = np.empty((B, n), np.int64)
    idx[:, 0] = rng.integers(0, n, B)
    newblk = rng.random((B, n)) < p
    rnd = rng.integers(0, n, (B, n))
    for t in range(1, n):
        idx[:, t] = np.where(newblk[:, t], rnd[:, t], (idx[:, t - 1] + 1) % n)
    return idx
