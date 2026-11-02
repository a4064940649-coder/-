#!/usr/bin/env python3
"""複製者 o9 核心：載入指定載體（o7 cmlib/cmstrat 原檔＋替換後輸入）、日期集合、自寫快速引擎、隨機安慰劑。"""
import os, sys, math, json
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
CAR = os.environ.get('O9_CAR', 'car_o7')
sys.path.insert(0, str(HERE / CAR))
import cmlib as L, cmstrat as T   # noqa: E402
assert Path(L.__file__).resolve().parent == (HERE / CAR).resolve()

FROZEN = 'F1_K5_N20_sqrt'
G = L.grow_exfix()
IDX, N = L.IDX, L.N
BUY, SELL = L.rates('normal')

# 漲停判斷快取（輸出與 L.limit_up_open 完全相同，只是避免重算）
_lu_orig = L.limit_up_open; _LU = {}
def _lu(t):
    if t not in _LU: _LU[t] = _lu_orig(t)
    return _LU[t]
L.limit_up_open = _lu

# ---------------- 日期 ----------------
TX = pd.DatetimeIndex(sorted(set(pd.read_csv(L.RAW / 'bench' / 'taiex_twii.csv', parse_dates=['date'])['date'])))
GOODSET = set(IDX[L.GOOD])
def deadline(m):
    s = str(m); day = 15 if s == '2019-02' else (12 if s == '2023-04' else 10)
    k = TX.searchsorted(pd.Timestamp(m.year, m.month, day)); return TX[k] if k < len(TX) else None
def day_for(m, j):
    D = deadline(m)
    if D is None: return None
    k = TX.get_loc(D) + j
    if k >= len(TX): return None
    s = TX[k]
    if s not in GOODSET: return None
    t = IDX.get_loc(s)
    if t + 1 >= len(IDX) or not L.GOOD[t + 1]: return None
    return s
def jdays(j, a, b, endm):
    out = []
    for m in pd.period_range(a, b, freq='M'):
        d = day_for(m, j)
        if d is not None: out.append(d)
    de = day_for(pd.Period(endm, 'M'), j)
    if de is None or de > IDX[-1]: de = IDX[-1]
    return [d for d in out if d < de], de
def offdays(a, b, end):
    ds = [d for d in L.official_days(a) if pd.Period(d, 'M') <= pd.Period(b, 'M') and d < pd.Timestamp(end)]
    return ds, pd.Timestamp(end)
def dateset(name, period):
    if period == 'MAIN':
        if name == 'OFF': return offdays('2019-05-01', '2026-09', '2026-10-12')
        return jdays(int(name[1:]), '2019-05', '2026-09', '2026-10')
    if period == 'PRE':
        if name == 'OFF':
            e = [d for d in L.official_days('2019-05-01') if pd.Period(d, 'M') == pd.Period('2019-05', 'M')][0]
            return offdays('2014-01-01', '2019-04', e)
        return jdays(int(name[1:]), '2014-01', '2019-04', '2019-05')
    if period == 'GATE':
        return offdays('2019-01-01', '2026-10', '2026-10-12')
    raise ValueError(period)
SETS = ['OFF'] + [f'J{j}' for j in range(13)]

# ---------------- 引擎 ----------------
class Windows:
    def __init__(self, days, end):
        self.days = list(days); self.end = end
        ts = [IDX.get_loc(d) for d in days]; te = IDX.get_loc(end)
        self.t0 = ts[0]; self.t1 = te
        self.ex = [t + 1 for t in ts]
        nxt = ts[1:] + [te]
        self.pre = []
        for k, e in enumerate(self.ex):
            tr = L.TRADED[e] & np.isfinite(L.On[e])
            with np.errstate(invalid='ignore', divide='ignore'):
                gap = np.where(tr, G[e] * L.On[e] / np.where(tr, L.Cn[e], 1.0), 1.0)
                oc = np.where(tr, L.Cn[e] / np.where(tr, L.On[e], 1.0), 1.0)
            F = np.prod(G[e + 1:nxt[k] + 1], axis=0)
            self.pre.append((tr, gap, oc, L.LAST_TRADE < e, F))

def solve_T(wv, htr, Vt, buy, sell):
    """解 T + buy·Σ(wT−h)+ + sell·Σ(h−wT)+ = Vt（分段線性、遞增），精確解。"""
    pos = wv > 0
    A = sell * htr[~pos].sum()
    w = wv[pos]; h = htr[pos]
    b = h / w; o = np.argsort(b, kind='stable'); b = b[o]; w = w[o]; h = h[o]
    # 在 T=b_k：前 k 個（含）為買方
    cw = np.concatenate([[0.0], np.cumsum(w)]); ch = np.concatenate([[0.0], np.cumsum(h)])
    Wt, Ht = cw[-1], ch[-1]
    # 斷點上的 f：買方 = 前 i 個（b ≤ T），賣方其餘
    i = np.arange(1, len(b) + 1)
    fb = b + A + buy * (b * cw[i] - ch[i]) + sell * ((Ht - ch[i]) - b * (Wt - cw[i]))
    k = int(np.searchsorted(fb, Vt, side='right'))   # 斷點 0..k−1 的 f ≤ Vt；段內前 k 個為買方
    slope = 1 + buy * cw[k] - sell * (Wt - cw[k])
    const = A - buy * ch[k] + sell * (Ht - ch[k])
    return (Vt - const) / slope

def sim(W, plan, buy=BUY, sell=SELL):
    """plan：list（與 W.ex 對齊）of (js, ws)。回傳 nav 於各訊號日與終點（長度 K+1）。"""
    val = np.zeros(N); cash = 1.0; nav = [1.0]
    for k, (js, ws) in enumerate(plan):
        tr, gap, oc, deadm, F = W.pre[k]
        h = val * gap
        frz = (h > 0) & ~tr
        dead = frz & deadm
        if dead.any():
            cash += h[dead].sum() * (1 - sell); h = np.where(dead, 0.0, h); frz &= ~dead
        htr = np.where(frz, 0.0, h)
        Vt = htr.sum() + cash
        wv = np.zeros(N); np.add.at(wv, js, ws); wv[frz] = 0.0
        s = wv.sum()
        if s <= 0: wv[:] = 0.0; Tt = 0.0
        else:
            wv = wv / s; Tt = solve_T(wv, htr, Vt, buy, sell)
        newval = np.where(frz, h, 0.0) + wv * Tt
        val = np.where(frz, newval, newval * oc) * F
        cash = 0.0
        nav.append(val.sum())
    return np.array(nav)

def plan_for(days, score_fn):
    plan = []; held = set()
    for d in days:
        cs = T.cross_section(d)
        sc = score_fn(cs)
        if sc is not None:
            c2 = dict(cs); c2['sc'] = sc
        else: c2 = cs
        js, ws, _, _ = T.target(c2, FROZEN, held)
        plan.append((np.asarray(js, int), np.asarray(ws, float))); held = set(js.tolist())
    return plan

def rand_score(rg):
    return lambda cs: np.where(cs['elig'], rg.random(N), np.nan)

def tstat(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x))))
