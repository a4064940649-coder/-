#!/usr/bin/env python3
"""V7X 零衝擊路徑（小數股、固定比例成本）。用來過零衝擊閘門：對 V7 原版逐日淨值相對差 ≤ 1e-9。
目標金額 T 的解法：f(T)=k·T＋b·Σ(T−h_i)+ ＋ s·Σ(h_i−T)+ ＋ s·Σ_非入選 h ＝ V，用排序後的斷點逐段求精確解（自寫）。
選項 disp：處置期出清以「收盤減一跳」計價；mdelay：現金對價併購以 0050 計值、下市日＋20 交易日撥款。"""
import numpy as np, pandas as pd
import x_data as D

FEE6 = 0.001425 * 0.6; TAX = 0.003


def rates(kind):
    if kind == 'stress': return 0.005, 0.005
    return FEE6 + 0.001, FEE6 + TAX + 0.001


def solve_T(hp, other, k, V, b, s):
    # f(T) = k T + b Σ (T-h)+ + s Σ (h-T)+ + s*other ; 遞增分段線性
    hs = np.sort(hp)
    n = len(hs)
    # 在區間 [hs[i-1], hs[i]] 內：有 i 個 h ≤ T（買進側）、n-i 個 h > T（賣出側）
    pref = np.concatenate([[0.0], np.cumsum(hs)])
    tot = pref[-1]
    for i in range(n + 1):
        # f(T) = kT + b(iT - pref[i]) + s((tot-pref[i]) - (n-i)T) + s*other
        slope = k + b * i - s * (n - i)
        cst = -b * pref[i] + s * (tot - pref[i]) + s * other
        T = (V - cst) / slope
        lo = hs[i - 1] if i > 0 else -np.inf
        hi = hs[i] if i < n else np.inf
        if lo - 1e-12 * max(1.0, abs(lo)) <= T <= hi + 1e-12 * max(1.0, abs(hi)):
            return T
    raise RuntimeError('no segment')


def simulate(pl, kind='normal', t0=None, t1=None, disp=False, mdelay=False):
    b, s = rates(kind)
    reb = {e: (pk, sg) for e, pk, sg in pl}
    t0 = pl[0][2] if t0 is None else t0
    t1 = len(D.IDX) - 1 if t1 is None else t1
    val = np.zeros(D.N); cash = 0.0; first = True
    mpend = []   # [金額, 撥款日 idx]
    used_m = set()
    nav = np.empty(t1 - t0 + 1)
    for t in range(t0, t1 + 1):
        if t == t0:
            nav[0] = 1.0; cash = 1.0; continue
        r50 = D.E50[t] / D.E50[t - 1] if np.isfinite(D.E50[t - 1]) and np.isfinite(D.E50[t]) else 1.0
        for m in mpend:
            if not (t in reb and m[1] < t): m[0] *= r50   # 解讀 v1：撥款後到再投入前仍停在 0050（再投入日以前收計）
        if t in reb:
            pk, _ = reb[t]
            tr = D.TR[t] & np.isfinite(D.O[t])
            with np.errstate(invalid='ignore', divide='ignore'):
                mult = np.where(tr, D.G[t] * D.O[t] / np.where(tr, D.C[t], 1.0), 1.0)
            h = val * mult
            frz = (h > 0) & ~tr
            for j in np.where(frz)[0]:
                if mdelay and j in D.MCASH and D.MCASH[j][1] <= t < D.MCASH[j][3]:
                    # 原規格解讀 v1：X 出清日轉為 0050 計值的待撥款（當日起計值），撥款日＝下市日＋20 交易日
                    mpend.append([h[j] * D.XR[j][0] * r50, D.MCASH[j][2]]); h[j] = 0.0; frz[j] = False
                    continue
                if j in D.XR and D.XR[j][1] <= t < D.XR[j][2]:
                    cash += h[j] * D.XR[j][0]; h[j] = 0.0; frz[j] = False
            # 撥款日已過的待撥款併入現金
            keep = []
            for m in mpend:
                if m[1] < t: cash += m[0]
                else: keep.append(m)
            mpend = keep
            pkset = np.zeros(D.N, bool); pkset[pk] = True
            dsell = np.zeros(D.N, bool)
            if disp:
                dsell = (h > 0) & tr & ~pkset & D.DISP[t]
                if dsell.any():
                    px = D.C[t] - D.tick(D.C[t])
                    with np.errstate(invalid='ignore', divide='ignore'):
                        h = np.where(dsell, h * px / D.O[t], h)
            htr = np.where(frz, 0.0, h)
            Vt = htr.sum() + cash
            hp = htr[pk]; other = htr.sum() - hp.sum()
            T = solve_T(hp, other, len(pk), Vt, b, s)
            nv = np.where(frz, h, 0.0); nv[pk] += T
            with np.errstate(invalid='ignore', divide='ignore'):
                val = np.where(frz, nv, nv * np.where(tr, D.C[t] / np.where(tr, D.O[t], 1.0), 1.0))
            cash = 0.0
        else:
            val = val * D.G[t]
        nav[t - t0] = val.sum() + cash + sum(m[0] for m in mpend)
    return pd.Series(nav, index=D.IDX[t0:t1 + 1])


def avg_nav(navs):
    R = pd.concat([n.pct_change() for n in navs], axis=1).mean(1).fillna(0)
    return (1 + R).cumprod()
