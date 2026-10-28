#!/usr/bin/env python3
"""o7 複製：每個訊號日的橫截面、F1 權重、引擎、統計。"""
import math
import numpy as np, pandas as pd
import rlib as L

IDX, NC, COLS = L.IDX, L.NC, L.COLS
_On = L._On; _Cn = L._Cn


def limit_up_open(i_sig, ex):
    """e 開盤是否在跳動點漲停價（以 d 收盤經除權息因子調整為參考價）。"""
    lim = 0.07 if IDX[ex] < pd.Timestamp('2015-06-01') else 0.10
    fac_d = L.ADJ.iloc[i_sig].to_numpy() / _Cn[i_sig]
    fac_e = L.ADJ.iloc[ex].to_numpy() / _Cn[ex]
    with np.errstate(invalid='ignore', divide='ignore'):
        ref = L.CF.iloc[i_sig].to_numpy() * fac_d / fac_e
        raw = ref * (1 + lim)
        tk = L.tick(raw)
        lp = np.floor(raw / tk + 1e-9) * tk
        tk2 = L.tick(lp); lp = np.floor(raw / tk2 + 1e-9) * tk2
        return _On[ex] >= lp - 1e-9


class XS:
    """訊號日 d 的橫截面。"""
    def __init__(self, d):
        self.d = d; self.i = L.POS[d]; self.ex = self.i + 1; self.e = IDX[self.ex]
        i = self.i
        self.cap, self.rank = L.cap_rank(i)
        self.R, self.mem, self.capm = L.replica_weights(i, self.e)
        v60 = L.V60.iloc[i].to_numpy(); px = _Cn[i]; s52 = L.S52.iloc[i].to_numpy()
        y3 = L.yoy3_at(d).to_numpy()
        with np.errstate(invalid='ignore'):
            el = (self.rank <= 300) & ~L.ISFIN & ~L.ISKY & (v60 >= 1e7) & (px >= 10) & np.isfinite(s52) & np.isfinite(y3)
        el &= ~L.disposed(d, self.e)
        el &= np.isfinite(_On[self.ex]) & np.isfinite(_Cn[self.ex])
        self.el = el; self.s52 = s52; self.y3 = y3
        sc = np.full(NC, np.nan)
        e_idx = np.where(el)[0]
        r52 = pd.Series(s52[e_idx]).rank(pct=True).to_numpy()
        rrv = pd.Series(y3[e_idx]).rank(pct=True).to_numpy()
        sc[e_idx] = 0.5 * r52 + 0.5 * rrv
        self.score = sc
        # 排序：分數高→營收高→代碼小
        self.order = sorted(e_idx, key=lambda j: (-sc[j], -y3[j], COLS[j]))
        self.lu = limit_up_open(i, self.ex)


def f1_weights(x, held, K=5, N=20, wmode='sqrt', act_cap=0.05, tier=None):
    """F1 層替換。tier=None → 標準；回傳權重向量。"""
    R = x.R; mem = x.mem
    memr = sorted(mem, key=lambda j: -x.capm[j])
    H = set(memr[:K])
    tsmc = L.CI['2330']
    if tsmc in mem: H.add(tsmc)
    H |= {j for j in mem if L.ISFIN[j]}
    w = np.zeros(NC)
    for j in H: w[j] = R[j]
    base = np.zeros(NC)
    for j in mem:
        if j not in H and R[j] > act_cap:
            base[j] = R[j] - act_cap
    w += base
    W = 1.0 - w.sum()
    cand = [j for j in x.order if x.rank[j] > K and j not in H]
    pick = []
    nskip = 0
    it = iter(cand)
    def nxt():
        nonlocal nskip
        for j in it:
            if x.lu[j] and j not in held:
                nskip += 1; continue
            return j
        return None
    while len(pick) < N:
        j = nxt()
        if j is None: break
        pick.append(j)
    ncap = 0
    if not pick:
        raise RuntimeError(f'no satellite candidates at {x.d.date()} el={int(x.el.sum())}')
    while True:
        p = np.array(pick)
        raw = np.sqrt(x.cap[p]) if wmode == 'sqrt' else np.ones(len(p))
        ub = act_cap + R[p] - base[p]
        s = np.zeros(len(p)); free = np.ones(len(p), bool); rem = W
        for _ in range(len(p) + 1):
            if rem <= 1e-15 or not free.any(): break
            alloc = rem * raw[free] / raw[free].sum()
            s_try = s.copy(); s_try[free] += alloc
            over = free & (s_try > ub + 1e-15)
            if not over.any():
                s = s_try; rem = 0; break
            s[over] = ub[over]; free &= ~over; rem = W - s.sum()
        if rem > 1e-12:
            ncap += 1
            j = nxt()
            if j is None: break
            pick.append(j); continue
        break
    w[np.array(pick)] += s
    return w, pick, nskip


def simulate(wdict, kind='normal', start=None, end=None, gapfix=True, delist_cash=True):
    """wdict: {訊號日 d: 目標權重}；d 收盤算、e 開盤成交。回傳日 NAV(以收盤)、周轉列表。"""
    br, sr = L.costs(kind)
    RT = L.RETX if gapfix else L._Rn
    GAP = L.GAPX if gapfix else L.GAPP
    INTR = L.INTR
    ds = sorted(wdict)
    exmap = {L.POS[d] + 1: d for d in ds}
    i0 = L.POS[ds[0]] if start is None else L.POS[start]
    i1 = len(IDX) - 1 if end is None else L.POS[end]
    h = np.zeros(NC); cash = 1.0
    nav = np.full(len(IDX), np.nan); nav[i0] = 1.0
    turn = []
    for t in range(i0 + 1, i1 + 1):
        trd = L.TRADE_OK[t]
        if t in exmap:
            # 隔夜
            g = np.where(trd & np.isfinite(GAP[t]), GAP[t], 0.0)
            h = h * (1 + g)
            # 下市凍結 → 現金
            if delist_cash:
                dead = (h != 0) & (L.LAST_TRADE < t)
                if dead.any():
                    cash += (h[dead] * (1 - sr)).sum(); h[dead] = 0
            Vpre = h.sum() + cash
            frozen = (h != 0) & ~trd
            Fv = h[frozen].sum()
            tw = wdict[exmap[t]].copy()
            tw[frozen] = 0
            tw[~trd] = 0
            if tw.sum() > 0: tw = tw / tw.sum()
            hf = np.where(frozen, 0, h)
            def total_after(Vx):
                tgt = tw * (Vx - Fv)
                dlt = tgt - hf
                c = br * np.clip(dlt, 0, None).sum() + sr * np.clip(-dlt, 0, None).sum()
                return Vpre - c
            lo, hi = 0.9 * Vpre, Vpre
            for _ in range(60):
                mid = (lo + hi) / 2
                if total_after(mid) >= mid: lo = mid
                else: hi = mid
            Vx = lo
            tgt = tw * (Vx - Fv)
            turn.append((IDX[t], np.abs(tgt - hf).sum() / 2 / Vpre, np.clip(tgt - hf, 0, None).sum() / Vpre))
            h = np.where(frozen, h, tgt)
            cash = Vx - Fv - tgt.sum()  # ≈0
            r = np.where(trd & np.isfinite(INTR[t]), INTR[t], 0.0)
            h = h * (1 + r)
        else:
            r = np.where(np.isfinite(RT[t]), RT[t], 0.0)
            h = h * (1 + r)
        nav[t] = h.sum() + cash
    return pd.Series(nav, index=IDX).iloc[i0:i1 + 1], turn


# ---------------- 統計 ----------------
def win_returns(nav, dates):
    """訊號窗：NAV(d_k) → NAV(d_{k+1})；以起點 d_k 標記。"""
    n = nav.reindex(dates)
    return (n.shift(-1) / n - 1).iloc[:-1]

def bench_win(dates):
    b = L.E50.reindex(dates)
    return (b.shift(-1) / b - 1).iloc[:-1]

def tstat(x):
    x = pd.Series(x).dropna(); return x.mean() / x.std(ddof=1) * math.sqrt(len(x))

PER = {'SEL': ('2019-01-01', '2022-12-31'), 'VAL': ('2023-01-01', '2026-09-30'),
       'POST2019': ('2019-01-01', '2026-09-30'), 'PRE': ('2006-04-01', '2018-12-31'),
       'ALL': ('2006-04-01', '2026-09-30'), 'V24': ('2024-01-01', '2026-09-30')}

def sel(s, p):
    a, b = PER[p]; return s[(s.index >= a) & (s.index <= b)]

def mdd(nav):
    return float((nav / nav.cummax() - 1).min())

def stats(nav, dates, other=None, periods=('SEL', 'VAL', 'V24', 'POST2019', 'PRE', 'ALL')):
    wr = win_returns(nav, dates)
    ob = bench_win(dates) if other is None else win_returns(other, dates)
    out = {}
    for p in periods:
        a = sel(wr, p); b = sel(ob, p); ex = (a - b).dropna()
        te = ex.std(ddof=1) * math.sqrt(12)
        d0 = a.index[0]; dl = dates[list(dates).index(a.index[-1]) + 1]
        seg = nav[(nav.index >= d0) & (nav.index <= dl)]
        n = len(a)
        ann = (1 + a).prod() ** (12 / n) - 1; annb = (1 + b).prod() ** (12 / n) - 1
        yrs = (dl - d0).days / 365.25
        annc = (1 + a).prod() ** (1 / yrs) - 1; annbc = (1 + b).prod() ** (1 / yrs) - 1
        bn = L.E50[(L.E50.index >= d0) & (L.E50.index <= dl)] if other is None else other[(other.index >= d0) & (other.index <= dl)]
        out[p] = dict(n=n, ex_pp=ex.mean() * 100, t=tstat(ex), IR=ex.mean() * 12 / te if te > 0 else np.nan, TE=te * 100,
                      ann=ann * 100, ann_b=annb * 100, ann_cal=annc * 100, ann_b_cal=annbc * 100,
                      mdd=mdd(seg) * 100, mdd_b=mdd(bn) * 100)
    return out, wr, ob
