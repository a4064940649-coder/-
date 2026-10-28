#!/usr/bin/env python3
"""獨立複製（o_hold_buffer_fresh / A20）：只 import 既有基礎設施（v2_core / drj_core / o3_core）取資料與主策略橫斷面；
緩衝選股、日頻引擎、成本、統計全部自己寫（不讀 hb_core）。"""
import sys, math
from pathlib import Path
import numpy as np, pandas as pd

AG = Path('/home/user/-/research/tw_stock_selection/agents')
sys.path.insert(0, str(AG / 'o_disposal_rejudge'))
import drj_core as R            # noqa
sys.path.pop(0)
sys.path.insert(0, str(AG / 'o_infra_v3_rebuild'))
import o3_core as O3            # noqa
sys.path.pop(0)
K, V = R.K, R.V
assert O3.K is K
NPATCH = O3.apply_patch(require_freeze=True)
XR = O3.exit_rules(O3.load_events(require_freeze=True), 'main')
DS = R.Disp(R.load_list())
YL0, Y50 = R.revenue()
YRD = O3.load_rdel()
IDX, N, COLS = K.IDX, K.N, K.COLS

def e1_mask(d):
    m = np.zeros(N, bool)
    for j in DS.at(d): m[j] = True
    return m

def make_S(unavail=None):
    S = V.V2('gapfix', YRD, Y50, exfin=True, unavail=unavail)
    orig = S.cross_section
    def cs(d, _o=orig):
        old = K.disposal_mask; K.disposal_mask = e1_mask
        try: return _o(d)
        finally: K.disposal_mask = old
    S.cross_section = cs
    return S

# ---------------- 排名（自己寫：分數 round 10 位 → 營收 3 月年增率 → 欄位順序） ----------------
def ranking(cs):
    el = cs['el']; idx = np.where(el)[0]
    sc = np.round(cs['sc'][idx], 10); y3 = np.nan_to_num(cs['y3'][idx], nan=-1e18)
    o = sorted(range(len(idx)), key=lambda i: (-sc[i], -y3[i], idx[i]))
    return idx[np.array(o, int)]

# ---------------- 緩衝選股 ----------------
def plan_buffer(S, n=10, Kbuf=None, h52=None):
    """Kbuf=None → 原版。回傳 list of dict(e, d, picks, kept, new, rank(dict j→rank), cand)。"""
    out = []; held = []
    for d in S.sig_days:
        cs = S.cross_section(d)
        el = cs['el']
        if el.sum() < 50: continue
        e = cs['e']; gu = K.gap_up_open(e)
        cand = ranking(cs)
        rk = {int(j): i + 1 for i, j in enumerate(cand)}
        keep = []
        if Kbuf is not None:
            for j in held:
                if el[j] and rk.get(j, 10 ** 9) <= Kbuf and (h52 is None or cs['s52'][j] >= h52):
                    keep.append(j)
            keep = sorted(keep, key=lambda j: rk[j])[:n]
        hs = set(held); ks = set(keep)
        pick = list(keep); new = []
        for j in cand:
            if len(pick) == n: break
            j = int(j)
            if j in ks: continue
            if gu[j] and j not in hs: continue
            pick.append(j)
            if j not in hs: new.append(j)
        out.append(dict(e=e, d=d, picks=np.array(sorted(pick), int), kept=keep, new=new, rank=rk, held_prev=list(held), s52=cs['s52'], el=el))
        held = list(pick)
    return out

# ---------------- 日頻引擎（自己寫） ----------------
def _solve_T(htr_pk, other_sell, Vt, k, bvec, svec):
    """解 k*T + Σ_pk [b_j (T-h_j)^+ + s_j (h_j-T)^+] + other_sell = Vt（逐段線性，精確解）。"""
    hs = np.sort(np.unique(np.r_[0.0, htr_pk]))
    def f(T):
        return k * T + (bvec * np.clip(T - htr_pk, 0, None)).sum() + (svec * np.clip(htr_pk - T, 0, None)).sum() + other_sell - Vt
    # 找段
    lo = 0.0
    pts = [p for p in hs if p > 0] + [Vt / k * 2 + hs.max()]
    for p in pts:
        if f(p) >= 0:
            hi = p; break
        lo = p
    flo, fhi = f(lo), f(hi)
    return lo + (hi - lo) * (-flo) / (fhi - flo) if fhi != flo else lo

def sim(plan, buy, sell, t0, t1, xrules=XR, cost_fn=None, G=None):
    """plan: list of (e, picks)。buy/sell 純量；cost_fn(t, htr, pk, T_guess)→(bvec_all, svec_all) 時逐檔成本（用二分）。
    回傳 nav, dict(turn_sell, turn_buy 每次（對總值）)。"""
    G = K.GROW if G is None else G
    Cn, On, TR = K.Cn, K.On, K.TRADED
    reb = {e: pk for e, pk in plan}
    rel = {j: (v[0], int(np.searchsorted(IDX, v[1])), int(np.searchsorted(IDX, v[3])) if len(v) > 3 else 10 ** 9) for j, v in (xrules or {}).items()}
    val = np.zeros(N); cash = 1.0
    nav = np.empty(t1 - t0 + 1); ts, tb = [], []
    for t in range(t0, t1 + 1):
        if t in reb:
            pk = np.asarray(reb[t], int)
            tr = TR[t] & np.isfinite(On[t])
            h = val * np.where(tr, G[t] * On[t] / np.where(tr, Cn[t], 1.0), 1.0)
            frz = (h > 0) & ~tr
            for j in np.where(frz)[0]:
                if j in rel and rel[j][1] <= t < rel[j][2]:
                    cash += h[j] * rel[j][0]; h[j] = 0.0; frz[j] = False
            Vfz = h[frz].sum()
            htr = np.where(frz, 0.0, h)
            Vt = htr.sum() + cash
            k = len(pk)
            inpk = np.zeros(N, bool); inpk[pk] = True
            if cost_fn is None:
                other = sell * htr[~inpk].sum()
                T = _solve_T(htr[pk], other, Vt, k, np.full(k, buy), np.full(k, sell))
            else:
                def g(T):
                    tgt = np.where(inpk, T, 0.0)
                    bv, sv = cost_fn(t, htr, tgt, Vt + Vfz)
                    return k * T + (bv * np.clip(tgt - htr, 0, None)).sum() + (sv * np.clip(htr - tgt, 0, None)).sum() - Vt
                lo, hi = 0.0, Vt / k
                for _ in range(60):
                    mid = 0.5 * (lo + hi)
                    if g(mid) > 0: hi = mid
                    else: lo = mid
                T = lo
            tgt = np.where(inpk, T, 0.0)
            ts.append(np.clip(htr - tgt, 0, None).sum() / (Vt + Vfz))
            tb.append(np.clip(tgt - htr, 0, None).sum() / (Vt + Vfz))
            nv = np.where(frz, h, tgt)
            val = np.where(frz, nv, nv * np.where(tr, Cn[t] / np.where(tr, On[t], 1.0), 1.0))
            cash = 0.0
        elif t > t0:
            val = val * G[t]
        nav[t - t0] = val.sum() + cash
    return pd.Series(nav, index=IDX[t0:t1 + 1]), dict(ts=ts, tb=tb)

# ---------------- 統計（自己寫） ----------------
E50 = K.E50
def tstat(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 3: return float('nan')
    s = x.std(ddof=1)
    return float(x.mean() / (s / math.sqrt(len(x))))

def winret(nav, sd):
    sd = [d for d in sd if d in nav.index]
    v = nav.reindex(pd.DatetimeIndex(sd)).to_numpy()
    return pd.Series(v[1:] / v[:-1] - 1, index=pd.DatetimeIndex(sd[:-1]))

def sel(s, a, b):
    return s[(s.index >= pd.Timestamp(a)) & (s.index <= pd.Timestamp(b))]

def cagr_dd(nav, a, b):
    i = np.where((nav.index >= pd.Timestamp(a)) & (nav.index <= pd.Timestamp(b)))[0]
    i0, i1 = i[0], i[-1]
    j0 = i0 - 1 if i0 > 0 else i0
    seg = nav.iloc[j0:i1 + 1] / nav.iloc[j0]
    yrs = (nav.index[i1] - nav.index[j0]).days / 365.25
    return (seg.iloc[-1] ** (1 / yrs) - 1) * 100, float((seg / seg.cummax() - 1).min()) * 100

PER = {'SEL': ('2006-04-01', '2018-12-31'), 'VAL': ('2019-01-01', '2023-12-31'), 'HOLD': ('2024-01-01', '2026-10-12'),
       'POST': ('2019-01-01', '2026-10-12'), 'ALL': ('2006-04-01', '2026-10-12')}

def cal_t(nav, a, b):
    me = nav.groupby(nav.index.to_period('M')).last()
    be = E50.reindex(nav.index).groupby(nav.index.to_period('M')).last()
    x = (me.pct_change() - be.pct_change()).dropna()
    x = x[(x.index >= pd.Period(a, 'M')) & (x.index <= pd.Period(b, 'M'))]
    return tstat(x)
