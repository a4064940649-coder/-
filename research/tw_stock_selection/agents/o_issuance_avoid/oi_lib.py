# -*- coding: utf-8 -*-
"""o_issuance_avoid shared library.
Imports main_v2/v2_core.py (read-only) for the panel, rolling signals (gapfix), main-strategy cross sections,
open-limit-up check, the daily engine (independent_audit/ia_core.simulate) and segment statistics.
"""
import os, sys, json, math, hashlib
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'main_v2'))
import v2_core as V
sys.path.pop(0)
K = V.K
IDX, COLS, N = K.IDX, K.COLS, K.N
CI = K.CI

SEL_END = pd.Timestamp('2019-04-30')      # selection period end (price-class split, BRIEF)
VAL_START = pd.Timestamp('2019-05-01')
HOLD_START = pd.Timestamp('2024-01-01')   # reported separately inside validation
EV_START = pd.Timestamp('2005-01-01')

S52, V60 = V.rolling_signals('gapfix')
R = np.where(K.TRADED, np.clip(np.nan_to_num(K.Rn, nan=0.0), -0.5, 0.5), np.nan)

# ---------------- same-pool equal weight (daily) ----------------
REB = K.rebalance_days('2005-01-01')
POOL = {}
def pool_on(d):
    """top-300 liquidity pool on rebalance day d (same base rule as main strategy, no disposal/fin filters)."""
    if d in POOL: return POOL[d]
    t = IDX.get_loc(d)
    v = V60[t]; px = K.Cn[t]
    base = np.isfinite(v) & (v >= 1e7) & np.isfinite(px) & (px >= 10)
    vv = np.where(base, v, -np.inf)
    o = np.argsort(-vv, kind='stable')[:min(300, int(base.sum()))]
    m = np.zeros(N, bool); m[o] = True; m &= base
    POOL[d] = m
    return m

def _pool_ew():
    pe = np.full(len(IDX), np.nan)
    inpool = np.zeros((len(IDX), N), bool)
    tr = [IDX.get_loc(d) for d in REB]
    for k, t in enumerate(tr):
        t2 = tr[k + 1] if k + 1 < len(tr) else len(IDX) - 1
        m = pool_on(IDX[t])
        seg = slice(t + 1, t2 + 1)
        x = R[seg][:, m]
        with np.errstate(all='ignore'):
            pe[seg] = np.nanmean(x, axis=1)
        inpool[seg] = m
    return pe, inpool
PEW, INPOOL = _pool_ew()

ADJ = K.MYADJ.to_numpy()

def tpos(d):
    """first trading index on/after date d"""
    i = IDX.searchsorted(pd.Timestamp(d))
    return int(i) if i < len(IDX) else None

def car(j, t0, a, b):
    """sum of (ret - poolEW) over t0+a..t0+b; None if window outside panel or stock untraded > 30% of days."""
    lo, hi = t0 + a, t0 + b
    if lo < 1 or hi >= len(IDX): return np.nan
    r = R[lo:hi + 1, j]; p = PEW[lo:hi + 1]
    ok = np.isfinite(r)
    if ok.mean() < 0.7: return np.nan
    return float(np.nansum(np.where(ok, r - np.nan_to_num(p), 0.0)))

def bhar(j, t0, a, b):
    lo, hi = t0 + a, t0 + b
    if lo < 1 or hi >= len(IDX): return np.nan
    r = R[lo:hi + 1, j]; p = PEW[lo:hi + 1]
    if np.isfinite(r).mean() < 0.7: return np.nan
    return float(np.prod(1 + np.nan_to_num(r)) - np.prod(1 + np.nan_to_num(p)))

def tstat(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 3: return float('nan')
    s = x.std(ddof=1)
    return float(x.mean() / (s / math.sqrt(len(x)))) if s > 0 else float('nan')

def cluster_t(vals, keys):
    """t of monthly (cluster) means: events grouped by anchor month."""
    s = pd.Series(np.asarray(vals, float), index=pd.Index(keys)).dropna()
    if len(s) == 0: return dict(n=0, mean=np.nan, t_cl=np.nan, n_cl=0, t_ev=np.nan)
    g = s.groupby(level=0).mean()
    return dict(n=int(len(s)), mean=float(s.mean()), median=float(s.median()), t_cl=tstat(g.to_numpy()), n_cl=int(len(g)), t_ev=tstat(s.to_numpy()))

# ---------------- events ----------------
EVENTS_SHA = None
def load_events():
    res = json.load(open(HERE / 'results.json'))
    sha = res['EVENTS_FROZEN']['sha256']['events.csv']
    f = HERE / 'events.csv'
    assert hashlib.sha256(f.read_bytes()).hexdigest() == sha, 'events.csv differs from frozen version'
    E = pd.read_csv(f, dtype={'code': str, 'cb_code': str}, parse_dates=['recv', 'eff', 'list_date', 'anchor'])
    E = E[E['code'].isin(CI)].copy()
    E['j'] = E['code'].map(CI)
    return E

def exrights_date(j, eff, max_td=45):
    """first panel ex-right/ex-div day (flag 1/2) within max_td trading days after the effective date."""
    t = tpos(eff)
    if t is None: return None
    f = K.Fn[t + 1:t + 1 + max_td, j]
    w = np.where((f == 1) | (f == 2))[0]
    return int(t + 1 + w[0]) if len(w) else None

def chars(j, t0):
    """characteristics known the day before t0."""
    t = t0 - 1
    a = ADJ[t, j]; a6 = ADJ[t - 126, j] if t - 126 >= 0 else np.nan
    return dict(s52=float(S52[t, j]) if np.isfinite(S52[t, j]) else np.nan,
                adv60=float(V60[t, j]) if np.isfinite(V60[t, j]) else np.nan,
                px=float(K.Cn[t, j]) if np.isfinite(K.Cn[t, j]) else np.nan,
                pre6m=float(a / a6 - 1) if np.isfinite(a) and np.isfinite(a6) and a6 > 0 else np.nan,
                in_pool=bool(INPOOL[t, j]))

# ---------------- main strategy with avoid list ----------------
_YL = _Y5 = None
def revenue():
    global _YL, _Y5
    if _YL is None:
        _YL, _Y5, _ = V.load_revenue()
    return _YL, _Y5

def strategy(exfin=True):
    YL, Y5 = revenue()
    return V.V2('gapfix', YL, Y5, exfin=exfin)

def plan_avoid(S, n, avoid_fn=None, sell_held=False):
    """V2.plan with an avoid list. avoid_fn(d) -> bool array (True = skip on signal day d).
    sell_held=False: an already-held stock that is avoided is kept if it is still in the top-n (no forced sale);
    sell_held=True : avoided names are sold even if held. Skipped names are replaced by the next-ranked name."""
    out = []; held = set(); nblk = 0; nav_ = 0; skipped = []
    for d in S.sig_days:
        cs = S.cross_section(d)
        el = cs['el']
        if el.sum() < 50: continue
        e = cs['e']; gu = K.gap_up_open(e)
        av = avoid_fn(d) if avoid_fn is not None else np.zeros(N, bool)
        cand = S.order(cs, 'official')[:int(el.sum())]
        pick = []
        # baseline top-n (for counting affected names)
        for j in cand:
            if av[j] and (sell_held or j not in held):
                skipped.append((d, j)); continue
            if gu[j] and j not in held: nblk += 1; continue
            pick.append(j)
            if len(pick) == n: break
        out.append((e, np.array(pick, int), np.where(el)[0], d))
        held = set(pick)
    S.nblk = nblk
    return out, skipped

def run_nav(S, plan, cost='normal', ew=None):
    t0, t1, sd = S.bounds()
    bb, ss = K.rates(0.001) if cost == 'normal' else (K.rates(0.001)[0] + 0.003, K.rates(0.001)[1] + 0.003)
    nav, lg = K.simulate([(p[0], p[1]) for p in plan], bb, ss, t0, t1)
    return nav, lg

def ew_nav(S, plan):
    t0, t1, sd = S.bounds()
    ew, _ = K.simulate([(p[0], p[2]) for p in plan], 0, 0, t0, t1)
    return ew

def monthly_rets(nav, sd):
    D = pd.DatetimeIndex([d for d in sd if d in nav.index])
    nd = nav.reindex(D)
    return (nd.shift(-1) / nd - 1).iloc[:-1]

def bench_monthly(sd):
    D = pd.DatetimeIndex(sd)
    b = K.E50.reindex(D)
    return (b.shift(-1) / b - 1).iloc[:-1]

def seg(nav, ew, a, b, sd, ad=None):
    return V.seg_stats(nav, ew, a, b, sd, ad=ad)

def dump(obj, name):
    def conv(o):
        if isinstance(o, (np.floating,)): return None if not np.isfinite(o) else round(float(o), 6)
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.bool_,)): return bool(o)
        if isinstance(o, (pd.Timestamp,)): return str(o.date())
        if isinstance(o, float): return None if not math.isfinite(o) else round(o, 6)
        return str(o)
    (HERE / name).write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=conv), encoding='utf-8')

def car_matched(j, t0, a, b, nq=5):
    """CAR vs same-pool peers in the same 52-week-high-ratio quintile (measured the day before t0).
    Removes the 52w-high / momentum premium that the main strategy itself harvests."""
    lo, hi = t0 + a, t0 + b
    t = t0 - 1
    if lo < 1 or hi >= len(IDX) or t < 1: return np.nan
    r = R[lo:hi + 1, j]
    if np.isfinite(r).mean() < 0.7: return np.nan
    pool = INPOOL[t].copy() if INPOOL[t].any() else np.isfinite(V60[t])
    s = S52[t]
    ok = pool & np.isfinite(s)
    if not np.isfinite(s[j]) or ok.sum() < 50: return np.nan
    qs = np.quantile(s[ok], np.linspace(0, 1, nq + 1))
    k = min(np.searchsorted(qs, s[j], side='right') - 1, nq - 1); k = max(k, 0)
    peer = ok & (s >= qs[k]) & (s <= qs[k + 1]); peer[j] = False
    pr = R[lo:hi + 1][:, peer]
    with np.errstate(all='ignore'):
        pm = np.nanmean(np.nansum(pr, axis=0)[np.isfinite(pr).mean(axis=0) >= 0.7])
    return float(np.nansum(r) - pm)
