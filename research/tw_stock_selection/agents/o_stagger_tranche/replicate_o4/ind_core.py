#!/usr/bin/env python3
"""o_stagger_tranche 獨立複製（第二次、從頭重寫；不讀研究員程式，也不使用本資料夾舊的 rcore.py）。
只讀 import 的資料層：o3_core（apply_patch／load_events／exit_rules／load_rdel；它再帶 v2_core 的 gapfix 滾動訊號、
FINMASK、PCOL、IS15 與 ia_core 的面板陣列／0050）、drj_core.load_list（官方處置名單）。
錨定日曆、AB、選股排序、日頻引擎（含分批與 c1）、統計全部在這裡自己寫。"""
import sys, math
from pathlib import Path
import numpy as np, pandas as pd

AG = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AG / 'o_infra_v3_rebuild')); import o3_core as O3; sys.path.pop(0)
sys.path.insert(0, str(AG / 'o_disposal_rejudge')); import drj_core as DRJ; sys.path.pop(0)
V = O3.V; K = O3.K

NPATCH = O3.apply_patch(require_freeze=True)
XR = O3.exit_rules(O3.load_events(require_freeze=True), 'main')
IDX, COLS, N = K.IDX, K.COLS, K.N
Cn, On, Rn, Fn = K.Cn, K.On, K.Rn, K.Fn
TR = K.TRADED; GR = K.GROW; PREVC = K.PREVC
GOODD = K.TRADED.sum(axis=1) >= 300
S52, V60 = V.rolling_signals('gapfix')
YL = O3.load_rdel()
Y5 = V.load_revenue()[1]
FIN, PCOL, IS15 = V.FINMASK, V.PCOL, V.IS15
E50 = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/etf_0050.csv', parse_dates=['date']).set_index('date')['adj'].sort_index()
E50 = E50.reindex(IDX.union(E50.index)).ffill().reindex(IDX)
END = pd.Timestamp('2026-10-12'); TEND = IDX.get_loc(END)
SW = pd.Timestamp('2021-03-01'); PSW = pd.Period('2021-03', 'M')
POS = {d: i for i, d in enumerate(IDX)}
FEE = 0.001425 * 0.6
NORMAL = (FEE + 0.001, FEE + 0.003 + 0.001)
FLAT = (0.005, 0.005)
P03 = (NORMAL[0] + 0.003, NORMAL[1] + 0.003)
GROSS = (0.0, 0.0)
def slip(s): return (FEE + s, FEE + 0.003 + s)

# ---------- 處置 E1 ----------
_D = DRJ.load_list()
_da, _ds, _de = _D.announce.values, _D.start.values, _D.end.values
_dj = np.array([K.CI[c] for c in _D.code])
_DMC = {}
def dmask(d):
    if d not in _DMC:
        x = np.datetime64(d)
        sel = (_da <= x) & (_ds <= x) & (_de >= x)
        m = np.zeros(N, bool); m[_dj[sel]] = True; _DMC[d] = m
    return _DMC[d]

# ---------- 日曆 ----------
TX = pd.DatetimeIndex(sorted(pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/taiex_twii.csv', parse_dates=['date'])['date'].unique()))
def offday(p):
    x = IDX[(IDX.year == p.year) & (IDX.month == p.month) & (IDX.day >= 11)]
    return x[0] if len(x) else None
def Dm(p):
    r = str(p - 1)
    if r == '2019-01': return pd.Timestamp('2019-02-15')
    if r == '2023-03': return pd.Timestamp('2023-04-12')
    return TX[TX >= pd.Timestamp(p.year, p.month, 10)][0]
def J(p, j):
    i = TX.get_loc(Dm(p)) + j
    return TX[i] if 0 <= i < len(TX) else None
def nxt_good(t):
    t += 1
    while not GOODD[t]: t += 1
    return t

def y3(Y, p):
    ms = [p - 1, p - 2, p - 3]
    if any(x not in Y.index for x in ms): return np.full(N, np.nan)
    Z = Y.reindex(index=ms, columns=COLS).to_numpy()
    o = Z.mean(axis=0); o[np.isnan(Z).any(axis=0)] = np.nan
    return o
def lastfull(Y):
    c = Y.notna().sum(axis=1); return c[c >= 0.8 * c.median()].index.max()
LFL, LF5 = lastfull(YL), lastfull(Y5)

ANCH = []
for p in pd.period_range('2005-01', '2026-10', freq='M'):
    d = offday(p)
    if d is None or d >= END: continue
    Y, lf = (Y5, LF5) if d >= SW else (YL, LFL)
    if p - 1 > lf: continue
    if np.isfinite(y3(Y, p)).sum() > 100: ANCH.append(p)
OFF = {p: offday(p) for p in ANCH}
OFFSD = [OFF[p] for p in ANCH] + [END]
OFFEXEC = {nxt_good(POS[OFF[p]]) for p in ANCH}

# ---------- AB ----------
D2020 = pd.Timestamp('2020-02-14')
def ab_kind(p, s):
    D = Dm(p)
    if str(p - 1) == '2020-01' and D <= s <= D2020: k = 'mc'
    elif s < D: k = 'all'
    elif s == D: k = 'mc'
    else: k = 'none'
    ins = False
    if '2026-01' <= str(p - 1) <= '2026-08':
        if s < TX[TX >= pd.Timestamp(p.year, p.month, 15)][0]: ins = True
    return k, ins

class AB:
    def __init__(self, seed=0, kindf=ab_kind):
        self.rng = np.random.default_rng(seed); self.c = {}; self.kindf = kindf; self.tag = ('AB', seed, id(self))
    def __call__(self, p, s):
        if (p, s) not in self.c:
            k, ins = self.kindf(p, s)
            u = np.zeros(N, bool)
            if k == 'all': u[:] = True
            elif k == 'mc': u = self.rng.random(N) < PCOL
            if ins: u = u | IS15
            self.c[(p, s)] = u
        return self.c[(p, s)]

# ---------- 選股 ----------
OPT = {'rev': 'anchor'}          # 'anchor'：有效營收月份 ≥ 2021-03 用 rev_5y；'signal'：訊號日 ≥ 2021-03-01 用 rev_5y
def yv(p, s, back):
    if OPT['rev'] == 'signal': Y = Y5 if s >= SW else YL
    else: Y = Y5 if p >= PSW else YL
    v = y3(Y, p - back)
    return np.where(FIN, np.nan, v)

def tick(x): return np.select([x < 10, x < 50, x < 100, x < 500, x < 1000], [0.01, 0.05, 0.1, 0.5, 1.0], 5.0)
def gapup(e):
    lim = 0.07 if IDX[e] < pd.Timestamp('2015-06-01') else 0.10
    ref = np.where(Fn[e] == 1, PREVC[e], Cn[e] / (1 + Rn[e]))
    raw = ref * (1 + lim); tk = tick(raw); lu = np.floor(raw / tk + 1e-9) * tk
    return np.isfinite(On[e]) & np.isfinite(ref) & (On[e] >= lu - 1e-6)

_RC = {}
def rank(p, s, U=None, utag=None, w=0.5, revfix=None):
    key = (p, s, utag, w, None if revfix is None else id(revfix))
    if key in _RC: return _RC[key]
    t = POS[s]; e = nxt_good(t)
    v = V60[t]; px = Cn[t]
    ok = np.isfinite(v) & (v >= 1e7) & np.isfinite(px) & (px >= 10)
    o = np.argsort(-np.where(ok, v, -np.inf), kind='stable')
    top = np.zeros(N, bool); top[o[:min(300, int(ok.sum()))]] = True; top &= ok
    yy = yv(p, s, 0)
    if U is not None and U.any(): yy = np.where(U, yv(p, s, 1), yy)
    s5 = S52[t]
    el = top & np.isfinite(yy) & np.isfinite(s5) & ~dmask(s) & np.isfinite(On[e]) & TR[e]
    if revfix is not None:                      # b1：營收腿百分位固定為 j=+1 批的橫斷面
        el &= np.isfinite(revfix)
    pr52 = pd.Series(np.where(el, s5, np.nan)).rank(pct=True).to_numpy()
    prr = pd.Series(np.where(el, yy, np.nan)).rank(pct=True).to_numpy() if revfix is None else revfix
    sc = w * pr52 + (1 - w) * prr; sc[~el] = np.nan
    order = np.lexsort((np.arange(N), -np.nan_to_num(yy, nan=-1e18), -np.round(np.nan_to_num(sc, nan=-9), 10)))
    r = dict(t=t, e=e, el=el, nel=int(el.sum()), order=order[:int(el.sum())], prr=prr, yy=yy)
    _RC[key] = r
    return r

def plan(days, ab=None, n=10, w=0.5, skip_fn=None, revfix_fn=None):
    """days：[(p, s)]。skip_fn(p, s)→set：a1 要跳過的欄位。回傳 [dict(e, pk, p, s)]。"""
    out = []; held = set()
    for p, s in days:
        if s is None or s not in POS: continue
        U = ab(p, s) if ab is not None else None
        rf = revfix_fn(p) if revfix_fn is not None else None
        r = rank(p, s, U, None if ab is None else ab.tag, w, rf)
        if r['nel'] < 50: continue
        gu = gapup(r['e']); pk = []
        for j in r['order']:
            if gu[j] and j not in held: continue
            pk.append(int(j))
            if len(pk) == n: break
        out.append(dict(e=r['e'], pk=np.array(pk, int), p=p, s=s, t=r['t'], order=r['order']))
        held = set(pk)
    return out

# ---------- 引擎 ----------
REL = {j: (v[0], int(np.searchsorted(IDX, v[1])), int(np.searchsorted(IDX, v[3])) if len(v) > 3 else 10 ** 9) for j, v in XR.items()}

def tsim(plans, buy, sell, t0, t1, c1=False, a1=False, cap=None):
    """plans：每批一個 plan（list of dict），批 i 起始資金 1/k。c1：官方成交日開盤先調回等額。
    a1：批 i 換股時跳過其他批「當下」持有的股票（plan 須以 order 形式提供，見 plan_a1）。
    回傳 nav, log。"""
    k = len(plans)
    reb = [{q['e']: q for q in pl} for pl in plans]
    first = [min(r) for r in reb]; allstart = max(first)
    val = [np.zeros(N) for _ in range(k)]; cash = [1.0 / k] * k
    nav = np.empty(t1 - t0 + 1); log = dict(tov=[], buys=[], c1cost=0.0, nc1=0); lastpk = [set() for _ in range(k)]
    for t in range(t0, t1 + 1):
        if t == t0:
            nav[0] = sum(cash); continue
        act = [i for i in range(k) if t in reb[i]]
        doc1 = c1 and k > 1 and (t in OFFEXEC) and t > allstart
        if not act and not doc1:
            for i in range(k): val[i] = val[i] * GR[t]
            nav[t - t0] = sum(v.sum() for v in val) + sum(cash); continue
        tr = TR[t] & np.isfinite(On[t])
        gap = np.where(tr, GR[t] * On[t] / np.where(tr, Cn[t], 1.0), 1.0)
        oc = np.where(tr, Cn[t] / np.where(tr, On[t], 1.0), 1.0)
        touched = set(act) | (set(range(k)) if doc1 else set())
        h = {i: val[i] * gap for i in touched}
        if doc1:
            frz = {i: (h[i] > 0) & ~tr for i in range(k)}
            F = np.array([h[i][frz[i]].sum() for i in range(k)])
            Ht = {i: np.where(frz[i], 0.0, h[i]) for i in range(k)}
            A = np.array([Ht[i].sum() + cash[i] for i in range(k)])
            W = F.sum() + A.sum()
            def ccost(Q):
                f = (Q - F) / A
                net = np.zeros(N)
                for i in range(k): net += (f[i] - 1.0) * Ht[i]
                return buy * net[net > 0].sum() - sell * net[net < 0].sum()
            lo, hi = 0.0, W / k
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                if mid * k + ccost(mid) > W: hi = mid
                else: lo = mid
            Q = lo; f = (Q - F) / A
            log['c1cost'] += W - k * Q; log['nc1'] += 1
            for i in range(k):
                h[i] = np.where(frz[i], h[i], Ht[i] * f[i]); cash[i] = cash[i] * f[i]
        tot_open = sum((h[i] if i in h else val[i] * gap).sum() + cash[i] for i in range(k))
        done = {}
        for i in act:
            q = reb[i][t]; hi_ = h[i]
            frz = (hi_ > 0) & ~tr
            for j in np.where(frz)[0]:
                if j in REL and REL[j][1] <= t < REL[j][2]:
                    cash[i] += hi_[j] * REL[j][0]; hi_[j] = 0.0; frz[j] = False
            Vfz = hi_[frz].sum()
            htr = np.where(frz, 0.0, hi_)
            Vt = htr.sum() + cash[i]
            pk = q['pk']
            if a1:
                others = set()
                for i2 in range(k):
                    if i2 == i: continue
                    hv = done[i2] if i2 in done else (h[i2] if i2 in h else val[i2])
                    others |= set(np.where(hv > 0)[0].tolist())
                pk = pick_a1(q, others, lastpk[i])
            lastpk[i] = set(int(j) for j in pk)
            kk = len(pk)
            def cost_of(T):
                tg = np.zeros(N); tg[pk] = T
                return buy * np.clip(tg - htr, 0, None).sum() + sell * np.clip(htr - tg, 0, None).sum()
            lo, hi = 0.0, Vt / kk
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if mid * kk + cost_of(mid) > Vt: hi = mid
                else: lo = mid
            T = lo
            tg = np.zeros(N); tg[pk] = T
            log['tov'].append((i, t, np.clip(htr - tg, 0, None).sum() / (Vt + Vfz)))
            log['buys'].append((i, t, q['t'], [int(j) for j in pk if htr[j] <= 0], T / tot_open))
            nv = np.where(frz, hi_, 0.0); nv[pk] += T
            done[i] = np.where(frz, nv, nv * oc); cash[i] = 0.0
        for i in range(k):
            if i in done: val[i] = done[i]
            elif i in h: val[i] = h[i] * oc          # 只做了 c1 調整、今天沒換股的批：開盤→收盤
            else: val[i] = val[i] * GR[t]
        nav[t - t0] = sum(v.sum() for v in val) + sum(cash)
    tv = [x[2] for x in log['tov']]
    log['turnover'] = float(np.mean(tv[k:])) if len(tv) > k else float('nan')
    return pd.Series(nav, index=IDX[t0:t1 + 1]), log

def pick_a1(q, others, held):
    gu = gapup(q['e']); pk = []
    for j in q['order']:
        if j in others: continue
        if gu[j] and j not in held: continue
        pk.append(int(j))
        if len(pk) == 10: break
    return np.array(pk, int)

# ---------- 統計 ----------
def tst(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 3: return float('nan')
    sd = x.std(ddof=1); return float(x.mean() / (sd / math.sqrt(len(x)))) if sd > 0 else float('nan')

def wret(nav, sd):
    D = pd.DatetimeIndex([d for d in sd if d in nav.index])
    v = nav.reindex(D); return (v.shift(-1) / v - 1).iloc[:-1]

def exw(nav, sd, bench=None):
    b = E50 if bench is None else bench
    return wret(nav, sd) - wret(b.reindex(nav.index), sd)

def cagr(nav, a=None, b=None):
    s = nav
    if a is not None: s = s[s.index >= pd.Timestamp(a)]
    if b is not None: s = s[s.index <= pd.Timestamp(b)]
    i0 = nav.index.get_loc(s.index[0])
    base_i = i0 - 1 if i0 > 0 else i0
    eq = nav.iloc[base_i:nav.index.get_loc(s.index[-1]) + 1]
    eq = eq / eq.iloc[0]
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    return float(eq.iloc[-1] ** (1 / yrs) - 1) * 100, float((eq / eq.cummax() - 1).min()) * 100

def calm(nav, bench=None):
    b = (E50 if bench is None else bench).reindex(nav.index)
    me = nav.groupby(nav.index.to_period('M')).last(); be = b.groupby(nav.index.to_period('M')).last()
    return (me.pct_change() - be.pct_change()).dropna()

def per(p):
    if p <= pd.Period('2018-12', 'M'): return 'SEL'
    if p <= pd.Period('2023-12', 'M'): return 'VAL'
    return 'HOLD'
