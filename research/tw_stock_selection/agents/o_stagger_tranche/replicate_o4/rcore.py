#!/usr/bin/env python3
"""o_stagger_tranche 獨立複製（replicate_o4）核心。
只讀 import 資料層：o3_core（apply_patch／exit_rules／load_rdel）、drj_core（官方處置名單 E1）、v2_core（gapfix 滾動訊號、FINMASK、PCOL、IS15）。
選股、AB、錨定日曆、日頻引擎（含分批與 c1 再平衡）、統計全部自己寫；不讀 o_stagger_tranche 研究員程式。"""
import sys, math
from pathlib import Path
import numpy as np, pandas as pd

AG = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AG / 'o_infra_v3_rebuild')); import o3_core as O3; sys.path.pop(0)
sys.path.insert(0, str(AG / 'o_disposal_rejudge')); import drj_core as R; sys.path.pop(0)
V = O3.V; K = O3.K
assert R.K is K

NPATCH = O3.apply_patch(require_freeze=True)
XR = O3.exit_rules(O3.load_events(require_freeze=True), 'main')
IDX, COLS, N = K.IDX, K.COLS, K.N
Cn, On, TRADED, GROW = K.Cn, K.On, K.TRADED, K.GROW
GOOD = K.GOOD
S52, V60 = V.rolling_signals('gapfix')
YL = O3.load_rdel()
Y5 = V.load_revenue()[1]
FINMASK, PCOL, IS15 = V.FINMASK, V.PCOL, V.IS15
E50 = K.E50
SWITCH = pd.Timestamp('2021-03-01')
END = pd.Timestamp('2026-10-12')
TEND = IDX.get_loc(END)
POS = {d: i for i, d in enumerate(IDX)}

# ---------------- 處置 E1（官方名單，公告日 ≤ d 且 start ≤ d ≤ end） ----------------
DL = R.load_list()
_dl_s, _dl_e, _dl_a = DL.start.values, DL.end.values, DL.announce.values
_dl_j = np.array([K.CI[c] for c in DL.code])
_DM = {}
def disp_mask(d):
    if d in _DM: return _DM[d]
    dv = np.datetime64(d)
    x = (_dl_s <= dv) & (_dl_e >= dv) & (_dl_a <= dv)
    m = np.zeros(N, bool); m[_dl_j[x]] = True
    _DM[d] = m
    return m

# ---------------- 日曆 ----------------
TX = pd.DatetimeIndex(sorted(pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/taiex_twii.csv', parse_dates=['date'])['date']))
def official_day(y, m):
    ii = IDX[(IDX.year == y) & (IDX.month == m) & (IDX.day >= 11)]
    return ii[0] if len(ii) else None

def deadline(m):
    """錨定月 m（Period）的營收期限 D_m（m−1 月營收）。"""
    s = str(m - 1)
    if s == '2019-01': return pd.Timestamp('2019-02-15')
    if s == '2023-03': return pd.Timestamp('2023-04-12')
    return TX[TX >= pd.Timestamp(m.year, m.month, 10)][0]

def jday(m, j):
    D = deadline(m); i = TX.get_loc(D)
    return TX[i + j] if 0 <= i + j < len(TX) else None

def next_good(t):
    e = t + 1
    while not GOOD[e]: e += 1
    return e

def yoy3(Y, m):
    ms = [m - 1, m - 2, m - 3]
    if any(x not in Y.index for x in ms): return np.full(N, np.nan)
    Z = Y.reindex(index=ms, columns=COLS).to_numpy()
    o = Z.mean(axis=0); o[np.isnan(Z).any(axis=0)] = np.nan
    return o

def last_full(Y):
    c = Y.notna().sum(axis=1); return c[c >= 0.8 * c.median()].index.max()
LF5, LFL = last_full(Y5), last_full(YL)

# 錨定月集合＝V3 官方訊號月
ANCHORS = []
for y in range(2006, 2027):
    for mo in range(1, 13):
        m = pd.Period(f'{y}-{mo:02d}', 'M')
        if m < pd.Period('2006-04', 'M'): continue
        d = official_day(y, mo)
        if d is None or d >= END: continue
        Y, lf = (Y5, LF5) if d >= SWITCH else (YL, LFL)
        if m - 1 > lf: continue
        if np.isfinite(yoy3(Y, m)).sum() > 100: ANCHORS.append(m)
OFFICIAL = {m: official_day(m.year, m.month) for m in ANCHORS}

# ---------------- AB ----------------
D20 = pd.Timestamp('2020-02-14')
def unavail_kind(m, s):
    """回傳 'none'（全已公告）、'all'（全未公告）、'mc'（抽樣），外加 INS15 是否未公告。"""
    D = deadline(m)
    if str(m - 1) == '2020-01' and s <= D20 and s >= D:
        k = 'mc'
    elif s < D: k = 'all'
    elif s == D: k = 'mc'
    else: k = 'none'
    ins = False
    if '2026-01' <= str(m - 1) <= '2026-08':
        D15 = TX[TX >= pd.Timestamp(m.year, m.month, 15)][0]
        if s < D15: ins = True
    return k, ins

_ABN = [0]
class AB:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed); self.cache = {}
        _ABN[0] += 1; self.uid = _ABN[0]
    def __call__(self, m, s):
        key = (m, s)
        if key in self.cache: return self.cache[key]
        k, ins = unavail_kind(m, s)
        u = np.zeros(N, bool)
        if k == 'all': u[:] = True
        elif k == 'mc': u = self.rng.random(N) < PCOL
        if ins: u |= IS15
        self.cache[key] = u
        return u

# ---------------- 選股 ----------------
REVRULE = {'mode': 'month'}   # 'month'：有效營收錨定月 (m−back) ≥ 2021-03 用 rev_5y；'signal'：訊號日 ≥ 2021-03-01
PSW = pd.Period('2021-03', 'M')
def y3_for(m, s, back=0):
    if REVRULE['mode'] == 'signal': Y = Y5 if s >= SWITCH else YL
    elif REVRULE['mode'] == 'anchor': Y = Y5 if m >= PSW else YL
    else: Y = Y5 if (m - back) >= PSW else YL
    y = yoy3(Y, m - back)
    return np.where(FINMASK, np.nan, y)

def tick(p):
    return np.select([p < 10, p < 50, p < 100, p < 500, p < 1000], [0.01, 0.05, 0.1, 0.5, 1.0], 5.0)
def gap_up(e):
    lim = 0.07 if IDX[e] < pd.Timestamp('2015-06-01') else 0.10
    c, r = Cn[e], K.Rn[e]
    ref = np.where(K.Fn[e] == 1, K.PREVC[e], c / (1 + r))
    raw = ref * (1 + lim); tk = tick(raw)
    lu = np.floor(raw / tk + 1e-9) * tk
    return np.isfinite(On[e]) & np.isfinite(ref) & (On[e] >= lu - 1e-6)

_CS = {}
W52 = {'w': 0.5}
def ranking(m, s, U=None, uid=None):
    """訊號日 s（面板日）收盤、錨定月 m 的排序（候選序列）、成交日 e、合格池。U：未公告遮罩。"""
    key = (m, s, uid) if W52['w'] == 0.5 else (m, s, uid, W52['w'])
    if key in _CS: return _CS[key]
    t = POS[s]; e = next_good(t)
    v60 = V60[t]; px = Cn[t]
    base = np.isfinite(v60) & (v60 >= 1e7) & np.isfinite(px) & (px >= 10)
    order = np.argsort(-np.where(base, v60, -np.inf), kind='stable')
    top = np.zeros(N, bool); top[order[:min(300, int(base.sum()))]] = True; top &= base
    y3 = y3_for(m, s)
    if U is not None and U.any(): y3 = np.where(U, y3_for(m, s, 1), y3)
    s52 = S52[t]
    el = top & np.isfinite(y3) & np.isfinite(s52) & ~disp_mask(s) & np.isfinite(On[e]) & TRADED[e]
    pr = lambda x: pd.Series(np.where(el, x, np.nan)).rank(pct=True).to_numpy()
    sc = W52['w'] * pr(s52) + (1 - W52['w']) * pr(y3); sc[~el] = np.nan
    scr = np.round(np.nan_to_num(sc, nan=-9), 10)
    cand = np.lexsort((np.arange(N), -np.nan_to_num(y3, nan=-1e18), -scr))[:int(el.sum())]
    out = dict(t=t, e=e, el=el, cand=cand, nel=int(el.sum()))
    _CS[key] = out
    return out

def make_plan(days, n=10, ab=None):
    """days：[(m, s)]（s＝訊號日 Timestamp）。回傳 [(e, picks, m, s)]。"""
    out = []; held = set()
    for m, s in days:
        if s not in POS: continue
        U = ab(m, s) if ab is not None else None
        uid = None if U is None else ab.uid
        r = ranking(m, s, U, uid)
        if r['nel'] < 50: continue
        gu = gap_up(r['e']); pick = []
        for j in r['cand']:
            if gu[j] and j not in held: continue
            pick.append(int(j))
            if len(pick) == n: break
        out.append((r['e'], np.array(pick, int), m, s))
        held = set(pick)
    return out

# ---------------- 引擎 ----------------
REL = {j: (v[0], int(np.searchsorted(IDX, v[1])), int(np.searchsorted(IDX, v[3])) if len(v) > 3 else 10 ** 9) for j, v in XR.items()}

def _solve(Vt, k, cost_of):
    lo, hi = 0.0, Vt / k
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if mid * k + cost_of(mid) > Vt: hi = mid
        else: lo = mid
    return lo

def tsim(plans, buy, sell, t0, t1, c1_days=None, G=None):
    """plans：每批一個 plan [(e, picks, ...)]；各批起始資金 1/k；c1_days：c1 再平衡的開盤日 index 集合。
    回傳 nav、log（turnover、每批 nav、c1 成本、新買名額）。"""
    G = GROW if G is None else G
    k = len(plans)
    reb = [{p[0]: p[1] for p in pl} for pl in plans]
    first = [min(r) if r else 10 ** 9 for r in reb]
    started_all = max(first)
    val = [np.zeros(N) for _ in range(k)]; cash = [1.0 / k for _ in range(k)]
    nav = np.empty(t1 - t0 + 1); tov = []; c1cost = 0.0; buys = []
    c1_days = set(c1_days or ())
    for t in range(t0, t1 + 1):
        act = [i for i in range(k) if t in reb[i]]
        doc1 = (t in c1_days) and t > started_all and k > 1
        if t > t0 and (act or doc1):
            tr = TRADED[t] & np.isfinite(On[t])
            gap = np.where(tr, G[t] * On[t] / np.where(tr, Cn[t], 1.0), 1.0)
            oc = np.where(tr, Cn[t] / np.where(tr, On[t], 1.0), 1.0)
            part = set(range(k)) if doc1 else set(act)
            tot_open = sum((val[i] * gap).sum() + cash[i] for i in range(k))
            h = {i: (val[i] * gap if i in part else None) for i in range(k)}
            if doc1:
                frz = [(h[i] > 0) & ~tr for i in range(k)]
                F = np.array([h[i][frz[i]].sum() for i in range(k)])
                htr = [np.where(frz[i], 0.0, h[i]) for i in range(k)]
                A = np.array([htr[i].sum() + cash[i] for i in range(k)])
                W = A.sum() + F.sum()
                def net_cost(Q):
                    f = (Q - F) / np.where(A > 0, A, 1.0)
                    net = sum((f[i] - 1) * htr[i] for i in range(k))
                    return buy * np.clip(net, 0, None).sum() + sell * np.clip(-net, 0, None).sum()
                Q = W / k
                for _ in range(50):
                    Qn = (W - net_cost(Q)) / k
                    if abs(Qn - Q) < 1e-15: Q = Qn; break
                    Q = Qn
                f = (Q - F) / np.where(A > 0, A, 1.0)
                c1cost += W - k * Q
                for i in range(k):
                    h[i] = np.where(frz[i], h[i], htr[i] * f[i]); cash[i] = cash[i] * f[i]
            for i in act:
                pk = reb[i][t]; hi_ = h[i]
                frz = (hi_ > 0) & ~tr
                for j in np.where(frz)[0]:
                    if j in REL and REL[j][1] <= t < REL[j][2]:
                        cash[i] += hi_[j] * REL[j][0]; hi_[j] = 0.0; frz[j] = False
                Vfz = hi_[frz].sum()
                htr = np.where(frz, 0.0, hi_)
                Vt = htr.sum() + cash[i]
                kk = len(pk)
                def cost_of(T, _htr=htr, _pk=pk):
                    tgt = np.zeros(N); tgt[_pk] = T
                    return buy * np.clip(tgt - _htr, 0, None).sum() + sell * np.clip(_htr - tgt, 0, None).sum()
                T = _solve(Vt, kk, cost_of)
                tgt = np.zeros(N); tgt[pk] = T
                tov.append((i, t, np.clip(htr - tgt, 0, None).sum() / (Vt + Vfz)))
                buys.append((i, t, [int(j) for j in pk if htr[j] <= 0], T, tot_open))
                nv = np.where(frz, hi_, 0.0); nv[pk] += T
                h[i] = nv; cash[i] = 0.0
            for i in range(k):
                val[i] = h[i] * oc if h[i] is not None else val[i] * G[t]
        elif t > t0:
            for i in range(k): val[i] = val[i] * G[t]
        nav[t - t0] = sum(v.sum() for v in val) + sum(cash)
    tv = [x[2] for x in tov]
    return pd.Series(nav, index=IDX[t0:t1 + 1]), dict(tov=tov, turnover=float(np.mean(tv[k:])) if len(tv) > k else np.nan, c1cost=c1cost, buys=buys)

NORMAL = (K.FEE + 0.001, K.FEE + 0.003 + 0.001)
FLAT = (0.005, 0.005)
P03 = (NORMAL[0] + 0.003, NORMAL[1] + 0.003)
ZERO = (0.0, 0.0)

# ---------------- 統計 ----------------
def tstat(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 3: return float('nan')
    s = x.std(ddof=1); return float(x.mean() / (s / math.sqrt(len(x))))

def winret(nav, sd):
    D = pd.DatetimeIndex([d for d in sd if d in nav.index])
    nd = nav.reindex(D); return (nd.shift(-1) / nd - 1).iloc[:-1]

def excess(nav, sd, bench=None):
    bench = E50 if bench is None else bench
    return winret(nav, sd) - winret(bench.reindex(nav.index), sd)

def cagr_mdd(nav, a, b):
    s = nav[(nav.index >= pd.Timestamp(a)) & (nav.index <= pd.Timestamp(b))]
    i0 = nav.index.get_loc(s.index[0])
    base = nav.iloc[i0 - 1] if i0 > 0 else s.iloc[0]
    start = nav.index[i0 - 1] if i0 > 0 else s.index[0]
    yrs = (s.index[-1] - start).days / 365.25
    eq = s / base
    return float(eq.iloc[-1] ** (1 / yrs) - 1) * 100, float((eq / eq.cummax() - 1).min()) * 100

def cal_t(nav):
    me = nav.groupby(nav.index.to_period('M')).last(); be = E50.reindex(nav.index).groupby(nav.index.to_period('M')).last()
    x = (me.pct_change() - be.pct_change()).dropna()
    return x
