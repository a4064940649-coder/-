"""strategy_ensemble 共用：策略淨值、投組層級組合模擬、統計、bootstrap 檢定。"""
import math, json, hashlib
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import minimize

H = Path(__file__).resolve().parent
RES = json.loads((H / 'results.json').read_text(encoding='utf-8'))
PR = RES['PREREG']
assert hashlib.sha256(json.dumps(PR, ensure_ascii=False, sort_keys=True).encode()).hexdigest() == RES['PREREG_sha256'], 'PREREG 被改過'
NAVDF = pd.read_parquet(H / 'strategy_nav.parquet')
ALL = PR['sets']['S_recent']
NAVS = NAVDF[ALL]
CAL = NAVS.index
P = NAVS.to_numpy()
RD = NAVS.pct_change().to_numpy()            # 日報酬（第一個有效日之前 NaN）
CI = {k: i for i, k in enumerate(ALL)}
FIRST = {k: int(np.argmax(np.isfinite(P[:, CI[k]]))) for k in ALL}
T = len(CAL)
_s = pd.Series(np.arange(T), index=CAL)
MONTH_END = set(_s.groupby([CAL.year, CAL.month]).last().to_numpy().tolist())
SIG_FLOOR = 0.05 / math.sqrt(252)
# 預先算 60 日波動（以 t−1 為止）：vol60[t] = std(RD[t-60..t-1])
_rd = pd.DataFrame(RD, index=CAL, columns=ALL)
VOL60 = _rd.rolling(60, min_periods=60).std().shift(1).to_numpy()
MOM126 = (NAVS / NAVS.shift(126)).shift(1).to_numpy() - 1

def pos(d): return int(CAL.searchsorted(pd.Timestamp(d)))

def reb_days(first_alloc):
    i0 = pos(first_alloc)
    return [i0] + sorted(i for i in MONTH_END if i > i0 and i < T - 1)

def has_hist(names, t, L):
    return all(FIRST[n] <= t - 1 - L for n in names)

def w_e1(names, t): return np.full(len(names), 1.0 / len(names))

def w_e2(names, t):
    if not has_hist(names, t, 60): return w_e1(names, t)
    s = np.maximum(VOL60[t, [CI[n] for n in names]], SIG_FLOOR)
    w = 1 / s; return w / w.sum()

def w_e3(names, t):
    n = len(names)
    if n == 2 or not has_hist(names, t, 252): return w_e1(names, t)
    X = RD[t - 252:t, [CI[k] for k in names]]
    S = np.cov(X, rowvar=False) + np.eye(n) * 1e-12
    cons = ({'type': 'eq', 'fun': lambda w: w.sum() - 1, 'jac': lambda w: np.ones(n)},)
    r = minimize(lambda w: w @ S @ w * 1e4, np.full(n, 1 / n), jac=lambda w: 2 * S @ w * 1e4, bounds=[(0, 0.5)] * n,
                 constraints=cons, method='SLSQP', options={'ftol': 1e-12, 'maxiter': 500})
    w = np.clip(r.x, 0, 0.5); return w / w.sum()

def w_e4(names, t):
    if not has_hist(names, t, 126): return w_e1(names, t)
    m = MOM126[t, [CI[k] for k in names]]
    top = np.argsort(-m)[:2]; w = np.zeros(len(names)); w[top] = 0.5; return w

def w_fixed(spec):
    def f(names, t): return np.array([spec[n] for n in names], float)
    return f

def w_e5(names, t):
    w = np.array([0.6 if n == 'N10' else 0.4 / (len(names) - 1) for n in names]); return w

METHODS = {'E1': w_e1, 'E2': w_e2, 'E3': w_e3, 'E4': w_e4, 'E5': w_e5}

def simulate(names, wfun, first_alloc, cost=0.005, return_w=False):
    """月底再平衡、月內漂移；回傳 [first_alloc, 最後日] 的日淨值（首日＝扣初始成本後）。"""
    names = list(names); J = [CI[k] for k in names]
    Pm = P[:, J]
    rb = reb_days(first_alloc); i0 = rb[0]
    nav = np.full(T, np.nan); A = None; wlog = []
    for s, t in enumerate(rb):
        w = wfun(names, t)
        if A is None:
            V = 1.0; h = np.zeros(len(names))
        else:
            h = A * Pm[t]; V = h.sum()
        turn = np.abs(w * V - h).sum() / V
        V2 = V * (1 - cost * turn)
        A = w * V2 / Pm[t]
        e = rb[s + 1] if s + 1 < len(rb) else T - 1
        seg = slice(t + 1, e + 1)
        nav[seg] = Pm[seg] @ A
        nav[t] = V2                                 # 成交日收盤淨值＝扣搬錢成本後（成本記在成交日）
        if return_w: wlog.append((CAL[t], dict(zip(names, np.round(w, 4))), round(turn, 4)))
    out = pd.Series(nav[i0:], index=CAL[i0:])
    return (out, wlog) if return_w else out

# ---------------- 統計 ----------------
E50 = NAVS['E0050']
def tstat(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 3 or x.std(ddof=1) == 0: return None
    return round(float(x.mean() / x.std(ddof=1) * math.sqrt(len(x))), 2)

def window(nav, a, b):
    a, b = pd.Timestamp(a), pd.Timestamp(b)
    pre = nav[nav.index < a]
    seg = nav[(nav.index >= a) & (nav.index <= b)]
    if len(pre): seg = pd.concat([pre.iloc[-1:], seg])
    return seg.dropna()

def monthly(nav, a, b):
    seg = window(nav, a, b); r = seg.pct_change().iloc[1:]
    return (1 + r).groupby([r.index.year, r.index.month]).prod() - 1

def stats(nav, a, b, refs=None):
    seg = window(nav, a, b)
    if len(seg) < 20: return None
    yrs = (seg.index[-1] - seg.index[0]).days / 365.25
    r = seg.pct_change().iloc[1:]
    cagr = float((seg.iloc[-1] / seg.iloc[0]) ** (1 / yrs) - 1)
    dd = float((seg / seg.cummax() - 1).min())
    o = {'CAGR': round(cagr, 4), 'vol': round(float(r.std() * math.sqrt(252)), 4), 'maxDD': round(dd, 4),
         'Calmar': round(cagr / abs(dd), 3) if dd < 0 else None}
    m = monthly(nav, a, b); o['n_months'] = int(len(m))
    for k, ref in (refs or {}).items():
        mr = monthly(ref, a, b).reindex(m.index)
        d = (m - mr).to_numpy(); o[f't_vs_{k}'] = tstat(d); o[f'mdiff_vs_{k}_pct'] = round(float(np.nanmean(d) * 100), 2)
    return o

def yearly(nav, a, b):
    seg = window(nav, a, b); r = seg.pct_change().iloc[1:]
    return {int(y): round(float((1 + g).prod() - 1) * 100, 1) for y, g in r.groupby(r.index.year)}

def bear(nav, wins):
    o = {}
    for k, (a, b) in wins.items():
        seg = window(nav, a, b)
        if len(seg) < 5: o[k] = None; continue
        o[k] = {'ret_pct': round(float(seg.iloc[-1] / seg.iloc[0] - 1) * 100, 1), 'maxDD_pct': round(float((seg / seg.cummax() - 1).min()) * 100, 1)}
    return o

# ---------------- bootstrap ----------------
def stationary_counts(n, B, blk, rng):
    """回傳 B×n 的抽樣次數矩陣（stationary bootstrap，平均區塊長 blk）。"""
    p = 1.0 / blk; C = np.zeros((B, n))
    for b in range(B):
        i = rng.integers(n); out = np.empty(n, int)
        for k in range(n):
            if k > 0:
                i = rng.integers(n) if rng.random() < p else (i + 1) % n
            out[k] = i
        C[b] = np.bincount(out, minlength=n)
    return C

def spa_maxt(D, B=2000, blk=6, seed=20261027):
    """D：n×K（正值＝組合較好）。回傳 SPA_c、SPA_l、RC p 值、max-t 族系 p 值、以及每個模型的單步 max-t 調整 p。"""
    D = np.asarray(D, float); n, K = D.shape
    rng = np.random.default_rng(seed)
    Cn = stationary_counts(n, B, blk, rng)
    dbar = D.mean(0); bm = Cn @ D / n                       # B×K
    om = np.sqrt(n * ((bm - dbar) ** 2).mean(0)); om[om <= 0] = np.inf
    tst = math.sqrt(n) * dbar / om
    Tspa = max(tst.max(), 0.0)
    thr = -math.sqrt(2 * math.log(math.log(n)))
    mu_c = np.where(tst <= thr, dbar, 0.0); mu_l = np.minimum(dbar, 0.0)
    Z = bm - dbar
    def p_of(mu):
        Tb = np.maximum((math.sqrt(n) * (Z + mu) / om).max(1), 0.0); return float((Tb >= Tspa).mean())
    maxb = (math.sqrt(n) * Z / om).max(1)
    adj = np.array([(maxb >= t).mean() for t in tst])
    return {'n_months': n, 'K': K, 'best_t': round(float(tst.max()), 2), 'best_model_index': int(tst.argmax()),
            'p_SPA_c': round(p_of(mu_c), 4), 'p_SPA_l': round(p_of(mu_l), 4), 'p_RC': round(p_of(np.zeros(K)), 4),
            'p_maxt_family': round(float((maxb >= tst.max()).mean()), 4), '_adj_p': adj, '_t': tst}
