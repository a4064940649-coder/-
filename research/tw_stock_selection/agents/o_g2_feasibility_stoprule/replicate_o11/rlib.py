"""Independent replication helpers (o11 G2 feasibility). Written from the frozen spec only."""
import math, re
import numpy as np, pandas as pd
from scipy import stats

AG = '/home/user/-/research/tw_stock_selection/agents'
SPA_DIR = AG + '/o_main_spa_breaks/data'
GRID_END = pd.Timestamp('2026-10-12')
EXCL_RE = re.compile(r'(2330|tsmc|TSMC|lev2|L2_|BENCH_)')


def bench():
    return pd.read_parquet(SPA_DIR + '/bench.parquet')


def grid():
    b = bench()
    return list(b.index) + [GRID_END]


def win_ret_from_nav(nav, g=None):
    """nav: Series daily. window return start->end on grid (nav at close of grid dates)."""
    g = g or grid()
    nav = nav.copy()
    idx = nav.index
    out = []
    for a, z in zip(g[:-1], g[1:]):
        i = idx.searchsorted(a); k = idx.searchsorted(z)
        out.append(nav.iloc[k] / nav.iloc[i] - 1)
    return np.array(out)


def javg_nav(df):
    """average daily returns across available j columns, chained (leading NaN = not started)."""
    r = df.pct_change()
    r = r.where(df.shift(1).notna())
    m = r.mean(axis=1, skipna=True).fillna(0.0)
    return (1 + m).cumprod()


def per_j_winret(df):
    """per-column window returns; before a column starts it is treated as cash (nav=1 at its start)."""
    d = df.bfill()  # leading NaN -> first value (cash before start); interior NaN none expected
    return np.column_stack([win_ret_from_nav(d[c]) for c in d.columns])


def ir(x, axis=0):
    x = np.asarray(x, float)
    return np.nanmean(x, axis=axis) / np.nanstd(x, axis=axis, ddof=1) * math.sqrt(12)


def tstat(x, axis=0):
    x = np.asarray(x, float)
    n = np.sum(~np.isnan(x), axis=axis)
    return np.nanmean(x, axis=axis) / np.nanstd(x, axis=axis, ddof=1) * np.sqrt(n)


def geo_ex(rs, rb, axis=0):
    """geometric annualized excess: (prod(1+rs))^(12/n) - (prod(1+rb))^(12/n). rs may be 2-D (n x k)."""
    n = rs.shape[axis]
    gs = np.exp(np.log1p(rs).sum(axis=axis) * 12 / n) - 1
    gb = np.exp(np.log1p(rb).sum() * 12 / n) - 1
    return gs - gb


def sb_indices(n, L, B, seed):
    """stationary bootstrap (Politis-Romano, circular) index matrix B x n."""
    rng = np.random.default_rng(seed)
    p = 1.0 / L
    idx = np.empty((B, n), dtype=np.int32)
    idx[:, 0] = rng.integers(0, n, B)
    newb = rng.random((B, n)) < p
    rnd = rng.integers(0, n, (B, n))
    for t in range(1, n):
        idx[:, t] = np.where(newb[:, t], rnd[:, t], (idx[:, t - 1] + 1) % n)
    return idx


def counts_from_idx(idx, n):
    B = idx.shape[0]
    W = np.zeros((B, n), dtype=np.float64)
    np.add.at(W, (np.repeat(np.arange(B), idx.shape[1]), idx.ravel()), 1.0)
    return W


def pr_kernel_lrv(D, L):
    """Politis-Romano stationary-bootstrap kernel long-run variance per column. D: n x k."""
    n = D.shape[0]
    Z = D - D.mean(axis=0)
    q = 1.0 / L
    v = (Z * Z).sum(axis=0) / n
    for i in range(1, n):
        k = (1 - i / n) * (1 - q) ** i + (i / n) * (1 - q) ** (n - i)
        if k < 1e-12:
            continue
        g = (Z[i:] * Z[:-i]).sum(axis=0) / n
        v = v + 2 * k * g
    return v


def spa_prep(D, W):
    """D: n x k excess. returns t-stats, recentered bootstrap max distribution (SPA_c) and per-col boot stats func."""
    n = D.shape[0]
    m = D.mean(axis=0)
    om = np.sqrt(pr_kernel_lrv(D, 6))
    t = math.sqrt(n) * m / om
    gc = np.where(t >= -math.sqrt(2 * math.log(math.log(n))), m, 0.0)
    return m, om, t, gc


def spa_boot_max(D, W, chunk=2000):
    n = D.shape[0]
    m, om, t, gc = spa_prep(D, W)
    B = W.shape[0]
    mx = np.empty(B)
    for s in range(0, B, chunk):
        bm = W[s:s + chunk] @ D / n
        z = math.sqrt(n) * (bm - gc) / om
        mx[s:s + chunk] = z.max(axis=1)
    return t, mx, (m, om, gc)


def dsr(x, sr_var, N, T=None, sr=None, g3=None, g4=None):
    x = np.asarray(x, float)
    T = T or len(x)
    sr = x.mean() / x.std(ddof=1) if sr is None else sr
    g3 = float(stats.skew(x)) if g3 is None else g3
    g4 = float(stats.kurtosis(x, fisher=False)) if g4 is None else g4
    emc = 0.5772156649
    N = max(float(N), 2.0)
    sr0 = math.sqrt(sr_var) * ((1 - emc) * stats.norm.ppf(1 - 1 / N) + emc * stats.norm.ppf(1 - 1 / (N * math.e)))
    z = (sr - sr0) * math.sqrt(T - 1) / math.sqrt(max(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2, 1e-12))
    return float(stats.norm.cdf(z)), sr0


def sr0_of(sr_var, N):
    emc = 0.5772156649
    return math.sqrt(sr_var) * ((1 - emc) * stats.norm.ppf(1 - 1 / N) + emc * stats.norm.ppf(1 - 1 / (N * math.e)))


def liji(D):
    n, N = D.shape
    Z = (D - D.mean(axis=0)) / D.std(axis=0, ddof=1)
    G = Z @ Z.T / (n - 1)
    lam = np.clip(np.linalg.eigvalsh(G), 0, None)
    return float(np.sum((lam >= 1).astype(float) + (lam - np.floor(lam))))


def npmle(y, se, lo=-2.0, hi=2.5, step=0.01, iters=5000, tol=1e-10):
    th = np.round(np.arange(lo, hi + step / 2, step), 10)
    Lk = stats.norm.pdf((y[:, None] - th[None, :]) / se[:, None]) / se[:, None]
    w = np.full(len(th), 1.0 / len(th))
    ll_old = -np.inf
    for it in range(iters):
        f = Lk @ w
        ll = np.log(f).sum()
        w = w * (Lk / f[:, None]).mean(axis=0)
        if abs(ll - ll_old) < tol:
            break
        ll_old = ll
    return th, w, Lk


def posterior(th, w, y, se):
    lk = stats.norm.pdf((y - th) / se)
    p = w * lk; p = p / p.sum()
    cdf = np.cumsum(p)
    return {'mean': float((p * th).sum()), 'median': float(th[np.searchsorted(cdf, 0.5)]),
            'q05': float(th[np.searchsorted(cdf, 0.05)]), 'q95': float(th[np.searchsorted(cdf, 0.95)]),
            'P>0.3': float(p[th > 0.3].sum()), 'P>=0.5': float(p[th >= 0.5 - 1e-9].sum())}
