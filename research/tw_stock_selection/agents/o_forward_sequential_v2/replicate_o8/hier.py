# Independent implementation of the hierarchical shrinkage versions (spec section 2-3).
import numpy as np, pandas as pd, os, sys
from scipy.optimize import minimize
from scipy.stats import norm, t as tdist
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rlib import hac_var_mean
VR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'vendor_o7', 'raw')
GRID = np.round(np.arange(-8, 4 + 1e-9, 0.02), 10)
L = 6

def load_family():
    f = pd.read_parquet(os.path.join(VR, 'spa/family_excess.parquet'))
    c = pd.read_csv(os.path.join(VR, 'o6shrink/clusters_k161.csv'))
    X = f[c.member_id].values * 100.0
    return f.index, X, c.cluster.values, c.is_medoid.values.astype(bool), c.member_id.values

def hv(x):  # HAC variance of the mean (NaN dropped)
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    return hac_var_mean(x, L) / len(x)

def ns_basis(x, df=5):
    K = df + 1
    knots = np.quantile(x, np.linspace(0, 1, K))
    def dk(k):
        return (np.maximum(x - knots[k], 0) ** 3 - np.maximum(x - knots[K - 1], 0) ** 3) / (knots[K - 1] - knots[k])
    cols = [x] + [dk(k) - dk(K - 2) for k in range(K - 2)]
    Q = np.column_stack(cols); Q = Q - Q.mean(0)
    U, s, Vt = np.linalg.svd(Q, full_matrices=False)
    return U / np.sqrt((U ** 2).sum(0))

QB = ns_basis(GRID)

def g2_fit(xh, se, c0=0.1):
    P = norm.pdf((xh[:, None] - GRID[None, :]) / se[:, None]) / se[:, None]
    def nll(a):
        e = QB @ a; e -= e.max(); g = np.exp(e); g /= g.sum()
        f = P @ g
        return -np.sum(np.log(f + 1e-300)) + c0 * np.sqrt(a @ a + 1e-12)
    r = minimize(nll, np.zeros(QB.shape[1]), method='BFGS')
    e = QB @ r.x; e -= e.max(); g = np.exp(e); return g / g.sum()

def g3_npmle(xh, se, it=500):
    P = norm.pdf((xh[:, None] - GRID[None, :]) / se[:, None]) / se[:, None]
    g = np.ones(len(GRID)) / len(GRID)
    for _ in range(it):
        f = P @ g; g = g * (P / f[:, None]).mean(0); g /= g.sum()
    return g

def post(g, x, se):
    w = g * norm.pdf((x - GRID) / se); s = w.sum()
    return w / s if s > 0 else g.copy()

def conv_normal(w, m, v):
    """density of theta + N(m, v) re-gridded."""
    if v <= 1e-12:
        return np.interp(GRID - m, GRID, w, left=0, right=0) / max(np.interp(GRID - m, GRID, w, left=0, right=0).sum(), 1e-300)
    K = norm.pdf((GRID[:, None] - GRID[None, :] - m) / np.sqrt(v))
    r = K @ w; return r / r.sum()

def conv_density(w, d):
    """theta + delta where delta has density d on centred grid DG."""
    r = np.convolve(w, d, mode='same'); return r / r.sum()

def stat_boot_idx(T, B, rng, Lb=6):
    idx = np.empty((B, T), int); i = rng.integers(T, size=B)
    jump = rng.random((B, T)) < 1.0 / Lb; new = rng.integers(T, size=(B, T))
    for t in range(T):
        if t > 0: i = np.where(jump[:, t], new[:, t], (i + 1) % T)
        idx[:, t] = i
    return idx

def max_bias(Y, rng, B=400, nq=200):
    """Y: T x m member series (NaN ok). demean, same-row stationary bootstrap, max of means."""
    Y = Y - np.nanmean(Y, 0)
    Y = np.where(np.isnan(Y), 0.0, Y)
    T = Y.shape[0]; idx = stat_boot_idx(T, B, rng)
    W = np.zeros((B, T)); np.add.at(W, (np.repeat(np.arange(B), T), idx.ravel()), 1.0)
    mb = (W @ Y / T).max(1)
    return np.quantile(mb, (np.arange(nq) + 0.5) / nq)

def sigma_w(X, cl, clusters):
    num = den = 0.0
    for c in clusters:
        Y = X[:, cl == c]; m = Y.shape[1]
        if m < 3: continue
        d = np.nanmean(Y, 0); Db = np.nanmean(Y, 1)
        hvbar = np.mean([hv(Y[:, k] - Db) for k in range(m)])
        num += (m - 1) * (np.var(d, ddof=1) - hvbar * m / (m - 1)); den += m - 1
    return np.sqrt(max(num / den, 0.0))

def sigma_w_one(Y):
    m = Y.shape[1]; d = np.nanmean(Y, 0); Db = np.nanmean(Y, 1)
    hvbar = np.mean([hv(Y[:, k] - Db) for k in range(m)])
    return np.sqrt(max(np.var(d, ddof=1) - hvbar * m / (m - 1), 0.0))

DG = GRID - GRID[len(GRID) // 2]  # centred grid for delta (step 0.02, span +-6)

def fit_all(X, cl, med, c, focus, versions=('V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7'), rng=None, Ymem=None):
    """X: T x M estimation-window family; c: held-out cluster; focus: T series. returns dict version->posterior on GRID"""
    keep = cl != c
    Mx = X[:, keep & med]; xh = np.nanmean(Mx, 0)
    Pser = np.nanmean(Mx, 1); vP = hv(Pser)
    se = np.sqrt(vP + np.array([hv(Mx[:, k] - Pser) for k in range(Mx.shape[1])]))
    g2 = g2_fit(xh, se)
    out = {}
    fx = np.nanmean(focus); fse = np.sqrt(vP + hv(focus - Pser))
    if Ymem is None: Ymem = X[:, cl == c]
    mser = X[:, (cl == (c if c != -999 else 114)) & med][:, 0]
    if 'V1' in versions: out['V1'] = post(g2, fx, fse)
    if 'V3' in versions:
        bq = max_bias(Ymem, rng)
        out['V3'] = np.mean([post(g2, fx - b, fse) for b in bq], 0)
    need_h = any(v in versions for v in ('V2', 'V4', 'V5', 'V6', 'V7'))
    if need_h:
        mx = np.nanmean(mser); mse = np.sqrt(vP + hv(mser - Pser))
        wm = post(g2, mx, mse)
        dser = focus - mser; d = np.nanmean(dser); sd2 = hv(dser) if np.nansum(np.abs(dser)) > 0 else 0.0
        clusters = [k for k in np.unique(cl[keep])]
        sw = sigma_w(X[:, keep], cl[keep], clusters)
        def hn(wbase, dd, sw_):
            if sd2 <= 0: return wbase
            Bf = 2 * sw_ ** 2 / (2 * sw_ ** 2 + sd2)
            return conv_normal(wbase, Bf * dd, Bf * sd2)
        if 'V2' in versions: out['V2'] = hn(wm, d, sw)
        if 'V4' in versions:
            D = Ymem - mser[:, None]
            bq = max_bias(D, rng)
            if sd2 <= 0 or 2 * sw ** 2 / (2 * sw ** 2 + sd2) * sd2 < 1e-8: out["V4"] = wm
            else:
                Bf = 2 * sw ** 2 / (2 * sw ** 2 + sd2)
                ker = norm.pdf((DG[:, None] - Bf * (d - bq[None, :])) / np.sqrt(Bf * sd2)).mean(1)
                out['V4'] = conv_density(wm, ker / ker.sum())
        if 'V5' in versions:
            if sd2 <= 0: out['V5'] = wm
            else:
                sc = np.sqrt(2) * sw / np.sqrt(3)
                pd_ = tdist.pdf(DG / max(sc, 1e-6), 3) * norm.pdf((d - DG) / np.sqrt(sd2)); pd_ /= pd_.sum()
                out['V5'] = conv_density(wm, pd_)
        if 'V6' in versions:
            g3 = g3_npmle(xh, se); out['V6'] = hn(post(g3, mx, mse), d, sw)
        if 'V7' in versions:
            m = Ymem.shape[1]
            sw7 = sigma_w_one(Ymem) if m >= 10 else sw
            out['V7'] = hn(wm, d, sw7)
    return out

def transform(w, s, k):
    mu = (GRID * w).sum(); th = s * mu + k * (GRID - mu)
    # density of transformed variable re-gridded via CDF interpolation
    cdf = np.cumsum(w); c2 = np.interp(GRID + 0.01, th + 0.01 * k, cdf, left=0, right=1)
    r = np.diff(np.concatenate([[0], c2])); r = np.maximum(r, 0); return r / r.sum()
