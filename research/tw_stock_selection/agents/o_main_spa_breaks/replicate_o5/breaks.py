"""斷點檢定（獨立實作）：OLS-CUSUM、遞迴 CUSUM、sup-F、Bai-Perron、Chow、MDE、檢定力。"""
import sys; sys.path.insert(0, '.')
from rlib import *
from scipy import stats as st

def hac_lr(e, lag=None):
    e = np.asarray(e, float); n = len(e)
    if lag is None: lag = int(np.floor(4 * (n / 100) ** (2 / 9)))
    v = (e * e).sum() / n
    for j in range(1, lag + 1):
        v += 2 * (1 - j / (lag + 1)) * (e[j:] * e[:-j]).sum() / n
    return v

def kolm_p(x):
    k = np.arange(1, 200)
    return float(min(1, max(0, 2 * np.sum((-1) ** (k - 1) * np.exp(-2 * k * k * x * x)))))

def ols_cusum(x, hac=True):
    x = np.asarray(x, float); n = len(x); e = x - x.mean()
    s2 = hac_lr(e) if hac else e.var()
    B = np.cumsum(e) / math.sqrt(n * s2)
    m = float(np.abs(B).max())
    return m, kolm_p(m)

def rec_cusum_stat(x):
    x = np.asarray(x, float); n = len(x)
    cm = np.cumsum(x)
    t = np.arange(1, n)
    w = (x[1:] - cm[:-1] / t) * np.sqrt(t / (t + 1))
    sig = w.std(ddof=1)
    W = np.cumsum(w) / sig
    nn = len(w); r = np.arange(1, nn + 1)
    bound = math.sqrt(nn) + 2 * r / math.sqrt(nn)
    return float((np.abs(W) / bound).max())       # > 0.948 → 越界

def fstats(x, trim=0.15, hac=True, ks=None):
    x = np.asarray(x, float); n = len(x)
    h = int(np.floor(trim * n))
    ks = range(h, n - h + 1) if ks is None else ks
    cs = np.cumsum(x); tot = cs[-1]
    F = {}
    for k in ks:
        m1 = cs[k - 1] / k; m2 = (tot - cs[k - 1]) / (n - k)
        e = np.concatenate([x[:k] - m1, x[k:] - m2])
        s2 = hac_lr(e) if hac else e.var(ddof=2)
        F[k] = (m1 - m2) ** 2 / (s2 * (1 / k + 1 / (n - k)))
    return F

def supF(x, hac=True):
    F = fstats(x, hac=hac); k = max(F, key=F.get)
    return F[k], k

def null_supF(n, hac, R=2000):
    rng = np.random.default_rng(777 + int(hac))
    return np.array([supF(rng.standard_normal(n), hac)[0] for _ in range(R)])

# ---- Bai-Perron（平均值模型）----
def ssr_mat(x, h):
    n = len(x); cs = np.concatenate([[0], np.cumsum(x)]); cs2 = np.concatenate([[0], np.cumsum(x * x)])
    S = np.full((n + 1, n + 1), np.inf)
    for i in range(n):
        for j in range(i + h, n + 1):
            m = j - i; s = cs[j] - cs[i]
            S[i, j] = cs2[j] - cs2[i] - s * s / m
    return S

def bp_dp(x, mmax=3, trim=0.15):
    n = len(x); h = int(np.floor(trim * n)); S = ssr_mat(x, h)
    best = {0: (S[0, n], [])}
    # f[m][j]: 最小 SSR，用 m 個斷點切 [0, j)
    f = {0: {j: (S[0, j], []) for j in range(h, n + 1)}}
    for m in range(1, mmax + 1):
        f[m] = {}
        for j in range((m + 1) * h, n + 1):
            bb = (np.inf, None)
            for k in range(m * h, j - h + 1):
                if k in f[m - 1]:
                    v = f[m - 1][k][0] + S[k, j]
                    if v < bb[0]: bb = (v, f[m - 1][k][1] + [k])
            f[m][j] = bb
        best[m] = f[m][n]
    return best, S, h

def bp_select(x, mmax=3):
    n = len(x); best, S, h = bp_dp(x, mmax)
    out = {}
    bic = {m: math.log(best[m][0] / n) + (2 * m + 1) * math.log(n) / n for m in best}
    c0, d0 = 0.299, 0.1
    lwz = {m: math.log(best[m][0] / (n - (2 * m + 1))) + (2 * m + 1) * c0 * (math.log(n)) ** (2 + d0) / n for m in best}
    out['BIC'] = min(bic, key=bic.get); out['LWZ'] = min(lwz, key=lwz.get)
    out['breaks'] = {m: best[m][1] for m in best}
    return out

def chow(x, k, hac=True):
    x = np.asarray(x, float); n = len(x)
    m1, m2 = x[:k].mean(), x[k:].mean()
    e = np.concatenate([x[:k] - m1, x[k:] - m2])
    s2 = hac_lr(e) if hac else e.var(ddof=2)
    se = math.sqrt(s2 * (1 / k + 1 / (n - k)))
    W = (m2 - m1) ** 2 / se ** 2
    mde = (st.norm.ppf(0.975) + st.norm.ppf(0.8)) * se
    return dict(pre=m1 * 100, post=m2 * 100, W=W, p=float(1 - st.chi2.cdf(W, 1)), mde_pp=mde * 100, se=se)
