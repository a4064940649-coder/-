"""Independent replication o6: estimators (HAC, G1 REML, G2 log-spline, G3 NPMLE, posterior)."""
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import norm

TH = np.round(np.arange(-8.0, 4.0 + 1e-9, 0.02), 10)
DT = 0.02


def hac(x, L=6):
    x = np.asarray(x, float)
    x = x - x.mean()
    n = len(x)
    v = x @ x / n
    for i in range(1, L + 1):
        v += 2 * (1 - i / (L + 1)) * (x[i:] @ x[:-i] / n)
    return v / n


def decompose(A, L=6):
    """A: n x K (pp). returns xhat, s2 (individual), m (row mean), vP."""
    m = A.mean(1)
    x = A.mean(0)
    s2 = np.array([hac(A[:, k] - m, L) for k in range(A.shape[1])])
    return x, s2, m, hac(m, L)


def ns_basis(th, df=5):
    knots = np.quantile(th, np.linspace(0, 1, df + 1))
    K = len(knots)
    xi_K = knots[-1]

    def d(k):
        return (np.clip(th - knots[k], 0, None) ** 3 - np.clip(th - xi_K, 0, None) ** 3) / (xi_K - knots[k])

    cols = [th]
    for k in range(K - 2):
        cols.append(d(k) - d(K - 2))
    X = np.column_stack(cols)
    X = X - X.mean(0)
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    U = U / np.sqrt((U ** 2).sum(0))
    return U


def Pmat(x, s2, th=TH):
    s = np.sqrt(s2)[:, None]
    return norm.pdf((x[:, None] - th[None, :]) / s) / s


def fit_g2(x, s2, c0=0.1, th=TH, Q=None):
    if Q is None:
        Q = ns_basis(th)
    P = Pmat(x, s2, th)

    def f(b):
        a = Q @ b
        a = a - a.max()
        w = np.exp(a)
        g = w / w.sum()
        Pg = P @ g
        nb = np.sqrt(b @ b)
        val = -np.sum(np.log(Pg)) + c0 * nb
        # gradient of sum log Pg wrt b
        R = (P * g[None, :]) / Pg[:, None]  # K x J posterior weights
        Qbar = g @ Q
        grad_ll = (R @ Q).sum(0) - R.sum(1).sum() * Qbar
        gp = c0 * b / nb if nb > 0 else np.zeros_like(b)
        return val, -grad_ll + gp

    r = minimize(f, np.zeros(Q.shape[1]), jac=True, method='BFGS')
    a = Q @ r.x
    w = np.exp(a - a.max())
    return w / w.sum(), r


def fit_g3(x, s2, th=TH, maxit=6000, tol=1e-10):
    P = Pmat(x, s2, th)
    g = np.ones(len(th)) / len(th)
    ll_old = None
    for it in range(1, maxit + 1):
        Pg = P @ g
        g = g * (P / Pg[:, None]).mean(0)
        g = g / g.sum()
        if it % 200 == 0:
            ll = np.sum(np.log(P @ g))
            if ll_old is not None and abs(ll - ll_old) <= tol * abs(ll_old):
                break
            ll_old = ll
    return g


def fit_g1(x, s2):
    def nreml(lt):
        t2 = np.exp(lt)
        v = t2 + s2
        w = 1 / v
        mu = np.sum(w * x) / np.sum(w)
        return 0.5 * (np.sum(np.log(v)) + np.log(np.sum(w)) + np.sum(w * (x - mu) ** 2))

    r = minimize_scalar(nreml, bounds=(np.log(1e-8), np.log(100)), method='bounded')
    t2 = np.exp(r.x)
    if t2 < 1e-6:
        t2 = 0.0
    w = 1 / (t2 + s2)
    mu = np.sum(w * x) / np.sum(w)
    return mu, t2, 1 / np.sum(w)


def post_grid(g, x, s2, vP, th=TH):
    lg = np.log(np.maximum(g, 1e-300)) - 0.5 * (x - th) ** 2 / s2
    p = np.exp(lg - lg.max())
    p = p / p.sum()
    if vP > 0:
        sd = np.sqrt(vP)
        h = int(np.ceil(6 * sd / DT))
        k = norm.pdf(np.arange(-h, h + 1) * DT / sd)
        k = k / k.sum()
        p = np.convolve(p, k, 'same')
        p = p / p.sum()
    return p


def summ(p, th=TH):
    c = np.cumsum(p)
    q = lambda a: float(np.interp(a, c, th))
    i0 = np.argmin(np.abs(th))
    return dict(mean=float(p @ th), P=float(p[th > 1e-9].sum() + 0.5 * p[i0]), q10=q(0.1), q90=q(0.9),
                sd=float(np.sqrt(p @ th ** 2 - (p @ th) ** 2)))


def g1_grid(mu, t2, vmu, x, s2, vP, th=TH):
    r = post_g1(mu, t2, vmu, x, s2, vP)
    v = ((r['q90'] - r['mean']) / 1.2816) ** 2
    p = norm.pdf(th, r['mean'], np.sqrt(v))
    return p / p.sum()


def post_g1(mu, t2, vmu, x, s2, vP):
    B = t2 / (t2 + s2)
    m = mu + B * (x - mu)
    v = t2 * s2 / (t2 + s2) + (1 - B) ** 2 * vmu + vP
    return dict(mean=float(m), P=float(1 - norm.cdf(0, m, np.sqrt(v))), q10=float(m - 1.2816 * np.sqrt(v)),
                q90=float(m + 1.2816 * np.sqrt(v)))
