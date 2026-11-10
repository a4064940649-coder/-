"""負二項（NB2）／Poisson GLM（scipy MLE＋數值 Hessian）。"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln
def nll(par, X, y, nb):
    b = par[:X.shape[1]]; eta = np.clip(X @ b, -30, 30); mu = np.exp(eta)
    if not nb:
        return -np.sum(y * eta - mu - gammaln(y + 1))
    a = np.exp(par[-1]); r = 1 / a
    return -np.sum(gammaln(y + r) - gammaln(r) - gammaln(y + 1) + r * np.log(r / (r + mu)) + y * np.log(mu / (r + mu)))
def hess(f, x, h=1e-4):
    n = len(x); H = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            e_i = np.zeros(n); e_j = np.zeros(n); e_i[i] = h; e_j[j] = h
            H[i, j] = H[j, i] = (f(x + e_i + e_j) - f(x + e_i - e_j) - f(x - e_i + e_j) + f(x - e_i - e_j)) / (4 * h * h)
    return H
def fit(X, y, nb=True):
    k = X.shape[1]; x0 = np.zeros(k + (1 if nb else 0)); x0[0] = np.log(max(y.mean(), 0.1))
    if nb: x0[-1] = np.log(0.5)
    f = lambda p: nll(p, X, y, nb)
    r = minimize(f, x0, method='BFGS', options=dict(maxiter=5000, gtol=1e-7))
    r = minimize(f, r.x, method='Nelder-Mead', options=dict(maxiter=20000, xatol=1e-9, fatol=1e-10)) if not r.success else r
    H = hess(f, r.x); cov = np.linalg.pinv(H)
    ll = -r.fun; npar = len(r.x)
    return dict(par=r.x, cov=cov, ll=ll, aic=2 * npar - 2 * ll, bic=npar * np.log(len(y)) - 2 * ll, success=bool(r.success), nb=nb, k=k)
def mu(F, X):
    return np.exp(np.clip(X @ F['par'][:F['k']], -30, 30))
def simulate(F, X, n=10000, seed=0, param_unc=True):
    """回傳 n × len(X) 的模擬計數。param_unc：係數（含 log α）多元常態抽樣。"""
    rng = np.random.default_rng(seed)
    P = rng.multivariate_normal(F['par'], F['cov'], size=n) if param_unc else np.tile(F['par'], (n, 1))
    eta = np.clip(P[:, :F['k']] @ X.T, -30, 30); m = np.exp(eta)
    if not F['nb']:
        return rng.poisson(m)
    a = np.exp(P[:, -1])[:, None]; r = 1 / a
    lam = rng.gamma(shape=np.broadcast_to(r, m.shape), scale=m / r)
    return rng.poisson(lam)
