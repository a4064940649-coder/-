#!/usr/bin/env python3
"""S25 獨立複製：統計小工具（複製員自寫，不依賴 statsmodels）。
未讀、未 import、未執行 s25lib.py／prep.py／describe.py／train.py／validate.py／posthoc.py。"""
import numpy as np


def tstat(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 2:
        return float('nan')
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))


def welch_t(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    return float((a.mean() - b.mean()) / np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)))


def ann_stats(r):
    """月報酬序列 → 年化平均（%）、年化波動（%）、Sharpe（月平均÷月標準差×√12）、t、n。"""
    r = np.asarray(r, float)
    m, s = r.mean(), r.std(ddof=1)
    return dict(mean_ann_pct=m * 1200, vol_ann_pct=s * np.sqrt(12) * 100, sharpe=m / s * np.sqrt(12), t_mean=tstat(r), n=int(len(r)))


def ols_nw(y, X, lags=4, correction=False):
    """OLS（X 不含常數，自動加常數）＋ Newey-West（Bartlett）標準誤；correction＝乘 n/(n−k)（預設不乘：與原研究 t 對齊後確認）。"""
    y = np.asarray(y, float)
    X = np.column_stack([np.ones(len(y))] + ([np.asarray(X, float).reshape(len(y), -1)] if X is not None else []))
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ X.T @ y
    e = y - X @ b
    Xe = X * e[:, None]
    S = Xe.T @ Xe
    for L in range(1, lags + 1):
        w = 1 - L / (lags + 1)
        G = Xe[L:].T @ Xe[:-L]
        S += w * (G + G.T)
    if correction:
        S *= n / (n - k)
    V = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(V))
    r2 = 1 - (e @ e) / (((y - y.mean()) ** 2).sum())
    return dict(coef=b.tolist(), t=(b / se).tolist(), r2=float(r2), n=int(n), alpha_ann_pct=float(b[0] * 1200), alpha_t=float(b[0] / se[0]))


def block_boot_sharpe_diff(a, b, block=6, n_boot=2000, seed=20261116):
    """配對移動區塊 bootstrap（區塊 6 個月、循環接續），回傳 Sharpe(a)−Sharpe(b) 的 90% 信賴區間與 P(差 ≤ 0)。"""
    a = np.asarray(a, float); b = np.asarray(b, float); n = len(a)
    rng = np.random.default_rng(seed)
    sh = lambda x: x.mean() / x.std(ddof=1) * np.sqrt(12)
    nb = int(np.ceil(n / block))
    d = np.empty(n_boot)
    for i in range(n_boot):
        st = rng.integers(0, n, nb)
        idx = (st[:, None] + np.arange(block)[None, :]).ravel()[:n] % n
        d[i] = sh(a[idx]) - sh(b[idx])
    return dict(diff=float(sh(a) - sh(b)), ci90=[float(np.quantile(d, 0.05)), float(np.quantile(d, 0.95))], p_le0=float((d <= 0).mean()))
