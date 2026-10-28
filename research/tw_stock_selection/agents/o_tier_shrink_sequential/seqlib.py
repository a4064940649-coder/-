#!/usr/bin/env python3
"""序貫判定規則的共用計算（前瞻監測與檢定力模擬共用，確保同一套數學）。
對一條線：先驗格點密度 prior（θ，pp／窗）、σ（pp／窗）、α_min。輸入前瞻月超額 x_1..x_n（pp），回傳逐月：
  post_P（P(α>0)）、E_up、E_down、累積財富比（若給 r50）。
E_up  ＝ Σ_θ π⁺(θ) exp(ll(θ; 0))，π⁺＝prior·1[θ>0] 正規化；
E_down＝ Σ_θ π⁻(θ) exp(ll(θ; α_min))，π⁻＝prior·1[θ<α_min] 正規化；
ll(θ; θ0)＝[(θ−θ0)·S_n − n(θ²−θ0²)/2] / σ²（常態、已知 σ 的對數概似比）。
"""
import numpy as np

def prep(prior, th, alpha_min, tol=1e-12):
    p = np.asarray(prior, float); p = p / p.sum()
    keep = p > tol
    th_k, p_k = th[keep], p[keep]
    wu = np.where(th_k > 0, p_k, 0.0); wu = wu / wu.sum() if wu.sum() > 0 else wu
    wd = np.where(th_k < alpha_min, p_k, 0.0); wd = wd / wd.sum() if wd.sum() > 0 else wd
    return th_k, p_k, wu, wd

def _lse(ll, w):
    """log Σ w exp(ll)，ll：(..., m)。"""
    m = ll.max(axis=-1, keepdims=True)
    return (m[..., 0] + np.log(np.maximum((w * np.exp(ll - m)).sum(axis=-1), 1e-300)))

def path_stats(xs, prior, th, sigma, alpha_min, r50=None):
    """xs：(P, n) 或 (n,)；回傳 dict of arrays (P, n)。log 值用以避免溢位。"""
    X = np.atleast_2d(np.asarray(xs, float))
    P_, n = X.shape
    th_k, p_k, wu, wd = prep(prior, th, alpha_min)
    S = np.cumsum(X, axis=1); N = np.arange(1, n + 1)[None, :]
    out = {'logE_up': np.empty((P_, n)), 'logE_down': np.empty((P_, n)), 'post_P': np.empty((P_, n))}
    lp = np.log(p_k)
    pos = th_k > 0
    for t in range(n):
        s = S[:, t][:, None]; k = N[0, t]
        ll0 = (th_k[None, :] * s - 0.5 * k * th_k[None, :] ** 2) / sigma ** 2          # 相對 θ0＝0
        out['logE_up'][:, t] = _lse(ll0, wu[None, :]) if wu.sum() > 0 else -np.inf
        lld = ((th_k[None, :] - alpha_min) * s - 0.5 * k * (th_k[None, :] ** 2 - alpha_min ** 2)) / sigma ** 2
        out['logE_down'][:, t] = _lse(lld, wd[None, :]) if wd.sum() > 0 else -np.inf
        lpost = lp[None, :] + ll0
        m = lpost.max(axis=1, keepdims=True); w = np.exp(lpost - m)
        out['post_P'][:, t] = (w * pos[None, :]).sum(axis=1) / w.sum(axis=1)
    if r50 is not None:
        R50 = np.atleast_2d(np.asarray(r50, float))
        rl = R50 + X / 100.0
        out['wealth_ratio'] = np.cumprod((1 + rl) / (1 + R50), axis=1)
    return out

def triggers(st, thr, n_months=None):
    """依門檻回傳逐月布林：upgrade、down、review（套用最短期間，未套用死區與每年一次——那是配置變更層級）。"""
    P_, n = st['post_P'].shape
    t = np.arange(1, n + 1)[None, :]
    up = (st['logE_up'] >= np.log(thr['E_up'])) & (t >= thr['up_min_months'])
    down_raw = (st['logE_down'] >= np.log(thr['E_down'])) | ((st['post_P'] < thr['P_down']) & (t >= thr['P_down_min_months']))
    down = down_raw & (t >= thr['change_min_months'])
    rev = (st['logE_down'] >= np.log(thr['E_review']))
    if 'wealth_ratio' in st: rev = rev | (st['wealth_ratio'] < thr['ratio_review'])
    return {'upgrade': up, 'down': down, 'review': rev}

def first_time(b):
    """每條路徑第一次為真的月份（1 起算；從未則 0）。"""
    any_ = b.any(axis=1)
    return np.where(any_, b.argmax(axis=1) + 1, 0)
