#!/usr/bin/env python3
"""自寫結構斷點檢定（均值模型 y_t = μ_j + e_t）。
- OLS-CUSUM（Ploberger & Krämer 1992）：sup|Σ(y−ȳ)|/(ω̂√n)，ω̂ 用 HAC；漸近分布＝布朗橋上確界（Kolmogorov）。
- 遞迴殘差 CUSUM（Brown-Durbin-Evans 1975）：5% 邊界 a=0.948；p 值用 iid 常態模擬。
- Andrews sup-F（修剪 15%）：每個候選斷點的 Wald 統計量（HAC），取最大；p 值用模擬的有限樣本虛無分布。
- Bai-Perron：動態規劃求全域最小 SSR，最多 3 個斷點、最短區段 15%；斷點數用 BIC 與 LWZ 選，
  另做 sup-F(l+1|l) 序列檢定與 UDmax；斷點日期 90% 信賴區間依 Bai（1997），臨界值自行模擬 argmax 分布。
- Chow（事前固定日期）：HAC Wald，χ²(1)。
HAC：Bartlett 核，落後 = floor(4(n/100)^(2/9))（Newey-West 1994 規則），殘差來自「允許斷點」的模型。
"""
import math
import numpy as np
from scipy import stats

def nw_lag(n): return int(math.floor(4 * (n / 100) ** (2 / 9)))

def lrv1(e, lag=None):
    e = np.asarray(e, float); n = len(e)
    if lag is None: lag = nw_lag(n)
    s = (e @ e) / n
    for i in range(1, lag + 1):
        s += 2 * (1 - i / (lag + 1)) * (e[i:] @ e[:-i]) / n
    return max(s, 1e-18)

def ols_cusum(y, hac=True):
    y = np.asarray(y, float); n = len(y); e = y - y.mean()
    om = math.sqrt(lrv1(e) if hac else e.var())
    W = np.cumsum(e) / (om * math.sqrt(n))
    s = float(np.abs(W).max())
    k = np.arange(1, 101)
    p = float(np.clip(2 * np.sum((-1) ** (k - 1) * np.exp(-2 * k ** 2 * s ** 2)), 0, 1))
    return {'stat': s, 'p': p, 'crit5': 1.358, 'argmax': int(np.abs(W).argmax())}

def rec_cusum_stat(y):
    y = np.asarray(y, float); n = len(y)
    cs = np.cumsum(y)
    t = np.arange(2, n + 1)
    mprev = cs[:-1] / np.arange(1, n)
    w = (y[1:] - mprev) * np.sqrt((t - 1) / t)
    sw = w.std(ddof=1)
    W = np.cumsum(w) / sw
    r = np.arange(1, len(w) + 1); m = len(w)
    bound = lambda a: a * (math.sqrt(m) + 2 * r / math.sqrt(m))
    ratio = float(np.max(np.abs(W) / (math.sqrt(m) + 2 * r / math.sqrt(m))))
    return ratio, W, bound

_RC_NULL = {}
def rec_cusum(y, nsim=5000, seed=11):
    n = len(y)
    ratio, W, bound = rec_cusum_stat(y)
    if n not in _RC_NULL:
        rng = np.random.default_rng(seed)
        _RC_NULL[n] = np.array([rec_cusum_stat(rng.standard_normal(n))[0] for _ in range(nsim)])
    p = float((_RC_NULL[n] >= ratio).mean())
    return {'stat_ratio': ratio, 'crosses_5pct': bool(ratio > 0.948), 'crosses_10pct': bool(ratio > 0.850), 'p_sim': p}

def _seg_stats(y):
    c1 = np.concatenate([[0.0], np.cumsum(y)]); c2 = np.concatenate([[0.0], np.cumsum(y * y)])
    return c1, c2
def ssr_seg(c1, c2, i, j):   # 區段 y[i:j]
    m = j - i; s = c1[j] - c1[i]
    return c2[j] - c2[i] - s * s / m

def supF(y, trim=0.15, hac=True, return_all=False):
    """單一斷點 sup-Wald（Andrews 1993），HAC 用允許斷點模型的殘差、共同長期變異數。"""
    y = np.asarray(y, float); n = len(y); h = int(math.floor(trim * n))
    c1, c2 = _seg_stats(y)
    best = (-1.0, None); allF = []
    for k in range(h, n - h + 1):
        m1, m2 = c1[k] / k, (c1[n] - c1[k]) / (n - k)
        e = np.concatenate([y[:k] - m1, y[k:] - m2])
        om2 = lrv1(e) if hac else e.var()
        F = (m1 - m2) ** 2 / (om2 * (1 / k + 1 / (n - k)))
        allF.append(F)
        if F > best[0]: best = (F, k)
    out = {'stat': float(best[0]), 'k': int(best[1]) if best[1] is not None else None}
    if return_all: out['F'] = np.array(allF)
    return out

def bp_dp(y, m_max=3, trim=0.15):
    """全域最小 SSR 的斷點（區段起點索引），回傳 {m: (ssr, [斷點])}，斷點 k 表示新區段從 y[k] 開始。"""
    y = np.asarray(y, float); n = len(y); h = max(2, int(math.floor(trim * n)))
    c1, c2 = _seg_stats(y)
    I = np.arange(n + 1)
    S = np.full((n + 1, n + 1), np.inf)
    for i in range(n + 1):
        j = np.arange(i + h, n + 1)
        if len(j) == 0: continue
        m = j - i; s = c1[j] - c1[i]
        S[i, j] = c2[j] - c2[i] - s * s / m
    res = {0: (float(S[0, n]), [])}
    cost = S[0].copy()            # cost[j]＝y[:j] 分 1 段
    back = []
    for m in range(1, m_max + 1):
        newc = np.full(n + 1, np.inf); arg = np.full(n + 1, -1)
        for j in range((m + 1) * h, n + 1):
            cand = cost[:j] + S[:j, j]
            i = int(np.argmin(cand)); newc[j] = cand[i]; arg[j] = i
        back.append(arg); cost = newc
        if not np.isfinite(cost[n]): break
        bks = []; j = n
        for a in reversed(back):
            i = int(a[j]); bks.append(i); j = i
        res[m] = (float(cost[n]), sorted(bks))
    return res

def seg_means(y, bks):
    y = np.asarray(y, float); edges = [0] + list(bks) + [len(y)]
    return [float(y[a:b].mean()) for a, b in zip(edges[:-1], edges[1:])], edges

def resid(y, bks):
    mu, edges = seg_means(y, bks)
    e = np.empty(len(y))
    for (a, b), m in zip(zip(edges[:-1], edges[1:]), mu): e[a:b] = np.asarray(y[a:b]) - m
    return e

def wald_breaks(y, bks, hac=True):
    """H0：所有區段均值相同 vs 給定斷點；HAC（共同長期變異數）Wald，自由度＝斷點數。"""
    y = np.asarray(y, float)
    mu, edges = seg_means(y, bks); e = resid(y, bks)
    om2 = lrv1(e) if hac else e.var()
    ns = np.diff(edges)
    m = len(mu)
    R = np.zeros((m - 1, m))
    for i in range(m - 1): R[i, i] = 1; R[i, i + 1] = -1
    Vb = np.diag(om2 / ns)
    d = R @ np.array(mu)
    Wd = float(d @ np.linalg.solve(R @ Vb @ R.T, d))
    return Wd / (m - 1)       # F 形式（Wald ÷ 斷點數）

def sup_f_l1(y, bks, trim=0.15, hac=True):
    """sup-F(l+1|l)：在現有 l 個斷點的每個區段內再找一個斷點，取最大的單斷點 F。"""
    y = np.asarray(y, float); n = len(y); h = int(math.floor(trim * n))
    edges = [0] + list(bks) + [n]
    best = 0.0; where = None
    for a, b in zip(edges[:-1], edges[1:]):
        seg = y[a:b]
        if len(seg) < 2 * h + 1: continue
        # 在區段內找斷點；HAC 用全樣本（含新斷點）殘差
        c1, c2 = _seg_stats(seg); L = len(seg)
        for k in range(h, L - h + 1):
            nb = sorted(list(bks) + [a + k])
            e = resid(y, nb); om2 = lrv1(e) if hac else e.var()
            m1, m2 = c1[k] / k, (c1[L] - c1[k]) / (L - k)
            F = (m1 - m2) ** 2 / (om2 * (1 / k + 1 / (L - k)))
            if F > best: best, where = F, a + k
    return float(best), where

def ic_select(y, res):
    n = len(y); out = {}
    for m, (ssr, bks) in res.items():
        pstar = (m + 1) * 1 + m
        out[m] = {'BIC': math.log(ssr / n) + pstar * math.log(n) / n,
                  'LWZ': math.log(ssr / (n - pstar)) + pstar * 0.299 * (math.log(n)) ** 2.1 / n}
    mb = min(out, key=lambda m: out[m]['BIC']); ml = min(out, key=lambda m: out[m]['LWZ'])
    return mb, ml, out

_ARGMAX = {}
def argmax_crit(level=0.90, nsim=20000, Lr=80.0, step=0.02, seed=5):
    """Bai（1997）：argmax_s {W(s) − |s|/2}（雙邊布朗運動）的對稱分位數 c，使 P(|ŝ| ≤ c)=level。"""
    key = (level, nsim)
    if key in _ARGMAX: return _ARGMAX[key]
    rng = np.random.default_rng(seed)
    m = int(Lr / step)
    s = np.arange(1, m + 1) * step
    res = np.empty(nsim)
    for i in range(nsim):
        Wr = np.cumsum(rng.standard_normal(m)) * math.sqrt(step) - s / 2
        Wl = np.cumsum(rng.standard_normal(m)) * math.sqrt(step) - s / 2
        jr, jl = int(np.argmax(Wr)), int(np.argmax(Wl))
        if max(Wr[jr], Wl[jl]) <= 0: res[i] = 0.0
        elif Wr[jr] >= Wl[jl]: res[i] = s[jr]
        else: res[i] = -s[jl]
    c = float(np.quantile(np.abs(res), level))
    _ARGMAX[key] = c
    return c

def break_ci(y, bks, level=0.90, hac=True):
    """每個斷點的 Bai（1997）信賴區間（共同變異數版）：k̂ ± c·ω²/δ² + 1。"""
    mu, edges = seg_means(y, bks); e = resid(y, bks)
    om2 = lrv1(e) if hac else e.var()
    c = argmax_crit(level)
    out = []
    for i, k in enumerate(bks):
        dlt = mu[i + 1] - mu[i]
        w = c * om2 / max(dlt * dlt, 1e-18)
        out.append((int(math.floor(k - w - 1)), int(math.ceil(k + w + 1))))
    return out, c

def chow(y, k, hac=True):
    y = np.asarray(y, float); n = len(y)
    m1, m2 = y[:k].mean(), y[k:].mean()
    e = np.concatenate([y[:k] - m1, y[k:] - m2])
    om2 = lrv1(e) if hac else e.var()
    se = math.sqrt(om2 * (1 / k + 1 / (n - k)))
    W = (m1 - m2) ** 2 / se ** 2
    mde80 = (stats.norm.ppf(0.975) + stats.norm.ppf(0.80)) * se
    return {'k': int(k), 'mean_before': float(m1), 'mean_after': float(m2), 'diff': float(m2 - m1), 'se_diff': float(se),
            'wald': float(W), 'p': float(1 - stats.chi2.cdf(W, 1)), 'MDE80_two_sided': float(mde80), 'n_before': int(k), 'n_after': int(n - k)}

def null_dists(n, nsim, seed, hac=True, trim=0.15):
    """iid 常態下的有限樣本虛無分布：supF(1)、UDmax（m≤3 的 supF(m) 取最大）、supF(2|1)、supF(3|2)。"""
    rng = np.random.default_rng(seed)
    out = {'supF1': [], 'UDmax': [], 'F21': [], 'F32': []}
    for _ in range(nsim):
        y = rng.standard_normal(n)
        s1 = supF(y, trim, hac)['stat']
        res = bp_dp(y, 3, trim)
        fm = [wald_breaks(y, res[m][1], hac) for m in (1, 2, 3) if m in res]
        out['supF1'].append(s1); out['UDmax'].append(max(fm))
        out['F21'].append(sup_f_l1(y, res[1][1], trim, hac)[0])
        out['F32'].append(sup_f_l1(y, res[2][1], trim, hac)[0])
    return {k: np.array(v) for k, v in out.items()}

def bai_perron(y, nulld, trim=0.15, hac=True, alpha=0.05):
    y = np.asarray(y, float)
    res = bp_dp(y, 3, trim)
    mb, ml, ic = ic_select(y, res)
    s1 = supF(y, trim, hac)
    fm = {m: wald_breaks(y, res[m][1], hac) for m in (1, 2, 3) if m in res}
    ud = max(fm.values())
    p = lambda k, v: float((nulld[k] >= v).mean())
    f21, w21 = sup_f_l1(y, res[1][1], trim, hac)
    f32, w32 = sup_f_l1(y, res[2][1], trim, hac)
    # 序列法：supF(1) 顯著才看 2|1，依此類推
    m_seq = 0
    if p('supF1', s1['stat']) < alpha:
        m_seq = 1
        if p('F21', f21) < alpha:
            m_seq = 2
            if p('F32', f32) < alpha: m_seq = 3
    out = {'supF1': s1['stat'], 'supF1_k': s1['k'], 'p_supF1': p('supF1', s1['stat']), 'UDmax': ud, 'p_UDmax': p('UDmax', ud),
           'F21': f21, 'p_F21': p('F21', f21), 'F32': f32, 'p_F32': p('F32', f32),
           'm_BIC': mb, 'm_LWZ': ml, 'm_seq5': m_seq, 'ic': ic,
           'breaks_by_m': {m: res[m][1] for m in res}, 'ssr_by_m': {m: res[m][0] for m in res}}
    for lab, m in (('BIC', mb), ('LWZ', ml), ('seq', m_seq)):
        bks = res[m][1] if m > 0 else []
        mu, edges = seg_means(y, bks)
        ci = break_ci(y, bks)[0] if bks else []
        out[f'sel_{lab}'] = {'m': m, 'breaks': bks, 'seg_means': mu, 'ci90': ci}
    return out
