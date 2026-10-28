#!/usr/bin/env python3
"""自寫的資料窺探檢定（White RC、Hansen SPA l/c/u、Romano-Wolf StepM、Deflated Sharpe）與有效試驗數。
記號：d[t,k]＝成員 k 在窗 t 對基準的超額（越大越好）；虛無假設 H0：max_k E d_k ≤ 0。
平穩 bootstrap（Politis-Romano 1994）：平均區塊長度 L，幾何分布區塊，環狀接續；同一組索引用於所有成員（保留相關）。
"""
import math
import numpy as np
from scipy import stats

def sb_counts(n, L, B, seed):
    """回傳 B×n 的抽樣次數矩陣（每列總和 n）。"""
    rng = np.random.default_rng(seed)
    p = 1.0 / L
    idx = np.empty((B, n), np.int64)
    idx[:, 0] = rng.integers(0, n, B)
    for t in range(1, n):
        new = rng.random(B) < p
        idx[:, t] = np.where(new, rng.integers(0, n, B), (idx[:, t - 1] + 1) % n)
    W = np.zeros((B, n), np.float64)
    rows = np.repeat(np.arange(B), n)
    np.add.at(W, (rows, idx.ravel()), 1.0)
    return W

def boot_means(W, D, chunk=1000):
    """W：B×n 次數；D：n×N（不可有 NaN）。回傳 B×N 的 bootstrap 平均（float32 以省記憶體）。"""
    n = D.shape[0]
    out = np.empty((W.shape[0], D.shape[1]), np.float32)
    for s in range(0, W.shape[0], chunk):
        out[s:s + chunk] = (W[s:s + chunk] @ D / n).astype(np.float32)
    return out

def lrv(D, kind, L):
    """各欄 √n·平均 的長期變異數估計 ω²。kind：'pr'＝與平穩 bootstrap 對應的 Politis-Romano 核（Hansen 2005 建議）；
       'nw'＝Bartlett 核、落後 L；'iid'＝樣本變異數。"""
    n = D.shape[0]; e = D - D.mean(axis=0)
    g0 = (e * e).sum(axis=0) / n
    if kind == 'iid': return g0
    s = g0.copy()
    if kind == 'pr':
        p = 1.0 / L
        for i in range(1, n):
            k = (n - i) / n * (1 - p) ** i + i / n * (1 - p) ** (n - i)
            if k < 1e-8 and i > 5 * L: break
            s += 2 * k * (e[i:] * e[:-i]).sum(axis=0) / n
    elif kind == 'nw':
        lag = int(L)
        for i in range(1, lag + 1):
            s += 2 * (1 - i / (lag + 1)) * (e[i:] * e[:-i]).sum(axis=0) / n
    else:
        raise ValueError(kind)
    return np.maximum(s, 1e-12 * np.maximum(g0, 1e-30))

def mcp_tests(D, W, L, var='pr', Mb=None, focus=None):
    """D：n×N 超額（無 NaN）；W：bootstrap 次數。回傳 RC、SPA_l/c/u、StepM（focus 欄的調整後 p）與統計量。
       var：ω 的估計法（'pr'、'nw'、'boot'）。"""
    n, N = D.shape
    dbar = D.mean(axis=0)
    if Mb is None: Mb = boot_means(W, D)
    B = Mb.shape[0]
    if var == 'boot':
        om2 = n * Mb.astype(np.float64).var(axis=0, ddof=0)
    else:
        om2 = lrv(D, var, L)
    om = np.sqrt(om2)
    sn = math.sqrt(n)
    tst = sn * dbar / om
    # White RC（不標準化）
    V = sn * dbar.max()
    Vb = sn * (Mb - dbar[None, :].astype(np.float32)).max(axis=1)
    p_rc = float((Vb >= V).mean())
    p_rc_focus = float((Vb >= sn * dbar[focus]).mean()) if focus is not None else None
    # 不標準化的 SPA（lower／consistent／upper；upper＝White RC）：供與 arch 8.0（其 SPA 統計量不標準化）核對
    thr0 = -np.sqrt(lrv(D, 'pr', L) / n * 2 * math.log(math.log(n)))
    _ns = {}
    for k, gk in (('l', np.maximum(dbar, 0.0)), ('c', np.where(dbar >= thr0, dbar, 0.0)), ('u', dbar)):
        Vk = sn * (Mb - gk[None, :].astype(np.float32)).max(axis=1)
        _ns[k] = float((Vk > V).mean())
    # Hansen SPA
    T = max(0.0, float(tst.max()))
    thr = -np.sqrt(om2 / n * 2 * math.log(math.log(n)))
    g = {'l': np.maximum(dbar, 0.0), 'c': np.where(dbar >= thr, dbar, 0.0), 'u': dbar}
    out = {'n': n, 'N': N, 'max_t': float(tst.max()), 'argmax': int(np.argmax(tst)), 'RC_stat': float(V), 'p_RC': p_rc, 'p_RC_focus': p_rc_focus, 'SPA_stat': T,
           'p_SPAns_l': _ns['l'], 'p_SPAns_c': _ns['c'], 'p_SPAns_u': _ns['u']}
    inv_om = (sn / om).astype(np.float32)
    for k, gk in g.items():
        Z = (Mb - gk[None, :].astype(np.float32)) * inv_om[None, :]
        Tb = np.maximum(Z.max(axis=1), 0.0)
        out[f'p_SPA_{k}'] = float((Tb >= T).mean()) if T > 0 else 1.0
        if focus is not None:
            tf = float(tst[focus])
            # 焦點策略的資料窺探校正 p：以焦點自己的標準化統計量對照「族群最大值」的重新置中 bootstrap 分布
            out[f'p_SPA_{k}_focus'] = float((Tb >= tf).mean()) if tf > 0 else 1.0
    out['n_kept_SPA_c'] = int((dbar >= thr).sum())
    # Romano-Wolf StepM（標準化、以 d̄ 置中）
    Zc = (Mb - dbar[None, :].astype(np.float32)) * inv_om[None, :]
    order = np.argsort(-tst)
    out['top_t'] = [float(tst[j]) for j in order[:10]]
    out['top_idx'] = [int(j) for j in order[:10]]
    if focus is not None:
        r = int(np.where(order == focus)[0][0])
        out['focus_rank'] = r + 1
        out['focus_t'] = float(tst[focus])
        top = order[:r + 1]
        rest = np.ones(N, bool); rest[top] = False
        mrest = Zc[:, rest].max(axis=1) if rest.any() else np.full(B, -np.inf)
        # 逐步調整 p：由排名第 r 往第 1 累積「其餘集合」的最大值（O(B·r)；與逐一重算子集合最大值的結果完全相同）
        padj = 0.0; run = mrest.copy()
        for i in range(r, -1, -1):
            run = np.maximum(run, Zc[:, top[i]])
            padj = max(padj, float((run >= tst[top[i]]).mean()))
        out['p_StepM_focus'] = padj
        # 單一檢定（無多重檢定校正）的 bootstrap p
        out['p_single_focus'] = float((Zc[:, focus] >= tst[focus]).mean())
    # StepM 5%：第一步臨界值與拒絕數
    c1 = float(np.quantile(Zc.max(axis=1), 0.95))
    rej = tst > c1; S = ~rej; steps = 1
    while rej.any() and S.any():
        c = float(np.quantile(Zc[:, S].max(axis=1), 0.95))
        new = S & (tst > c)
        if not new.any(): break
        rej |= new; S = ~rej; steps += 1
    out['StepM_5pct_n_reject'] = int(rej.sum()); out['StepM_crit_step1'] = c1
    if focus is not None: out['StepM_5pct_rejects_focus'] = bool(rej[focus])
    return out

def eff_trials(D):
    """相關矩陣特徵值法的有效試驗數：Nyholt（2004）與 Li-Ji（2005）。D：n×N（無 NaN）。"""
    n, N = D.shape
    Z = (D - D.mean(axis=0)) / D.std(axis=0, ddof=1)
    G = Z @ Z.T / (n - 1)                 # n×n，非零特徵值與 N×N 相關矩陣相同
    lam = np.clip(np.linalg.eigvalsh(G), 0, None)
    lam_all = np.concatenate([lam, np.zeros(max(N - n, 0))]) if N > n else lam
    M = N
    var_l = float(np.mean(lam_all ** 2) - np.mean(lam_all) ** 2)
    nyholt = 1 + (M - 1) * (1 - var_l / M)
    liji = float(np.sum((lam_all >= 1).astype(float) + (lam_all - np.floor(lam_all))))
    return {'N': int(N), 'M_eff_Nyholt': float(nyholt), 'M_eff_LiJi': liji, 'top_eig_share': float(lam.max() / N)}

def dsr(x, sr_all, n_trials):
    """Deflated Sharpe Ratio（Bailey & López de Prado 2014）。x：焦點策略的超額序列；sr_all：族群各成員的 SR（同頻率）。"""
    x = np.asarray(x, float); T = len(x)
    sr = x.mean() / x.std(ddof=1)
    g3 = float(stats.skew(x)); g4 = float(stats.kurtosis(x, fisher=False))
    v = float(np.var(sr_all, ddof=1))
    emc = 0.5772156649
    N = max(float(n_trials), 2.0)
    sr0 = math.sqrt(v) * ((1 - emc) * stats.norm.ppf(1 - 1 / N) + emc * stats.norm.ppf(1 - 1 / (N * math.e)))
    z = (sr - sr0) * math.sqrt(T - 1) / math.sqrt(max(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2, 1e-12))
    return {'SR': float(sr), 'SR0': float(sr0), 'skew': g3, 'kurt': g4, 'var_SR': v, 'N_used': N, 'DSR': float(stats.norm.cdf(z)), 'p_DSR': float(1 - stats.norm.cdf(z))}
