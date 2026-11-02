#!/usr/bin/env python3
"""前瞻序貫規則族 R1～R4、三種資料生成過程（DGP）與操作特性（OC）模擬。前瞻監測（make_forward.py）與 OC 模擬共用本檔，確保同一套數學。

符號：x_t＝第 t 個前瞻月的決策線月超額（pp）；S_n＝Σ_{t≤n} x_t；σ＝事前凍結的設計 sd（pp／月）；z_n＝S_n／(σ√n)。
所有規則：第 12 個月之前不做任何動作；先觸發者吸收（一旦升級或下修，該路徑後續不再判定）。
R1 貝氏後驗門檻（6 個月檢視一次：12、18、…）：先驗＝格點密度（校準後），概似＝常態已知 σ；
    升級：P(α > α_min | 資料) ≥ up_P；下修：P(α < 0 | 資料) ≥ down_P。（門檻固定，不校準）
R2 群組序貫（6 個月一次）：升級＝O'Brien-Fleming 型 z_n ≥ c_u·√(H_c/n)（等價於 S_n ≥ c_u·σ·√H_c）；下修＝Pocock 型 z_n ≤ −c_d。
R3 單邊 CUSUM 下修：C_0＝0，C_t＝min(0, C_{t−1} ＋ (x_t − k_ref)/σ)，k_ref＝−0.5pp；C_t ≤ −h（t ≥ 12）即下修；升級同 R1。
R4 固定檢視（12、24、36、48、60…每 12 個月）：升級 z_n ≥ c_u4；下修 z_n ≤ −c_d4。
校準（事前寫死）：c_u、c_d、h、c_u4、c_d4 用設計 DGP（iid 常態、σ）200,000 條路徑、seed 固定，以分位數精確解：
    α＝0 時 H_c 個月內誤升級＝target_up；α＝−1 時 H_c 個月內下修＝target_pow（吸收前的邊際機率，所以升級門檻偏保守）。
DGP：D1 iid 常態；D2 參考期殘差的 6 個月區塊 bootstrap（環狀、不縮放）；D3 t(4)（單位變異）×波動狀態（乘數 0.7／1.4，
    高波動穩態機率 0.347，高→低 0.06、低→高 0.0319；E[乘數²]＝1）。
"""
import math
import numpy as np
from scipy import stats

ALPHAS = [-1.0, -0.5, 0.0, 0.5, 1.0]
REG_M = np.array([0.7, 1.4]); REG_P_HL, REG_P_LH = 0.06, 0.0319


def gen(dgp, alpha, sigma, P, H, rng, resid=None, block=6):
    if dgp == 'D1':
        return alpha + sigma * rng.standard_normal((P, H))
    if dgp == 'D2':
        e = np.asarray(resid, float); e = e - e.mean(); n = len(e)
        nb = -(-H // block)
        st = rng.integers(0, n, (P, nb))
        idx = (st[:, :, None] + np.arange(block)[None, None, :]) % n
        return alpha + e[idx.reshape(P, nb * block)[:, :H]]
    if dgp == 'D3':
        ph = REG_P_LH / (REG_P_LH + REG_P_HL)
        s = (rng.random(P) < ph).astype(int)
        out = np.empty((P, H))
        tt = stats.t.rvs(4, size=(P, H), random_state=rng) / math.sqrt(2.0)
        for t in range(H):
            out[:, t] = alpha + sigma * REG_M[s] * tt[:, t]
            u = rng.random(P)
            s = np.where(s == 1, np.where(u < REG_P_HL, 0, 1), np.where(u < REG_P_LH, 1, 0))
        return out
    raise ValueError(dgp)


def looks6(H, first=12):
    return list(range(first, H + 1, 6))


def looks12(H, first=12):
    return list(range(first, H + 1, 12))


def _z(X, sigma):
    S = np.cumsum(X, axis=1); n = np.arange(1, X.shape[1] + 1)[None, :]
    return S / (sigma * np.sqrt(n)), S


def _first(mask):
    """mask：P×H 布林（t＝1..H）；回傳第一次為真的月份（從未＝0）。"""
    any_ = mask.any(axis=1)
    return np.where(any_, mask.argmax(axis=1) + 1, 0)


def _lookmask(H, looks):
    m = np.zeros(H, bool); m[[l - 1 for l in looks if l <= H]] = True
    return m


def post_probs(X, sigma, prior, th, looks, alpha_min):
    """回傳 dict(look → (P(α>α_min), P(α<0)))，各為長度 P 的陣列。"""
    p = np.asarray(prior, float); p = p / p.sum(); keep = p > 1e-12
    thk, lp = th[keep], np.log(p[keep])
    S = np.cumsum(X, axis=1)
    out = {}
    up_m = thk > alpha_min; dn_m = thk < 0
    for n in looks:
        if n > X.shape[1]: continue
        s = S[:, n - 1][:, None]
        ll = lp[None, :] + (thk[None, :] * s - 0.5 * n * thk[None, :] ** 2) / sigma ** 2
        w = np.exp(ll - ll.max(axis=1, keepdims=True)); w /= w.sum(axis=1, keepdims=True)
        out[n] = ((w * up_m).sum(axis=1), (w * dn_m).sum(axis=1))
    return out


def prior_probs(prior, th, alpha_min):
    p = np.asarray(prior, float); p = p / p.sum()
    return float(p[th > alpha_min].sum()), float(p[th < 0].sum()), float(p[th > 0].sum())


def raw_triggers(rule, X, sigma, par, prior=None, th=None):
    """回傳 (up_mask, down_mask)：P×H 布林，不含吸收。"""
    P_, H = X.shape
    z, S = _z(X, sigma)
    up = np.zeros((P_, H), bool); dn = np.zeros((P_, H), bool)
    if rule in ('R1', 'R3'):
        L = looks6(H)
        pp = post_probs(X, sigma, prior, th, L, par['alpha_min'])
        for n, (pu, pd_) in pp.items():
            up[:, n - 1] = pu >= par['up_P']
            if rule == 'R1': dn[:, n - 1] = pd_ >= par['down_P']
    if rule == 'R2':
        lm = _lookmask(H, looks6(H)); n = np.arange(1, H + 1)
        bu = par['c_u'] * np.sqrt(par['H_c'] / n)
        up[:, lm] = (z >= bu[None, :])[:, lm]
        dn[:, lm] = (z <= -par['c_d'])[:, lm]
    if rule == 'R3':
        C = np.zeros(P_)
        for t in range(H):
            C = np.minimum(0.0, C + (X[:, t] - par['k_ref']) / sigma)
            if t + 1 >= 12: dn[:, t] = C <= -par['h']
    if rule == 'R4':
        lm = _lookmask(H, looks12(H))
        up[:, lm] = (z >= par['c_u4'])[:, lm]
        dn[:, lm] = (z <= -par['c_d4'])[:, lm]
    if rule == 'E7':      # o7 原規則（e-process），作為迭代基準
        import fslib; seqlib = fslib.SQ
        st = seqlib.path_stats(X, prior, th, sigma, par['alpha_min_e'])
        tr = seqlib.triggers(st, par['thr'])
        up, dn = tr['upgrade'], tr['down']
    return up, dn


def first_events(up, dn):
    tu, td = _first(up), _first(dn)
    u = np.where((tu > 0) & ((td == 0) | (tu < td)), tu, 0)
    d = np.where((td > 0) & ((tu == 0) | (td < tu)), td, 0)
    return u, d


def oc_curve(u, d, horizons):
    return {int(h): dict(up=float(np.mean((u > 0) & (u <= h))), down=float(np.mean((d > 0) & (d <= h)))) for h in horizons}


# ---------------------------------------------------------------- 設計校準
def calibrate(sigma, H_c, target_up, target_pow, Hsim, P=200_000, seed=20261102, k_ref=-0.5):
    rng = np.random.default_rng(seed)
    X0 = gen('D1', 0.0, sigma, P, Hsim, rng)
    X1 = gen('D1', -1.0, sigma, P, Hsim, rng)
    out = {}
    L6 = [l for l in looks6(Hsim) if l <= H_c]; L12 = [l for l in looks12(Hsim) if l <= H_c]
    z0, _ = _z(X0, sigma); z1, _ = _z(X1, sigma)
    n = np.arange(1, Hsim + 1)
    # R2：OBF 升級、Pocock 下修
    M0 = (z0[:, [l - 1 for l in L6]] * np.sqrt(np.array(L6) / H_c)[None, :]).max(axis=1)
    out['c_u'] = float(np.quantile(M0, 1 - target_up))
    m1 = z1[:, [l - 1 for l in L6]].min(axis=1)
    out['c_d'] = float(-np.quantile(m1, target_pow))
    # R4
    out['c_u4'] = float(np.quantile(z0[:, [l - 1 for l in L12]].max(axis=1), 1 - target_up))
    out['c_d4'] = float(-np.quantile(z1[:, [l - 1 for l in L12]].min(axis=1), target_pow))
    # R3：CUSUM
    C = np.zeros(P); mn = np.full(P, np.inf)
    for t in range(H_c):
        C = np.minimum(0.0, C + (X1[:, t] - k_ref) / sigma)
        if t + 1 >= 12: mn = np.minimum(mn, C)
    out['h'] = float(-np.quantile(mn, target_pow))
    out['k_ref'] = k_ref; out['H_c'] = H_c
    return out


def run_oc(rules, par, sigma, prior, th, dgps, P, Hsim, seed, resid=None, horizons=None, alphas=ALPHAS):
    horizons = horizons or list(range(12, Hsim + 1, 6))
    res = {}
    for dgp in dgps:
        for a in alphas:
            rng = np.random.default_rng([seed, int(round((a + 2) * 100)), ord(dgp[-1])])
            X = gen(dgp, a, sigma, P, Hsim, rng, resid=resid)
            for r in rules:
                up, dn = raw_triggers(r, X, sigma, par, prior, th)
                u, d = first_events(up, dn)
                res[(r, dgp, a)] = oc_curve(u, d, horizons)
    return res


def oc_table(res):
    rows = []
    for (r, dgp, a), cur in res.items():
        for h, v in cur.items():
            rows.append(dict(rule=r, dgp=dgp, alpha=a, h=h, up=v['up'], down=v['down']))
    return rows


def check_constraints(res, rule, dgps, H_c=36, pow_min=0.50, up_max=0.05):
    ok, detail = True, {}
    for dgp in dgps:
        pw = res[(rule, dgp, -1.0)][H_c]['down']; fu = res[(rule, dgp, 0.0)][H_c]['up']; fd = res[(rule, dgp, 0.0)][H_c]['down']
        pu1 = res[(rule, dgp, 1.0)][H_c]['up']
        detail[dgp] = dict(power_down_a_m1=pw, false_up_a0=fu, false_down_a0=fd, power_up_a_p1=pu1)
        ok &= (pw >= pow_min) and (fu <= up_max)
    return bool(ok), detail
