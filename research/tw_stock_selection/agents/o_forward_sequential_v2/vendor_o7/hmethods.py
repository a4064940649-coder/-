#!/usr/bin/env python3
"""階層收縮方法族（PREREG_DESIGN 的 V1～V7）。輸入一個族群矩陣（n 窗 × K 成員，pp）、群標籤、medoid，
對任一焦點成員（或外部序列＋指定群）回傳 α 的格點事後密度。
V1_o6   G2×medoids，焦點直接收縮（o6 主欄）
V2_HN   G2 群間＋群內常態 δ~N(0,2σ_w²)
V3_G2MAX V1＋群內最大值偏誤 b（成員絕對序列）
V4_HNMAX V2＋群內最大值偏誤 b（差序列）
V5_HT   G2 群間＋群內 t3
V6_HNG3 G3 群間＋群內常態
V7_HNC  V2＋大群（m_c ≥ 10）用自己的 σ_w
"""
import math
import numpy as np, pandas as pd
from scipy import stats
import tlib as T

VERSIONS = ['V1_o6', 'V2_HN', 'V3_G2MAX', 'V4_HNMAX', 'V5_HT', 'V6_HNG3', 'V7_HNC']
NB_Q = 200          # 最大值偏誤分布取 200 個分位點做混合

class Fam:
    """一個族群（可為合成）上的共用量：medoid 分解、G2／G3 先驗、σ_w 動差估計。"""
    def __init__(self, X, labels, medoids, G=T.G_DEFAULT, c0=0.1, L=6, need_g3=True, sw_scale=1.0, exclude_clusters=()):
        self.X = X if isinstance(X, np.ndarray) else np.asarray(X, float)
        self.cols = list(labels.index) if isinstance(labels, pd.Series) else None
        self.lab = np.asarray(labels, int)
        self.med = dict(medoids)                    # cluster → 欄位索引
        self.G, self.L = G, L
        self.th = G.th
        excl = set(exclude_clusters)
        mcl = [c for c in sorted(self.med) if c not in excl]
        self.mcl = mcl
        A = self.X[:, [self.med[c] for c in mcl]]
        self.mser = A.mean(axis=1)                  # 族群（medoid）平均序列
        xhat = A.mean(axis=0)
        s2 = T.hac_var_mean(A - self.mser[:, None], L)
        self.vP = float(T.hac_var_mean(self.mser, L))
        self.g2 = T.fit_g2(xhat, s2, c0=c0, G=G)
        self.g3 = T.fit_g3(xhat, s2, G=G) if need_g3 else None
        self.prior_xhat, self.prior_s2 = xhat, s2
        # σ_w 動差估計（m_c ≥ 3 的群，拿掉 exclude）。PREREG_DESIGN_AMEND1：
        #   群內非 medoid 成員 d_k 的橫斷面變異＝σ_w²（v_med 對同群所有 d_k 是共同平移）＋雜訊的橫斷面離散；
        #   雜訊離散＝群內「差序列減群內平均差序列」的 HAC L6 變異平均（×m/(m−1)），不是單一差序列的雜訊變異。
        num = den = nz = 0.0
        self.sw_c = {}
        for c in sorted(self.med):
            if c in excl: continue
            ii = np.where(self.lab == c)[0]
            if len(ii) < 3: continue
            nm = ii[ii != self.med[c]]
            Dm = self.X[:, nm] - self.X[:, [self.med[c]]]
            dk = Dm.mean(axis=0)
            Dc = Dm - Dm.mean(axis=1, keepdims=True)
            nv = float(np.mean(T.hac_var_mean(Dc, L))) * len(nm) / (len(nm) - 1)
            v = float(np.var(dk, ddof=1))
            w = len(ii) - 1
            num += v * w; nz += nv * w; den += w
            if len(ii) >= 10:
                self.sw_c[c] = math.sqrt(max(v - nv, 0.0))
        self.sigma_w = sw_scale * math.sqrt(max(num / den - nz / den, 0.0)) if den > 0 else 0.0
        self.sw_c = {c: sw_scale * v for c, v in self.sw_c.items()}

    # ------------------------------------------------ 基本件
    def _xs(self, x):
        x = np.asarray(x, float)
        return float(x.mean()), float(T.hac_var_mean(x - self.mser, self.L))

    def post_direct(self, x, g='g2', shift=0.0):
        xh, s2 = self._xs(x)
        pr = self.g2 if g == 'g2' else self.g3
        return T.posterior_g(pr['g'], self.th, xh - shift, s2, self.vP, self.G.step)

    def post_medoid(self, c, g='g2'):
        return self.post_direct(self.X[:, self.med[c]], g)

    def _d(self, x, c):
        dser = np.asarray(x, float) - self.X[:, self.med[c]]
        return float(dser.mean()), float(T.hac_var_mean(dser, self.L))

    def _add_normal(self, dens, mu, var):
        d = T.shift_dens(dens, self.th, mu)
        return T.convolve_dens(d, math.sqrt(var), self.G.step) if var > 0 else d

    def maxbias(self, c, kind='abs', B=400, seed=0, extra=None):
        ii = np.where(self.lab == c)[0]
        Xc = self.X[:, ii]
        if extra is not None: Xc = np.column_stack([Xc, extra])
        if Xc.shape[1] < 2: return np.zeros(NB_Q)
        if kind == 'abs':
            b = T.max_noise_boot(Xc, B=B, L=self.L, seed=seed)
        else:
            b = T.max_noise_boot_diff(Xc, self.X[:, self.med[c]], B=B, L=self.L, seed=seed)
        return np.quantile(b, (np.arange(NB_Q) + 0.5) / NB_Q)

    # ------------------------------------------------ 方法
    def posterior(self, version, x, c, is_medoid=False, bq=None, seed=0, extra_members=None):
        """x：焦點序列（pp，n）；c：焦點所屬群；is_medoid：焦點就是 medoid；bq：預先算好的偏誤分位點（可省略）。
        extra_members：外部新群時，群內成員矩陣（n×m）由呼叫端另外處理（見 posterior_newcluster）。"""
        th = self.th
        if version == 'V1_o6':
            return self.post_direct(x)
        if version == 'V3_G2MAX':
            if bq is None: bq = self.maxbias(c, 'abs', seed=seed)
            xh, s2 = self._xs(x)
            acc = np.zeros_like(th)
            for b in bq:
                acc += T.posterior_g(self.g2['g'], th, xh - b, s2, self.vP, self.G.step)
            return acc / acc.sum()
        g = 'g3' if version == 'V6_HNG3' else 'g2'
        medd = self.post_medoid(c, g)
        if is_medoid:
            return medd
        sw = self.sw_c.get(c, self.sigma_w) if version == 'V7_HNC' else self.sigma_w
        d, sd2 = self._d(x, c)
        tau2 = 2 * sw * sw
        if version in ('V2_HN', 'V6_HNG3', 'V7_HNC'):
            if tau2 <= 0: return medd
            Bf = tau2 / (tau2 + sd2)
            return self._add_normal(medd, Bf * d, Bf * sd2)
        if version == 'V4_HNMAX':
            if bq is None: bq = self.maxbias(c, 'diff', seed=seed)
            if tau2 <= 0: return medd
            Bf = tau2 / (tau2 + sd2)
            acc = np.zeros_like(th)
            for b in bq:
                acc += self._add_normal(medd, Bf * (d - b), Bf * sd2)
            return acc / acc.sum()
        if version == 'V5_HT':
            if tau2 <= 0: return medd
            step = self.G.step
            W = 4.0
            dg = np.round(np.arange(-W, W + step / 2, step), 10)
            scale = math.sqrt(tau2) / math.sqrt(3.0)
            lp = stats.t.logpdf(dg / scale, 3) - 0.5 * (d - dg) ** 2 / sd2
            pd_ = np.exp(lp - lp.max()); pd_ = pd_ / pd_.sum()
            full = np.convolve(medd, pd_, mode='full')
            gx = th[0] + dg[0] + np.arange(len(full)) * step
            out = np.interp(th, gx, full, left=0.0, right=0.0)
            return out / out.sum()
        raise ValueError(version)

def posterior_newcluster(fam, version, members, focus_idx, seed=0):
    """外部新群（本輪新策略的版本族）：members＝n×m（pp），focus_idx＝凍結版的欄位。
    新群的 medoid＝群內平均相關最高者；群水準先驗用 fam 的 g_B（不重估），σ_w 用 fam 的合併值。"""
    M = np.asarray(members, float)
    C = np.corrcoef(M, rowvar=False)
    mi = int(np.argmax(C.mean(axis=1))) if M.shape[1] > 1 else 0
    th = fam.th
    x = M[:, focus_idx]
    if version == 'V1_o6':
        return fam.post_direct(x), mi
    if version == 'V3_G2MAX':
        b = np.quantile(T.max_noise_boot(M, B=400, L=fam.L, seed=seed), (np.arange(NB_Q) + 0.5) / NB_Q) if M.shape[1] > 1 else np.zeros(1)
        xh, s2 = fam._xs(x)
        acc = np.zeros_like(th)
        for bb in b: acc += T.posterior_g(fam.g2['g'], th, xh - bb, s2, fam.vP, fam.G.step)
        return acc / acc.sum(), mi
    g = 'g3' if version == 'V6_HNG3' else 'g2'
    medd = fam.post_direct(M[:, mi], g)
    if focus_idx == mi: return medd, mi
    dser = x - M[:, mi]; d = float(dser.mean()); sd2 = float(T.hac_var_mean(dser, fam.L))
    tau2 = 2 * fam.sigma_w ** 2
    if tau2 <= 0: return medd, mi
    Bf = tau2 / (tau2 + sd2)
    if version == 'V4_HNMAX':
        b = np.quantile(T.max_noise_boot_diff(M, M[:, mi], B=400, L=fam.L, seed=seed), (np.arange(NB_Q) + 0.5) / NB_Q)
        acc = np.zeros_like(th)
        for bb in b: acc += fam._add_normal(medd, Bf * (d - bb), Bf * sd2)
        return acc / acc.sum(), mi
    if version == 'V5_HT':
        step = fam.G.step; dg = np.round(np.arange(-4.0, 4.0 + step / 2, step), 10)
        scale = math.sqrt(tau2) / math.sqrt(3.0)
        lp = stats.t.logpdf(dg / scale, 3) - 0.5 * (d - dg) ** 2 / sd2
        p_ = np.exp(lp - lp.max()); p_ /= p_.sum()
        full = np.convolve(medd, p_, mode='full'); gx = th[0] + dg[0] + np.arange(len(full)) * step
        out = np.interp(th, gx, full, left=0.0, right=0.0); return out / out.sum(), mi
    return fam._add_normal(medd, Bf * d, Bf * sd2), mi

def load_real():
    """真實族群：X（DataFrame）、labels（Series，index＝member_id）、medoid（cluster→欄位索引）。"""
    X, M = T.load_family()
    cl = T.load_clusters().set_index('member_id').loc[X.columns]
    lab = cl['cluster'].astype(int)
    cols = list(X.columns)
    med = {int(c): cols.index(m) for c, m in cl[cl.is_medoid]['cluster'].items() for m in [c]} if False else \
          {int(r.cluster): cols.index(mid) for mid, r in cl[cl.is_medoid].iterrows()}
    return X, lab, med
