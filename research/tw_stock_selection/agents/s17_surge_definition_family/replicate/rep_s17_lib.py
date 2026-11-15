# -*- coding: utf-8 -*-
"""S17（SURGE 定義家族）獨立複製：自寫函式庫。
只依 s17_surge_definition_family/PREREG.json、data_check.json 與 FINDINGS.md 第 2 節（定義家族）、第 8 節（規格）的文字。
未讀、未 import 研究員的 s17lib／s10lib／s9lib／s4lib／s2lib 及其 run／prep／posthoc／prereg_write。
import 的現成程式：s14_corporate_events/replicate/rep_s14_lib.py（S14 複製員：0050 停牌日修正 attach_e50_fixed、
引擎 simulate_s14＝rep_s10 引擎＋listmax）→ rep_s10 → rep_s9 → … → rep_s1（S1 複製員的 load_rev、surge、rev_avail、統計）。
價格面板：s1 複製員 rep_s1_panel.py；創新板名稱：s3 複製員 rep_s3_names.py。

本檔自己寫的部分：
  1. family()：7 個 SURGE 定義（STD、W12、W36、RATE、SM3、NOMU、ROB）。
     營收矩陣＝rep_s1_lib.load_rev（千元、同代碼同月優先上市、營收 ≤ 0 缺值）。
     年增量：D(m)＝R(m)−R(m−12)（RATE：R(m)÷R(m−12)−1；SM3：R(m)+R(m−1)+R(m−2)−R(m−12)−R(m−13)−R(m−14)），
       用到 2012 年（含）以前營收的一律缺值（一般 D：m ≤ 2013-12；SM3：m ≤ 2014-02）。
     視窗 w＝D(m−W…m−1)，W＝24（W12：12；W36：36），視窗內全部有值且 D(m) 有值才算。
     μ＝Σw÷W；σ＝sqrt(Σ(w−μ)²÷(W−1))（逐窗兩段式，不用 pandas rolling）。
     σ ≤ 1e-9 × mean(|w|) → 缺值（全 0 視窗 mean|w|＝0 也缺值）。
     STD/W12/W36/RATE/SM3：(D−μ)÷σ；NOMU：D÷σ；ROB：(D−med)÷(1.4826×MAD)，MAD＝median(|w−med|)，尺度同樣的 σ≈0 規則。
     訊號月份 < 2016-01 一律缺值。
  2. py_check()：純 Python 迴圈逐格重算（驗收 A2）。
  3. make_lists_cnt()：rep_s3_lib.make_lists 同一邏輯，但「股池內有值 ≥ 50 檔」用另一個矩陣判斷（審查 B 用：限制後的 NOMU
     仍用原 NOMU 判斷是否換股）。
  4. month_paired()：兩條日淨值的逐月報酬差（月底對月底）在 [a, b] 月份區間的平均與 t。
  5. spearman_rows()：截面 Spearman（平均名次處理同分）。
"""
import sys, math
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/user/-/research/tw_stock_selection/agents/s14_corporate_events/replicate")
import rep_s14_lib as K14  # noqa: E402
K10 = K14.K10
K9 = K10.K9
S, Q, R, L = K14.S, K14.Q, K14.R, K14.L
ST, X2 = K14.ST, K14.X2
simulate = K14.simulate_s14
attach_e50_fixed = K14.attach_e50_fixed
attach_e50_orig = K14.attach_e50_orig

DEFS = ["STD", "W12", "W36", "RATE", "SM3", "NOMU", "ROB"]
EPS_REL = 1e-9
FIRST_SIG = 2016 * 12 + 0  # 2016-01


def yoy(Rv, mi0, kind):
    nm, N = Rv.shape
    D = np.full((nm, N), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        if kind == "amt":
            D[12:] = Rv[12:] - Rv[:-12]
            cut = 2013 * 12 + 11            # m ≤ 2013-12 用到 2012 年營收
        elif kind == "rate":
            D[12:] = Rv[12:] / Rv[:-12] - 1.0
            cut = 2013 * 12 + 11
        elif kind == "sm3":
            s3 = np.full((nm, N), np.nan)
            s3[2:] = Rv[2:] + Rv[1:-1] + Rv[:-2]
            D[12:] = s3[12:] - s3[:-12]
            cut = 2014 * 12 + 1             # m ≤ 2014-02 用到 2012 年營收（m−14）
        else:
            raise ValueError(kind)
    a = cut - mi0
    if a >= 0:
        D[:a + 1] = np.nan
    return D


def zscore(D, mi0, W=24, mode="z"):
    """mode：z＝(D−μ)÷σ；nomu＝D÷σ；rob＝(D−med)÷(1.4826 MAD)。回傳 (S, mu, sd)。"""
    nm, N = D.shape
    Sx = np.full((nm, N), np.nan); MU = np.full((nm, N), np.nan); SD = np.full((nm, N), np.nan)
    for m in range(W, nm):
        if mi0 + m < FIRST_SIG:
            continue
        w = D[m - W:m]
        ok = np.isfinite(w).all(axis=0) & np.isfinite(D[m])
        if not ok.any():
            continue
        ww = w[:, ok]; d = D[m, ok]
        absm = np.abs(ww).sum(axis=0) / W
        if mode == "rob":
            med = np.median(ww, axis=0)
            mad = np.median(np.abs(ww - med), axis=0)
            sc = 1.4826 * mad
            good = sc > EPS_REL * absm
            with np.errstate(invalid="ignore", divide="ignore"):
                z = np.where(good, (d - med) / sc, np.nan)
            mu, sd = med, sc
        else:
            mu = ww.sum(axis=0) / W
            sd = np.sqrt(((ww - mu) ** 2).sum(axis=0) / (W - 1))
            good = sd > EPS_REL * absm
            with np.errstate(invalid="ignore", divide="ignore"):
                z = np.where(good, ((d - mu) if mode == "z" else d) / sd, np.nan)
        Sx[m, ok] = z
        MU[m, ok] = np.where(good, mu, np.nan); SD[m, ok] = np.where(good, sd, np.nan)
    return Sx, MU, SD


def family(Rv, mi0):
    Da = yoy(Rv, mi0, "amt"); Dr = yoy(Rv, mi0, "rate"); D3 = yoy(Rv, mi0, "sm3")
    out = {}; aux = {}
    out["STD"], aux["mu"], aux["sd"] = zscore(Da, mi0, 24, "z")
    out["W12"] = zscore(Da, mi0, 12, "z")[0]
    out["W36"] = zscore(Da, mi0, 36, "z")[0]
    out["RATE"] = zscore(Dr, mi0, 24, "z")[0]
    out["SM3"] = zscore(D3, mi0, 24, "z")[0]
    out["NOMU"] = zscore(Da, mi0, 24, "nomu")[0]
    out["ROB"] = zscore(Da, mi0, 24, "rob")[0]
    aux["D_amt"] = Da
    # σ≈0 被剔除的格數（全市場、全部月份 ≥ 2016-01）
    return out, aux


def sigma0_count(Rv, mi0):
    """各定義：視窗與 D(m) 都有值、但 σ（或 MAD 尺度）≈0 被當缺值的格數（訊號月份 ≥ 2016-01）。"""
    Da = yoy(Rv, mi0, "amt"); Dr = yoy(Rv, mi0, "rate"); D3 = yoy(Rv, mi0, "sm3")
    spec = dict(STD=(Da, 24, "z"), W12=(Da, 12, "z"), W36=(Da, 36, "z"), RATE=(Dr, 24, "z"), SM3=(D3, 24, "z"),
                NOMU=(Da, 24, "nomu"), ROB=(Da, 24, "rob"))
    res = {}
    for k, (D, W, mode) in spec.items():
        nm = D.shape[0]; n0 = 0; cells = []
        for m in range(W, nm):
            if mi0 + m < FIRST_SIG:
                continue
            w = D[m - W:m]
            ok = np.isfinite(w).all(axis=0) & np.isfinite(D[m])
            ww = w[:, ok]
            absm = np.abs(ww).sum(axis=0) / W
            if mode == "rob":
                med = np.median(ww, axis=0); sc = 1.4826 * np.median(np.abs(ww - med), axis=0)
            else:
                mu = ww.sum(axis=0) / W; sc = np.sqrt(((ww - mu) ** 2).sum(axis=0) / (W - 1))
            bad = ~(sc > EPS_REL * absm)
            n0 += int(bad.sum())
            for j in np.where(ok)[0][bad]:
                cells.append((m, int(j)))
        res[k] = (n0, cells)
    return res


# ------------------------------------------------------------------ 純 Python 重算（A2）
def _d_py(Rv, mi0, kind, m, j):
    def r(k):
        if k < 0:
            return None
        v = float(Rv[k, j])
        return v if v == v else None
    if kind == "amt" or kind == "rate":
        if mi0 + m <= 2013 * 12 + 11:
            return None
        a, b = r(m), r(m - 12)
        if a is None or b is None:
            return None
        return a - b if kind == "amt" else a / b - 1.0
    if mi0 + m <= 2014 * 12 + 1:
        return None
    xs = [r(m - k) for k in (0, 1, 2, 12, 13, 14)]
    if any(x is None for x in xs):
        return None
    return (xs[0] + xs[1] + xs[2]) - (xs[3] + xs[4] + xs[5])


def py_score(Rv, mi0, name, m, j):
    kind = {"RATE": "rate", "SM3": "sm3"}.get(name, "amt")
    W = {"W12": 12, "W36": 36}.get(name, 24)
    if mi0 + m < FIRST_SIG or m < W:
        return None
    d = _d_py(Rv, mi0, kind, m, j)
    if d is None:
        return None
    w = []
    for k in range(m - W, m):
        x = _d_py(Rv, mi0, kind, k, j)
        if x is None:
            return None
        w.append(x)
    absm = sum(abs(x) for x in w) / W
    if name == "ROB":
        s = sorted(w)
        med = (s[W // 2 - 1] + s[W // 2]) / 2.0
        a = sorted(abs(x - med) for x in w)
        mad = (a[W // 2 - 1] + a[W // 2]) / 2.0
        sc = 1.4826 * mad
        if not sc > EPS_REL * absm:
            return None
        return (d - med) / sc
    mu = sum(w) / W
    sd = math.sqrt(sum((x - mu) ** 2 for x in w) / (W - 1))
    if not sd > EPS_REL * absm:
        return None
    return (d / sd) if name == "NOMU" else (d - mu) / sd


# ------------------------------------------------------------------ 名單
def make_lists_cnt(P, S_, Scnt, mi0, shift=0, excl=None, min_n=50, ma_len=60, ret_len=20):
    """rep_s3_lib.make_lists，但「股池內有值 < min_n 不換股」用 Scnt 判斷。"""
    out = {}
    codes = np.array(P.codes)
    mav = R.ma(P, ma_len)
    for m in range(S_.shape[0]):
        d10, j, s0 = L.rev_avail(P, mi0 + m)
        s = s0 + shift
        if s >= P.T - 1 or s0 < P.t0 or s0 >= P.T - 1:
            continue
        sv = S_[m]
        u0 = R.universe(P, s0)
        if excl is not None:
            u0 = u0 & ~excl(s0)
        if int((u0 & np.isfinite(Scnt[m])).sum()) < min_n:
            continue
        u = R.universe(P, s)
        if excl is not None:
            u = u & ~excl(s)
        with np.errstate(invalid="ignore", divide="ignore"):
            ok = u & np.isfinite(sv) & (P.G[s] > mav[s])
            r20 = P.G[s] / P.G[s - ret_len] - 1.0
        idx = np.where(ok)[0]
        order = idx[np.lexsort((codes[idx], -sv[idx]))]
        out[s] = dict(lma=order, r20=r20, m=m, n_surge=int((u & np.isfinite(sv)).sum()))
    return out


# ------------------------------------------------------------------ 統計
def month_rets(P, nav):
    s = pd.Series(np.asarray(nav[P.t0:], float), index=P.dates[P.t0:])
    me = s.groupby(s.index.to_period("M")).last()
    r = me / me.shift(1) - 1.0
    return r.iloc[1:]


def month_paired(P, nav_a, nav_b, a, b=None):
    d = month_rets(P, nav_a) - month_rets(P, nav_b)
    d = d[d.index >= pd.Period(a, "M")]
    if b is not None:
        d = d[d.index <= pd.Period(b, "M")]
    n = len(d); mu = float(d.mean()); sd = float(d.std(ddof=1))
    return dict(n=n, mean=mu, t=mu / (sd / math.sqrt(n)), n_pos=int((d > 0).sum()), first=str(d.index[0]),
                last=str(d.index[-1]))


def _rank_avg(x):
    return pd.Series(x).rank(method="average").values


def spearman(a, b):
    ra = _rank_avg(a); rb = _rank_avg(b)
    ra = ra - ra.mean(); rb = rb - rb.mean()
    return float((ra * rb).sum() / math.sqrt((ra * ra).sum() * (rb * rb).sum()))
