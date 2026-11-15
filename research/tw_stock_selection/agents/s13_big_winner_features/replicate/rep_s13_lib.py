# -*- coding: utf-8 -*-
"""S13（大贏家能否事前辨識）獨立複製：自寫函式庫。
只依 s13_big_winner_features/PREREG.json、FROZEN.json、data_check.json 與 FINDINGS.md 第 1 節的文字規格。
未讀、未 import 研究員的 s13lib／s12lib／s10lib／s9lib 及其 train／validate／posthoc／prep／s0_datacheck／prereg_write；
未 import o_pead 的程式（只讀它的資料檔 fin_long.parquet、avail_dates.parquet：rep_s11_lib.load_fin 讀格式與可得日，
本檔另讀 gp_s、rev_s 算毛利率）。
import 的現成程式：s12_surge_size_tiers/replicate/rep_s12_lib.py（S12 複製員）→ rep_s11 → rep_s10 → rep_s9 → … → rep_s1。

本檔自己寫的部分：
  1. candidates：rep_s3 make_lists（位移 0、排除創新板）每個訊號列的 L_MA 前 20 名；F1＝rep_s3 cell_plans('C10')。
  2. features：ACC、REVHI12、GMYOY、NEAR52、RET60、ADVCHG（定義見各函式）。
  3. outcome_e0：E0 持有期超額（同 rep_s9 引擎的出場時序：下市檢查 → 待賣且可賣就開盤賣 → 訊號列收盤重新標記）。
  4. outcome_fixed：審查 C 用，k＋1 開盤 → 下一個訊號列＋1 開盤。
  5. 統計：AUC（Mann–Whitney 平均名次）、五組、門檻規則月份群聚 t、Newey-West、區塊自助法。
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/user/-/research/tw_stock_selection/agents/s12_surge_size_tiers/replicate")
import rep_s12_lib as K12  # noqa: E402
K11 = K12.K11
K10 = K12.K10
K9 = K10.K9
C, S, Q, R, L = K10.C, K10.S, K10.Q, K10.R, K10.L
ST, X2 = K10.ST, K10.X2
limit_dn, limit_up = S.limit_dn, S.limit_up

FEATS = ["ACC", "REVHI12", "GMYOY", "NEAR52", "RET60", "ADVCHG"]
PEAD = K11.PEAD


# ------------------------------------------------------------------ 候選
def candidates(P, lists, f1buy, train_end="2021-12-31"):
    rows = []
    for s in sorted(lists):
        lma = lists[s]["lma"][:20]
        b = set(int(i) for i in f1buy[s])
        for r, i in enumerate(lma):
            rows.append(dict(s=int(s), i=int(i), code=P.codes[i], sig=P.dstr[s], rank=r + 1, f1=int(int(i) in b),
                             m=int(lists[s]["m"]), period="train" if P.dstr[s] <= train_end else "valid"))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 特徵
def load_gm(P):
    """單季毛利率（%）矩陣 GM[季索引, 代號]、格式 FIN、可得日 A（rep_s11_lib.load_fin）。"""
    F = K11.load_fin(P)
    d = pd.read_parquet(PEAD + "/fin_long.parquet", columns=["code", "y", "q", "gp_s", "rev_s"])
    d = d[d.code.isin(P.cidx)]
    qi = (d.y.values * 4 + d.q.values - 1).astype(int)
    ok = (qi >= F["q0"]) & (qi < F["q0"] + F["nq"])
    d = d[ok]; qi = qi[ok]
    GP = np.full((F["nq"], P.N), np.nan); RV = np.full((F["nq"], P.N), np.nan)
    col = d.code.map(P.cidx).values
    GP[qi - F["q0"], col] = d.gp_s.values
    RV[qi - F["q0"], col] = d.rev_s.values
    with np.errstate(invalid="ignore", divide="ignore"):
        GM = np.where(RV > 0, GP / RV * 100.0, np.nan)
    GM = np.where(F["FIN"], np.nan, GM)
    F["GM"] = GM
    return F


def gmyoy_at(P, F, s, idx, fallback_missing=False, maxback=4, info=None):
    """訊號列 s：法定期限 < 訊號日、且 A（缺→期限日）≤ 訊號日 的最近一季；晚交改用更早一季。
    fallback_missing=True 時，該季毛利率年變化缺值也往前找（變體，只拿來對涵蓋率）。"""
    ds = P.dstr[s]
    q_hi = F["q0"] + F["nq"] - 1
    qs = [q for q in range(q_hi, F["q0"] + 3, -1) if K11.deadline_str(q) < ds][:maxback]
    out = np.full(len(idx), np.nan)
    used = [None] * len(idx)
    for n, i in enumerate(idx):
        for q in qs:
            j = q - F["q0"]
            a = F["A"][j, i] or K11.deadline_str(q)
            if a > ds:
                continue
            v = F["GM"][j, i] - F["GM"][j - 4, i]
            if fallback_missing and not np.isfinite(v):
                continue
            out[n] = v
            y, qq = divmod(q, 4)
            used[n] = "%dQ%d" % (y, qq + 1)
            break
    if info is not None:
        info.extend(used)
    return out


def features(P, cand, S_, Rv, F, variants=False):
    """回傳特徵 DataFrame（與 cand 列對齊）。"""
    G = P.G
    with np.errstate(invalid="ignore", divide="ignore"):
        AH = P.Hb * G / P.Cf                                   # 還原最高價
    AHm = pd.DataFrame(AH).rolling(240, min_periods=240).max().values
    Gm = pd.DataFrame(G).rolling(240, min_periods=240).max().values
    amt = np.nan_to_num(P.AMT, nan=0.0)
    cs = np.vstack([np.zeros((1, P.N)), np.cumsum(amt, axis=0)])
    out = {k: np.full(len(cand), np.nan) for k in FEATS}
    var = {k: np.full(len(cand), np.nan) for k in ["NEAR52_G", "NEAR52_Hbars", "GMYOY_fb", "ADVCHG_inbar"]}
    gq = [None] * len(cand)
    for s, g in cand.groupby("s"):
        ii = g.i.values; rr = g.index.values; m = int(g.m.iloc[0])
        with np.errstate(invalid="ignore", divide="ignore"):
            out["ACC"][rr] = S_[m, ii] - S_[m - 1, ii]
            w = Rv[m - 12:m, ii]
            nok = np.isfinite(w).sum(axis=0)
            mx = np.where(nok > 0, np.nanmax(np.where(np.isfinite(w), w, -np.inf), axis=0), np.nan)
            out["REVHI12"][rr] = np.where(nok >= 10, Rv[m, ii] / mx - 1.0, np.nan)
            out["NEAR52"][rr] = G[s, ii] / AHm[s, ii]
            out["RET60"][rr] = G[s, ii] / G[s - 60, ii] - 1.0
            a20 = (cs[s + 1, ii] - cs[s - 19, ii]) / 20.0
            a60 = (cs[s - 19, ii] - cs[s - 79, ii]) / 60.0
            out["ADVCHG"][rr] = np.where((a20 > 0) & (a60 > 0), np.log(a20 / a60), np.nan)
        info = []
        out["GMYOY"][rr] = gmyoy_at(P, F, s, ii, info=info)
        for n, r in enumerate(rr):
            gq[r] = info[n]
        if variants:
            with np.errstate(invalid="ignore", divide="ignore"):
                var["NEAR52_G"][rr] = G[s, ii] / Gm[s, ii]
                hb = P.barno[s, ii] >= 240
                var["NEAR52_Hbars"][rr] = np.where(hb, G[s, ii] / np.nanmax(AH[s - 239:s + 1, ii], axis=0), np.nan)
                inb = P.inbar[s - 79:s + 1, ii]
                x = amt[s - 79:s + 1, ii]
                a20b = x[60:].sum(0) / np.maximum(inb[60:].sum(0), 1)
                a60b = x[:60].sum(0) / np.maximum(inb[:60].sum(0), 1)
                var["ADVCHG_inbar"][rr] = np.where((a20b > 0) & (a60b > 0), np.log(a20b / a60b), np.nan)
            var["GMYOY_fb"][rr] = gmyoy_at(P, F, s, ii, fallback_missing=True)
    df = pd.DataFrame(out, index=cand.index)
    df["gm_q"] = gq
    if variants:
        for k, v in var.items():
            df[k] = v
    return df


# ------------------------------------------------------------------ 結果變數
def outcome_e0(P, rows_sorted, top20, s, i, tc):
    """E0 持有期。回傳 dict 或 None（進場日沒有開盤價）。tc＝估值截止列（含）。"""
    e = s + 1
    o = P.O[e, i]
    if not (o == o):
        return None
    ao_e = P.G[e, i] * o / P.Cf[e, i]
    e50_e = P.e50o[e]
    lim_up = bool(o >= limit_up(P.ref[e, i]) - 1e-9) if P.ref[e, i] == P.ref[e, i] else False
    pend = False
    rowset = top20
    last = int(P.last[i])
    x = None; kind = None; mark = -1
    for t in range(e + 1, tc + 1):
        if t > last:
            v = P.G[last, i]
            if not (v == v):
                v = pd.Series(P.G[:last + 1, i]).ffill().values[-1]
            x = (last, float(v), float(P.e50c[last])); kind = "delist"
            break
        if pend:
            ot = P.O[t, i]; rf = P.refe[t, i]
            if (ot == ot) and (rf == rf) and ot > limit_dn(P.ref[t, i]) + 1e-9:
                x = (t, float(P.G[t, i] * ot / P.Cf[t, i]), float(P.e50o[t])); kind = "sig"
                break
        if t in rowset:
            pend = i not in rowset[t]
            if pend:
                mark = t
    if x is None:
        col = P.G[:tc + 1, i]
        k = np.where(np.isfinite(col))[0]
        x = (tc, float(col[k[-1]]), float(P.e50c[tc])); kind = "trunc"
    ret = x[1] / ao_e - 1.0
    r50 = x[2] / e50_e - 1.0
    return dict(e=e, x=int(x[0]), kind=kind, ret=ret, r50=r50, exc=ret - r50, hold=int(x[0] - e), lim_up=lim_up,
                mark=int(mark))


def outcome_fixed(P, AO, Gf, s, s_next, i, tc):
    """審查 C：k＋1 開盤進場 → 下一個訊號列＋1 開盤（AO：沒開盤用最後 G；下市＝最後 G）；超過 tc 就截在 tc 收盤。"""
    e = s + 1
    o = P.O[e, i]
    if not (o == o):
        return None
    ao_e = P.G[e, i] * o / P.Cf[e, i]
    x = None if s_next is None else s_next + 1
    if x is None or x > tc:
        v = Gf[tc, i]; v50 = P.e50c[tc]; kind = "trunc"; xx = tc
    else:
        v = AO[x, i]; v50 = P.e50o[x]; kind = "fixed"; xx = x
    if P.last[i] < xx:
        v = Gf[int(P.last[i]), i]; v50 = P.e50c[int(P.last[i])]; kind = "delist"
    ret = v / ao_e - 1.0; r50 = v50 / P.e50o[e] - 1.0
    return dict(ret=ret, r50=r50, exc=ret - r50, kind=kind)


# ------------------------------------------------------------------ 統計
def auc(y, x):
    """y：bool（正例＝1），x：分數；缺值剔除。AUC＝P(正例分數 > 負例分數)，平手算一半。"""
    y = np.asarray(y, bool); x = np.asarray(x, float)
    ok = np.isfinite(x)
    y = y[ok]; x = x[ok]
    n1 = int(y.sum()); n0 = int((~y).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    r = pd.Series(x).rank(method="average").values
    return float((r[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def quintiles(df, f, bw_col="bw", exc_col="exc"):
    d = df[np.isfinite(df[f]) & np.isfinite(df[exc_col])]
    g = pd.qcut(d[f], 5, labels=False)
    out = []
    for k in range(5):
        x = d[g == k]
        out.append(dict(q=k + 1, n=int(len(x)), lo=float(x[f].min()), hi=float(x[f].max()),
                        bw_share=float(x[bw_col].mean()), mean_exc=float(x[exc_col].mean()),
                        median_exc=float(x[exc_col].median()), sd_exc=float(x[exc_col].std(ddof=1))))
    return out


def tstat(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))


def month_d(df, keep, exc_col="exc"):
    """每月 d_m＝被留下的平均超額 − 當月全部平均（只用有被留下的月份）。回傳 Series（index＝訊號列 s）。"""
    d = df[np.isfinite(df[exc_col])]
    k = keep.reindex(d.index).fillna(False).astype(bool)
    allm = d.groupby("s")[exc_col].mean()
    km = d[k].groupby("s")[exc_col].mean()
    return (km - allm.reindex(km.index)).sort_index()


def rule_keep(df, f, direction, thr):
    x = df[f]
    with np.errstate(invalid="ignore"):
        k = (x >= thr) if direction > 0 else (x <= thr)
    return k & np.isfinite(x)


def nw_t(x, lag):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x); mu = x.mean(); z = x - mu
    v = z @ z / n
    for l in range(1, lag + 1):
        w = 1.0 - l / (lag + 1.0)
        v += 2.0 * w * (z[l:] @ z[:-l]) / n
    se = np.sqrt(v / n)
    return float(mu / se), float(se)


def block_boot_t(x, block, nrep=20000, seed=13):
    """移動區塊自助法（循環）：t＝平均 ÷ 自助平均的標準差；p＝自助平均（減去樣本平均後）≥ 樣本平均的比例。"""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x); mu = x.mean()
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    xx = np.concatenate([x, x[:block]])
    st = rng.integers(0, n, size=(nrep, nb))
    idx = (st[:, :, None] + np.arange(block)[None, None, :]).reshape(nrep, -1)[:, :n]
    means = xx[idx].mean(axis=1)
    se = means.std(ddof=1)
    p = float(((means - mu) >= mu).mean())
    return float(mu / se), float(se), p
