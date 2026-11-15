# -*- coding: utf-8 -*-
"""S15（強勢股拉回買進）獨立複製：自寫函式庫。
只依 s15_strong_pullback/PREREG.json、data_check.json 與 FINDINGS.md 第 2 節（規格）的文字。
未讀、未 import 研究員的 s15lib／s11lib／s10lib／s9lib 及其 run／prep／posthoc／s0_datacheck／prereg_write。
import 的現成程式：s14_corporate_events/replicate/rep_s14_lib.py（S14 複製員：0050 停牌日修正 attach_e50_fixed、
引擎 simulate_s14＝rep_s10 引擎＋名單掃描長度 listmax）→ rep_s10 → rep_s9 → … → rep_s1；rep_s11_lib.adj_open_matrix。

本檔自己寫的部分：
  1. signal_mats：六條件（股池、RS120 前 20%、距 240 日高 ≤ 15%、前一日 G/MA20 在 0.97～1.02、前一日 5 日均量 < 20 日均量、
     收盤 > 前一日還原最高價）的向量版，回傳訊號布林矩陣與 RS120。
     G＝含息累積指數（rep_s1 面板 P.G）；還原最高價＝最高價 × G ÷ 收盤（無成交日 高＝收）；量＝成交股數，無成交記 0。
     RS120 排名池＝當日股池（rep_s2 universe 且非創新板）內 RS120 有值者；名次 ≤ ceil(0.2 × 檔數)。
     RS120 的 G[t−120]：用「未遮罩」的含息累積指數 Gu＝cumprod(1＋日報酬)，上市前＝1（rep_s1 的 P.G 在上市前是 NaN）。
     這樣上市未滿 120 根的股票也有 RS120（＝上市以來報酬 − 0050 的 120 日報酬），會進排名池。
     理由（凍結前、只看 data_check）：照字面（上市前 NaN、未滿 121 根不進池）訊號 12,201 個；用 Gu 則條件累加
     318,570／197,840／55,769／42,230／12,188、逐年訊號數與獨立事件 5,124 筆和 data_check 完全相同（strong 一項逐筆相同）。
     研究員風險 3（任務說明轉述）也說排名池含上市未滿 120 日的股票。字面版當作審查 A。
     pool_filter（審查 A）：排名池另需為 True（P.barno ≥ 121，即 G[t−120] 在上市後）。
  2. pb_plans：位移 sh：訊號列 t → 名單列 t＋sh（t＋sh＋1 開盤買）；可買＝t＋sh 在股池內；同列依 RS120[t] 由大到小
     （同值依代號）；order＝"random" 時同列隨機排序（審查 C）。
  3. make_eod_hold：每日收盤，持股 t − 進場列 ≥ H − 1 就標記待賣（隔日開盤賣），why＝end。
  4. check_conditions_py：逐格純 Python 重算 6 條件（驗收 A1-ii）。
"""
import sys, math
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/user/-/research/tw_stock_selection/agents/s14_corporate_events/replicate")
import rep_s14_lib as K14  # noqa: E402
K10 = K14.K10
S, Q, R, L = K14.S, K14.Q, K14.R, K14.L
ST, X2 = K14.ST, K14.X2
simulate = K14.simulate_s14
attach_e50_fixed = K14.attach_e50_fixed
adj_open_matrix = K14.adj_open_matrix
tstat = K14.tstat


def univ_matrix(P, excl):
    U = np.zeros((P.T, P.N), bool)
    for t in range(P.T):
        U[t] = R.universe(P, t) & ~excl(t)
    return U


def twse_tenure(P):
    """連續在上市市場（M＝1）的交易日數（含當日）。"""
    tw = P.is_twse & P.inbar
    out = np.zeros((P.T, P.N), np.int32)
    run = np.zeros(P.N, np.int32)
    for t in range(P.T):
        run = np.where(tw[t], run + 1, 0)
        out[t] = run
    return out


def signal_mats(P, U, pool_filter=None):
    """回傳 dict：sig（訊號布林）、rs（RS120）、各條件（累加用）。pool_filter：T×N 布林，排名池另需為 True。"""
    T, N = P.T, P.N
    G = P.G
    e50 = P.e50.values.astype(float)
    r0 = np.where(np.isfinite(P.ret), P.ret, 0.0)
    Gu = np.cumprod(1.0 + r0, axis=0)
    del r0
    with np.errstate(invalid="ignore", divide="ignore"):
        rs = np.full((T, N), np.nan)
        rs[120:] = Gu[120:] / Gu[:-120] - 1.0 - (e50[120:] / e50[:-120] - 1.0)[:, None]
        rs[~P.inbar] = np.nan
        del Gu
        pool = U & np.isfinite(rs)
        if pool_filter is not None:
            pool &= pool_filter
        strong = np.zeros((T, N), bool)
        for t in range(T):
            idx = np.where(pool[t])[0]
            n = len(idx)
            if n == 0:
                continue
            k = int(math.ceil(0.2 * n))
            o = idx[np.argsort(-rs[t, idx], kind="mergesort")]
            strong[t, o[:k]] = True
        hadj = np.where(P.inbar, P.Hb * G / P.Cf, np.nan)
        hi240 = pd.DataFrame(hadj).rolling(240, min_periods=240).max().values
        near = G / hi240 >= 0.85
        del hi240
        ma20 = pd.DataFrame(G).rolling(20, min_periods=20).mean().values
        x = G / ma20
        pb = np.zeros((T, N), bool)
        pb[1:] = (x[:-1] >= 0.97) & (x[:-1] <= 1.02)
        del ma20, x
        V = P.Vb
        v5 = pd.DataFrame(V).rolling(5, min_periods=5).mean().values
        v20 = pd.DataFrame(V).rolling(20, min_periods=20).mean().values
        vd = np.zeros((T, N), bool)
        vd[1:] = v5[:-1] < v20[:-1]
        del v5, v20
        turn = np.zeros((T, N), bool)
        turn[1:] = G[1:] > hadj[:-1]
        del hadj
    base = U & P.traded
    c1 = base & strong
    c2 = c1 & near
    c3 = c2 & pb
    c4 = c3 & vd
    sig = c4 & turn
    alone = dict(strong=strong, near=near, pb1=pb, vd1=vd, turn=turn)
    return dict(sig=sig, rs=rs, base=base, cum=dict(strong=c1, near=c2, pb1=c3, vd1=c4, turn=sig), alone=alone,
                pool=pool)


def pb_plans(P, sig, rs, U, shift=0, order="rs", seed=None, start_row=None):
    """回傳 (buy, info)。buy：{名單列 s＝t＋shift: 代號索引陣列}；info：{(s, i): t}（訊號列）。"""
    t0 = P.t0 if start_row is None else start_row
    codes = np.array(P.codes)
    rng = np.random.default_rng(seed) if order == "random" else None
    buy = {}; info = {}
    for t in np.where(sig.any(axis=1))[0]:
        t = int(t)
        if t < t0:
            continue
        s = t + shift
        if s >= P.T - 1:
            continue
        ii = np.where(sig[t])[0]
        ii = ii[U[s, ii]]
        if len(ii) == 0:
            continue
        if order == "rs":
            ii = ii[np.lexsort((codes[ii], -rs[t, ii]))]
        else:
            ii = ii[np.argsort(codes[ii], kind="mergesort")]
            ii = rng.permutation(ii)
        buy[s] = ii.astype(int)
        for i in ii:
            info[(s, int(i))] = t
    return buy, info


def make_eod_hold(H):
    def eod(t, pos):
        for i, p in pos.items():
            if p["pend"]:
                continue
            if t - p["tb"] >= H - 1:
                p["pend"] = True; p["why"] = "end"; p["mk"] = t
    return eod


def check_conditions_py(P, U, rs_pool_rank_ok, t, i):
    """逐格純 Python 重算 6 條件（rs_pool_rank_ok 由呼叫端以純 Python 名次算好傳入）。"""
    G = P.G; e = P.e50.values

    def g(tt):
        return float(G[tt, i])

    def hadj(tt):
        return float(P.Hb[tt, i]) * float(G[tt, i]) / float(P.Cf[tt, i])
    c1 = bool(U[t, i]) and bool(P.traded[t, i])
    c2 = bool(rs_pool_rank_ok)
    if P.barno[t, i] >= 240:
        hi = max(hadj(tt) for tt in range(t - 239, t + 1))
        c3 = g(t) / hi >= 0.85
    else:
        c3 = False
    if P.barno[t - 1, i] >= 20:
        m = sum(g(tt) for tt in range(t - 20, t)) / 20.0
        c4 = 0.97 <= g(t - 1) / m <= 1.02
    else:
        c4 = False
    if P.barno[t - 1, i] >= 20:
        v5 = sum(float(P.Vb[tt, i]) for tt in range(t - 5, t)) / 5.0
        v20 = sum(float(P.Vb[tt, i]) for tt in range(t - 20, t)) / 20.0
        c5 = v5 < v20
    else:
        c5 = False
    c6 = (P.barno[t - 1, i] >= 1) and g(t) > hadj(t - 1)
    return [c1, c2, c3, c4, c5, c6]
