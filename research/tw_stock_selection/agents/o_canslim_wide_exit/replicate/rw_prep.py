# -*- coding: utf-8 -*-
"""獨立複製（o_canslim_wide_exit）第 1 步：資料與進場訊號 → 精簡暫存檔。
- 資料載入與進場訊號用前一題獨立複製員的 rep_lib（load_panel / fundamentals / tech / signals）。
- 出場條件矩陣（移動停損、均線下穿、50 日線）在這裡依 FINDINGS 2.3 自己寫，不用 rep_lib 的 ex1/ex2。
- 不讀、不 import 研究員的 cwlib.py / run.py / cslib.py。
用法：python3 -I -B rw_prep.py <scratch_dir>   （先跑 rep_panel.py 與 rw_e50.py）
輸出：<scratch_dir>/prep.npz
"""
import sys, os, resource
import numpy as np, pandas as pd

sys.path.insert(0, "/home/user/-/research/tw_stock_selection/agents/o_ext_canslim_xs/replicate")
import rep_lib as L

ROOT = "/home/user/-/research/tw_stock_selection/"
EXT_START = "2015-12-25"


def roll_sum_exact(X, n):
    """sum_{k=0..n-1} X[t-k]，依 k 由小到大逐項相加；視窗不足或含 NaN → NaN。"""
    T = X.shape[0]
    s = np.full(X.shape, np.nan)
    acc = np.zeros((T - n + 1, X.shape[1]))
    for k in range(n):
        acc += X[n - 1 - k: T - k]
    s[n - 1:] = acc
    return s


def roll_max(X, n):
    T = X.shape[0]
    s = np.full(X.shape, np.nan)
    acc = X[n - 1:].copy()
    for k in range(1, n):
        acc = np.fmax(acc, X[n - 1 - k: T - k]) if False else np.maximum(acc, X[n - 1 - k: T - k])  # NaN 會傳遞＝需滿 n 根
    s[n - 1:] = acc
    return s


def back_eps():
    """FY2011、FY2012 年度 EPS＝隔年 4/10～4/30 每日（收盤÷本益比）的中位數。"""
    per = pd.read_parquet(ROOT + "agents/o_pead_margin_surprise_low_attention/data/per_long.parquet", columns=["date", "code", "per"])
    per = per[(per.date >= "2012-04-10") & (per.date <= "2013-04-30")]
    cl = pd.read_parquet(ROOT + "data_panel/panel/close_raw.parquet")
    out = {}
    stats = {}
    for fy in (2011, 2012):
        a, b = f"{fy + 1}-04-10", f"{fy + 1}-04-30"
        p = per[(per.date >= a) & (per.date <= b) & (per.per > 0)]
        c = cl.loc[a:b].stack().rename("close").reset_index()
        c.columns = ["date", "code", "close"]
        m = p.merge(c, on=["date", "code"], how="inner")
        m = m[m.close > 0]
        m["eps"] = m.close / m.per
        med = m.groupby("code").eps.median()
        for code, v in med.items():
            out[(str(code), fy)] = float(v)
        stats[fy] = int(len(med))
    return out, stats


def main(S):
    P = L.load_panel(S)
    T, N = P.T, P.N
    K = L.tech(P)
    keep = dict(condN=K["condN"], condS=K["condS"], condL=K["condL"], mok=K["mok"], excess=K["excess"])
    del K
    del P.C, P.Hb, P.Vb, P.barno
    FP = L.fundamentals(P, True)
    sig_all = L.signals(P, FP, keep, "both")
    sig_tw = L.signals(P, FP, keep, "twse")
    assert sig_tw.sum() <= sig_all.sum()
    sig_tp = sig_all & (P.M == 2)
    del FP
    extra, st = back_eps()
    t_ext = P.dstr.index(EXT_START)
    FE = L.fundamentals(P, True, extra_epsy=extra)
    # 只補 Y3：extra 只含 FY2011/2012，fundamentals 以 setdefault 併入（fin_long 自 2013 起，無重疊）
    sig_ext = L.signals(P, FE, keep, "twse", t_start=t_ext)
    n_ext_only = int(sig_ext[t_ext:P.t0].sum())
    same_after = bool((sig_ext[P.t0:] == sig_tw[P.t0:]).all())
    del FE
    pool = keep["condN"] & keep["condS"] & keep["condL"] & keep["mok"][:, None] & (P.M == 1) & P.inbar & (P.Cf > 0)
    pool[:P.t0] = False
    # ---- 出場條件（FINDINGS 2.3）
    CF = P.Cf
    mx20 = roll_max(CF, 20)
    with np.errstate(invalid="ignore"):
        tr = {p: (CF <= mx20 * (1 - p / 100)) for p in (12, 15, 20)}
        for p in (6,):
            tr[p] = CF <= mx20 * (1 - p / 100)
        del mx20
        s8 = roll_sum_exact(CF, 8) / 8
        s20 = roll_sum_exact(CF, 20) / 20
        cross = np.zeros((T, N), bool)
        cross[1:] = (s8[1:] < s20[1:]) & (s8[:-1] >= s20[:-1])
        del s8, s20
        s50 = roll_sum_exact(CF, 50) / 50
        ma50 = CF < s50
        del s50
    # ---- 0050
    e = pd.read_csv(os.path.join(S, "e50_ohlc.csv"), parse_dates=["date"]).set_index("date").reindex(P.dates)
    adj = P.e50.values
    r50 = np.zeros(T); r50[1:] = adj[1:] / adj[:-1] - 1
    o50 = e["open"].values.astype(float); c50 = e["close"].values.astype(float)
    has = np.isfinite(o50) & np.isfinite(c50) & (o50 > 0)
    intr = np.where(has, c50 / np.where(has, o50, 1.0) - 1, 0.0)
    ov = (1 + r50) / (1 + intr) - 1
    tk = np.where(o50 < 50, 0.01, 0.05)
    slip50 = np.where(has, 0.5 * tk / np.where(has, o50, 1.0), 0.0003)
    np.savez_compressed(
        os.path.join(S, "prep.npz"), dates=np.array(P.dstr), codes=np.array(P.codes), O=P.O, Cf=P.Cf, ref=P.ref, ret=P.ret,
        AMT=P.AMT, M=P.M, last=P.last, first=P.first, excess=keep["excess"], sig_tw=sig_tw, sig_all=sig_all, sig_tp=sig_tp,
        sig_ext=sig_ext, pool=pool, tr6=tr[6], tr12=tr[12], tr15=tr[15], tr20=tr[20], cross=cross, ma50=ma50,
        e50=adj, ov=ov, intr=intr, slip50=slip50, has50=has, o50=np.where(has, o50, np.nan), idx=P.idx.values,
        t0=P.t0, t_odd=P.t_odd, t_ext=t_ext)
    print("T,N", T, N, "t0", P.dstr[P.t0], "t_odd", P.dstr[P.t_odd], "t_ext", P.dstr[t_ext])
    print("signals all", int(sig_all.sum()), "twse", int(sig_tw.sum()), "tpex", int(sig_tp.sum()), "pool rows", int(pool.sum()),
          "pool stocks", int(pool.any(axis=0).sum()))
    print("ext-only twse signals", n_ext_only, "same after t0", same_after, "back-eps codes", st, "cells", len(extra))
    print("0050 missing open days", int((~has).sum()), "ov mean bp", float(ov[P.t0 + 1:].mean() * 1e4), "intr mean bp", float(intr[P.t0 + 1:].mean() * 1e4),
          "prod ratio", float(np.prod((1 + ov[P.t0 + 1:]) * (1 + intr[P.t0 + 1:])) / (adj[-1] / adj[P.t0])))
    print("bad ret cells", P.n_bad, "idx missing", P.idx_missing, "e50 missing", P.e50_missing)
    print("peak RSS MB", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)


if __name__ == "__main__":
    main(sys.argv[1])
