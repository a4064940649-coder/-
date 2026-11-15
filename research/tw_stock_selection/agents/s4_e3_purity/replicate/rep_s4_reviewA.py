# -*- coding: utf-8 -*-
"""審查 A（凍結後）：E3' 換股日位移 0～4 的路徑拆解——哪幾個月、哪幾筆交易讓位移 2 天特別差。
用法：python3 -I -B rep_s4_reviewA.py <scratch_dir>    輸出：review_A_detail.json（本資料夾）
"""
import sys, os, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import rep_s4_lib as S
Q, R, L = S.Q, S.R, S.L
ST = S.STRICT


def main(sc):
    P = R.build(sc); S.attach_e50(P, sc); excl = S.innov_mask_fn(P, sc)
    Rv, mi0 = L.load_rev(P); S_, _ = L.surge(Rv, mi0, null2013=True); del Rv
    i19 = L.didx(P, "2018-12-28")
    navs = {}; trs = {}; lists_by = {}
    for sh in range(0, 5):
        lk = Q.make_lists(P, S_, mi0, shift=sh, excl=excl)
        x = S.simulate(P, *Q.cell_plans(lk, "E3"), **ST)
        navs[sh] = x["nav"]
        t = x["trades"].copy()
        t["m"] = [lk[int(b) - 1]["m"] for b in t.tb]
        rk, r20 = Q.lma_rank(lk, t); t["rank"] = rk; t["r20"] = r20
        # 每筆對進場前一日淨值的貢獻（近似）：損益 ÷ 進場前一日收盤淨值
        t["contrib"] = t.pnl.values / x["nav"][t.tb.values.astype(int) - 1]
        trs[sh] = t
        # 未平倉部位也算
        op = [(c, b, co, v) for c, b, co, v in x["open_pos"]]
        trs[str(sh) + "_open"] = op
        lists_by[sh] = lk
    out = {}
    # 1) 月報酬（2019 起）：位移 2 − 位移 1、3 平均，取對數差
    mr = {sh: R.month_ret(P, navs[sh]) for sh in navs}
    df = pd.DataFrame({sh: np.log1p(mr[sh]) for sh in mr})
    df = df[df.index >= pd.Period("2019-01", "M")]
    d2 = df[2] - (df[1] + df[3]) / 2
    out["total_log_gap_2019"] = float(d2.sum())
    out["n_months"] = int(len(d2))
    worst = d2.sort_values().head(12)
    out["worst_months"] = [dict(month=str(k), gap_log=float(v), r0=float(df[0][k]), r1=float(df[1][k]),
                                r2=float(df[2][k]), r3=float(df[3][k]), r4=float(df[4][k])) for k, v in worst.items()]
    best = d2.sort_values().tail(6)
    out["best_months"] = [dict(month=str(k), gap_log=float(v)) for k, v in best.items()]
    cum = d2.sort_values().cumsum()
    out["share_of_gap_from_worst_k"] = {k: float(cum.iloc[k - 1] / d2.sum()) for k in (1, 3, 5, 10)}
    # 年度
    out["yearly_log_gap"] = {str(y): float(v) for y, v in d2.groupby(d2.index.year).sum().items()}
    # 2) 交易層：以（代碼, 營收月）比對，各位移的貢獻
    key = lambda t: t.code + "_" + t.m.astype(str)
    agg = {}
    for sh in range(0, 5):
        t = trs[sh][trs[sh].tb > i19]
        g = t.assign(k=key(t)).groupby("k").agg(contrib=("contrib", "sum"), ret=("ret", "sum"),
                                                buy=("buy_date", "first"), rank=("rank", "first"))
        agg[sh] = g
    allk = sorted(set().union(*[set(g.index) for g in agg.values()]))
    M = pd.DataFrame(index=allk)
    for sh in range(0, 5):
        M["c%d" % sh] = agg[sh].contrib.reindex(allk)
        M["r%d" % sh] = agg[sh].ret.reindex(allk)
        M["b%d" % sh] = agg[sh].buy.reindex(allk)
    M["gap"] = M.c2.fillna(0) - (M.c1.fillna(0) + M.c3.fillna(0)) / 2
    out["trade_gap_sum"] = float(M.gap.sum())
    w = M.sort_values("gap").head(15)
    out["worst_trade_gaps"] = [dict(key=k, gap=float(r.gap), **{"c%d" % s: (None if pd.isna(r["c%d" % s]) else float(r["c%d" % s])) for s in range(5)},
                                    **{"r%d" % s: (None if pd.isna(r["r%d" % s]) else float(r["r%d" % s])) for s in range(5)},
                                    **{"b%d" % s: (None if pd.isna(r["b%d" % s]) else r["b%d" % s]) for s in range(5)})
                               for k, r in w.iterrows()]
    b = M.sort_values("gap").tail(8)
    out["best_trade_gaps"] = [dict(key=k, gap=float(r.gap), c1=None if pd.isna(r.c1) else float(r.c1),
                                   c2=None if pd.isna(r.c2) else float(r.c2), c3=None if pd.isna(r.c3) else float(r.c3))
                              for k, r in b.iterrows()]
    # 只出現在位移 1 與 3、不在位移 2 的（代碼, 營收月）
    miss = M[M.c2.isna() & M.c1.notna() & M.c3.notna()]
    out["missing_in_2_present_1_3"] = dict(n=int(len(miss)), sum_c1=float(miss.c1.sum()), sum_c3=float(miss.c3.sum()),
                                           top=[dict(key=k, c1=float(r.c1), c3=float(r.c3), r1=float(r.r1), r3=float(r.r3))
                                                for k, r in miss.sort_values("c1", ascending=False).head(8).iterrows()])
    only2 = M[M.c2.notna() & M.c1.isna() & M.c3.isna()]
    out["only_in_2"] = dict(n=int(len(only2)), sum_c2=float(only2.c2.sum()),
                            worst=[dict(key=k, c2=float(r.c2), r2=float(r.r2), b2=r.b2)
                                   for k, r in only2.sort_values("c2").head(8).iterrows()])
    # 3) 訊號日 +k 落在哪幾天（月中第幾日的分布）、與除權息旺季（7～9 月）
    sig0 = sorted(lists_by[0]); sig2 = sorted(lists_by[2])
    out["buy_day_of_month_shift2"] = pd.Series([P.dates[s + 1].day for s in sig2]).describe().to_dict()
    # 4) 各位移每年交易筆數、平均報酬（2019 起）
    out["per_shift_trade_stats_2019"] = {}
    for sh in range(0, 5):
        t = trs[sh][trs[sh].tb > i19]
        out["per_shift_trade_stats_2019"][sh] = dict(n=int(len(t)), mean=float(t.ret.mean()), median=float(t.ret.median()),
                                                     sum_contrib=float(t.contrib.sum()),
                                                     top5_contrib=float(t.contrib.nlargest(5).sum()),
                                                     open=[(c, b, round(v / co - 1, 3)) for c, b, co, v in trs[str(sh) + "_open"]])
    json.dump(out, open(os.path.join(HERE, "review_A_detail.json"), "w"), ensure_ascii=False, indent=1, default=str)
    print(json.dumps(out, ensure_ascii=False, indent=1, default=str)[:9000])


if __name__ == "__main__":
    main(sys.argv[1])
