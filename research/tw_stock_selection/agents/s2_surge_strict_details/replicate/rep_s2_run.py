# -*- coding: utf-8 -*-
"""S2 獨立複製：凍結用主程式。用法：python3 -I -B rep_s2_run.py <scratch_dir>
輸出（本資料夾）：my_numbers.json、my_numbers.sha256、my_trades.csv.gz、my_navs.csv.gz、my_targets.json
這支程式不讀研究員的任何輸出檔。
"""
import sys, os, json, hashlib, resource
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import rep_s2_lib as R
L = R.L

STRICT = dict(cmult=1.0, extra=0.003)
X2 = dict(cmult=2.0, extra=0.006)


def brief(s):
    return {k: s[k] for k in ("ann_2019", "ann_full", "ann_2024", "mdd_full", "n_trades")}


def main(scratch):
    P = R.build(scratch)
    Rv, mi0 = L.load_rev(P)
    S, D = L.surge(Rv, mi0, null2013=True)
    e = P.T - 1
    i19 = L.didx(P, "2018-12-28"); i24 = L.didx(P, "2023-12-29")
    out = dict(meta=dict(T=P.T, N=P.N, first=P.dstr[0], last=P.dstr[-1], t0=P.dstr[P.t0]))
    e50 = P.e50.values
    out["bench_0050"] = dict(ann_2019=L.ann(P, e50, i19, e), ann_full=L.ann(P, e50, P.t0, e), ann_2024=L.ann(P, e50, i24, e),
                             ann_2019_2023=L.ann(P, e50, i19, i24), yearly=L.yearly(P, e50))
    plans = {}; info = {}
    plans["B0"], info["B0"] = R.make_plans(P, S, mi0)
    plans["D1"], info["D1"] = R.make_plans(P, S, mi0, ma_len=60)
    out["meta"].update(n_signals=len(plans["B0"]), first_signal=info["B0"][0]["signal"], last_signal=info["B0"][-1]["signal"],
                       min_n_surge=min(x["n_surge"] for x in info["B0"]),
                       avg_n_univ=float(np.mean([x["n_univ"] for x in info["B0"]])),
                       avg_d1_pass_ratio=float(np.mean([x["n_pass"] / x["n_surge"] for x in info["D1"]])))
    res = {}; navs = {}
    # ---- 1、2：嚴成本基準；3：成本加倍；輕成本只當參考
    for g in ("B0", "D1"):
        res[g] = R.simulate(P, plans[g], **STRICT)
        out[g + "_strict"] = R.stats(P, res[g]); navs[g + "_strict"] = res[g]["nav"]
        x = R.simulate(P, plans[g], **X2)
        out[g + "_x2"] = R.stats(P, x); navs[g + "_x2"] = x["nav"]
        lt = R.simulate(P, plans[g])
        out[g + "_light"] = R.stats(P, lt); navs[g + "_light"] = lt["nav"]
        # ---- 4：整段持有期禁買基準跑損益金額最大的 5／10 筆
        for k in (5, 10):
            sp, top = R.span_of(res[g]["trades"], k)
            b = R.simulate(P, plans[g], block_span=sp, **STRICT)
            out["%s_block%d" % (g, k)] = R.stats(P, b); out["%s_block%d" % (g, k)]["blocked_trades"] = top
            navs["%s_block%d" % (g, k)] = b["nav"]
    # ---- 5：配對差
    out["paired_D1_minus_B0"] = dict(from_2016_03=R.paired(P, res["D1"]["nav"], res["B0"]["nav"], "2016-03"),
                                     from_2019_01=R.paired(P, res["D1"]["nav"], res["B0"]["nav"], "2019-01"))
    # ---- 6：換股日位移
    for g, ml in (("D1", 60), ("B0", None)):
        for k in (3, 5, 10):
            pk, _ = R.make_plans(P, S, mi0, ma_len=ml, shift=k)
            x = R.simulate(P, pk, **STRICT)
            out["%s_shift%d" % (g, k)] = R.stats(P, x)
    # ---- 7：股池等權
    for nm, hold in (("eqw_hold", True), ("eqw_daily_rebal", False)):
        q = R.eqw_nav(P, plans["B0"], hold=hold)
        out[nm] = dict(ann_2019_2023=L.ann(P, q, i19, i24), ann_2024=L.ann(P, q, i24, e), ann_2019=L.ann(P, q, i19, e),
                       ann_full=L.ann(P, q, P.t0, e), yearly=L.yearly(P, q))
        navs[nm] = q
    # ---- 審查 A：均線長度
    out["reviewA_ma_len"] = {}
    for n in (40, 50, 60, 80, 100, 120):
        pk, ik = R.make_plans(P, S, mi0, ma_len=n)
        x = R.simulate(P, pk, **STRICT)
        s = R.stats(P, x)
        s["paired_vs_B0"] = R.paired(P, x["nav"], res["B0"]["nav"], "2016-03")
        s["avg_pass_ratio"] = float(np.mean([q["n_pass"] / q["n_surge"] for q in ik]))
        out["reviewA_ma_len"][str(n)] = s
        navs["D1_ma%d" % n] = x["nav"]
    # ---- 審查 B：逐年差、拿掉 D1 換進來的最賺 5 筆
    yb = out["B0_strict"]["yearly"]; yd = out["D1_strict"]["yearly"]
    rb = {}
    rb["yearly_diff"] = {str(y): yd[y] - yb[y] for y in yd}
    tb = res["B0"]["trades"]; td = res["D1"]["trades"].copy()
    kb = set(zip(tb.code, tb.buy_date))
    td["only_d1"] = [(c, d) not in kb for c, d in zip(td.code, td.buy_date)]
    # 較嚴的定義：D1 的持有期間內 B0 完全沒持有過同一檔
    bsp = {}
    for c, a, b in zip(tb.code, tb.tb, tb.ts):
        bsp.setdefault(c, []).append((a, b))
    td["no_overlap"] = [not any(a <= y and x <= b for a, b in bsp.get(c, ())) for c, x, y in zip(td.code, td.tb, td.ts)]
    kd = set(zip(td.code, td.buy_date))
    b_only = tb[[(c, d) not in kd for c, d in zip(tb.code, tb.buy_date)]]
    rb["n_d1_only"] = int(td.only_d1.sum()); rb["d1_only_mean_ret"] = float(td.ret[td.only_d1].mean())
    rb["d1_only_pnl_sum"] = float(td.pnl[td.only_d1].sum()); rb["d1_total_pnl"] = float(td.pnl.sum())
    rb["n_b0_only"] = int(len(b_only)); rb["b0_only_mean_ret"] = float(b_only.ret.mean())
    rb["b0_only_pnl_sum"] = float(b_only.pnl.sum()); rb["b0_total_pnl"] = float(tb.pnl.sum())
    rb["n_common"] = int((~td.only_d1).sum())
    rb["n_d1_no_overlap"] = int(td.no_overlap.sum())
    for nm, col in (("drop5_d1_only", "only_d1"), ("drop5_d1_no_overlap", "no_overlap")):
        for k in (5, 10):
            sp, top = R.span_of(td[td[col]], k)
            x = R.simulate(P, plans["D1"], block_span=sp, **STRICT)
            s = R.stats(P, x)
            key = nm if k == 5 else nm.replace("drop5", "drop10")
            rb[key] = dict(blocked_trades=top, ann_2019=s["ann_2019"], ann_full=s["ann_full"], ann_2024=s["ann_2024"],
                           mdd_full=s["mdd_full"], n_trades=s["n_trades"],
                           diff_full_vs_B0=s["ann_full"] - out["B0_strict"]["ann_full"],
                           diff_2019_vs_B0=s["ann_2019"] - out["B0_strict"]["ann_2019"],
                           paired=R.paired(P, x["nav"], res["B0"]["nav"], "2016-03"),
                           top_pnl_share_of_d1_only=float(sum(q["pnl"] for q in top) / td.pnl[td[col]].sum()))
            navs["D1_" + key] = x["nav"]
    # 對照：B0 換掉（B0 有、D1 沒有）的交易中賠最多的不算，只列 B0-only 最賺 5 筆供參考
    rb["b0_only_top5"] = R.span_of(b_only, 5)[1]
    # 配對差月份分布
    dm = (R.month_ret(P, res["D1"]["nav"]) - R.month_ret(P, res["B0"]["nav"]))
    dm = dm[dm.index >= pd.Period("2016-03", "M")]
    srt = dm.sort_values(ascending=False)
    rb["paired_top5_months"] = [(str(i), float(v)) for i, v in srt.head(5).items()]
    rb["paired_bottom5_months"] = [(str(i), float(v)) for i, v in srt.tail(5).items()]
    rb["paired_sum"] = float(dm.sum()); rb["paired_sum_ex_top5"] = float(srt.iloc[5:].sum())
    rb["paired_pos_months"] = int((dm > 0).sum()); rb["paired_neg_months"] = int((dm < 0).sum()); rb["paired_zero_months"] = int((dm == 0).sum())
    out["reviewB"] = rb
    # 名單（給事後逐筆比對用）
    tg = {}
    for g in ("B0", "D1"):
        tg[g] = {x["signal"]: [(P.codes[int(i)], round(float(S[x["m"], int(i)]), 4)) for i in plans[g][x["s"]][:10]] for x in info[g]}
    json.dump(tg, open(os.path.join(HERE, "my_targets.json"), "w"), ensure_ascii=False)
    out["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
    t1 = res["B0"]["trades"].assign(run="B0_strict"); t2 = td.assign(run="D1_strict")
    pd.concat([t1, t2]).drop(columns=["tb", "ts"]).to_csv(os.path.join(HERE, "my_trades.csv.gz"), index=False)
    pd.DataFrame(dict(date=P.dstr[P.t0:], **{k: v[P.t0:] for k, v in navs.items()}, e50=e50[P.t0:])).to_csv(
        os.path.join(HERE, "my_navs.csv.gz"), index=False, float_format="%.4f")
    fn = os.path.join(HERE, "my_numbers.json")
    json.dump(out, open(fn, "w"), ensure_ascii=False, indent=1)
    h = hashlib.sha256(open(fn, "rb").read()).hexdigest()
    open(os.path.join(HERE, "my_numbers.sha256"), "w").write(h + "  my_numbers.json\n")
    pr = lambda v: "%.2f%%" % (v * 100)
    for k in ("B0_strict", "D1_strict", "B0_x2", "D1_x2", "B0_light", "D1_light", "B0_block5", "B0_block10", "D1_block5", "D1_block10",
              "D1_shift3", "D1_shift5", "D1_shift10", "B0_shift3", "B0_shift5", "B0_shift10"):
        s = out[k]
        print(k, pr(s["ann_2019"]), pr(s["ann_full"]), pr(s["ann_2024"]), pr(s["mdd_full"]), s["mdd_peak"], s["mdd_trough"], s["n_trades"])
    print("yearly D1", {y: round(v * 100, 1) for y, v in out["D1_strict"]["yearly"].items()})
    print("yearly B0", {y: round(v * 100, 1) for y, v in out["B0_strict"]["yearly"].items()})
    print("paired", out["paired_D1_minus_B0"])
    print("eqw", {k: (pr(out[k]["ann_2019_2023"]), pr(out[k]["ann_2024"])) for k in ("eqw_hold", "eqw_daily_rebal")})
    print("0050", {k: pr(v) for k, v in out["bench_0050"].items() if k != "yearly"})
    for n, s in out["reviewA_ma_len"].items():
        print("MA", n, pr(s["ann_2019"]), pr(s["ann_full"]), pr(s["ann_2024"]), pr(s["mdd_full"]), s["n_trades"],
              "paired %.3f%% t %.2f" % (s["paired_vs_B0"]["mean"] * 100, s["paired_vs_B0"]["t"]), "pass %.2f" % s["avg_pass_ratio"])
    print("yearly diff", {y: round(v * 100, 1) for y, v in rb["yearly_diff"].items()})
    for k in ("drop5_d1_only", "drop10_d1_only", "drop5_d1_no_overlap", "drop10_d1_no_overlap"):
        q = rb[k]
        print(k, pr(q["ann_2019"]), pr(q["ann_full"]), "diff_full %.2f diff_2019 %.2f" % (q["diff_full_vs_B0"] * 100, q["diff_2019_vs_B0"] * 100),
              "paired %.3f%% t %.2f" % (q["paired"]["mean"] * 100, q["paired"]["t"]), "share %.2f" % q["top_pnl_share_of_d1_only"],
              [(x["code"], x["buy"], round(x["pnl"])) for x in q["blocked_trades"]])
    print({k: v for k, v in rb.items() if k.startswith(("n_", "d1_", "b0_only_m", "b0_only_p", "b0_t", "paired"))})
    print(out["meta"], out["peak_rss_mb"], h)
    print(out["B0_strict"]["cnt"]); print(out["D1_strict"]["cnt"])


if __name__ == "__main__":
    main(sys.argv[1])
