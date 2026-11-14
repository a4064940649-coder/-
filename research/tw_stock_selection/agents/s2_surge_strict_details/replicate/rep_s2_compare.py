# -*- coding: utf-8 -*-
"""S2 獨立複製：凍結之後才跑。用法：python3 -I -B rep_s2_compare.py <scratch_dir>
(1) 讀研究員的 trades.csv.gz，和 my_trades.csv.gz 逐筆比對（B0、D1；嚴成本、空位現金）。
(2) 審查 B 的對稱對照（事後）：B0 也拿掉「B0 有而 D1 沒有」的最賺 5 筆，再比。
輸出：compare_vs_researcher.json
"""
import sys, os, json, hashlib
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import rep_s2_lib as R
L = R.L
UP = os.path.dirname(HERE)

fn = os.path.join(HERE, "my_numbers.json")
h = hashlib.sha256(open(fn, "rb").read()).hexdigest()
assert open(os.path.join(HERE, "my_numbers.sha256")).read().split()[0] == h, "凍結檔被改過"
mine = json.load(open(fn))
out = dict(frozen_sha256=h)

# ---------------- (1) 逐筆比對
rt = pd.read_csv(os.path.join(UP, "trades.csv.gz"), dtype={"code": str}, low_memory=False)
mt = pd.read_csv(os.path.join(HERE, "my_trades.csv.gz"), dtype={"code": str})
for g in ("B0", "D1"):
    r = rt[rt.run == "%s|cash|strict" % g].copy()
    m = mt[mt.run == "%s_strict" % g].copy()
    why = r.why.value_counts().to_dict()
    last = r.exit_date.max()
    r_closed = r[~r.why.astype(str).str.contains("open|end", case=False)]
    j = r.merge(m, left_on=["code", "ent_date"], right_on=["code", "buy_date"], how="outer", suffixes=("_r", "_m"), indicator=True)
    both = j[j._merge == "both"]
    only_r = j[j._merge == "left_only"]; only_m = j[j._merge == "right_only"]
    d = dict(n_r_rows=int(len(r)), n_r_closed=int(len(r_closed)), n_m_closed=int(len(m)), why=why, n_both=int(len(both)),
             exit_same=int((both.exit_date == both.sell_date).sum()),
             shares_same=int((both.shares == both.sh).sum()),
             max_abs_ret_diff=float((both.ret_r - both.ret_m).abs().max()),
             max_abs_cost_diff=float((both.cost_r - both.cost_m).abs().max()),
             max_abs_pnl_diff=float((both.pnl_r - both.pnl_m).abs().max()),
             sum_pnl_r=float(both.pnl_r.sum()), sum_pnl_m=float(both.pnl_m.sum()),
             only_r=[(a, b, c, w) for a, b, c, w in zip(only_r.code, only_r.ent_date, only_r.exit_date, only_r.why)],
             only_m=[(a, b, c) for a, b, c in zip(only_m.code, only_m.buy_date, only_m.sell_date)])
    bd = both[(both.exit_date != both.sell_date) | (both.shares != both.sh)].sort_values("ent_date")
    d["first_diff"] = None if len(bd) == 0 else bd.iloc[0][["code", "ent_date", "exit_date", "sell_date", "shares", "sh", "ret_r", "ret_m"]].to_dict()
    d["n_diff_rows"] = int(len(bd))
    # 單筆報酬差最大的一筆
    k = (both.ret_r - both.ret_m).abs().idxmax()
    d["max_ret_diff_row"] = both.loc[k, ["code", "ent_date", "exit_date", "sell_date", "shares", "sh", "ret_r", "ret_m", "cost_r", "cost_m"]].to_dict()
    cd = both[(both.cost_r - both.cost_m).abs() > 0.5].sort_values("ent_date")
    d["n_cost_diff_gt_0.5"] = int(len(cd))
    d["first_cost_diff"] = None if len(cd) == 0 else cd.iloc[0][["code", "ent_date", "shares", "sh", "cost_r", "cost_m"]].to_dict()
    out["trades_" + g] = d
    print(g, {k: v for k, v in d.items() if k not in ("only_r",)}); print("only_r", d["only_r"])

# ---------------- (2) 審查 B 對稱對照（事後）
P = R.build(sys.argv[1])
Rv, mi0 = L.load_rev(P)
S, D = L.surge(Rv, mi0, null2013=True)
pb, _ = R.make_plans(P, S, mi0); pd1, _ = R.make_plans(P, S, mi0, ma_len=60)
ST = dict(cmult=1.0, extra=0.003)
b = R.simulate(P, pb, **ST); d1 = R.simulate(P, pd1, **ST)
assert abs(R.stats(P, b)["ann_2019"] - mine["B0_strict"]["ann_2019"]) < 1e-12
assert abs(R.stats(P, d1)["ann_full"] - mine["D1_strict"]["ann_full"]) < 1e-12
tb, td = b["trades"], d1["trades"]
kb = set(zip(tb.code, tb.buy_date)); kd = set(zip(td.code, td.buy_date))
b_only = tb[[(c, x) not in kd for c, x in zip(tb.code, tb.buy_date)]]
d_only = td[[(c, x) not in kb for c, x in zip(td.code, td.buy_date)]]
sym = {}
for k in (5, 10):
    spb, topb = R.span_of(b_only, k); spd, topd = R.span_of(d_only, k)
    xb = R.simulate(P, pb, block_span=spb, **ST); xd = R.simulate(P, pd1, block_span=spd, **ST)
    sb, sd = R.stats(P, xb), R.stats(P, xd)
    sym[str(k)] = dict(b0_blocked=topb, b0_ann_2019=sb["ann_2019"], b0_ann_full=sb["ann_full"],
                       d1_ann_2019=sd["ann_2019"], d1_ann_full=sd["ann_full"],
                       diff_2019=sd["ann_2019"] - sb["ann_2019"], diff_full=sd["ann_full"] - sb["ann_full"],
                       paired=R.paired(P, xd["nav"], xb["nav"], "2016-03"))
    print("sym", k, "B0' %.2f/%.2f D1' %.2f/%.2f diff %.2f/%.2f" % tuple(100 * v for v in (
        sb["ann_2019"], sb["ann_full"], sd["ann_2019"], sd["ann_full"], sym[str(k)]["diff_2019"], sym[str(k)]["diff_full"])),
        sym[str(k)]["paired"], [(q["code"], q["buy"], round(q["pnl"])) for q in topb])
out["reviewB_symmetric_posthoc"] = sym
# D1-only／B0-only 交易依進場年度的損益與筆數
ya = {}
for nm, t in (("d1_only", d_only), ("b0_only", b_only)):
    y = t.buy_date.str[:4]
    ya[nm] = {k: dict(n=int(v.shape[0]), pnl=float(v.pnl.sum()), mean_ret=float(v.ret.mean())) for k, v in t.groupby(y)}
out["only_trades_by_entry_year"] = ya
print({k: {y: (v["n"], round(v["pnl"]), round(v["mean_ret"] * 100, 2)) for y, v in q.items()} for k, q in ya.items()})
# D1-only 交易損益集中度
dp = d_only.pnl.sort_values(ascending=False)
out["d1_only_conc"] = dict(n=int(len(dp)), total=float(dp.sum()), top5=float(dp.head(5).sum()), top10=float(dp.head(10).sum()),
                           median_ret=float(d_only.ret.median()), win=float((d_only.ret > 0).mean()),
                           mean_ret_ex_top5_by_ret=float(d_only.ret.sort_values(ascending=False).iloc[5:].mean()),
                           b_only_mean_ret_ex_top5_by_ret=float(b_only.ret.sort_values(ascending=False).iloc[5:].mean()),
                           b_only_median_ret=float(b_only.ret.median()), b_only_win=float((b_only.ret > 0).mean()))
print(out["d1_only_conc"])
json.dump(out, open(os.path.join(HERE, "compare_vs_researcher.json"), "w"), ensure_ascii=False, indent=1, default=str)
