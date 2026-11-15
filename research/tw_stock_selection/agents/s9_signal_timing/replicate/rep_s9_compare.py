# -*- coding: utf-8 -*-
"""S9 獨立複製：凍結後比對。先驗 my_numbers.json 的 sha256，再讀研究員 FINDINGS（數字手抄自第 4～9 節表格）與 trades.csv.gz。
results*.json、navs.csv.gz 不讀。用法：python3 -I -B rep_s9_compare.py
輸出：compare_vs_researcher.json
"""
import os, json, hashlib
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
UP = os.path.dirname(HERE)
h = hashlib.sha256(open(os.path.join(HERE, "my_numbers.json"), "rb").read()).hexdigest()
assert h == open(os.path.join(HERE, "my_numbers.sha256")).read().split()[0], "my_numbers.json 被改過"
M = json.load(open(os.path.join(HERE, "my_numbers.json")))

# ---- 研究員（FINDINGS 主要數字表、第 4～7 節）：2019 起／全期／2024 起／最大回撤（%）；路徑中位數 報酬／回撤
RO = {  # open
    ("A08", "W10_70"): (35.70, 24.96, 66.52, -34.60, 35.01, -33.15), ("A08", "WN_70"): (40.93, 27.74, 77.28, -36.99, 38.04, -37.88),
    ("A09", "W10_70"): (36.35, 25.78, 67.78, -34.45, 33.90, -33.15), ("A09", "WN_70"): (42.93, 30.15, 80.30, -36.98, 37.66, -38.66),
    ("A10", "W10_70"): (48.13, 34.08, 70.16, -33.40, 46.23, -34.01), ("A10", "WN_70"): (62.24, 44.55, 80.41, -39.50, 55.33, -41.93),
    ("A11", "W10_70"): (46.91, 32.67, 64.77, -37.05, 43.17, -33.69), ("A11", "WN_70"): (59.33, 40.91, 66.89, -44.69, 53.38, -42.47),
    ("A12", "W10_70"): (44.73, 32.22, 63.13, -35.29, 42.41, -33.85), ("A12", "WN_70"): (55.52, 40.13, 64.09, -43.91, 49.71, -43.61),
    ("A13", "W10_70"): (45.64, 31.80, 63.37, -32.73, 41.45, -32.64), ("A13", "WN_70"): (55.03, 36.54, 60.90, -43.69, 47.08, -40.81),
    ("A15", "W10_70"): (43.43, 31.00, 63.55, -32.59, 42.43, -30.17), ("A15", "WN_70"): (48.70, 33.49, 59.91, -39.39, 48.84, -38.26),
    ("B11opt", "W10_70"): (49.48, 35.24, 74.19, -30.06, 45.79, -33.63), ("B11opt", "WN_70"): (64.17, 46.27, 88.48, -35.14, 56.85, -42.47),
    ("B11pes", "W10_70"): (53.63, 37.29, 83.24, -31.86, 51.94, -33.17), ("B11pes", "WN_70"): (70.32, 48.07, 105.22, -37.85, 65.35, -39.33),
}
# 收盤 − 開盤（第 6 節）：2019 起差／路徑中位數差／回撤差（正＝收盤較淺）
CO = {
    "W10_70": {"A08": (-0.62, -0.83, 2.62), "A09": (-0.91, -0.06, 1.95), "A10": (-2.88, -1.85, -5.47), "A11": (-3.88, 0.00, 0.81),
               "A12": (-2.46, 1.24, -1.86), "A13": (-4.62, 0.17, -5.55), "A15": (6.16, 0.63, 5.87)},
    "WN_70": {"A08": (-1.00, -1.54, 3.49), "A09": (-3.62, -1.38, -1.28), "A10": (-3.58, -0.52, -7.85), "A11": (-3.86, -2.83, 2.51),
              "A12": (-4.06, 1.19, -1.85), "A13": (-6.58, 1.46, -4.79), "A15": (8.64, 1.04, 5.14)},
}
# 收盤直接給的（第 4、7 節）
RC_DIRECT = {("A10", "W10_70"): (45.24, -38.88, 44.38, -36.98), ("A10", "WN_70"): (58.66, -47.35, 54.81, -47.21),
             ("B11opt", "W10_70"): (45.46, -36.81, 43.69, -36.87), ("B11opt", "WN_70"): (57.82, -44.67, 53.75, -44.51),
             ("B11pes", "W10_70"): (46.06, -39.09, 47.91, -34.02), ("B11pes", "WN_70"): (58.75, -48.20, 58.59, -41.79)}
# 家族（第 4、5 節）
FAM = {
    "W10_70|open": dict(fam_median=44.73, fam_min=35.70, A10_minus_median_pp=3.40, A10_rank_ret_7=1, A10_rank_dd_7=3,
                        fam5_median=45.64, A10_rank_ret_5=1, dd_median=-34.45, dd_deepest=-37.05,
                        p140_ret_median=42.07, p140_ret_p10=33.54, p140_dd_median=-32.83, p140_dd_p10=-35.63,
                        p100_ret_median=43.01, p100_ret_p10=40.64, p100_dd_median=-32.61, p100_dd_p10=-36.31),
    "W10_70|close": dict(fam_median=42.27, fam_min=35.08, A10_minus_median_pp=2.97, A10_rank_ret_7=2, A10_rank_dd_7=7,
                         fam5_median=43.03, A10_rank_ret_5=2, dd_median=-36.24, dd_deepest=-38.88,
                         p140_ret_median=42.42, p140_ret_p10=33.50, p140_dd_median=-34.24, p140_dd_p10=-38.10,
                         p100_ret_median=43.22, p100_ret_p10=40.72, p100_dd_median=-34.37, p100_dd_p10=-38.31),
    "WN_70|open": dict(fam_median=55.03, fam_min=40.93, A10_minus_median_pp=7.21, A10_rank_ret_7=1, A10_rank_dd_7=4,
                       fam5_median=55.52, A10_rank_ret_5=1, dd_median=-39.50, dd_deepest=-44.69,
                       p140_ret_median=47.72, p140_ret_p10=36.94, p140_dd_median=-39.39, p140_dd_p10=-44.84,
                       p100_ret_median=50.66, p100_ret_p10=44.15, p100_dd_median=-40.38, p100_dd_p10=-45.27),
    "WN_70|close": dict(fam_median=51.46, fam_min=39.32, A10_minus_median_pp=7.20, A10_rank_ret_7=1, A10_rank_dd_7=6,
                        fam5_median=55.47, A10_rank_ret_5=1, dd_median=-42.18, dd_deepest=-48.48,
                        p140_ret_median=48.94, p140_ret_p10=35.59, p140_dd_median=-41.13, p140_dd_p10=-47.36,
                        p100_ret_median=50.71, p100_ret_p10=46.30, p100_dd_median=-41.92, p100_dd_p10=-47.45),
}
RULE_PP = (0.5, 0.5, 0.02)   # 年化差、回撤差（個百分點）、筆數相對差


def main():
    out = dict(sha256_verified=h, thresholds=dict(ann_pp=0.5, mdd_pp=0.5, trades_rel=0.02))
    tr = pd.read_csv(os.path.join(UP, "trades.csv.gz"), dtype={"code": str})
    mt = pd.read_csv(os.path.join(HERE, "my_trades.csv.gz"), dtype={"code": str})
    rows = []
    for key, o in M["combos"].items():
        d, ex, cell = key.split("|")
        p = M["paths20"][key]
        r = dict(combo=key, my_2019=o["ann_2019"] * 100, my_full=o["ann_full"] * 100, my_2024=o["ann_2024"] * 100,
                 my_mdd=o["mdd_full"] * 100, my_n=o["n_trades"], my_path_ret=p["ann_2019_median"] * 100,
                 my_path_dd=p["mdd_median"] * 100, my_limit_up=o["limit_up_skips"], my_cls=o["cls"])
        rr = tr[(tr.run == key) & (tr.why == "rule")]
        r["res_n"] = int(len(rr))
        if ex == "open":
            v = RO[(d, cell)]
            r.update(res_2019=v[0], res_full=v[1], res_2024=v[2], res_mdd=v[3], res_path_ret=v[4], res_path_dd=v[5],
                     src="FINDINGS 主要數字表")
        else:
            if (d, cell) in RC_DIRECT:
                v = RC_DIRECT[(d, cell)]
                r.update(res_2019=v[0], res_mdd=v[1], res_path_ret=v[2], res_path_dd=v[3], src="FINDINGS 第 4／7 節")
            elif d in CO[cell]:
                vo = RO[(d, cell)]; c_ = CO[cell][d]
                r.update(res_2019=round(vo[0] + c_[0], 2), res_mdd=round(vo[3] + c_[2], 2),
                         res_path_ret=round(vo[4] + c_[1], 2), src="開盤值＋第 6 節差（±0.01 捨入）")
        for k in ("2019", "full", "2024", "mdd", "path_ret", "path_dd"):
            if ("res_" + k) in r:
                r["d_" + k] = r["my_" + k] - r["res_" + k]
        r["d_n_rel"] = (r["my_n"] - r["res_n"]) / r["res_n"]
        okk = [abs(r[k]) <= RULE_PP[0] for k in ("d_2019", "d_full", "d_2024") if k in r]
        okk += [abs(r["d_mdd"]) <= RULE_PP[1]] if "d_mdd" in r else []
        okk += [abs(r["d_n_rel"]) <= RULE_PP[2]]
        r["pass"] = bool(all(okk))
        # 逐筆交易：同代號、進場日、出場日、股數
        mm = mt[mt.combo == key]
        a = set(zip(mm.code, mm.buy_date, mm.sell_date, mm.sh.astype(int)))
        b = set(zip(rr.code, rr.ent_date, rr.exit_date, rr.shares.astype(int)))
        a2 = set(zip(mm.code, mm.buy_date, mm.sell_date)); b2 = set(zip(rr.code, rr.ent_date, rr.exit_date))
        r.update(trades_exact_match=len(a & b), trades_match_no_shares=len(a2 & b2), my_only=len(a2 - b2), res_only=len(b2 - a2))
        rows.append(r)
    df = pd.DataFrame(rows)
    out["combos"] = df.round(4).to_dict("records")
    out["n_combos"] = int(len(df)); out["n_pass"] = int(df["pass"].sum())
    out["max_abs"] = {k: float(df[k].abs().max()) for k in ("d_2019", "d_full", "d_2024", "d_mdd", "d_path_ret", "d_path_dd", "d_n_rel")
                      if k in df}
    # 家族統計
    fam = {}
    for k, v in FAM.items():
        mine = M["family"][k]
        fam[k] = {}
        for kk, rv in v.items():
            mv = mine[kk]
            if isinstance(rv, int) and "rank" in kk:
                fam[k][kk] = dict(mine=mv, res=rv, same=(mv == rv))
            else:
                mvp = mv if kk.endswith("_pp") else mv * 100
                fam[k][kk] = dict(mine=round(mvp, 2), res=rv, diff=round(mvp - rv, 2))
    out["family"] = fam
    # 審查用：研究員敘述的數字對照
    out["review_refs"] = dict(
        B11pes_minus_B11opt_pp=dict(res={"W10_70": 4.15, "WN_70": 6.15},
                                    mine={c: M["reviewA"][c]["pes_minus_opt_2019_pp"] for c in ("W10_70", "WN_70")}),
        slope_single_path=dict(res={"W10_70": -1.51, "WN_70": -4.03},
                               mine={c: M["reviewB"][c]["main_slope_pp_per_day"] for c in ("W10_70", "WN_70")},
                               note="研究員含 B11opt 共 6 點；我的只用 A10～A15 五點"),
        slope_path_median=dict(res={"W10_70": -1.12, "WN_70": -2.46},
                               mine={c: M["reviewB"][c]["path_slopes"]["median"] for c in ("W10_70", "WN_70")}),
        calendar=dict(res=dict(A08=(25, -1.32), A09=(47, -0.64), A11=(49, 0.62), A12=(24, 1.29), A13=(7, 1.98), A15=(3, 3.36),
                               B11=(80, -0.38), early=49, months=129),
                      mine={d: (M["calendar"][d]["same_as_A10"], round(M["calendar"][d]["mean_lag"], 2)) for d in
                            ("A08", "A09", "A11", "A12", "A13", "A15", "B11opt")}),
        limit_up_close_A10=dict(res="29～30", mine={c: M["combos"]["A10|close|%s" % c]["limit_up_skips"] for c in ("W10_70", "WN_70")}),
        limit_up_close_others=dict(res="17～38", mine={k: v for k, v in M["limit_up_main"].items() if "|close|" in k}),
    )
    # 凍結後推導（標「凍結後」）：研究員口徑的 6 點迴歸（含 B11opt）、家族差距拆解
    pr = pd.read_csv(os.path.join(HERE, "my_paths.csv.gz"))
    lag = {"B11opt": -0.38, **{k: M["calendar"][k]["mean_lag"] for k in ("A10", "A11", "A12", "A13", "A15")}}
    pf = {}
    for c in ("W10_70", "WN_70"):
        for src in ("main", "path_median"):
            y = np.array([(M["combos"]["%s|open|%s" % (k, c)]["ann_2019"] if src == "main" else
                           M["paths20"]["%s|open|%s" % (k, c)]["ann_2019_median"]) * 100 for k in lag])
            x = np.array(list(lag.values())); bb = np.polyfit(x, y, 1); rs = y - np.polyval(bb, x)
            pf["6pt_%s_%s" % (src, c)] = dict(slope_pp=float(bb[0]), A10_resid_pp=float(rs[1]), resid_sd_pp=float(rs.std(ddof=2)))
        g = pr[(pr.cell == c) & (pr.ex == "open") & pr.d.isin(list(lag)[1:])]
        x = g.d.map(lag).values + g["shift"].values; y = g.ann_2019.values * 100
        bb = np.polyfit(x, y, 1); a10 = M["combos"]["A10|open|%s" % c]["ann_2019"] * 100
        pf["gap_decomp_%s" % c] = dict(total_pp=float(a10 - np.median(y)), A10_above_line_pp=float(a10 - bb[1]),
                                       lag_pp=float(bb[1] - y.mean()), mean_minus_median_pp=float(y.mean() - np.median(y)),
                                       mean_eff_lag=float(x.mean()), slope_pp=float(bb[0]))
    out["post_freeze_derived"] = pf
    print(json.dumps(pf))
    json.dump(out, open(os.path.join(HERE, "compare_vs_researcher.json"), "w"), ensure_ascii=False, indent=1,
              default=lambda o: o.item() if hasattr(o, "item") else str(o))
    pd.set_option("display.width", 250)
    print(df[["combo", "my_2019", "res_2019", "d_2019", "d_full", "d_2024", "d_mdd", "d_path_ret", "d_path_dd", "my_n", "res_n",
              "d_n_rel", "trades_match_no_shares", "my_only", "res_only", "pass"]].round(3).to_string())
    print("pass", out["n_pass"], "/", out["n_combos"], out["max_abs"])
    for k, v in fam.items():
        print(k, {kk: (vv.get("diff"), vv.get("same")) for kk, vv in v.items()})
    print(json.dumps(out["review_refs"], ensure_ascii=False))


if __name__ == "__main__":
    main()
