# -*- coding: utf-8 -*-
"""S10 獨立複製：凍結後比對。先驗 my_numbers.sha256，再比對研究員 FINDINGS 的數字（手抄自 FINDINGS 主要數字表、第 4～7 節）
與 trades.csv.gz（逐筆）。results*.json、navs.csv.gz 不讀。
用法：python3 -I -B rep_s10_compare.py
"""
import os, json, hashlib
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
UP = os.path.dirname(HERE)
js = open(os.path.join(HERE, "my_numbers.json")).read()
h = hashlib.sha256(js.encode("utf-8")).hexdigest()
assert open(os.path.join(HERE, "my_numbers.sha256")).read().split()[0] == h
M = json.loads(js)
V = ["E0", "R10", "R15", "R30", "MA20", "MANO", "DMA60", "TS15", "TS25", "TS15M", "RQ6"]
CELLS = ["W10_70", "WN_70"]

# 研究員（FINDINGS 主要數字表）：主跑 2019／全期／2024／回撤；路徑報酬 最小／中位數／最大；路徑回撤 最深／中位數／最淺；
# 配對報酬（條數、中位數）；配對回撤（條數、中位數）；判定
RS = {
 "W10_70": {
  "E0": ([48.13, 34.08, 70.16, -33.40], [43.24, 46.23, 48.51], [-39.27, -34.01, -30.25], None, None, None),
  "R10": ([44.47, 31.76, 64.49, -32.27], [40.26, 43.15, 45.99], [-36.12, -32.96, -29.13], (0, -2.93), (20, 1.37), ("比標準差", "不能")),
  "R15": ([48.54, 34.22, 69.71, -32.03], [44.09, 46.21, 49.21], [-37.79, -32.40, -30.07], (13, 0.51), (20, 1.31), ("差不多", "不能")),
  "R30": ([45.77, 32.05, 64.02, -35.93], [42.00, 44.80, 46.49], [-43.57, -36.60, -29.81], (1, -1.31), (4, -2.93), ("差不多", "不能")),
  "MA20": ([48.70, 34.51, 73.45, -31.00], [45.68, 47.45, 49.40], [-34.79, -31.64, -29.98], (18, 1.14), (18, 1.10), ("差不多", "不能")),
  "MANO": ([49.44, 35.07, 79.27, -31.13], [46.27, 48.87, 50.58], [-37.27, -33.48, -30.70], (20, 2.50), (10, 0.68), ("比標準好", "不能")),
  "DMA60": ([46.71, 32.86, 73.78, -31.15], [42.61, 45.03, 47.41], [-34.51, -32.51, -29.33], (4, -1.42), (20, 1.53), ("差不多", "不能")),
  "TS15": ([47.18, 33.72, 66.13, -32.40], [42.25, 45.18, 47.60], [-35.17, -33.23, -29.82], (3, -1.20), (20, 0.80), ("差不多", "不能")),
  "TS25": ([48.67, 34.56, 71.37, -34.36], [45.26, 47.23, 49.58], [-38.35, -34.08, -30.59], (20, 0.80), (10, 0.14), ("差不多", "不能")),
  "TS15M": ([49.41, 34.93, 74.50, -32.81], [44.94, 48.02, 50.50], [-36.78, -33.17, -30.07], (19, 1.54), (20, 0.77), ("差不多", "不能")),
  "RQ6": ([49.77, 35.11, 74.87, -33.40], [44.67, 47.39, 50.13], [-39.28, -34.01, -30.14], (20, 1.42), (10, 0.00), ("差不多", "不能")),
 },
 "WN_70": {
  "E0": ([62.24, 44.55, 80.41, -39.50], [52.26, 55.33, 63.49], [-47.60, -41.93, -37.18], None, None, None),
  "R10": ([56.75, 40.13, 69.03, -38.03], [50.15, 52.42, 58.69], [-43.41, -39.00, -30.90], (1, -3.59), (20, 4.24), ("比標準差", "能壓回撤")),
  "R15": ([63.16, 43.77, 79.37, -38.27], [52.28, 57.01, 65.11], [-45.33, -40.84, -36.96], (16, 0.70), (18, 1.75), ("差不多", "不能")),
  "R30": ([59.76, 41.40, 74.69, -44.08], [49.84, 55.70, 62.17], [-51.56, -46.38, -40.30], (3, -1.38), (0, -4.06), ("差不多", "不能")),
  "MA20": ([63.93, 44.75, 84.07, -32.87], [55.14, 59.77, 65.07], [-37.75, -34.89, -29.80], (18, 2.98), (20, 9.02), ("比標準好", "有效率地壓回撤")),
  "MANO": ([65.53, 47.15, 100.80, -33.14], [57.04, 60.51, 66.90], [-39.27, -38.06, -32.85], (20, 4.12), (19, 6.02), ("比標準好", "有效率地壓回撤")),
  "DMA60": ([58.90, 42.01, 83.62, -33.47], [52.67, 55.54, 59.43], [-39.27, -35.01, -32.29], (9, -0.38), (20, 7.27), ("差不多", "有效率地壓回撤")),
  "TS15": ([58.12, 42.60, 75.13, -36.86], [51.26, 53.81, 59.82], [-42.59, -38.67, -35.69], (3, -2.93), (20, 3.33), ("比標準差", "能壓回撤")),
  "TS25": ([61.63, 44.79, 80.33, -41.13], [53.02, 58.76, 63.26], [-45.67, -41.96, -37.97], (14, 1.03), (11, 0.69), ("差不多", "不能")),
  "TS15M": ([62.86, 44.61, 84.21, -38.71], [53.11, 58.25, 64.45], [-43.29, -39.32, -36.06], (18, 1.48), (20, 2.85), ("差不多", "有效率地壓回撤")),
  "RQ6": ([63.91, 45.58, 83.80, -39.47], [54.16, 57.09, 65.00], [-47.62, -41.82, -38.25], (20, 1.67), (10, -0.01), ("差不多", "不能")),
 }}
R_RANK = {"W10_70": dict(path_ann=6, path_mdd=9, main_ann=7, main_mdd=9), "WN_70": dict(path_ann=9, path_mdd=9, main_ann=6, main_mdd=9)}
R_FAMILY = {"W10_70": dict(path_ann_med=46.23, path_mdd_med=-33.23, main_ann_med=48.54, main_mdd_med=-32.40),
            "WN_70": dict(path_ann_med=57.01, path_mdd_med=-39.32, main_ann_med=62.24, main_mdd_med=-38.27)}
R_OVERALL = {"MANO": "穩定地比較好", "MA20": "只有一格較好"}
R_TS = "部分"
R_POST = {
    "ma_only_events": {"W10_70": (41, 3.56, 2.13), "WN_70": (44, 3.25, None)},
    "mano_1923": {"W10_70": (0, -1.88, -3.49, -0.53), "WN_70": (0, -2.67, -5.00, -0.23)},
    "mano_no2408_all": {"W10_70": (19, 1.10), "WN_70": (20, 3.20)},
    "mano_block5_main_run_diff": {"W10_70": -1.69, "WN_70": -1.02},
    "pre_2024_06_mdd_main": {"W10_70": {"E0": -28.79, "MA20": -31.00, "MANO": -31.13, "DMA60": -31.15, "TS15": -32.40},
                             "WN_70": {"E0": -29.60, "MANO": -33.14, "DMA60": -31.70, "TS15": -31.85, "R10": -26.11, "RQ6": -27.62}},
    "minfrac_changed": 0,
    "turnover_W10": {"E0": (3.18, 3.97), "TS15": (3.46, 4.52)},
}

out = dict(my_sha256=h, combos={}, flags=[])
mx = dict(ann=0.0, mdd=0.0, ntr=0.0)
for c in CELLS:
    for v in V:
        rs = RS[c][v]
        o = M["combos"]["%s|%s" % (v, c)]; p = M["paths20"]["%s|%s|0.5" % (v, c)]
        mine_main = [o["ann_2019"] * 100, o["ann_full"] * 100, o["ann_2024"] * 100, o["mdd_full"] * 100]
        mine_pa = [p["ann_2019_min"] * 100, p["ann_2019_median"] * 100, p["ann_2019_max"] * 100]
        mine_pd = [p["mdd_deepest"] * 100, p["mdd_median"] * 100, p["mdd_shallowest"] * 100]
        d_main = [round(a - b, 3) for a, b in zip(mine_main, rs[0])]
        d_pa = [round(a - b, 3) for a, b in zip(mine_pa, rs[1])]
        d_pd = [round(a - b, 3) for a, b in zip(mine_pd, rs[2])]
        rec = dict(mine_main=[round(x, 2) for x in mine_main], res_main=rs[0], d_main=d_main,
                   mine_path_ann=[round(x, 2) for x in mine_pa], res_path_ann=rs[1], d_path_ann=d_pa,
                   mine_path_mdd=[round(x, 2) for x in mine_pd], res_path_mdd=rs[2], d_path_mdd=d_pd)
        mx["ann"] = max(mx["ann"], max(abs(x) for x in d_main[:3] + d_pa))
        mx["mdd"] = max(mx["mdd"], max(abs(x) for x in [d_main[3]] + d_pd))
        if rs[3] is not None:
            pp = M["paired"]["%s|%s|0.5" % (v, c)]
            rec.update(mine_pair=(pp["ann_n_pos"], round(pp["ann_median"] * 100, 2)), res_pair=rs[3],
                       mine_pair_dd=(pp["mdd_n_shallower"], round(pp["mdd_median"] * 100, 2)), res_pair_dd=rs[4],
                       mine_cls=(pp["cls"], pp["dd_cls"]), res_cls=rs[5], cls_same=bool((pp["cls"], pp["dd_cls"]) == rs[5]))
            if not rec["cls_same"]:
                out["flags"].append("判定不同 %s|%s" % (v, c))
        out["combos"]["%s|%s" % (v, c)] = rec
out["max_abs_diff_pp"] = mx

# 逐筆交易
rt = pd.read_csv(os.path.join(UP, "trades.csv.gz"), dtype={"code": str})
rt = rt[rt.why != "open_end"]
mt = pd.read_csv(os.path.join(HERE, "my_trades.csv.gz"), dtype={"code": str})
WHY = {"rule": "sig", "trail": "ts", "ma_daily": "dma", "requal": "rq"}
trd = {}
for c in CELLS:
    for v in V:
        a = rt[rt.run == "%s|%s" % (v, c)]
        b = mt[(mt.v == v) & (mt.cell == c)]
        ka = set(zip(a.code, a.ent_date, a.exit_date)); kb = set(zip(b.code, b.buy_date, b.sell_date))
        kab = set(zip(a.code, a.ent_date)); kbb = set(zip(b.code, b.buy_date))
        wa = a.why.map(WHY).value_counts().to_dict()
        wb = b.why.replace({"tsm": "ts"}).value_counts().to_dict()
        aa = a.set_index(["code", "ent_date", "exit_date"]).ret
        bb = b.set_index(["code", "buy_date", "sell_date"]).ret
        com = sorted(ka & kb)
        dret = float(np.max(np.abs(aa.loc[com].values - bb.loc[com].values))) if com else None
        n_r, n_m = len(a), len(b)
        trd["%s|%s" % (v, c)] = dict(n_res=n_r, n_mine=n_m, n_diff_pct=round(100.0 * (n_m - n_r) / n_r, 3),
                                     exact_match=len(com), only_res=len(ka - kb), only_mine=len(kb - ka),
                                     entries_only_res=len(kab - kbb), entries_only_mine=len(kbb - kab),
                                     why_res=wa, why_mine=wb, max_abs_ret_diff_common=dret)
        mx["ntr"] = max(mx["ntr"], abs(trd["%s|%s" % (v, c)]["n_diff_pct"]))
out["trades"] = trd

# 排名、家族、整體判定
rk = {}
for c in CELLS:
    r = M["decisions"]["ranks"][c]
    rk[c] = dict(mine=dict(path_ann=r["path_ann"]["E0"], path_mdd=r["path_mdd"]["E0"], main_ann=r["main_ann"]["E0"],
                           main_mdd=r["main_mdd"]["E0"]), res=R_RANK[c],
                 family_mine=dict(path_ann_med=round(r["family_path_ann"]["median"] * 100, 2),
                                  path_mdd_med=round(r["family_path_mdd"]["median"] * 100, 2),
                                  main_ann_med=round(r["family_main_ann"]["median"] * 100, 2),
                                  main_mdd_med=round(r["family_main_mdd"]["median"] * 100, 2)),
                 family_res=R_FAMILY[c])
out["ranks"] = rk
pv = M["decisions"]["per_variant"]
out["overall"] = dict(mine={v: pv[v]["overall"] for v in pv}, res_named=R_OVERALL,
                      mine_ts=M["decisions"]["ts_can_reduce_dd"], res_ts=R_TS,
                      mine_xq=M["decisions"]["xq"], res_xq=["MANO"],
                      mine_e0_best=M["decisions"]["e0_best"], res_e0_best="不是（6／11、9／11）")
# 事後項目
ra = M["review_A"]
post = {}
post["ma_only_events"] = {c: dict(mine=ra["ma_only_events_" + c]["source_E0_run"]["all"], res=R_POST["ma_only_events"][c]) for c in CELLS}
post["mano_1923"] = {c: dict(mine=ra["segments_all_variants"]["MANO|" + c]["ann_1923"], res=R_POST["mano_1923"][c]) for c in CELLS}
post["mano_no2408_all"] = {c: dict(mine=ra["block_tests_MANO_minus_E0"][c]["no2408_all"], res=R_POST["mano_no2408_all"][c]) for c in CELLS}
post["mano_block5"] = {c: dict(mine_main_run=round((M["combos"]["MANO|" + c]["block5"]["ann_2019"] - M["combos"]["E0|" + c]["block5"]["ann_2019"]) * 100, 2),
                               mine_paths=ra["block_tests_MANO_minus_E0"][c]["own5"], res_main_run=R_POST["mano_block5_main_run_diff"][c]) for c in CELLS}
post["pre_2024_mdd"] = {c: dict(mine={v: round(M["review_B"]["%s|%s" % (v, c)]["main"]["pre_2024-07"] * 100, 2) for v in R_POST["pre_2024_06_mdd_main"][c]},
                                res=R_POST["pre_2024_06_mdd_main"][c]) for c in CELLS}
post["minfrac_changed"] = dict(mine=M["review_C"], res=R_POST["minfrac_changed"])
post["turnover_W10"] = dict(mine={v: (round(M["combos"]["%s|W10_70" % v]["turnover_1way"], 2), round(M["combos"]["%s|W10_70" % v]["cost_all_pct"] * 100, 2)) for v in ("E0", "TS15")},
                            res=R_POST["turnover_W10"])
out["posthoc"] = post
out["verdict"] = dict(thresholds="年化差 ≤ 0.5、回撤差 ≤ 0.5 個百分點、筆數差 ≤ 2%",
                      ann_ok=bool(mx["ann"] <= 0.5), mdd_ok=bool(mx["mdd"] <= 0.5), ntr_ok=bool(mx["ntr"] <= 2.0),
                      decisions_same=bool(not out["flags"]))
json.dump(out, open(os.path.join(HERE, "compare_vs_researcher.json"), "w"), ensure_ascii=False, indent=1, default=str)
print(json.dumps(out["max_abs_diff_pp"]), out["flags"], out["verdict"])
for k, x in trd.items():
    print(k, x["n_res"], x["n_mine"], x["exact_match"], x["only_res"], x["only_mine"], x["why_res"], x["why_mine"], x["max_abs_ret_diff_common"])
print(json.dumps(rk, ensure_ascii=False))
print(json.dumps(post, ensure_ascii=False, default=str)[:3000])
