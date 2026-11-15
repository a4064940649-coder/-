# -*- coding: utf-8 -*-
"""S8 獨立複製：凍結後比對。先驗 my_numbers.sha256，再讀研究員 FINDINGS（數字手抄於下）與 trades.csv.gz。
results*.json、navs.csv.gz 不讀。用法：python3 -I -B rep_s8_compare.py
"""
import os, json, hashlib
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
UP = os.path.dirname(HERE)

h = hashlib.sha256(open(os.path.join(HERE, "my_numbers.json"), "rb").read()).hexdigest()
assert open(os.path.join(HERE, "my_numbers.sha256")).read().split()[0] == h, "凍結檔被改過"
me = json.load(open(os.path.join(HERE, "my_numbers.json")))

# 研究員 FINDINGS 第 3、4、5.1、7 節（%）
RS = {  # 2019 起, 全期, 2024 起, 最大回撤, 筆數
    "WN_70L_C12": (58.13, 41.57, 73.05, -40.47, 358), "WN_70L_C10": (51.23, 36.21, 67.08, -37.86, 373),
    "WN_70L_C08": (49.13, 34.69, 68.53, -34.62, 382), "WN_60L": (55.61, 39.77, 74.79, -36.44, 341),
    "WN_60L_C10": (52.18, 38.11, 68.32, -37.36, 359), "W10_70L": (47.49, 33.68, 68.14, -33.38, 378),
    "WN_70L": (60.22, 43.07, 78.19, -39.34, 342),
}
RS_ROB = {  # 成本加倍 2019/全期, 禁買 5 筆 2019/全期, 前半/後半
    "WN_70L_C12": (49.80, 35.06, 51.25, 37.28, 34.52, 48.94), "WN_70L_C10": (46.71, 31.89, 45.52, 32.64, 28.82, 43.99),
    "WN_70L_C08": (43.75, 30.57, 44.47, 31.77, 27.38, 42.38), "WN_60L": (48.40, 34.26, 47.53, 34.71, 31.93, 48.03),
    "WN_60L_C10": (47.94, 32.46, 47.46, 35.13, 31.94, 44.53), "W10_70L": (41.70, 29.33, 43.32, 31.06, 26.52, 41.21),
    "WN_70L": (51.02, 35.47, 52.51, 38.28, 35.30, 51.24),
}
RS_CAP = {"WN_70L_C12": (249, 48), "WN_70L_C10": (331, 22), "WN_70L_C08": (363, 3), "WN_60L_C10": (263, 45),
          "WN_70L": (0, 72)}  # 觸發次數, 跳過次數
RS_PATH = {  # 最深, 中位數, 最淺, −36% 內條數, 位移 0/1/2/3 中位數；報酬 最小/中位數/最大
    "WN_70L_C12": (-45.87, -41.79, -36.90, 0, (-40.5, -43.8, -42.8, -40.4), (46.23, 51.63, 60.06)),
    "WN_70L_C10": (-43.02, -39.76, -36.90, 0, (-37.9, -42.1, -41.3, -37.8), (44.50, 49.47, 53.06)),
    "WN_70L_C08": (-39.72, -35.32, -34.37, 11, (-34.4, -39.7, -36.7, -34.5), (42.54, 44.94, 49.35)),
    "WN_60L": (-43.52, -42.52, -36.29, 0, (-36.4, -42.5, -43.0, -42.6), (42.57, 52.57, 56.83)),
    "WN_60L_C10": (-42.68, -39.50, -36.69, 0, (-37.4, -42.4, -39.7, -38.2), (42.33, 47.60, 53.55)),
    "W10_70L": (-39.15, -34.25, -29.99, 15, (-33.2, -38.3, -34.7, -31.1), (41.72, 45.36, 47.86)),
    "WN_70L": (-47.12, -44.17, -37.79, 0, (-38.7, -44.7, -45.6, -43.7), (44.86, 56.35, 62.16)),
}
RS_PAIR = {"pair_main_vs_WN_70L": (-3.79, 4, 1.62, 14), "pair_main_vs_W10_70L": (8.33, 20, -7.22, 0),
           "pair_WN_60L_vs_WN_70L": (-4.71, 0, 2.12, 19), "pair_WN_70L_C10_vs_WN_70L": (-7.38, 1, 3.84, 20),
           "pair_WN_70L_C08_vs_WN_70L": (-10.71, 0, 7.04, 20), "pair_WN_60L_C10_vs_WN_70L": (-7.81, 0, 3.72, 20)}
RS_DLY = {("WN_70L_C12", 1): (52.00, 37.25, 62.92, -43.28), ("WN_70L_C12", 2): (49.37, 36.16, 57.99, -44.91),
          ("W10_70L", 1): (43.87, 31.08, 62.35, -38.01), ("W10_70L", 2): (42.21, 30.06, 58.40, -36.37),
          ("WN_70L", 1): (55.05, 39.11, 67.97, -42.35), ("WN_70L", 2): (52.60, 38.17, 59.51, -45.50)}
RS_DLY_CAPS = {("WN_70L_C12", 1): 53.69, ("WN_70L_C12", 2): 49.37, ("W10_70L", 1): 43.87, ("W10_70L", 2): 42.21}

out = dict(frozen_sha256=h, cells={}, robust={}, cap={}, paths={}, pairs={}, delay={}, trades={})
mx = dict(ann=0.0, dd=0.0, ntr=0.0)
for c, r in RS.items():
    m = me[c]
    v = (m["ann_2019"] * 100, m["ann_full"] * 100, m["ann_2024"] * 100, m["mdd_full"] * 100, m["n_trades"])
    d = [round(v[k] - r[k], 3) for k in range(5)]
    out["cells"][c] = dict(mine=[round(x, 3) for x in v], researcher=list(r), diff=d)
    mx["ann"] = max(mx["ann"], abs(d[0]), abs(d[1])); mx["dd"] = max(mx["dd"], abs(d[3]))
    mx["ntr"] = max(mx["ntr"], abs(d[4]) / r[4])
for c, r in RS_ROB.items():
    rb = me["robust"][c]; m = me[c]
    v = (rb["x2"]["ann_2019"], rb["x2"]["ann_full"], rb["block5"]["ann_2019"], rb["block5"]["ann_full"],
         m["ann_h1"], m["ann_h2"])
    v = [x * 100 for x in v]
    out["robust"][c] = dict(mine=[round(x, 3) for x in v], researcher=list(r), diff=[round(v[k] - r[k], 3) for k in range(6)],
                            blocked_mine=[b["code"] for b in rb["block5"]["blocked"]], cls_mine=rb["cls"])
for c, r in RS_CAP.items():
    out["cap"][c] = dict(mine=(me[c]["cap_hits"], me[c]["short_skip"]), researcher=r)
for c, r in RS_PATH.items():
    p = me["paths20"][c]
    v = (p["mdd_deepest"] * 100, p["mdd_median"] * 100, p["mdd_shallowest"] * 100, p["n_within_36"],
         [round(p["mdd_median_by_shift"][str(k)] * 100, 2) for k in range(4)],
         (p["ret_min"] * 100, p["ret_median"] * 100, p["ret_max"] * 100))
    out["paths"][c] = dict(mine=[round(v[0], 2), round(v[1], 2), round(v[2], 2), v[3], v[4], [round(x, 2) for x in v[5]]],
                           researcher=r, diff_dd=[round(v[k] - r[k], 3) for k in range(3)],
                           diff_ret=[round(v[5][k] - r[5][k], 3) for k in range(3)])
for k, r in RS_PAIR.items():
    p = me["paths20"][k]
    out["pairs"][k] = dict(mine=(round(p["ret_diff_median_pp"], 2), p["ret_n_pos"], round(p["dd_shallower_median_pp"], 2),
                                 p["dd_n_shallower"]), researcher=r)
for (c, d), r in RS_DLY.items():
    x = me["delay"][c]["d%d" % d]
    v = (x["ann_2019"] * 100, x["ann_full"] * 100, x["ann_2024"] * 100, x["mdd_full"] * 100)
    o = dict(mine=[round(y, 3) for y in v], researcher=r, diff=[round(v[k] - r[k], 3) for k in range(4)])
    if (c, d) in RS_DLY_CAPS:
        o["caps_median_mine"] = round(x["caps_median_2019"] * 100, 3); o["caps_median_researcher"] = RS_DLY_CAPS[(c, d)]
    out["delay"][(c + "_d%d" % d)] = o
    mx["ann"] = max(mx["ann"], abs(o["diff"][0]), abs(o["diff"][1])); mx["dd"] = max(mx["dd"], abs(o["diff"][3]))

# 逐筆交易
rt = pd.read_csv(os.path.join(UP, "trades.csv.gz"), dtype={"code": str})
mt = pd.read_csv(os.path.join(HERE, "my_trades.csv.gz"), dtype={"code": str})
rt = rt[rt.why != "open_end"]
for c in RS:
    a = rt[rt.run == c + "|strict"]; b = mt[mt.run == c]
    ka = set(zip(a.code, a.ent_date, a.exit_date)); kb = set(zip(b.code, b.buy_date, b.sell_date))
    mg = a.merge(b, left_on=["code", "ent_date", "exit_date"], right_on=["code", "buy_date", "sell_date"])
    shd = (mg.sh - mg.shares).abs() / mg.shares
    out["trades"][c] = dict(n_researcher=len(a), n_mine=len(b), matched=len(ka & kb), only_researcher=len(ka - kb),
                            only_mine=len(kb - ka), shares_diff_n=int((shd > 0).sum()),
                            shares_diff_gt_0p1pct=int((shd > 0.001).sum()), shares_diff_max=float(shd.max()),
                            examples_only_researcher=sorted(ka - kb)[:5], examples_only_mine=sorted(kb - ka)[:5])
for c in ("WN_70L_C12", "W10_70L"):
    for d in (1, 2):
        a = rt[rt.run == "%s|strict_d%d" % (c, d)] if ("%s|strict_d%d" % (c, d)) in set(rt.run) else None
        out["trades"]["%s_d%d" % (c, d)] = "研究員未存延遲版交易" if a is None or len(a) == 0 else len(a)
out["runs_in_researcher_trades"] = sorted(set(rt.run))
out["max_abs"] = dict(ann_pp=mx["ann"], dd_pp=mx["dd"], ntrades_frac=mx["ntr"])
out["verdict"] = "可重現" if (mx["ann"] <= 0.5 and mx["dd"] <= 0.5 and mx["ntr"] <= 0.02) else "有超出門檻的差異"
json.dump(out, open(os.path.join(HERE, "compare_vs_researcher.json"), "w"), ensure_ascii=False, indent=1, default=str)
print(json.dumps(out["max_abs"]), out["verdict"])
for k in ("cells", "robust", "cap", "pairs", "delay", "trades"):
    for c, v in out[k].items():
        print(k, c, v if not isinstance(v, dict) else {a: b for a, b in v.items() if not a.startswith("examples")})
for c, v in out["paths"].items():
    print("paths", c, v["mine"], v["diff_dd"], v["diff_ret"])

# ---- 凍結後由凍結數字推導（不重跑）：只含底倉鎖定格的路徑前緣；審查 B 加「鎖定」虛擬變數
LOCKED = ["WN_70L_C12", "WN_70L_C10", "WN_70L_C08", "WN_60L", "WN_60L_C10", "W10_70L", "WN_70L", "W10_80L", "WN_80L",
          "W7_70L", "W7_80L"]
pz = me["paths20"]


def frontier(pts):
    on = []
    for a, (ra, da) in pts.items():
        if not any((rb >= ra and db >= da and (rb > ra or db > da)) for b, (rb, db) in pts.items() if b != a):
            on.append(a)
    return sorted(on, key=lambda k: pts[k][1])


pts = {c: (pz[c]["ret_median"], pz[c]["mdd_median"]) for c in LOCKED}
pts1 = {c: (me[c]["ann_2019"], me[c]["mdd_full"]) for c in LOCKED}
ALL17 = list(me["reviewB"]["table"])
x = np.array([me[c]["per_stock_w_daily"] for c in ALL17]); y = np.array([pz[c]["mdd_median"] for c in ALL17])
lk = np.array([1.0 if c in LOCKED else 0.0 for c in ALL17])
A = np.vstack([np.ones_like(x), x, lk]).T
cf, *_ = np.linalg.lstsq(A, y, rcond=None); yh = A @ cf
r2 = 1 - ((y - yh) ** 2).sum() / ((y - y.mean()) ** 2).sum()
xl = x[lk == 1]; yl = y[lk == 1]
cl = np.polyfit(xl, yl, 1); r2l = 1 - ((yl - np.polyval(cl, xl)) ** 2).sum() / ((yl - yl.mean()) ** 2).sum()
out["post_freeze_derived"] = dict(
    locked_only_path_frontier_2019=frontier(pts), locked_only_single_path_frontier_2019=frontier(pts1),
    reviewB_with_lock_dummy=dict(intercept=float(cf[0]), slope=float(cf[1]), lock=float(cf[2]), r2=float(r2)),
    reviewB_locked_only=dict(slope=float(cl[0]), intercept=float(cl[1]), r2=float(r2l), n=int(len(xl))))
json.dump(out, open(os.path.join(HERE, "compare_vs_researcher.json"), "w"), ensure_ascii=False, indent=1, default=str)
print(out["post_freeze_derived"])
