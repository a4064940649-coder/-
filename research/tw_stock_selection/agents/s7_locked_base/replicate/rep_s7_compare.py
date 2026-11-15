# -*- coding: utf-8 -*-
"""S7 獨立複製：凍結後比對。先驗證 my_numbers.json 的 sha256 與凍結值一致，再讀研究員 trades.csv.gz 逐筆比對；
FINDINGS.md 的數字手抄在 RESEARCHER（讀 FINDINGS 全文後抄錄；不讀 results*.json、navs.csv.gz）。
用法：python3 -I -B rep_s7_compare.py
輸出：compare_vs_researcher.json
"""
import os, json, hashlib
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
RD = os.path.dirname(HERE)

h = hashlib.sha256(open(os.path.join(HERE, "my_numbers.json"), "rb").read()).hexdigest()
frozen = open(os.path.join(HERE, "my_numbers.sha256")).read().split()[0]
assert h == frozen, "my_numbers.json 在凍結後被改過"
M = json.load(open(os.path.join(HERE, "my_numbers.json")))

# 研究員 FINDINGS 數字（%；回撤為負）
RESEARCHER = {
    "W10_70L": dict(ann_2019=47.49, ann_full=33.68, ann_2024=68.14, mdd_full=-33.38, x2=(41.70, 29.33), block5=(43.32, 31.06), h=(26.52, 41.21), paths=(41.72, 45.36, 47.86)),
    "W7_70L": dict(ann_2019=52.92, ann_full=36.72, ann_2024=67.74, mdd_full=-37.26),
    "WN_70L": dict(ann_2019=60.22, ann_full=43.07, ann_2024=78.19, mdd_full=-39.34, x2=(51.02, 35.47), block5=(52.51, 38.28), h=(35.30, 51.24), paths=(44.86, 56.35, 62.16)),
    "W10_80L": dict(ann_2019=50.51, ann_full=35.58, ann_2024=70.41, mdd_full=-34.71),
    "W7_80L": dict(ann_2019=57.89, ann_full=41.78, ann_2024=69.12, mdd_full=-39.38),
    "WN_80L": dict(ann_2019=66.27, ann_full=46.79, ann_2024=82.84, mdd_full=-40.32, x2=(54.83, 37.94), block5=(54.70, 39.65), h=(38.65, 55.35), paths=(46.77, 59.59, 66.27)),
    "W10_70": dict(ann_2019=48.13, ann_full=34.08, ann_2024=70.16, mdd_full=-33.40, paths=(43.24, 46.23, 48.51)),
    "WN_70": dict(ann_2019=62.24, ann_full=44.55, ann_2024=80.41, mdd_full=-39.50, paths=(52.26, 55.33, 63.49)),
    "WN_100": dict(ann_2019=74.68, ann_full=51.25, ann_2024=90.75, mdd_full=-42.91),
}
R_LOCK = {  # 底倉平均／中位數、<10%、<20%、拉回上／下、縮小／跳過（天）、個股部位
    "W10_70L": (26.9, 27.1, 0.0, 1.2, 2, 0, 5, 1, 6, 37.3), "W7_70L": (28.6, 29.3, 0.0, 1.8, 3, 0, 19, 38, 29, 48.1),
    "WN_70L": (26.3, 25.5, 0.0, 4.3, 4, 0, 39, 72, 62, 59.7), "W10_80L": (17.3, 18.2, 0.6, 76.3, 1, 0, 5, 1, 6, 43.0),
    "W7_80L": (17.4, 17.5, 1.9, 69.1, 2, 0, 17, 47, 34, 57.6), "WN_80L": (16.5, 16.5, 1.3, 80.2, 2, 0, 31, 75, 59, 66.6),
    "W10_70": (26.8, 27.3, 0.0, 2.0, 3, 0, 0, 0, 0, 37.4), "WN_70": (19.6, 22.4, 21.0, 35.9, 45, 0, 11, 9, 13, 67.0),
}
R_DELAY = {  # d1 (2019, full, 2024, mdd), d2 (...)
    "WN_70L": ((55.05, 39.11, 67.97, -42.35), (52.60, 38.17, 59.51, -45.50), (60.22, 56.29, 52.60)),
    "W10_70L": ((43.87, 31.08, 62.35, -38.01), (42.21, 30.06, 58.40, -36.37), (47.49, 43.87, 42.21)),
    "WN_70": ((56.66, 39.70, 68.35, -45.17), (52.81, 38.28, 60.90, -44.99), (62.97, 56.27, 52.81)),
}
R_DELAY_OTHER = {"W7_70L": (4.03, 7.38), "W10_80L": (3.85, 6.44), "W7_80L": (4.95, 8.48), "WN_80L": (6.45, 9.34),
                 "W10_70": (3.64, 5.49), "WN_100": (4.32, 13.13)}
R_DELAY_MDD_WN100 = (-51.31, -52.52)
R_REVA = dict(diff_2019=-2.02, diff_full=-1.49, keep_2019=93.9, keep_full=93.6, dd_saved=0.16,
              paths_median=-2.05, paths_min=-8.18, paths_max=5.12, paths_npos=6,
              w10_diff_2019=-0.64, w10_paths_median=-0.88, w10_paths_min=-1.65, w10_paths_max=-0.45, w10_npos=0,
              yearly_diff={2016: 1.1, 2017: 0.8, 2018: -4.0, 2019: -2.5, 2020: -3.8, 2021: 5.9, 2022: -0.4,
                           2023: -10.7, 2024: -8.7, 2025: 11.8, 2026: -10.7},
              wn80_paths_median=2.57, wn80_npos=15)
R_YEARLY = {
    "WN_70L": {2016: 13.0, 2017: 27.1, 2018: 12.5, 2019: 26.9, 2020: 184.0, 2021: 60.0, 2022: -17.3, 2023: 64.2, 2024: 62.0, 2025: 67.9, 2026: 86.8},
    "WN_70": {2016: 12.0, 2017: 26.3, 2018: 16.5, 2019: 29.5, 2020: 187.8, 2021: 54.1, 2022: -16.9, 2023: 74.9, 2024: 70.7, 2025: 56.1, 2026: 97.6},
    "W10_70L": {2016: 15.2, 2017: 22.2, 2018: 1.8, 2019: 33.6, 2020: 99.4, 2021: 46.4, 2022: -18.9, 2023: 52.7, 2024: 60.2, 2025: 51.0, 2026: 78.4},
    "WN_80L": {2016: 11.0, 2017: 28.0, 2018: 13.9, 2019: 32.2, 2020: 212.4, 2021: 65.3, 2022: -16.4, 2023: 70.6, 2024: 64.4, 2025: 72.0, 2026: 93.2},
}
R_DD = {"WN_70L": (78.3, 19.2, 65.6, 22.1), "WN_70": (100.0, 0.0, 70.2, 18.7), "W10_70L": (76.1, 23.9, 45.9, 25.6),
        "WN_80L": (81.8, 13.7, 68.3, 15.9)}

out = dict(my_sha256=h)
pp = lambda x: round(100 * x, 2)
# ---- 主表
tab = {}
worst = dict(ann=0.0, mdd=0.0, ntr=0.0)
tr_r = pd.read_csv(os.path.join(RD, "trades.csv.gz"), dtype={"code": str})
tr_r = tr_r[tr_r.why != "open_end"]
tr_m = pd.read_csv(os.path.join(HERE, "my_trades.csv.gz"), dtype={"code": str})
for cid, r in RESEARCHER.items():
    m = M[cid]
    row = {}
    for k in ("ann_2019", "ann_full", "ann_2024", "mdd_full"):
        row[k] = dict(mine=pp(m[k]), researcher=r[k], diff=round(pp(m[k]) - r[k], 2))
        key = "mdd" if k == "mdd_full" else "ann"
        if k != "ann_2024":
            worst[key] = max(worst[key], abs(pp(m[k]) - r[k]))
    rt = tr_r[tr_r.run == cid + "|strict"]
    mt = tr_m[tr_m.run == cid]
    row["n_trades"] = dict(mine=int(len(mt)), researcher=int(len(rt)),
                           diff_pct=round(100 * (len(mt) - len(rt)) / len(rt), 2))
    worst["ntr"] = max(worst["ntr"], abs(row["n_trades"]["diff_pct"]))
    # 逐筆：代號＋進場日＋出場日
    a = set(zip(mt.code, mt.buy_date, mt.sell_date)); b = set(zip(rt.code, rt.ent_date, rt.exit_date))
    a2 = set(zip(mt.code, mt.buy_date)); b2 = set(zip(rt.code, rt.ent_date))
    row["trade_match"] = dict(exact_code_entry_exit=len(a & b), only_mine=len(a - b), only_researcher=len(b - a),
                              same_code_entry=len(a2 & b2),
                              first_mismatch_mine=sorted(a - b)[:5], first_mismatch_researcher=sorted(b - a)[:5])
    if len(a & b):
        mm = mt.set_index(["code", "buy_date", "sell_date"]); rr = rt.set_index(["code", "ent_date", "exit_date"])
        ks = sorted(a & b)
        dret = np.array([mm.loc[k, "ret"] if np.ndim(mm.loc[k, "ret"]) == 0 else mm.loc[k, "ret"].iloc[0] for k in ks]) - \
            np.array([rr.loc[k, "ret"] if np.ndim(rr.loc[k, "ret"]) == 0 else rr.loc[k, "ret"].iloc[0] for k in ks])
        row["trade_match"]["matched_ret_absdiff_max"] = float(np.abs(dret).max())
        row["trade_match"]["matched_ret_absdiff_median"] = float(np.median(np.abs(dret)))
    for k in ("x2", "block5", "h", "paths"):
        if k in r:
            if k == "paths":
                mine = (pp(M["paths20"][cid]["min"]), pp(M["paths20"][cid]["median"]), pp(M["paths20"][cid]["max"]))
            elif k == "h":
                mine = (pp(m["ann_h1"]), pp(m["ann_h2"]))
            else:
                x = M["robust"][cid][k]; mine = (pp(x["ann_2019"]), pp(x["ann_full"]))
            row[k] = dict(mine=mine, researcher=r[k], diff=[round(u - v, 2) for u, v in zip(mine, r[k])])
    if cid in R_LOCK:
        q = R_LOCK[cid]
        mine = (round(100 * m["base_w_mean"], 1), round(100 * m["base_w_median"], 1), round(100 * m["base_lt10"], 1),
                round(100 * m["base_lt20"], 1), m["reb_up"], m["reb_down"], m["short_shrink"], m["short_skip"],
                m["short_days"], round(100 * m["avg_stock_w"], 1))
        row["lock_stats"] = dict(fields="底倉平均,中位數,<10%,<20%,拉回上,拉回下,縮小,跳過,受影響日,個股部位",
                                 mine=mine, researcher=q, identical=[u == v for u, v in zip(mine, q)])
    tab[cid] = row
out["cells"] = tab
out["worst_abs_diff"] = dict(ann_2019_or_full_pp=round(worst["ann"], 3), mdd_pp=round(worst["mdd"], 3),
                             n_trades_pct=worst["ntr"])
# ---- 延遲
dl = {}
for cid, (r1, r2, rc) in R_DELAY.items():
    o = {}
    for d, r in ((1, r1), (2, r2)):
        x = M["delay"][cid]["d%d" % d]
        mine = (pp(x["ann_2019"]), pp(x["ann_full"]), pp(x["ann_2024"]), pp(x["mdd_full"]))
        o["d%d" % d] = dict(mine=mine, researcher=r, diff=[round(u - v, 2) for u, v in zip(mine, r)])
    mine = (pp(M["delay"][cid]["d0"]["caps_median_2019"]), pp(M["delay"][cid]["d1"]["caps_median_2019"]),
            pp(M["delay"][cid]["d2"]["caps_median_2019"]))
    o["caps_median_d0_d1_d2"] = dict(mine=mine, researcher=rc, diff=[round(u - v, 2) for u, v in zip(mine, rc)])
    dl[cid] = o
for cid, r in R_DELAY_OTHER.items():
    b0 = M[cid]["ann_2019"]
    mine = (pp(b0 - M["delay"][cid]["d1"]["ann_2019"]), pp(b0 - M["delay"][cid]["d2"]["ann_2019"]))
    dl[cid] = dict(loss_d1_d2_2019=dict(mine=mine, researcher=r, diff=[round(u - v, 2) for u, v in zip(mine, r)]))
dl["WN_100"]["mdd_d1_d2"] = dict(mine=(pp(M["delay"]["WN_100"]["d1"]["mdd_full"]), pp(M["delay"]["WN_100"]["d2"]["mdd_full"])),
                                 researcher=R_DELAY_MDD_WN100)
out["delay"] = dl
# ---- 審查 A
le = M["lock_effect"]["WN_70L_vs_WN_70"]; pa = M["paths20"]["diff_WN_70L_minus_WN_70"]
pw = M["paths20"]["diff_W10_70L_minus_W10_70"]; p8 = M["paths20"]["diff_WN_80L_minus_WN_70"]
yd = M["reviewA"]["WN_70L_minus_WN_70"]["yearly_diff"]
out["reviewA"] = dict(
    mine=dict(diff_2019=pp(le["diff_2019"]), diff_full=pp(le["diff_full"]), keep_2019=pp(le["keep_2019"]),
              keep_full=pp(le["keep_full"]), dd_saved=pp(le["dd_saved"]), paths_median=pp(pa["median"]),
              paths_min=pp(pa["min"]), paths_max=pp(pa["max"]), paths_npos=pa["n_pos"],
              w10_diff_2019=pp(M["lock_effect"]["W10_70L_vs_W10_70"]["diff_2019"]), w10_paths_median=pp(pw["median"]),
              w10_paths_min=pp(pw["min"]), w10_paths_max=pp(pw["max"]), w10_npos=pw["n_pos"],
              yearly_diff={int(k): round(100 * v, 1) for k, v in yd.items()},
              wn80_paths_median=pp(p8["median"]), wn80_npos=p8["n_pos"]),
    researcher=R_REVA)
out["yearly"] = {cid: dict(mine={int(k): round(100 * v, 1) for k, v in M[cid]["yearly"].items() if int(k) >= 2016},
                           researcher=r,
                           diff={k: round(100 * M[cid]["yearly"][str(k)] - r[k], 1) for k in r}) for cid, r in R_YEARLY.items()}
out["reviewB_dd"] = {cid: dict(mine=(pp(M["reviewB"][cid]["stock_w_at_peak"]), pp(M["reviewB"][cid]["base_w_at_peak"]),
                                     pp(M["reviewB"][cid]["stock_w_dd"]), pp(M["reviewB"][cid]["base_w_dd"])),
                               researcher=r) for cid, r in R_DD.items()}
# ---- 分歧追查：WN_70L 2025 年逐筆差
for cid in ("WN_70L", "WN_100"):
    rt = tr_r[tr_r.run == cid + "|strict"]; mt = tr_m[tr_m.run == cid]
    a = set(zip(mt.code, mt.buy_date, mt.sell_date)); b = set(zip(rt.code, rt.ent_date, rt.exit_date))
    out.setdefault("divergence", {})[cid] = dict(only_mine=sorted(a - b), only_researcher=sorted(b - a))
# 份額比對（同代號同進場）：股數差異
for cid in ("WN_70L", "WN_80L", "W7_70L"):
    rt = tr_r[tr_r.run == cid + "|strict"].set_index(["code", "ent_date"])
    mt = tr_m[tr_m.run == cid].set_index(["code", "buy_date"])
    ks = sorted(set(rt.index) & set(mt.index))
    dif = [(k[0], k[1], int(mt.loc[k, "sh"] if np.ndim(mt.loc[k, "sh"]) == 0 else mt.loc[k, "sh"].iloc[0]),
            int(rt.loc[k, "shares"] if np.ndim(rt.loc[k, "shares"]) == 0 else rt.loc[k, "shares"].iloc[0])) for k in ks]
    bad = [x for x in dif if x[2] != x[3]]
    out.setdefault("shares_check", {})[cid] = dict(n=len(dif), n_diff=len(bad), first=bad[:10])
json.dump(out, open(os.path.join(HERE, "compare_vs_researcher.json"), "w"), ensure_ascii=False, indent=1, default=str)
print(json.dumps(out["worst_abs_diff"]))
for cid, r in tab.items():
    print(cid, {k: (v["diff"] if "diff" in v else v.get("diff_pct")) for k, v in r.items() if k not in ("trade_match", "lock_stats")},
          {k: r["trade_match"][k] for k in ("exact_code_entry_exit", "only_mine", "only_researcher")},
          r.get("lock_stats", {}).get("identical"))
print(json.dumps(out["delay"], ensure_ascii=False))
print(json.dumps(out["reviewA"], ensure_ascii=False))
print(json.dumps(out["yearly"], ensure_ascii=False))
print(json.dumps(out["reviewB_dd"]))
print(json.dumps(out["divergence"], ensure_ascii=False)[:3000])
print(json.dumps(out["shares_check"], ensure_ascii=False))
