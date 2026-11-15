# -*- coding: utf-8 -*-
"""S17 獨立複製：凍結後比對。my_numbers.json（sha256 見 my_numbers.sha256）凍結後，才讀研究員 FINDINGS.md 全文與 trades.csv.gz。
研究員的數字從 FINDINGS.md 文字抄入（RS 字典，單位 %）；results*.json、navs.csv.gz 不讀。
用法：python3 -I -B rep_s17_compare.py
輸出：compare_vs_researcher.json
"""
import os, json, hashlib
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RS_DIR = os.path.dirname(HERE)
CELLS = ["W10_70", "WN_70"]
DEFS = ["STD", "W12", "W36", "RATE", "SM3", "NOMU", "ROB"]

# FINDINGS 第 3 節：路徑中位數、最小、最大、回撤中位數、2019～2023 中位數、2024 起中位數、主跑 2019／全期／回撤、對 STD 正的條數、中位數
T3 = {
    "W10_70": {
        "NOMU": (51.03, 48.50, 52.89, -32.40, 38.67, 75.16, 51.83, 35.92, -31.95, 20, 4.92),
        "STD": (46.23, 43.24, 48.52, -34.01, 35.69, 65.98, 48.13, 34.08, -33.40, None, None),
        "SM3": (45.79, 40.92, 48.55, -38.78, 38.65, 58.81, 46.35, 32.67, -39.79, 9, -1.45),
        "W36": (44.34, 40.84, 47.66, -37.49, 39.32, 53.75, 46.99, 32.67, -36.04, 3, -1.47),
        "ROB": (41.17, 37.00, 44.28, -39.02, 38.34, 47.83, 41.99, 29.89, -35.40, 0, -5.15),
        "RATE": (36.90, 32.92, 41.32, -36.96, 28.06, 53.94, 39.53, 28.17, -37.61, 0, -7.87),
        "W12": (35.50, 33.62, 41.70, -34.37, 24.48, 57.85, 40.72, 29.48, -34.18, 0, -9.17)},
    "WN_70": {
        "NOMU": (67.11, 60.82, 73.90, -37.96, 54.98, 90.25, 67.97, 46.66, -37.25, 20, 9.34),
        "STD": (55.33, 52.26, 63.49, -41.93, 51.82, 63.13, 62.24, 44.56, -39.50, None, None),
        "SM3": (54.02, 48.71, 61.41, -48.72, 53.10, 56.13, 52.08, 37.00, -51.65, 8, -1.98),
        "W36": (53.03, 45.63, 57.93, -48.19, 54.43, 45.54, 56.13, 40.20, -43.22, 0, -4.98),
        "ROB": (51.01, 42.07, 54.97, -47.15, 51.81, 49.09, 51.52, 37.69, -41.31, 0, -6.12),
        "RATE": (42.80, 39.57, 47.62, -45.99, 37.21, 52.05, 45.07, 32.45, -44.16, 0, -13.04),
        "W12": (39.19, 33.43, 52.70, -41.22, 31.58, 54.32, 51.28, 38.93, -39.94, 0, -16.43)}}
FAM = {"W10_70": dict(family_median=44.34, all140_median=43.27, all140_p10=34.98, P3=1.88),
       "WN_70": dict(family_median=53.03, all140_median=52.63, all140_p10=40.36, P3=2.30)}
SEG = {"W10_70": dict(NOMU=(2.83, 20, 10.57, 20), W36_1923=3.48), "WN_70": dict(NOMU=(4.47, 18, 21.36, 20), W36_1923=3.45)}
MF = {"W10_70": {0.45: 4.90, 0.55: 4.80}, "WN_70": {0.45: 9.47, 0.55: 8.80}}
BEAR = {  # 第 4 節（20 條路徑中位數）；括號判定
    "W10_70": {"STD": (-16.8, -26.5, -29.7, -34.0), "NOMU": (-18.4, -26.7, -30.2, -32.4), "W12": (-16.1, -28.0, -27.7, -34.4),
               "W36": (-18.7, -26.3, -27.0, -37.5), "RATE": (-17.8, -26.7, -28.5, -37.0), "SM3": (-17.9, -28.3, -25.3, -38.8),
               "ROB": (-17.9, -25.5, -28.1, -39.0)},
    "WN_70": {"STD": (-25.6, -23.6, -29.6, -41.9), "NOMU": (-34.1, -25.1, -31.5, -38.0), "W12": (-24.7, -28.1, -23.7, -41.2),
              "W36": (-33.9, -23.6, -25.8, -48.2), "RATE": (-30.9, -24.6, -29.2, -46.0), "SM3": (-35.2, -27.7, -23.5, -48.7),
              "ROB": (-30.2, -22.5, -28.4, -47.1)}}
BEAR_CLS = {
    "W10_70": {"NOMU": "----", "W12": "-D--", "W36": "D-SD", "RATE": "---D", "SM3": "DDSD", "ROB": "--SD"},
    "WN_70": {"NOMU": "DDDS", "W12": "-DSS", "W36": "D-SD", "RATE": "DD-D", "SM3": "DDSD", "ROB": "D--D"}}
BK = ["2018", "2020-03", "2022", "2024-07~2025-04"]
X2B = {"W10_70": dict(STD=(42.28, 29.70, 43.42), NOMU=(46.33, 32.07, 46.50), RATE=(34.60, 24.34, 33.79)),
       "WN_70": dict(STD=(53.44, 36.69, 51.47), NOMU=(58.29, 39.74, 56.05), RATE=(36.96, 25.95, 34.61))}
YR = {"W10_70": {"STD": [15.2, 22.2, 1.8, 33.6, 99.4, 46.4, -18.9, 52.7, 60.2, 56.5, 78.1],
                 "W12": [18.1, 20.9, 1.0, 37.9, 86.8, 22.2, -14.2, 29.8, 68.2, 31.2, 86.5],
                 "W36": [19.6, 17.9, -4.3, 34.8, 91.8, 57.3, -15.1, 52.1, 62.9, 24.3, 90.9],
                 "RATE": [12.9, 22.9, -1.1, 27.7, 84.4, 36.1, -17.1, 30.5, 53.4, 44.1, 76.2],
                 "SM3": [18.3, 26.8, -6.8, 41.6, 58.5, 64.3, -9.1, 53.6, 46.6, 51.5, 71.6],
                 "NOMU": [22.0, 19.7, -5.6, 36.0, 98.1, 70.0, -20.9, 43.5, 73.8, 45.2, 99.2],
                 "ROB": [22.4, 21.6, -6.3, 34.4, 76.5, 75.3, -20.2, 36.3, 54.2, 56.6, 41.9]},
      "WN_70": {"STD": [12.0, 26.3, 16.5, 29.5, 187.8, 54.1, -16.9, 74.9, 70.7, 56.1, 97.6],
                "NOMU": [18.9, 22.2, 1.8, 44.6, 160.0, 90.0, -19.6, 50.8, 86.4, 46.6, 143.3],
                "SM3": [14.6, 37.8, -5.8, 44.5, 83.1, 77.7, -0.8, 59.8, 41.7, 52.6, 64.3],
                "W36": [19.6, 21.9, 8.0, 28.3, 155.3, 80.1, -12.7, 61.9, 74.6, 5.0, 112.7]}}
TOP5 = {"W10_70": [24.4, 23.7, 21.6, 23.5, 24.6, 31.3, 28.6], "WN_70": [33.1, 35.3, 30.9, 39.0, 35.9, 39.4, 46.2]}
OVL = dict(W12=(62, 63), W36=(69, 71), RATE=(69, 76), SM3=(52, 57), NOMU=(68, 67), ROB=(63, 70))
RA = {"W10_70": dict(pre=(6.23, 5.96, -0.44, 8), t19=(0.23, 0.82), t1923=(0.11, 0.37), t24=(0.45, 0.76),
                     blk_union=(4.26, 20), blk_2408=(4.56, 20), yearly_wins=5),
      "WN_70": dict(pre=(8.69, 5.56, -3.53, 6), t19=(0.30, 0.63), t1923=(-0.07, -0.13), t24=(0.96, 1.05),
                    blk_union=(7.71, 20), blk_2408=(8.73, 20), yearly_wins=5)}
RB = dict(mu_sd=(0.76, 0.28), rev12_yi=(95, 83), price=(87, 64))
RC = dict(nomu_r20=(2.50, 2.16, 2.88), std_r20=2.01, nomu_r10=(3.19, 2.80, 3.58), std_r10=2.5)


def main():
    h = hashlib.sha256(open(os.path.join(HERE, "my_numbers.json"), "rb").read()).hexdigest()
    hs = open(os.path.join(HERE, "my_numbers.sha256")).read().split()[0]
    assert h == hs, (h, hs)
    M = json.load(open(os.path.join(HERE, "my_numbers.json")))
    out = dict(my_numbers_sha256=h, note="研究員數字抄自 FINDINGS.md（單位 %、四捨五入到 0.01）；results*.json、navs.csv.gz 未讀。")
    P20, MAIN, PR, DEC = M["paths20"], M["main"], M["paired"], M["decisions"]
    tab = {}; worst = dict(ann=0.0, mdd=0.0)
    for c in CELLS:
        for d in DEFS:
            r = T3[c][d]; k = "%s|%s" % (d, c); p = P20[k]; m = MAIN[k]
            mine = [p["ann_2019_median"] * 100, p["ann_2019_min"] * 100, p["ann_2019_max"] * 100, p["mdd_median"] * 100,
                    p["ann_1923_median"] * 100, p["ann_2024_median"] * 100, m["ann_2019"] * 100, m["ann_full"] * 100,
                    m["mdd_full"] * 100]
            if d != "STD":
                pc = PR["%s-STD|%s" % (d, c)]["ann_2019"]
                mine += [pc["n_pos"], pc["median"] * 100]
            else:
                mine += [None, None]
            names = ["path_med", "path_min", "path_max", "mdd_med", "seg1923_med", "seg2024_med", "main2019", "mainfull",
                     "mainmdd", "n_pos", "paired_med"]
            row = {}
            for nm, a, b in zip(names, mine, r):
                if b is None:
                    continue
                row[nm] = dict(mine=round(a, 2), researcher=b, diff=round(a - b, 2))
                if nm in ("path_med", "path_min", "path_max", "seg1923_med", "seg2024_med", "main2019", "mainfull", "paired_med"):
                    worst["ann"] = max(worst["ann"], abs(a - b))
                if nm in ("mdd_med", "mainmdd"):
                    worst["mdd"] = max(worst["mdd"], abs(a - b))
            tab[k] = row
    out["table3"] = tab
    fam = {}
    for c in CELLS:
        o = DEC[c]
        mine = dict(family_median=o["P1"]["family_median"] * 100, all140_median=o["P1"]["all140_median"] * 100,
                    all140_p10=o["P1"]["all140_p10"] * 100, P3=o["P3"] * 100)
        fam[c] = {k: dict(mine=round(v, 2), researcher=FAM[c][k], diff=round(v - FAM[c][k], 2)) for k, v in mine.items()}
        fam[c]["rank_return"] = dict(mine=o["P2"]["rank_return"], researcher=2)
        fam[c]["rank_mdd"] = dict(mine=o["P2"]["rank_mdd"], researcher=2 if c == "W10_70" else 3)
        fam[c]["order"] = dict(mine=o["P2"]["order_return"], researcher=["NOMU", "STD", "SM3", "W36", "ROB", "RATE", "W12"])
        fam[c]["optimism"] = dict(mine=o["optimism"], researcher="不必因定義打折")
        pc = PR["NOMU-STD|%s" % c]
        s = SEG[c]["NOMU"]
        fam[c]["NOMU_segments"] = dict(mine=(round(pc["ann_1923"]["median"] * 100, 2), pc["ann_1923"]["n_pos"],
                                             round(pc["ann_2024"]["median"] * 100, 2), pc["ann_2024"]["n_pos"]), researcher=s)
        fam[c]["W36_seg1923"] = dict(mine=round(PR["W36-STD|%s" % c]["ann_1923"]["median"] * 100, 2), researcher=SEG[c]["W36_1923"])
        fam[c]["NOMU_minfrac"] = {str(mf): dict(mine=(round(M["paired_minfrac"]["NOMU-STD|%s|%.2f" % (c, mf)]["median"] * 100, 2),
                                                      M["paired_minfrac"]["NOMU-STD|%s|%.2f" % (c, mf)]["n_pos"]),
                                                researcher=(v, 20)) for mf, v in MF[c].items()}
        fam[c]["cls"] = {d: o["vs_std"][d]["cls"] for d in DEFS[1:]}
    out["family"] = fam
    out["decisions_mine"] = dict(std_is_best=DEC["std_is_best"], stable=DEC["stable"], xq=DEC["xq"])
    out["decisions_researcher"] = dict(std_is_best="前段（兩格第 2）", stable=dict(NOMU="穩定地比較好", others="否"),
                                       cls=dict(W12="比標準差／比標準差", RATE="比標準差／比標準差", ROB="比標準差／比標準差",
                                                W36="差不多／比標準差", SM3="差不多／差不多", NOMU="比標準好／比標準好"),
                                       eight="14 個全過")
    # 第 4 節
    bt = {}; bmax = 0.0; cls_diff = []
    for c in CELLS:
        for d in DEFS:
            mm = [P20["%s|%s" % (d, c)]["bears_median"][b] * 100 for b in BK]
            bt["%s|%s" % (d, c)] = dict(mine=[round(x, 2) for x in mm], researcher=BEAR[c][d])
            bmax = max(bmax, max(abs(a - b) for a, b in zip(mm, BEAR[c][d])))
            if d != "STD":
                mc = "".join({"較深": "D", "較淺": "S", "差不多": "-"}[DEC[c]["vs_std"][d]["bears"][b]["cls"]] for b in BK)
                if mc != BEAR_CLS[c][d]:
                    cls_diff.append(dict(cell=c, defn=d, mine=mc, researcher=BEAR_CLS[c][d],
                                         detail={b: DEC[c]["vs_std"][d]["bears"][b] for b in BK}))
    out["bears"] = dict(table=bt, max_abs_diff_pp=round(bmax, 3), note="研究員取到 0.1，差 ≤ 0.05 視為相同", cls_diffs=cls_diff)
    # 第 5 節
    x2 = {}
    for c in CELLS:
        for d, v in X2B[c].items():
            m = MAIN["%s|%s" % (d, c)]
            mine = (m["x2"]["ann_2019"] * 100, m["x2"]["ann_full"] * 100, m["block5"]["ann_2019"] * 100)
            x2["%s|%s" % (d, c)] = dict(mine=[round(x, 2) for x in mine], researcher=v, diff=[round(a - b, 2) for a, b in zip(mine, v)])
    out["x2_block5"] = x2
    out["eight_checks_mine"] = M["decisions"]["eight_checks"]
    # 第 6 節
    yr = {}; ymax = 0.0
    for c, dd in YR.items():
        for d, v in dd.items():
            mine = [MAIN["%s|%s" % (d, c)]["yearly"][str(y)] * 100 for y in range(2016, 2027)]
            yr["%s|%s" % (d, c)] = dict(mine=[round(x, 1) for x in mine], researcher=v)
            ymax = max(ymax, max(abs(a - b) for a, b in zip(mine, v)))
    out["yearly"] = dict(table=yr, max_abs_diff_pp=round(ymax, 2))
    out["top5_share_paths_median"] = {c: dict(mine=[round(P20["%s|%s" % (d, c)]["top5_share_median"] * 100, 1) for d in DEFS],
                                              researcher=TOP5[c]) for c in CELLS}
    ov = M["posthoc"]["overlap_vs_STD_shift0"]
    out["overlap"] = {d: dict(mine=(round(ov[d]["buy_overlap_of_std"] * 100), round(ov[d]["top20_overlap"] * 100), ov[d]["n_months"]),
                              researcher=OVL[d] + (125,)) for d in OVL}
    # 第 7 節（審查 A／B／C 對照）
    ra = {}
    for c in CELLS:
        o = M["reviewA"][c]; r = RA[c]
        ra[c] = dict(pre2019=dict(mine=(round(o["pre2019_paths"]["std_median"] * 100, 2), round(o["pre2019_paths"]["nomu_median"] * 100, 2),
                                        round(o["pre2019_paths"]["paired_median"] * 100, 2), o["pre2019_paths"]["paired_n_pos"]),
                                  researcher=r["pre"]),
                     t_from2019=dict(mine=(round(o["monthly_t"]["from2019"]["mean"] * 100, 2), round(o["monthly_t"]["from2019"]["t"], 2)), researcher=r["t19"]),
                     t_2019_2023=dict(mine=(round(o["monthly_t"]["2019_2023"]["mean"] * 100, 2), round(o["monthly_t"]["2019_2023"]["t"], 2)), researcher=r["t1923"]),
                     t_from2024=dict(mine=(round(o["monthly_t"]["from2024"]["mean"] * 100, 2), round(o["monthly_t"]["from2024"]["t"], 2)), researcher=r["t24"]),
                     block_union=dict(mine=(round(o["blocks"]["top5_union"]["paths_2019_median"] * 100, 2), o["blocks"]["top5_union"]["paths_2019_n_pos"]),
                                      researcher=r["blk_union"]),
                     block_2408=dict(mine=(round(o["blocks"]["no_2408"]["paths_2019_median"] * 100, 2), o["blocks"]["no_2408"]["paths_2019_n_pos"]),
                                     researcher=r["blk_2408"]),
                     yearly_wins=dict(mine=o["yearly_main_wins"], researcher=r["yearly_wins"]))
    out["reviewA_vs_researcher"] = ra
    b = M["reviewB"]["buy"]
    out["reviewB_vs_researcher"] = dict(mu_sd=dict(mine=(round(b["NOMU"]["mu_over_sd_median"], 2), round(b["STD"]["mu_over_sd_median"], 2)), researcher=RB["mu_sd"]),
                                        rev12_yi=dict(mine=(round(b["NOMU"]["rev12m_median_kTWD"] / 1e5, 1), round(b["STD"]["rev12m_median_kTWD"] / 1e5, 1)), researcher=RB["rev12_yi"]),
                                        price=dict(mine=(b["NOMU"]["price_median"], b["STD"]["price_median"]), researcher=RB["price"]))
    c_ = M["reviewC"]
    out["reviewC_vs_researcher"] = dict(nomu_r20=dict(mine=(round(c_["NOMU"]["rank20"]["median"], 2), round(c_["NOMU"]["rank20"]["q25"], 2), round(c_["NOMU"]["rank20"]["q75"], 2)), researcher=RC["nomu_r20"]),
                                        std_r20=dict(mine=round(c_["STD"]["rank20"]["median"], 2), researcher=RC["std_r20"]),
                                        nomu_r10=dict(mine=(round(c_["NOMU"]["rank10"]["median"], 2), round(c_["NOMU"]["rank10"]["q25"], 2), round(c_["NOMU"]["rank10"]["q75"], 2)), researcher=RC["nomu_r10"]),
                                        std_r10=dict(mine=round(c_["STD"]["rank10"]["median"], 2), researcher=RC["std_r10"]))
    # 逐筆交易
    rt = pd.read_csv(os.path.join(RS_DIR, "trades.csv.gz"), dtype={"code": str})
    mt = pd.read_csv(os.path.join(HERE, "my_trades.csv.gz"), dtype={"code": str})
    trd = {}; nmax = 0.0
    for c in CELLS:
        for d in DEFS:
            k = "%s|%s" % (d, c)
            r = rt[rt.run == k]; rc_ = r[r.why == "rule"]; ro = r[r.why == "open_end"]
            m = mt[(mt["def"] == d) & (mt.cell == c)]
            a = m.set_index(["code", "buy_date", "sell_date"]); bb = rc_.set_index(["code", "ent_date", "exit_date"])
            com = a.index.intersection(bb.index)
            only_m = sorted(set(a.index) - set(bb.index)); only_r = sorted(set(bb.index) - set(a.index))
            dret = (a.loc[com, "ret"] - bb.loc[com, "ret"]).abs()
            n_my_open = MAIN[k]["n_open"]
            nd = abs(len(m) - len(rc_)) / len(rc_)
            nmax = max(nmax, nd)
            trd[k] = dict(mine_closed=int(len(m)), researcher_closed=int(len(rc_)), mine_open=n_my_open, researcher_open=int(len(ro)),
                          matched=int(len(com)), only_mine=[list(x) for x in only_m][:10], only_researcher=[list(x) for x in only_r][:10],
                          n_only_mine=len(only_m), n_only_researcher=len(only_r),
                          ret_absdiff_max_pp=round(float(dret.max()) * 100, 3) if len(dret) else None,
                          ret_absdiff_n_gt_0p1pp=int((dret > 0.001).sum()), count_diff_pct=round(nd * 100, 3))
    out["trades"] = trd
    out["summary"] = dict(max_abs_ann_diff_pp=round(worst["ann"], 3), max_abs_mdd_diff_pp=round(worst["mdd"], 3),
                          max_trade_count_diff_pct=round(nmax * 100, 3),
                          pass_ann=bool(worst["ann"] <= 0.5), pass_mdd=bool(worst["mdd"] <= 0.5), pass_count=bool(nmax <= 0.02),
                          decisions_same=bool(DEC["std_is_best"] == "前段" and DEC["stable"]["NOMU"] == "穩定地比較好" and
                                              all(DEC["stable"][d] == "否" for d in DEFS[1:] if d != "NOMU")))
    json.dump(out, open(os.path.join(HERE, "compare_vs_researcher.json"), "w"), ensure_ascii=False, indent=1)
    print(json.dumps(out["summary"], ensure_ascii=False))
    print(json.dumps(out["family"], ensure_ascii=False))
    print(json.dumps(out["bears"]["cls_diffs"], ensure_ascii=False), out["bears"]["max_abs_diff_pp"], out["yearly"]["max_abs_diff_pp"])
    for k, v in tab.items():
        print(k, {n: x["diff"] for n, x in v.items()})
    for k, v in trd.items():
        print(k, {n: x for n, x in v.items() if n not in ("only_mine", "only_researcher")}, v["only_mine"][:3], v["only_researcher"][:3])
    print(json.dumps(out["x2_block5"], ensure_ascii=False))
    print(json.dumps(out["top5_share_paths_median"]), json.dumps(out["overlap"]))
    print(json.dumps(ra, ensure_ascii=False)); print(out["reviewB_vs_researcher"], out["reviewC_vs_researcher"])


if __name__ == "__main__":
    main()
