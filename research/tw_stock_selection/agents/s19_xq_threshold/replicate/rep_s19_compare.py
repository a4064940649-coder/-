# -*- coding: utf-8 -*-
"""S19 獨立複製：凍結後比對（只讀研究員 FINDINGS.md 全文的數字與 trades.csv.gz；results*.json、navs.csv.gz 不讀）。
用法：python3 -I -B rep_s19_compare.py [<scratch_dir>]（給 scratch_dir 時另做分歧診斷：腳本股池逐列比對）
輸出：compare_vs_researcher.json
"""
import sys, os, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np, pandas as pd

AG = os.path.dirname(HERE)
# 研究員 FINDINGS 第 3 節（主跑 2019／全期／2024／回撤；路徑中位數；回撤中位數）
RES = {
    "E0|W10_70": (48.13, 34.08, 70.16, -33.40, 46.23, -34.01), "XR|W10_70": (49.48, 35.24, 74.19, -30.06, 45.79, -33.63),
    "XU|W10_70": (47.05, 32.72, 62.44, -38.53, 43.98, -39.12), "XT|W10_70": (46.47, 32.32, 62.45, -38.58, 44.01, -39.60),
    "XT0|W10_70": (50.85, 35.68, 60.35, -39.88, 47.38, -40.53), "XDYN|W10_70": (46.22, 32.04, 69.23, -39.93, 45.07, -40.51),
    "XS3|W10_70": (48.21, 34.16, 69.61, -38.24, 45.96, -38.76),
    "E0|SAT": (41.61, 30.10, 56.53, -25.08, 39.54, -26.46), "XR|SAT": (44.10, 32.46, 63.14, -23.82, 38.36, -26.35),
    "XU|SAT": (45.03, 30.09, 49.88, -32.42, 40.14, -34.21), "XT|SAT": (40.46, 26.74, 49.86, -32.35, 39.98, -34.36),
    "XT0|SAT": (57.03, 39.14, 61.23, -44.56, 49.22, -44.09), "XDYN|SAT": (42.96, 28.09, 57.56, -27.89, 41.22, -32.09),
    "XS3|SAT": (39.32, 27.17, 48.69, -28.43, 34.68, -28.75)}
RES_EIGHT = {"XT|SAT": "有條件", "XDYN|SAT": "有條件", "XS3|SAT": "有條件"}
RES_PAIR = {  # 2019 起中位數（為正條數）、2019～2023、2024 起、回撤
    "XT-E0|W10_70": (-1.64, 2, 1.24, -7.73, -4.80), "XR-E0|W10_70": (1.10, 13, 0.80, 1.22, 0.40),
    "XU-XR|W10_70": (-2.06, 2, 1.92, -7.92, -5.42), "XT-XU|W10_70": (0.06, 10, -0.17, 0.09, -0.29),
    "XT0-E0|W10_70": (1.27, 15, 8.70, -13.42, -6.79), "XDYN-E0|W10_70": (-1.26, 6, 2.01, -2.68, -7.11),
    "XS3-E0|W10_70": (-0.14, 9, 2.31, -1.16, -4.88),
    "XT-E0|SAT": (0.06, 11, 6.21, -6.53, -6.97), "XR-E0|SAT": (0.71, 13, 0.61, -1.00, 0.76),
    "XU-XR|SAT": (0.89, 12, 5.52, -8.11, -8.01), "XT-XU|SAT": (0.36, 12, 0.12, 0.66, 0.01),
    "XT0-E0|SAT": (11.27, 20, 20.99, -7.53, -18.15), "XDYN-E0|SAT": (2.25, 19, 6.10, -3.52, -5.50),
    "XS3-E0|SAT": (-3.86, 0, 1.69, -13.32, -2.58)}
RES_J = {"J1_W10_70": "略有不同（較低）", "J1_SAT": "接近", "J2_W10_70": "threshold", "J2_SAT": "三者都小",
         "J3_W10_70": {"2018": "差不多", "2020-03": "差不多", "2022": "較淺", "2024-07~2025-04": "差不多"},
         "J3_SAT": {"2018": "較深", "2020-03": "差不多", "2022": "較淺", "2024-07~2025-04": "較深"},
         "J4": "不必改動態", "J5": "XT", "J5_warn_xt0": False}
RES_LIST = {"XU": (6.7, 4, 42, 24, 4, 0.83, 0.62), "XT": (6.5, 3, 43, 24, 5, 0.79, 0.58), "XT0": (50.3, 38, 218, 120, 0, 0.95, 0.11),
            "XDYN": (5.6, 3, 32, 22, 8, 0.78, 0.55), "XS3": (3.5, 2, 21, 12, 22, 0.58, 0.47)}
RES_YEAR = {  # XT 純衛星、E0 純衛星、XT W10、E0 W10
    2016: (-1.55, 9.01, 17.15, 15.23), 2017: (8.37, 10.95, 20.01, 22.22), 2018: (-3.95, 6.93, -4.12, 1.83),
    2019: (7.15, 7.88, 36.25, 33.61), 2020: (79.13, 109.86, 82.04, 99.37), 2021: (113.55, 46.86, 91.84, 46.42),
    2022: (-3.21, -11.56, -19.32, -18.92), 2023: (14.93, 46.15, 31.33, 52.74), 2024: (35.62, 32.97, 63.47, 60.20),
    2025: (39.21, 58.18, 44.68, 56.45), 2026: (65.38, 67.80, 65.65, 78.11)}
RES_S5 = {"extra_buys": (72, -1.59, -0.86), "whole": (72, 11.52, 1.28), "extra_holds": (82, 3.30, 1.17),
          "missed_all": (169, 4.14, 2.92), "missed_full10": (31, 11.14, 2.49)}


def main(sc=None):
    me = json.load(open(os.path.join(HERE, "my_numbers.json")))
    out = {}
    # ---------------- 主跑
    mt = {}
    for k, r in RES.items():
        m = me["main"][k]; p = me["paths20"][k]
        mine = (m["ann_2019"] * 100, m["ann_full"] * 100, m["ann_2024"] * 100, m["mdd_full"] * 100,
                p["ann_2019_median"] * 100, p["mdd_median"] * 100)
        d = [a - b for a, b in zip(mine, r)]
        mt[k] = dict(mine=[round(x, 2) for x in mine], researcher=r, diff=[round(x, 2) for x in d],
                     within_0_5pp=bool(abs(d[0]) <= 0.5 and abs(d[1]) <= 0.5 and abs(d[3]) <= 0.5),
                     eight_mine=me["main"][k]["J7"], eight_res=RES_EIGHT.get(k, "值得上 XQ 測"))
    out["main"] = mt
    # ---------------- 交易筆數與逐筆
    rt = pd.read_csv(os.path.join(AG, "trades.csv.gz"), dtype={"code": str})
    my = pd.read_csv(os.path.join(HERE, "my_trades.csv.gz"), dtype={"code": str})
    my["run"] = my.strat + "|" + my.fund
    tc = {}
    for run, g in rt.groupby("run"):
        gm = my[my.run == run]
        a = set(zip(g.code, g.ent_date)); b = set(zip(gm.code, gm.buy_date))
        ga = g[g.why != "open"] if "open" in set(g.why) else g
        n_res = len(g); n_me = len(gm)
        tc[run] = dict(n_res_rows=n_res, n_me_rows=n_me, n_diff_pct=round(100.0 * (n_me - n_res) / n_res, 2),
                       why_res=g.why.value_counts().to_dict(), common_entries=len(a & b), only_res=len(a - b), only_me=len(b - a),
                       jaccard=round(len(a & b) / len(a | b), 4),
                       only_res_first=sorted(a - b, key=lambda x: x[1])[:12], only_me_first=sorted(b - a, key=lambda x: x[1])[:12])
        both = g.merge(gm, left_on=["code", "ent_date"], right_on=["code", "buy_date"], suffixes=("_r", "_m"))
        both = both[both.exit_date.notna() & (both.sell_date.fillna("") != "")]
        if len(both):
            tc[run]["same_exit_share"] = round(float((both.exit_date == both.sell_date).mean()), 4)
            tc[run]["max_abs_ret_diff_same_exit"] = float((both[both.exit_date == both.sell_date].ret_r -
                                                           both[both.exit_date == both.sell_date].ret_m).abs().max())
    out["trades"] = tc
    # ---------------- 配對
    pc = {}
    for k, r in RES_PAIR.items():
        q = me["paired"][k]
        mine = (q["ann_2019"]["median"] * 100, q["ann_2019"]["n_pos"], q["ann_1923"]["median"] * 100, q["ann_2024"]["median"] * 100,
                q["mdd_full"]["median"] * 100)
        pc[k] = dict(mine=[round(x, 2) for x in mine], researcher=r, diff_2019=round(mine[0] - r[0], 2))
    out["paired"] = pc
    # ---------------- 判定
    dec = me["decisions"]
    jm = {"J1_W10_70": dec["W10_70"]["J1"]["ann_2019"]["cls"], "J1_SAT": dec["SAT"]["J1"]["ann_2019"]["cls"],
          "J2_W10_70": dec["W10_70"]["J2"]["J2"], "J2_SAT": dec["SAT"]["J2"]["J2"],
          "J3_W10_70": {k: v["cls"] for k, v in dec["W10_70"]["J3"].items()}, "J3_SAT": {k: v["cls"] for k, v in dec["SAT"]["J3"].items()},
          "J4": dec["J4"]["cls"], "J5": dec["J5"]["recommend"], "J5_warn_xt0": dec["J5"]["warn_xt0"]}
    out["decisions"] = dict(mine=jm, researcher=RES_J,
                            same={k: (str(jm[k]).replace("較", "") == str(RES_J[k]).replace("較", "")) for k in RES_J},
                            J1_segments_mine={f: {w: dec[f]["J1"][w]["cls"] for w in ("ann_1923", "ann_2024")} for f in ("W10_70", "SAT")},
                            J7_mine=dec["J7"])
    # ---------------- 名單
    lc = {}
    for c, r in RES_LIST.items():
        q = me["list_diff"][c]
        lc[c] = dict(mine=(round(q["cand_mean"], 2), q["cand_median"], q["cand_max"], q["months_cand_gt10"], q["months_cand0"],
                           round(q["share_xr_in_cand"], 3), round(q["jaccard_mean"], 3)), researcher=r)
    out["list_diff"] = lc
    # ---------------- 逐年
    yc = {}
    for y, r in RES_YEAR.items():
        ys = str(y)
        mine = (me["yearly"]["XT|SAT"][ys]["main"] * 100, me["yearly"]["E0|SAT"][ys]["main"] * 100,
                me["yearly"]["XT|W10_70"][ys]["main"] * 100, me["yearly"]["E0|W10_70"][ys]["main"] * 100)
        yc[y] = dict(mine=[round(x, 2) for x in mine], researcher=r, diff=[round(a - b, 2) for a, b in zip(mine, r)])
    out["yearly"] = yc
    s5 = me["s5_reviewA"]["W10_70"]
    out["s5"] = dict(mine=dict(extra_buys=(s5["a_extra_buys"]["n"], s5["a_extra_buys"]["mean"] * 100, s5["a_extra_buys"]["t_naive"]),
                               whole=(s5["a_whole_trade"]["n"], s5["a_whole_trade"]["mean"] * 100, s5["a_whole_trade"]["t_naive"]),
                               extra_holds=(s5["b_extra_holds"]["n"], s5["b_extra_holds"]["mean"] * 100, s5["b_extra_holds"]["t_naive"]),
                               missed_all=(s5["c_missed_all"]["n"], s5["c_missed_all"]["mean"] * 100, s5["c_missed_all"]["t_naive"]),
                               missed_full10=(s5["c_missed_full10"]["n"], s5["c_missed_full10"]["mean"] * 100, s5["c_missed_full10"]["t_naive"],
                                              s5["c_missed_full10"]["t_cluster_month"], s5["c_missed_full10"]["n_months"]),
                               missed_noslot=(s5["c_missed_noslot"]["n"], s5["c_missed_noslot"]["mean"] * 100, s5["c_missed_noslot"]["t_naive"],
                                              s5["c_missed_noslot"]["t_cluster_month"], s5["c_missed_noslot"]["n_months"])),
                     researcher=RES_S5)
    # ---------------- 分歧診斷：腳本股池
    if sc:
        out["diag"] = diag(sc, rt, my)
        f60 = os.path.join(sc, "f60", "my_numbers.json")
        if os.path.exists(f60):
            out["diag_full60"] = f60_summary(json.load(open(f60)), rt)
    js = json.dumps(out, ensure_ascii=False, indent=1, default=str)
    open(os.path.join(HERE, "compare_vs_researcher.json"), "w").write(js)
    print(js[:20000])


def f60_summary(f, rt):
    """rep_s19_f60.py（腳本股池＋full60）的結果對研究員。逐筆比對用 f60 沒有存交易檔，改比主跑數字、配對、判定。"""
    o = dict(note="腳本股池加上 full60（近 60 列都有資料）後重跑整套；只用來找分歧來源，不是複製員的主結果。")
    mm = {}
    for k, r in RES.items():
        m = f["main"][k]; p = f["paths20"][k]
        v = (m["ann_2019"] * 100, m["ann_full"] * 100, m["ann_2024"] * 100, m["mdd_full"] * 100, p["ann_2019_median"] * 100, p["mdd_median"] * 100)
        mm[k] = dict(f60=[round(x, 2) for x in v], researcher=r, diff=[round(a - b, 2) for a, b in zip(v, r)], n_trades=m["n_trades"],
                     n_res=int((rt.run == k).sum()), eight=m["J7"], n_checks=m["n_checks_pass"])
    o["main"] = mm
    o["paired"] = {k: dict(f60=[round(f["paired"][k]["ann_2019"]["median"] * 100, 2), f["paired"][k]["ann_2019"]["n_pos"],
                                round(f["paired"][k]["ann_1923"]["median"] * 100, 2), round(f["paired"][k]["ann_2024"]["median"] * 100, 2),
                                round(f["paired"][k]["mdd_full"]["median"] * 100, 2)], researcher=r) for k, r in RES_PAIR.items()}
    o["list_diff"] = {c: dict(f60=(round(f["list_diff"][c]["cand_mean"], 2), f["list_diff"][c]["cand_median"], f["list_diff"][c]["cand_max"],
                                   f["list_diff"][c]["months_cand_gt10"], f["list_diff"][c]["months_cand0"],
                                   round(f["list_diff"][c]["share_xr_in_cand"], 3), round(f["list_diff"][c]["jaccard_mean"], 3)), researcher=r)
                      for c, r in RES_LIST.items()}
    s5 = f["s5_reviewA"]["W10_70"]
    o["s5"] = {k: s5[k] for k in ("a_extra_buys", "a_whole_trade", "b_extra_holds", "c_missed_all", "c_missed_full10",
                                  "c_missed_full10_in_cand", "c_missed_noslot")}
    o["yearly_XT"] = {y: dict(SAT=round(f["yearly"]["XT|SAT"][str(y)]["main"] * 100, 2), W10_70=round(f["yearly"]["XT|W10_70"][str(y)]["main"] * 100, 2),
                              researcher=(RES_YEAR[y][0], RES_YEAR[y][2])) for y in RES_YEAR}
    o["decisions"] = dict(J1={fd: f["decisions"][fd]["J1"]["ann_2019"]["cls"] for fd in ("W10_70", "SAT")},
                          J2={fd: f["decisions"][fd]["J2"]["J2"] for fd in ("W10_70", "SAT")}, J4=f["decisions"]["J4"]["cls"],
                          J5=f["decisions"]["J5"])
    o["reviewB"] = {k: dict(main=v.get("main", {}) and {a: b for a, b in v["main"].items() if a != "yearly"},
                            paths_2019_median=v.get("paths_2019_median"), diff_vs_XT_2019_median=v.get("diff_vs_XT_2019_median"))
                    for k, v in f["reviewB"].items()}
    return o


def diag(sc, rt, my):
    import rep_s19_lib as K
    S, Q, R, L = K.S, K.Q, K.R, K.L
    P = R.build(sc)
    excl = S.innov_mask_fn(P, sc)
    K.script_panel(P)
    da = np.array(P.dstr)
    o = {}
    # XT｜W10_70：研究員有、我沒有的進場 → 在訊號日的腳本股池條件
    g = rt[rt.run == "XT|W10_70"]; gm = my[my.run == "XT|W10_70"]
    a = set(zip(g.code, g.ent_date)); b = set(zip(gm.code, gm.buy_date))
    rows = []
    for code, ent in sorted(a - b, key=lambda x: x[1])[:40]:
        i = P.cidx.get(code)
        t = int(np.searchsorted(da, ent)); s = t - 1
        if i is None:
            rows.append((code, ent, "not_in_panel")); continue
        rows.append((code, ent, dict(Cadj=round(float(P.Cadj[s, i]), 3), C=float(P.C[s, i]), adv_adj=round(float(P.adv_adj[s, i]) / 1e7, 3),
                                     adv60=round(float(P.adv60[s, i]) / 1e7, 3), traded=bool(P.traded[s, i]),
                                     res_univ=bool(K.research_universe(P, s, excl)[i]), scr_univ=bool(K.script_universe(P, s, excl)[i]))))
    o["only_res_XT_W10"] = rows
    rows = []
    for code, ent in sorted(b - a, key=lambda x: x[1])[:40]:
        i = P.cidx.get(code)
        t = int(np.searchsorted(da, ent)); s = t - 1
        rows.append((code, ent, dict(Cadj=round(float(P.Cadj[s, i]), 3), C=float(P.C[s, i]), adv_adj=round(float(P.adv_adj[s, i]) / 1e7, 3),
                                     adv60=round(float(P.adv60[s, i]) / 1e7, 3),
                                     res_univ=bool(K.research_universe(P, s, excl)[i]), scr_univ=bool(K.script_universe(P, s, excl)[i]))))
    o["only_me_XT_W10"] = rows
    return o


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
