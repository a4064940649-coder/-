# -*- coding: utf-8 -*-
"""S15 獨立複製：凍結後比對（my_numbers.json 已凍結、sha256 已寫）。
讀：my_numbers.json、my_trades.csv.gz、my_events.csv.gz、研究員 FINDINGS.md 全文（數字手抄進 RES）與 trades.csv.gz。
不讀 results*.json、navs.csv.gz。
另做一個凍結後診斷（不改凍結數字）：0050 掃入（閒錢 ≥ 2 萬買 0050）不收最低手續費 20 元的引擎變體，
看 P／P40／E0／C 的交易清單是否和研究員逐筆相同。需要 scratch 面板。
用法：python3 -I -B rep_s15_compare.py <scratch_dir>
輸出：compare_vs_researcher.json
"""
import sys, os, json, hashlib
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np, pandas as pd

RDIR = os.path.dirname(HERE)
THR_ANN, THR_MDD, THR_N = 0.5, 0.5, 0.02

# 研究員 FINDINGS 第 3、4、6、7 節（%／個百分點）
RES = {
    "main": {  # 2019 起、全期、2019～2023、2024 起、最大回撤、筆數（含未平倉，trades.csv.gz 列數）
        "E0": [48.13, 34.08, 37.01, 70.16, -33.40, 387], "P": [19.98, 13.32, 19.40, 21.04, -38.06, 1297],
        "C": [38.49, 25.74, 28.61, 57.97, -32.33, 1609], "C0": [39.19, 27.48, 26.34, 65.35, -30.32, 341],
        "P40": [23.85, 16.71, 12.87, 46.07, -52.15, 660]},
    "bench": [28.91, 21.14, 16.43, 54.48, -33.96],
    "paths": {  # 中位數、最少、最多、贏 0050 條數、回撤中位數、2019～2023 中位數、2024 起中位數
        "E0": [46.23, 43.24, 48.51, 20, -34.01, 35.69, 65.98], "P": [23.34, 17.25, 26.76, 0, -38.97, 22.68, 21.84],
        "C": [36.78, 34.34, 40.49, 20, -33.60, 28.72, 54.18], "C0": [37.84, 35.92, 41.12, 20, -31.57, 26.65, 61.80],
        "P40": [25.92, 21.91, 30.75, 3, -45.81, 14.93, 48.19]},
    "paired": {  # 2019 起（正條數、中位數）、2019～2023（正、中位數）、2024 起（正、中位數）、回撤較淺（條數、中位數）
        "P-0050": [0, -5.57, 18, 6.25, 0, -32.63, 0, -5.01], "C-C0": [0, -1.14, 20, 2.00, 0, -7.40, 0, -2.64],
        "C-E0": [0, -8.89, 0, -8.25, 0, -11.43, 8, -0.17], "P40-P": [18, 1.75, 2, -7.24, 20, 22.81, 0, -7.36]},
    "paired_range_2019": {"P-0050": [-11.66, -2.15], "C-C0": [-1.83, -0.17], "C-E0": [-12.31, -6.24]},
    "bears_median": {"0050": [-15.99, -28.22, -33.96, -27.48], "E0": [-16.79, -26.47, -29.69, -34.01],
                     "P": [-25.10, -33.60, -31.08, -38.67], "C": [-16.93, -28.93, -30.04, -33.60],
                     "C0": [-16.16, -27.60, -30.53, -31.57]},
    "bears_CC0": [[2, -0.96, "較深"], [0, -1.37, "較深"], [14, 0.68, "差不多"], [0, -2.64, "較深"]],
    "bears_P0050": [[0, -9.11, "較深"], [0, -5.39, "較深"], [17, 2.88, "較抗跌"], [0, -11.19, "較深"]],
    "eight": {  # x2 2019、x2 全期、禁買 2019、禁買全期、前半、後半、通過數、分類
        "E0": [42.28, 29.70, 43.42, 31.12, 26.52, 42.05, 8, "值得上 XQ 測"],
        "P": [9.80, 4.69, 16.50, 11.03, 11.18, 15.49, 0, "不值得"],
        "C": [33.70, 21.19, 34.98, 23.52, 16.59, 35.55, 7, "有條件"],
        "C0": [35.94, 25.15, 36.28, 25.62, 19.58, 35.85, 8, "值得上 XQ 測"],
        "P40": [17.04, 11.02, 19.18, 13.65, 15.43, 17.98, 0, "不值得"]},
    "decisions": {"P": "輸 0050", "C": "沒有補充價值", "xq": "不建議加進現行策略", "P40": "輸 0050"},
    "other": {"top5_share_P": 38.4, "top5_share_E0": 24.9, "corr_full": 0.36, "corr_2019": 0.37, "corr_paths_med": 0.47,
              "overlap_share": 3.56, "overlap_days": 752, "P_trades_overlap_E0": 78, "P_win": 44.1, "P_avg_win": 14.23,
              "P_avg_loss": 8.97, "P_exp": 1.27, "turnover": 8.07, "cost_share": 10.14, "P_mean_npos": 9.37,
              "P_full_share": 86.0},
    "event": {"n": 5104, "cum": {1: -0.36, 5: -0.09, 10: 0.50, 20: 1.70, 40: 3.50, 60: 5.72},
              "exc": {1: -0.44, 5: -0.56, 10: -0.45, 20: -0.29, 40: -0.50, 60: -0.31},
              "exc_med": {1: -0.69, 5: -1.24, 10: -1.67, 20: -2.19, 40: -3.55, 60: -4.61},
              "t": {1: -7.55, 5: -5.06, 10: -3.07, 20: -1.34, 40: -1.68, 60: -0.82},
              "gap": 0.39, "oo20_exc": -0.06, "oo20_t": -0.26, "all_n": 12152, "all_oo20_exc": 0.01, "all_oo20_t": 0.10,
              "period_oo20": {"2015_2018": [-0.23, -0.66], "2019_2023": [0.83, 2.70], "2024_": [-1.43, -3.01]},
              "year_oo20": {"2016": -1.32, "2017": 1.13, "2018": -0.95, "2019": -0.73, "2020": -0.45, "2021": 2.99,
                            "2022": 0.43, "2023": 1.78, "2024": -2.12, "2025": -1.95, "2026": 0.70},
              "rs_terc": {"high": [0.40, 1.36], "mid": [0.01, None], "low": [-0.23, None]}},
    "yearly_P_minus_0050": {"2016": -10.15, "2017": 9.00, "2018": -16.36, "2019": -13.67, "2020": 7.60, "2021": 11.55,
                            "2022": 4.34, "2023": 4.54, "2024": -27.30, "2025": -39.61, "2026": -22.28},
    "yearly_C_minus_C0": {"2016": -7.40, "2017": 0.98, "2018": -5.65, "2019": -7.76, "2020": 8.90, "2021": 0.62,
                          "2022": 3.17, "2023": 7.27, "2024": -6.55, "2025": -8.36, "2026": -4.87},
}
BK = ["2018", "2020-03", "2022", "2024-07~2025-04"]


def pct(x):
    return None if x is None else round(100.0 * x, 2)


def main(sc):
    mine = json.load(open(os.path.join(HERE, "my_numbers.json")))
    h = hashlib.sha256(open(os.path.join(HERE, "my_numbers.json"), "rb").read()).hexdigest()
    assert open(os.path.join(HERE, "my_numbers.sha256")).read().split()[0] == h
    out = dict(my_numbers_sha256=h, thresholds=dict(ann_pp=THR_ANN, mdd_pp=THR_MDD, n_rel=THR_N))
    cells = mine["cells"]
    # ---------------- 主跑
    main_ = {}
    fails = []
    for c, r in RES["main"].items():
        o = cells[c]
        m = [pct(o["ann_2019"]), pct(o["ann_full"]), pct(o["ann_1923"]), pct(o["ann_2024"]), pct(o["mdd_full"]),
             o["n_trades_incl_open"]]
        d = [round(a - b, 2) for a, b in zip(m[:5], r[:5])]
        nrel = (m[5] - r[5]) / r[5]
        ok = all(abs(x) <= THR_ANN for x in d[:4]) and abs(d[4]) <= THR_MDD and abs(nrel) <= THR_N
        main_[c] = dict(mine=m, res=r, diff=d, n_rel=round(nrel, 4), within=ok)
        if not ok:
            fails.append("main:" + c)
    out["main"] = main_
    # ---------------- 20 條路徑
    p20 = {}
    for c, r in RES["paths"].items():
        o = mine["paths20"][c]
        m = [pct(o["ann_2019_median"]), pct(o["ann_2019_min"]), pct(o["ann_2019_max"]), o["n_beat_0050"],
             pct(o["mdd_median"]), pct(o["ann_1923_median"]), pct(o["ann_2024_median"])]
        d = [round(a - b, 2) for a, b in zip(m, r)]
        ok = abs(d[0]) <= THR_ANN and abs(d[4]) <= THR_MDD and abs(d[5]) <= THR_ANN and abs(d[6]) <= THR_ANN and d[3] == 0
        p20[c] = dict(mine=m, res=r, diff=d, within=ok)
        if not ok:
            fails.append("paths:" + c)
    out["paths20"] = p20
    # ---------------- 逐條配對
    pr = {}
    for k, r in RES["paired"].items():
        o = mine["paired"][k]
        m = [o["ann_2019"]["n_pos"], pct(o["ann_2019"]["median"]), o["ann_1923"]["n_pos"], pct(o["ann_1923"]["median"]),
             o["ann_2024"]["n_pos"], pct(o["ann_2024"]["median"]), o["mdd_full"]["n_pos"], pct(o["mdd_full"]["median"])]
        pr[k] = dict(mine=m, res=r, diff=[round(a - b, 2) for a, b in zip(m, r)],
                     range_2019_mine=[pct(o["ann_2019"]["min"]), pct(o["ann_2019"]["max"])],
                     range_2019_res=RES["paired_range_2019"].get(k))
    out["paired"] = pr
    # ---------------- 空頭
    bm = {}
    for c, r in RES["bears_median"].items():
        if c == "0050":
            m = [pct(mine["bench_0050"]["bears"][k]) for k in BK]
        else:
            m = [pct(mine["paths20"][c]["bears_median"][k]) for k in BK]
        bm[c] = dict(mine=m, res=r, diff=[round(a - b, 2) for a, b in zip(m, r)])
    out["bears_median"] = bm
    dec = mine["decisions"]
    out["bears_cls"] = dict(
        C_C0=[dict(seg=k, mine=[dec["bears_C_vs_C0"][k]["n_shallower"], pct(dec["bears_C_vs_C0"][k]["median"]),
                                dec["bears_C_vs_C0"][k]["cls"]], res=RES["bears_CC0"][j]) for j, k in enumerate(BK)],
        P_0050=[dict(seg=k, mine=[dec["bears_P_vs_0050"][k]["n_shallower"], pct(dec["bears_P_vs_0050"][k]["median"]),
                                  dec["bears_P_vs_0050"][k]["cls"]], res=RES["bears_P0050"][j]) for j, k in enumerate(BK)])
    out["bears_cls_same"] = all(x["mine"][2] == x["res"][2] for v in out["bears_cls"].values() for x in v)
    # ---------------- 八項
    ei = {}
    for c, r in RES["eight"].items():
        o = cells[c]
        m = [pct(o["x2"]["ann_2019"]), pct(o["x2"]["ann_full"]), pct(o["block5"]["ann_2019"]), pct(o["block5"]["ann_full"]),
             pct(o["ann_h1"]), pct(o["ann_h2"]), o["n_checks_pass"], o["eight_cls"]]
        ei[c] = dict(mine=m, res=r, diff=[round(a - b, 2) for a, b in zip(m[:6], r[:6])], same_cls=m[7] == r[7])
    out["eight"] = ei
    # ---------------- 判定
    out["decisions"] = dict(
        P=dict(mine=dec["P_vs_0050"], res=RES["decisions"]["P"]), C=dict(mine=dec["C_supplement"]["cls"], res=RES["decisions"]["C"]),
        xq=dict(mine=dec["xq"], res=RES["decisions"]["xq"]), P40=dict(mine=dec["P40_vs_0050_desc"], res=RES["decisions"]["P40"]))
    out["decisions_all_same"] = all(v["mine"] == v["res"] for v in out["decisions"].values()) and \
        all(v["same_cls"] for v in ei.values()) and out["bears_cls_same"]
    # ---------------- 其他
    co = mine["corr_overlap"]; ts = cells["P"]["trade_stats"]; pe = mine["P_main_engine"]
    out["other"] = dict(
        mine=dict(top5_share_P=pct(cells["P"]["top5_share_of_gain"]), top5_share_E0=pct(cells["E0"]["top5_share_of_gain"]),
                  corr_full=round(co["P_E0_excess_full"], 3), corr_2019=round(co["P_E0_excess_2019"], 3),
                  corr_paths_med=round(co["paths20_median"], 3), overlap_share=pct(co["C_overlap_mean_share_of_P_half"]),
                  overlap_days=co["C_overlap_days_any"], P_trades_overlap_E0=co["P_main_trades_overlapping_E0_trades"],
                  P_win=pct(ts["win"]), P_avg_win=pct(ts["avg_win"]), P_avg_loss=pct(-ts["avg_loss"]), P_exp=pct(ts["exp"]),
                  turnover=round(pe["turnover_buy_per_year_over_mean_nav"], 2),
                  cost_share=pct(pe["cost_total_per_year_over_mean_nav"]), P_mean_npos=round(pe["mean_npos"], 2),
                  P_full_share=pct(pe["share_days_full"])),
        res=RES["other"], top5_codes_mine=[x["code"] for x in cells["P"]["block5_top"]],
        top5_codes_res=["2243", "4764", "1721", "3037", "6770"])
    # ---------------- 事件層
    ev = mine["event_level"]["independent_buyable"]
    evo = dict(n=[ev["n"], RES["event"]["n"]])
    for hz in (1, 5, 10, 20, 40, 60):
        evo["d%d" % hz] = dict(cum=[pct(ev["cum%d" % hz]["mean"]), RES["event"]["cum"][hz]],
                               exc=[pct(ev["exc%d" % hz]["mean"]), RES["event"]["exc"][hz]],
                               exc_med=[pct(ev["exc%d" % hz]["median"]), RES["event"]["exc_med"][hz]],
                               t=[round(ev["exc%d" % hz]["t"], 2), RES["event"]["t"][hz]])
    evo["gap_raw"] = [pct(ev["gap"]["mean"]), RES["event"]["gap"]]
    evo["oo20_exc"] = [pct(ev["oo20_exc"]["mean"]), RES["event"]["oo20_exc"], round(ev["oo20_exc"]["t"], 2), RES["event"]["oo20_t"]]
    al = mine["event_level"]["all_buyable"]
    evo["all_oo20_exc"] = [al["oo20_exc"]["n"], pct(al["oo20_exc"]["mean"]), round(al["oo20_exc"]["t"], 2),
                           RES["event"]["all_n"], RES["event"]["all_oo20_exc"], RES["event"]["all_oo20_t"]]
    evo["period_oo20"] = {k: dict(mine=[pct(v["oo20_exc"]["mean"]), round(v["oo20_exc"]["t"], 2)], res=RES["event"]["period_oo20"][k])
                          for k, v in mine["event_level"]["independent_by_period"].items()}
    evo["year_oo20"] = {y: [pct(v["oo20_exc_mean"]), RES["event"]["year_oo20"].get(y)]
                        for y, v in mine["event_level"]["independent_by_year"].items() if y in RES["event"]["year_oo20"]}
    # RS 三分組（全部可買訊號；用開盤到開盤 20 日超額，研究員 7.3 節的口徑）
    me = pd.read_csv(os.path.join(HERE, "my_events.csv.gz"), dtype={"code": str})
    me = me[me.buyable].copy()
    g = me.groupby("sig_date")
    r_ = g.rs.rank(ascending=False, method="first"); n_ = g.rs.transform("count")
    me["grp"] = np.where(n_ < 3, -1, np.minimum((r_ - 1) * 3 // n_, 2))

    def ds(x):
        x = x.dropna().values
        return [pct(x.mean()), round(float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))), 2), int(len(x))]
    evo["rs_terc_oo20_mine"] = {nm: ds(me.oo20_exc[me.grp == k]) for nm, k in (("high", 0), ("mid", 1), ("low", 2))}
    evo["rs_terc_oo20_mine"]["days_lt3_signals"] = ds(me.oo20_exc[me.grp == -1])
    evo["rs_terc_res"] = RES["event"]["rs_terc"]
    out["event"] = evo
    # ---------------- 逐年
    y = mine["posthoc"]["yearly"]; y5 = mine["posthoc"]["yearly_0050"]
    out["yearly"] = dict(
        P_minus_0050={k: [pct(y["P"][k] - y5[k]), v] for k, v in RES["yearly_P_minus_0050"].items()},
        C_minus_C0={k: [pct(y["C"][k] - y["C0"][k]), v] for k, v in RES["yearly_C_minus_C0"].items()})
    # ---------------- 逐筆交易
    rt = pd.read_csv(os.path.join(RDIR, "trades.csv.gz"), dtype={"code": str})
    mt = pd.read_csv(os.path.join(HERE, "my_trades.csv.gz"), dtype={"code": str})
    tcmp = {}
    for nm, (rr, ss, mc, mh) in {"E0": ("E0", "E0", "E0", "E0"), "P": ("P", "P", "P", "P"), "P40": ("P40", "P40", "P40", "P40"),
                                 "C_E0": ("C", "E0", "C", "E0"), "C_P": ("C", "P", "C", "P")}.items():
        r = rt[(rt.run == rr) & (rt.sleeve == ss) & (rt.why != "open_end")]
        m = mt[(mt.cell == mc) & (mt.half == mh)]
        A = set(zip(r.code, r.ent_date)); B = set(zip(m.code, m.buy_date))
        j = r.rename(columns={"ent_date": "d"}).merge(m.rename(columns={"buy_date": "d"}), on=["code", "d"], suffixes=("", "_m"))
        first = sorted(A ^ B, key=lambda z: z[1])[:4]
        tcmp[nm] = dict(res_closed=len(r), mine_closed=len(m), common=len(A & B), only_res=len(A - B), only_mine=len(B - A),
                        first_diffs=first, common_same_exit=float((j.exit_date == j.sell_date).mean()) if len(j) else None,
                        common_same_shares=float((j.shares == j.sh).mean()) if len(j) else None)
    out["trades"] = tcmp
    # ---------------- 凍結後診斷：0050 掃入不收最低手續費
    out["diag_e50_sweep_no_minfee"] = diag(sc, rt)
    out["fails"] = fails
    with open(os.path.join(HERE, "compare_vs_researcher.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x))
    print(json.dumps(dict(fails=fails, decisions_all_same=out["decisions_all_same"], trades=tcmp,
                          diag=out["diag_e50_sweep_no_minfee"]), ensure_ascii=False, indent=1, default=str))


def diag(sc, rt):
    import rep_s15_lib as K
    K14 = K.K14
    L = K.L
    P = K.R.build(sc); K.attach_e50_fixed(P, sc)
    excl = K.S.innov_mask_fn(P, sc)
    U = K.univ_matrix(P, excl)
    M = K.signal_mats(P, U)
    b, _ = K.pb_plans(P, M["sig"], M["rs"], U, 0)
    Rv, mi0 = L.load_rev(P); S_, _ = L.surge(Rv, mi0, null2013=True)
    bE, kE = K.Q.cell_plans(K.Q.make_lists(P, S_, mi0, shift=0, excl=excl), "C10", x=0.10)
    a = "        f = max(minfee, fee * cash / (1.0 + fee))"
    s = K14._SRC.replace("def simulate_s14(", "def simv(")
    assert s.count(a) == 1
    s = s.replace(a, "        f = fee * cash / (1.0 + fee)")
    ns = dict(vars(K14.K10)); exec(compile(s, "<diag>", "exec"), ns)
    sim = ns["simv"]
    i19 = L.didx(P, "2018-12-28"); i24 = L.didx(P, "2023-12-29"); e = P.T - 1
    o = {}
    x = {}
    for nm, cap, H in (("P", 1e6, 20), ("P40", 1e6, 40), ("P_half", 5e5, 20)):
        x[nm] = sim(P, b, {}, sat_w=0.7, sizing="W10", capital=cap, eod=K.make_eod_hold(H), listmax=9999, **K.ST)
    x["E0"] = sim(P, bE, kE, sat_w=0.7, sizing="W10", capital=1e6, **K.ST)
    x["E0_half"] = sim(P, bE, kE, sat_w=0.7, sizing="W10", capital=5e5, **K.ST)
    navs = {k: v["nav"] for k, v in x.items()}
    navs["C"] = navs["E0_half"] + navs["P_half"]
    for nm, (rr, ss) in {"P": ("P", "P"), "P40": ("P40", "P40"), "E0": ("E0", "E0"), "P_half": ("C", "P")}.items():
        r = rt[(rt.run == rr) & (rt.sleeve == ss) & (rt.why != "open_end")]
        A = set(zip(r.code, r.ent_date)); B = set(zip(x[nm]["trades"].code, x[nm]["trades"].buy_date))
        o[nm] = dict(common=len(A & B), only_res=len(A - B), only_mine=len(B - A), first_diffs=sorted(A ^ B, key=lambda z: z[1])[:2])
    for nm in ("P", "P40", "E0", "C"):
        nav = navs[nm]
        o.setdefault(nm, {}).update(ann_2019=round(100 * L.ann(P, nav, i19, e), 2), ann_full=round(100 * L.ann(P, nav, P.t0, e), 2),
                                    ann_1923=round(100 * L.ann(P, nav, i19, i24), 2), ann_2024=round(100 * L.ann(P, nav, i24, e), 2),
                                    mdd=round(100 * L.maxdd(P, nav, P.t0, e)[0], 2))
    o["note"] = "凍結後診斷，不改凍結數字：引擎唯一差別是閒錢掃入 0050 時手續費不設 20 元下限。"
    return o


if __name__ == "__main__":
    main(sys.argv[1])
