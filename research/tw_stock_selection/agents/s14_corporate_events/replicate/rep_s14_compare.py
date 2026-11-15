# -*- coding: utf-8 -*-
"""S14 獨立複製：凍結後比對。研究員數字取自 FINDINGS.md 全文（手抄）與 trades.csv.gz；不讀 results*.json、navs.csv.gz。
另做凍結後的補充（標「凍結後補充」）：事件層延長／非延長的月份分群 t（研究員 FINDINGS 用月份分群 t）、2019 年可買事件掛零月數。
用法：python3 -I -B rep_s14_compare.py <scratch_dir>
輸出：compare_vs_researcher.json
"""
import sys, os, json, hashlib
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import rep_s14_lib as K

RS = os.path.dirname(HERE)
TOL_ANN = 0.005; TOL_MDD = 0.005; TOL_N = 0.02

# ---- 研究員（FINDINGS.md 第 3、4、5、6、7 節）
RES = {
    "main": {  # 2019 起、全期、2019～2023、2024 起、最大回撤
        "0050": (0.2891, 0.2114, 0.1643, 0.5448, -0.3396), "E0": (0.4813, 0.3408, 0.3701, 0.7016, -0.3340),
        "D": (0.4260, 0.2997, 0.2912, 0.7013, -0.3443), "DA": (0.4228, 0.3008, 0.2844, 0.7065, -0.3435),
        "C": (0.4382, 0.3025, 0.3069, 0.7051, -0.3142), "CA": (0.4341, 0.3022, 0.3009, 0.7053, -0.3143),
        "C0": (0.3919, 0.2748, 0.2634, 0.6535, -0.3032)},
    "n_trades_incl_open": {"E0": 387, "D": 594, "DA": 669, "C": 565 + 341, "CA": 640 + 341, "C0": 341},
    "eight": {"D": dict(x2=(0.3350, 0.2367), block5=(0.4026, 0.2849), h1=0.1866, h2=0.4229),
              "C": dict(x2=(0.3725, 0.2570), block5=(0.4122, 0.2862)), "DA": dict(x2=(0.3190, 0.2309))},
    "paths20": {  # 中位數、最少、最多、贏 0050 條數、回撤中位數、2019～2023 中位數、2024 起中位數
        "E0": (0.4623, 0.4324, 0.4851, 20, -0.3401, 0.3569, 0.6598),
        "D": (0.3687, 0.3047, 0.4459, 20, -0.3798, 0.2980, 0.4708),
        "DA": (0.4232, 0.3639, 0.4678, 20, -0.3702, 0.3236, 0.5680),
        "C": (0.3958, 0.3676, 0.4618, 20, -0.3373, 0.3038, 0.5751),
        "CA": (0.4173, 0.3924, 0.4590, 20, -0.3253, 0.3142, 0.6127),
        "C0": (0.3784, 0.3592, 0.4112, 20, -0.3157, 0.2665, 0.6180)},
    "paired": {  # (2019 正條數, 中位數), 全期, 2019～2023, 2024 起, 回撤較淺條數
        "C-C0": ((18, 0.0206), (20, 0.0194), (20, 0.0433), (5, -0.0471), 5),
        "CA-C0": ((20, 0.0420), (20, 0.0290), (20, 0.0533), (8, -0.0070), 5),
        "C-E0": ((0, -0.0589), (0, -0.0340), (0, -0.0496), (3, -0.0810), 13),
        "D-0050": ((20, 0.0796), (20, 0.0644), (20, 0.1337), (5, -0.0739), 0),
        "DA-D": ((18, 0.0545), (20, 0.0399), (17, 0.0357), (20, 0.0897), 14)},
    "bears_median": {"0050": (-0.1599, -0.2822, -0.3396, -0.2748), "E0": (-0.1679, -0.2647, -0.2969, -0.3401),
                     "D": (-0.1673, -0.2817, -0.3798, -0.3130), "C": (-0.1622, -0.2747, -0.3373, -0.3184),
                     "C0": (-0.1616, -0.2760, -0.3053, -0.3157)},
    "bears_pair": {"C-C0": ((11, 0.0001), (10, 0.0004), (0, -0.0325), (10, 0.0004)),
                   "D-0050": ((2, -0.0074), (10, 0.0005), (0, -0.0402), (0, -0.0381))},
    "corr": (0.33, 0.32, 0.43), "overlap": 0.052,
    "top5_share": {"D": 0.143, "C": 0.137, "E0": 0.249, "C0": 0.176},
    "D_stats": dict(n=594, win=0.534, hold_median=10, mean_npos=2.26, stock_w=0.159),
    "decisions": dict(D="贏 0050", DA="贏 0050", C="有補充價值，但只靠一段", CA="有補充價值，但只靠一段",
                      xq="不建議加進現行策略", eight="6 格都 8／8"),
    "event_level": dict(n=727, ext_n=91, ext_net=0.1627, ext_t=3.86, non_n=636, non_net=0.0023, non_t=0.66,
                        all_net=0.0224, all_t=2.08),
    "precollect": dict(ge10=138, ge30=23),
    "yearly_D": {2015: -0.100, 2016: 0.193, 2017: 0.191, 2018: -0.054, 2019: 0.345, 2020: 0.443, 2021: 0.744,
                 2022: -0.219, 2023: 0.359, 2024: 0.541, 2025: 0.429, 2026: 1.027},
    "zero_months_2019_eligible": 7,
}
BK = ["2018", "2020-03", "2022", "2024-07~2025-04"]


def main(sc):
    my = json.load(open(os.path.join(HERE, "my_numbers.json")))
    h = hashlib.sha256(open(os.path.join(HERE, "my_numbers.json"), "rb").read()).hexdigest()
    frozen = open(os.path.join(HERE, "my_numbers.sha256")).read().split()[0]
    out = dict(my_numbers_sha256=h, frozen_sha256=frozen, frozen_ok=bool(h == frozen),
               tolerances=dict(ann=TOL_ANN, mdd=TOL_MDD, n_rel=TOL_N))
    # ---- 主跑
    rows = []
    allok = True
    for c, v in RES["main"].items():
        m = my["bench_0050"] if c == "0050" else my["cells"][c]
        mine = (m["ann_2019"], m["ann_full"], m["ann_1923"], m["ann_2024"], m["mdd_full"])
        d = [a - b for a, b in zip(mine, v)]
        ok = all(abs(x) <= TOL_ANN for x in d[:4]) and abs(d[4]) <= TOL_MDD
        r = dict(cell=c, mine=[round(x, 4) for x in mine], researcher=list(v), diff_pp=[round(x * 100, 3) for x in d], ok=ok)
        if c != "0050":
            nm = m["n_trades"] + m.get("n_open_end", 0)
            nr = RES["n_trades_incl_open"][c]
            r.update(n_mine_incl_open=nm, n_researcher=nr, n_rel_diff=round((nm - nr) / nr, 4))
            r["ok"] = ok = ok and abs(nm - nr) / nr <= TOL_N
        allok &= ok
        rows.append(r)
    out["main"] = rows
    # 八項明細
    e8 = {}
    for c, v in RES["eight"].items():
        m = my["cells"][c]
        o = {}
        if "x2" in v:
            o["x2"] = dict(mine=(m["x2"]["ann_2019"], m["x2"]["ann_full"]), researcher=v["x2"])
        if "block5" in v:
            o["block5"] = dict(mine=(m["block5"]["ann_2019"], m["block5"]["ann_full"]), researcher=v["block5"])
        if "h1" in v:
            o["h1h2"] = dict(mine=(m["ann_h1"], m["ann_h2"]), researcher=(v["h1"], v["h2"]))
        o["max_abs_diff_pp"] = round(max(abs(a - b) for k_, x in o.items() for a, b in zip(x["mine"], x["researcher"])) * 100, 3)
        o["checks_mine"] = m["n_checks_pass"]
        e8[c] = o
    out["eight_detail"] = e8
    # ---- 20 條
    p = {}
    for c, v in RES["paths20"].items():
        m = my["paths20"][c]
        mine = (m["ann_2019_median"], m["ann_2019_min"], m["ann_2019_max"], m["n_beat_0050"], m["mdd_median"],
                m["ann_1923_median"], m["ann_2024_median"])
        d = [a - b for a, b in zip(mine, v)]
        p[c] = dict(mine=[round(x, 4) for x in mine], researcher=list(v), diff_pp=[round(x * 100, 3) if i != 3 else x for i, x in enumerate(d)],
                    ok=bool(all(abs(x) <= TOL_ANN for i, x in enumerate(d) if i != 3) and d[3] == 0))
        allok &= p[c]["ok"]
    out["paths20"] = p
    pr = {}
    for k, v in RES["paired"].items():
        m = my["paired"][k]
        mine = ((m["ann_2019"]["n_pos"], m["ann_2019"]["median"]), (m["ann_full"]["n_pos"], m["ann_full"]["median"]),
                (m["ann_1923"]["n_pos"], m["ann_1923"]["median"]), (m["ann_2024"]["n_pos"], m["ann_2024"]["median"]),
                m["mdd_full"]["n_pos"])
        pr[k] = dict(mine=[(a, round(b, 4)) for a, b in mine[:4]] + [mine[4]], researcher=list(v),
                     n_pos_identical=bool(all(a[0] == b[0] for a, b in zip(mine[:4], v[:4])) and mine[4] == v[4]),
                     max_median_diff_pp=round(max(abs(a[1] - b[1]) for a, b in zip(mine[:4], v[:4])) * 100, 3))
    out["paired"] = pr
    bm = {}
    for c, v in RES["bears_median"].items():
        mine = [my["bench_0050"]["bears"][k] for k in BK] if c == "0050" else [my["paths20"][c]["bears_median"][k] for k in BK]
        bm[c] = dict(mine=[round(x, 4) for x in mine], researcher=list(v),
                     max_diff_pp=round(max(abs(a - b) for a, b in zip(mine, v)) * 100, 3))
    out["bears_median"] = bm
    bp = {}
    for k, v in RES["bears_pair"].items():
        m = my["paired"][k]
        mine = [(m["bear_" + b]["n_pos"], round(m["bear_" + b]["median"], 4)) for b in BK]
        bp[k] = dict(mine=mine, researcher=list(v), n_identical=bool(all(a[0] == b[0] for a, b in zip(mine, v))))
    out["bears_pair"] = bp
    co = my["corr_overlap"]
    out["corr_overlap"] = dict(mine=(round(co["D_E0_excess_full"], 3), round(co["D_E0_excess_2019"], 3),
                                     round(co["paths20_median"], 3), round(co["C_overlap_mean_share_of_D_half"], 4)),
                               researcher=RES["corr"] + (RES["overlap"],))
    out["top5_share"] = {c: dict(mine=round(my["cells"][c]["top5_share_of_gain"], 4), researcher=v)
                         for c, v in RES["top5_share"].items()}
    ts = my["cells"]["D"]["trade_stats"]
    out["D_stats"] = dict(mine=dict(n=my["cells"]["D"]["n_trades"], win=round(ts["win"], 4), hold_median=ts["hold_median"],
                                    mean_npos=round(co["D_main_mean_npos"], 3), stock_w=round(co["D_main_mean_stock_weight"], 4)),
                          researcher=RES["D_stats"])
    dec = my["decisions"]
    out["decisions"] = dict(mine=dict(D=dec["D_vs_0050"], DA=dec["DA_vs_0050"], C=dec["C_supplement"]["cls"],
                                      CA=dec["CA_supplement"]["cls"], xq=dec["xq"],
                                      eight={k: my["cells"][k]["n_checks_pass"] for k in my["cells"]},
                                      bears_C_vs_C0={k: v["cls"] for k, v in dec["bears_C_vs_C0"].items()}),
                            researcher=RES["decisions"])
    out["decisions"]["identical"] = bool(dec["D_vs_0050"] == RES["decisions"]["D"] and dec["DA_vs_0050"] == RES["decisions"]["DA"]
                                         and dec["C_supplement"]["cls"] == RES["decisions"]["C"]
                                         and dec["CA_supplement"]["cls"] == RES["decisions"]["CA"]
                                         and dec["xq"] == RES["decisions"]["xq"]
                                         and all(my["cells"][k]["n_checks_pass"] == 8 for k in my["cells"]))
    out["yearly_D"] = {y: dict(mine=round(my["cells"]["D"]["yearly"][str(y)], 4), researcher=v)
                       for y, v in RES["yearly_D"].items()}
    out["yearly_D_max_diff_pp"] = round(max(abs(my["cells"]["D"]["yearly"][str(y)] - v) for y, v in RES["yearly_D"].items()) * 100, 2)
    rA = my["reviewA"]
    out["event_level"] = dict(mine=dict(n=rA["event_level_net_excess"]["all"]["n"],
                                        ext_n=rA["event_level_net_excess"]["extended"]["n"],
                                        ext_net=rA["event_level_net_excess"]["extended"]["mean"],
                                        ext_t_iid=rA["event_level_net_excess"]["extended"]["t"],
                                        non_n=rA["event_level_net_excess"]["not_extended"]["n"],
                                        non_net=rA["event_level_net_excess"]["not_extended"]["mean"],
                                        non_t_iid=rA["event_level_net_excess"]["not_extended"]["t"],
                                        all_net=rA["event_level_net_excess"]["all"]["mean"],
                                        all_t_iid=rA["event_level_net_excess"]["all"]["t"]),
                              researcher=RES["event_level"])
    out["precollect"] = dict(mine=my["reviewB"]["D_main_lots"], researcher=RES["precollect"])
    out["all_main_paths_within_tol"] = bool(allok)

    # ---- 逐筆交易
    t = pd.read_csv(os.path.join(RS, "trades.csv.gz"), dtype={"code": str})
    m = pd.read_csv(os.path.join(HERE, "my_trades.csv.gz"), dtype={"code": str})
    tm = {}
    for (run, sleeve), mine_key in ((("E0", "E0"), ("E0", "E0")), (("D", "D"), ("D", "D")), (("DA", "DA"), ("DA", "DA")),
                                    (("C", "D"), ("C", "D")), (("C", "E0"), ("C", "E0"))):
        r = t[(t.run == run) & (t.sleeve == sleeve)].copy()
        mm = m[(m.cell == mine_key[0]) & (m.half == mine_key[1])].copy()
        rc = r[r.why != "open_end"]
        bb = mm.rename(columns={"buy_date": "ent_date"})[["code", "ent_date", "sell_date", "sh", "ret"]]
        j = rc.merge(bb, on=["code", "ent_date"], how="outer", suffixes=("_r", "_m"), indicator=True).set_index(["code", "ent_date"])
        both = j[j._merge == "both"]
        tm["%s|%s" % (run, sleeve)] = dict(
            researcher_closed=int(len(rc)), researcher_open_end=int((r.why == "open_end").sum()), mine_closed=int(len(mm)),
            matched=int(len(both)), only_researcher=int((j._merge == "left_only").sum()),
            only_mine=int((j._merge == "right_only").sum()),
            exit_date_equal=int((both.exit_date == both.sell_date).sum()),
            shares_equal=int((both.shares == both.sh).sum()),
            max_abs_ret_diff=float((both.ret_r - both.ret_m).abs().max()) if len(both) else None,
            n_ret_diff_gt_1e3=int(((both.ret_r - both.ret_m).abs() > 1e-3).sum()),
            examples_only_researcher=[list(x) for x in j[j._merge == "left_only"].reset_index()[["code", "ent_date", "exit_date"]].head(5).values],
            examples_only_mine=[list(x) for x in j[j._merge == "right_only"].reset_index()[["code", "ent_date", "sell_date"]].head(5).values],
            ret_diff_examples=[list(x) for x in both[(both.ret_r - both.ret_m).abs() > 1e-3].reset_index()[
                ["code", "ent_date", "exit_date", "sell_date", "ret_r", "ret_m"]].head(5).values])
    out["trades"] = tm
    # 研究員 D 的持有天數 vs 鏈：延長鏈的出場日
    # ---- 凍結後補充：事件層月份分群 t、2019 可買事件掛零月數
    P = K.R.build(sc); K.attach_e50_fixed(P, sc); excl = K.S.innov_mask_fn(P, sc)
    disp, _ = K.load_disp(); ch = K.build_chains(P, disp, (1,)); cs = K.adv20(P)
    AO, Gf = K.adj_open_matrix(P)
    e = P.T - 1
    ev = []
    for r in ch.itertuples(index=False):
        s = int(r.k0)
        if s < P.t0 or s >= P.T - 1:
            continue
        if not (K.R.universe(P, s) & ~excl(s))[r.i]:
            continue
        a = s + 1; i = int(r.i); o = P.O[a, i]
        if not (o == o) or o >= K.S.limit_up(P.ref[a, i]) - 1e-9 or r.L is None or pd.isna(r.L) or int(r.L) + 1 > e:
            continue
        x = int(r.L) + 1
        ox = P.O[x, i] if P.O[x, i] == P.O[x, i] else P.Cf[x, i]
        net = K.strict_net(AO[a, i], AO[x, i], o, ox) - (P.e50o[x] / P.e50o[a] - 1.0)
        ev.append(dict(ym=P.dstr[a][:7], ext=bool(r.extended), exc=net))
    evd = pd.DataFrame(ev)

    def mt(g):
        mm_ = g.groupby("ym").exc.mean()
        return dict(n=int(len(g)), months=int(len(mm_)), mean=float(g.exc.mean()), t_month=K.tstat(mm_.values))
    elig = pd.Series([P.dstr[int(r.k0) + 1][:7] for r in ch.itertuples(index=False)
                      if P.t0 <= int(r.k0) < P.T - 1 and (K.R.universe(P, int(r.k0)) & ~excl(int(r.k0)))[r.i]])
    months = pd.period_range("2015-07", "2026-10", freq="M").astype(str)
    em = elig.value_counts().reindex(months, fill_value=0)
    out["postfreeze_supplement"] = dict(
        note="凍結後補充（研究員用月份分群 t；複製員凍結數字是逐筆 t）",
        event_level_month_cluster=dict(extended=mt(evd[evd.ext]), not_extended=mt(evd[~evd.ext]), all=mt(evd)),
        eligible_zero_months_by_year={y: int((em[[mm_ for mm_ in months if mm_.startswith(y)]] == 0).sum())
                                      for y in [str(z) for z in range(2016, 2027)]},
        eligible_2019_by_month=em[[mm_ for mm_ in months if mm_.startswith("2019")]].to_dict())
    with open(os.path.join(HERE, "compare_vs_researcher.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x))
    print(json.dumps(out, ensure_ascii=False, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x))[:20000])


if __name__ == "__main__":
    main(sys.argv[1])
