# -*- coding: utf-8 -*-
"""S20 獨立複製：凍結後比對。先核對 my_numbers.sha256，再讀研究員 FINDINGS（數字手抄於 RES）與 trades.csv.gz。
results*.json 不讀。另做一項凍結後追加：F1 內「留下組 − 當月全部」月份群聚 t（研究員的口徑），需要 scratch 面板。
用法：python3 -I -B rep_s20_compare.py <scratch_dir>
輸出：compare_vs_researcher.json
"""
import sys, os, json, hashlib
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np, pandas as pd

SPEC = os.path.dirname(HERE)
js = open(os.path.join(HERE, "my_numbers.json")).read()
h = hashlib.sha256(js.encode("utf-8")).hexdigest()
assert h == open(os.path.join(HERE, "my_numbers.sha256")).read().split()[0]
M = json.loads(js)

# 研究員數字（FINDINGS.md 結論、第 2、3、4 節；百分比）
RES = {
    "train_CUR_median": 33.4, "train_CUR_min": 30.1, "train_CUR_max": 35.7, "train_0050": 19.9,
    "train": {  # 年化中位數, 配對差中位數, n_pos, 範圍 min, max, 1618, 1618npos, 1921, 1921npos, dd差
        "MS33": (30.4, -2.94, 0, -4.34, -0.91, 3.02, 20, -10.51, 0, -1.37),
        "MU": (30.4, -2.94, 0, -4.34, -0.95, 3.02, 20, -10.51, 0, -1.22),
        "NP6": (29.4, -3.89, 0, -4.73, -2.26, 1.63, 19, -10.50, 0, -0.10),
        "MS50": (27.1, -6.34, 0, -7.93, -3.14, 0.68, 12, -15.23, 0, -1.29),
        "NP8": (24.9, -8.82, 0, -10.24, -6.05, 1.03, 16, -20.92, 0, -0.62),
        "NP9": (24.1, -9.22, 0, -10.23, -6.36, 0.80, 13, -21.10, 0, -1.96),
        "MS67": (23.8, -9.61, 0, -10.67, -5.83, -0.72, 8, -21.09, 0, -1.85),
        "LOW": (25.2, -7.86, 0, -11.54, -4.96, -1.51, 1, -16.57, 0, -0.62)},
    "frozen": "MS33", "T0": False,
    "valid": dict(V1_npos=10, V2_median=-0.70, V2_min=-5.10, V2_max=1.60, V3_median=-0.57, V3_npos=10, V3_cur=34.44, V3_fz=33.71,
                  V4_2022=-3.09, V4_2022_npos=0, V4_2024=6.04, V4_2024_npos=20, CUR_med=39.86, FZ_med=39.40, LOW_med=35.46,
                  LOW_diff=-4.74, LOW_npos=0, mdd_CUR=-33.99, mdd_FZ=-33.04, mdd_LOW=-37.74,
                  bear2022_CUR=-30.2, bear2022_FZ=-33.0, bear2024_CUR=-34.0, bear2024_FZ=-27.9, passed=False),
    "fresh_main": {"CUR": dict(ann=42.99, mdd=-32.80, n=197, open=8, hold=33.4, bl=36.07, x2=37.26, npos=5.73),
                   "FZ": dict(ann=41.49, mdd=-32.98, n=134, open=6, hold=34.8, bl=34.29, x2=37.79, npos=4.08),
                   "LOW": dict(ann=38.04, mdd=-37.48, n=89, open=2, hold=32.8, bl=27.53, x2=35.43, npos=2.52)},
    "cont_main": {"CUR": dict(a19=48.13, af=34.08, h1=26.52, h2=42.05, mdd=-33.40, n=379, hold=34.2, x2_19=42.28, x2_f=29.70,
                              bl_19=43.42, bl_f=31.12, checks=8, bears=(-16.4, -26.9, -28.8, -33.4)),
                  "FZ": dict(a19=44.42, af=32.54, h1=24.92, h2=40.58, mdd=-31.95, n=247, hold=36.1, x2_19=39.95, x2_f=29.35,
                             bl_19=39.80, bl_f=29.60, checks=8, bears=(-17.7, -26.1, -31.9, -28.1)),
                  "LOW": dict(a19=38.16, af=26.93, h1=20.11, h2=34.10, mdd=-37.05, n=194, hold=32.6, x2_19=35.74, x2_f=25.01,
                              bl_19=31.24, bl_f=22.51, checks=8, bears=(-14.6, -28.7, -29.6, -37.0))},
    "yearly": {"CUR": [15.2, 22.2, 1.8, 33.6, 99.4, 46.4, -18.9, 52.7, 60.2, 56.5, 78.1],
               "FZ": [18.2, 30.2, -0.3, 35.5, 73.4, 42.4, -22.2, 44.5, 66.8, 60.8, 75.3],
               "LOW": [15.7, 15.9, -0.8, 30.9, 59.5, 27.6, -18.5, 49.9, 50.2, 55.1, 65.1]},
    "cont_paths_FZ": dict(d2019=-3.85, d2019_npos=0, dfull=-1.55, dfull_npos=1, mdd=1.28, mdd_npos=15),
    "cont_paths_LOW_2019": -8.66,
    "reviewA": dict(CUR_train=34.22, CUR_valid=42.69, FZ_train=32.29, FZ_valid=41.36, LOW_train=24.24, LOW_valid=37.95,
                    rnd67_train=(31.77, 29.6, 33.4), rnd67_valid=(39.32, 34.7, 42.1), rnd67_2019=(44.15, 40.2, 46.5),
                    rnd46_train=(27.70, 26.1, 29.2), rnd46_valid=(35.31, 33.3, 39.1), rnd46_2019=(39.41, 36.8, 41.6)),
    "reviewB": dict(all_train=(279, 7.25), all_valid=(245, 5.75), kept_train=(186, 8.14), kept_valid=(169, 6.62),
                    drop_train=(93, 5.48), drop_valid=(76, 3.82), drop_top5_train=-1.97, drop_top5_valid=-1.54,
                    mugt0_train=8.3, mule0_train=5.3, mugt0_valid=6.6, mule0_valid=3.8,
                    kept_vs_all_t_train=0.52, kept_vs_all_t_valid=0.81),
}


def pp(x):
    return None if x is None else round(100.0 * x, 3)


cmp = dict(my_numbers_sha256=h, note="差＝複製員 − 研究員，單位個百分點（研究員數字多為 2 位小數，四捨五入誤差 ±0.005～0.05）。")
# ---------------- 特徵與門檻
f = M["features"]
cmp["features"] = dict(thr_max_abs_diff_vs_prereg=f["thr_max_abs_diff"],
                       train_q_MS=f["train"]["MS"]["q"], train_q_NPOS12=f["train"]["NPOS12"]["q"],
                       keep_frac=f["keep_frac"], boundary=f["MS33_boundary_entries"],
                       f1_counts=f["f1"], spearman_train=f["train"]["spearman"])
# ---------------- 訓練
tr = {}
for k, v in RES["train"].items():
    m = M["train"]["rules"][k]
    mine = (pp(m["train_median"]), pp(m["score_median"]), m["npos"], pp(m["diff_min"]), pp(m["diff_max"]),
            pp(m["y1618_median"]), m["y1618_npos"], pp(m["y1921_median"]), m["y1921_npos"], pp(m["dd_diff_median"]))
    tr[k] = dict(mine=mine, researcher=v, diff=[None if isinstance(a, int) and isinstance(b, int) else round(a - b, 3)
                                                for a, b in zip(mine, v)])
tr["CUR"] = dict(mine=pp(M["train"]["rules"]["CUR"]["train_median"]), researcher=RES["train_CUR_median"])
tr["selected"] = dict(mine=M["train"]["selected"], researcher=RES["frozen"], T0_mine=M["train"]["T0_pass"], T0_res=RES["T0"],
                      same=bool(M["train"]["selected"] == RES["frozen"] and M["train"]["T0_pass"] == RES["T0"]),
                      ranking_mine=M["train"]["ranking"])
cmp["train"] = tr
# ---------------- 驗證
V = M["validation"]; RV = RES["valid"]
pm = V["path_medians"]
cmp["validation"] = dict(
    V1_npos=(V["V1"]["n_pos"], RV["V1_npos"]), V2_median=(pp(V["V2"]["median"]), RV["V2_median"]),
    V2_range=((pp(V["V1"]["min"]), pp(V["V1"]["max"])), (RV["V2_min"], RV["V2_max"])),
    V3=((pp(V["V3"]["median"]), V["V3"]["n_pos"], pp(V["V3"]["cur_median"]), pp(V["V3"]["fz_median"])),
        (RV["V3_median"], RV["V3_npos"], RV["V3_cur"], RV["V3_fz"])),
    V4_2022=((pp(V["V4"]["2022"]["median_diff"]), V["V4"]["2022"]["n_shallower"]), (RV["V4_2022"], RV["V4_2022_npos"])),
    V4_2024=((pp(V["V4"]["2024-07~2025-04"]["median_diff"]), V["V4"]["2024-07~2025-04"]["n_shallower"]), (RV["V4_2024"], RV["V4_2024_npos"])),
    medians=({k: pp(pm[k]["ann_valid"]) for k in ["CUR", "FZ", "LOW"]}, dict(CUR=RV["CUR_med"], FZ=RV["FZ_med"], LOW=RV["LOW_med"])),
    mdd_medians=({k: pp(pm[k]["mdd_valid"]) for k in ["CUR", "FZ", "LOW"]}, dict(CUR=RV["mdd_CUR"], FZ=RV["mdd_FZ"], LOW=RV["mdd_LOW"])),
    LOW_diff=((pp(V["LOW_minus_CUR"]["median"]), V["LOW_minus_CUR"]["n_pos"]), (RV["LOW_diff"], RV["LOW_npos"])),
    passed=(V["passed"], RV["passed"]), same_verdict=bool(V["passed"] == RV["passed"] and
                                                         (V["V1"]["ok"], V["V2"]["ok"], V["V3"]["ok"], V["V4"]["ok"]) == (False,) * 4))
# ---------------- 主跑
mm = {}
maxd = dict(ann=0.0, mdd=0.0, ntr_rel=0.0)
for k, r in RES["fresh_main"].items():
    o = M["main"]["fresh|" + k]
    mine = dict(ann=pp(o["ann_valid"]), mdd=pp(o["mdd_valid"]), n=o["n_trades"], open=o["n_open"], hold=round(o["hold_mean"], 2),
                bl=pp(o["block5"]["ann_valid"]), x2=pp(o["x2"]["ann_valid"]), npos=round(o["npos_mean_valid"], 2))
    mm["fresh|" + k] = dict(mine=mine, researcher=r, diff={a: round(mine[a] - r[a], 3) for a in r})
    maxd["ann"] = max(maxd["ann"], abs(mine["ann"] - r["ann"]), abs(mine["bl"] - r["bl"]), abs(mine["x2"] - r["x2"]))
    maxd["mdd"] = max(maxd["mdd"], abs(mine["mdd"] - r["mdd"]))
    maxd["ntr_rel"] = max(maxd["ntr_rel"], abs(mine["n"] - r["n"]) / r["n"])
for k, r in RES["cont_main"].items():
    o = M["main"]["cont|" + k]
    mine = dict(a19=pp(o["ann_2019"]), af=pp(o["ann_full"]), h1=pp(o["ann_h1"]), h2=pp(o["ann_h2"]), mdd=pp(o["mdd_full"]),
                n=o["n_trades"], hold=round(o["hold_mean"], 2), x2_19=pp(o["x2"]["ann_2019"]), x2_f=pp(o["x2"]["ann_full"]),
                bl_19=pp(o["block5"]["ann_2019"]), bl_f=pp(o["block5"]["ann_full"]), checks=o["n_checks_pass"],
                bears=tuple(round(100 * o["bears"][b], 1) for b in ["2018", "2020-03", "2022", "2024-07~2025-04"]))
    mm["cont|" + k] = dict(mine=mine, researcher=r,
                           diff={a: (round(mine[a] - r[a], 3) if a != "bears" else [round(x - y, 1) for x, y in zip(mine[a], r[a])]) for a in r})
    for a in ["a19", "af", "h1", "h2", "x2_19", "x2_f", "bl_19", "bl_f"]:
        maxd["ann"] = max(maxd["ann"], abs(mine[a] - r[a]))
    maxd["mdd"] = max(maxd["mdd"], abs(mine["mdd"] - r["mdd"]))
    maxd["ntr_rel"] = max(maxd["ntr_rel"], abs(mine["n"] - r["n"]) / r["n"])
yr = {}
for k, r in RES["yearly"].items():
    my = [round(100 * M["main"]["cont|" + k]["yearly"][str(y)], 1) for y in range(2016, 2027)]
    yr[k] = dict(mine=my, researcher=r, max_abs_diff=round(max(abs(a - b) for a, b in zip(my, r)), 2))
mm["yearly"] = yr
cp = M["cont_paths"]["MS33"]
mm["cont_paths_FZ"] = dict(mine=dict(d2019=pp(cp["paired_2019"]["median"]), d2019_npos=cp["paired_2019"]["n_pos"],
                                     dfull=pp(cp["paired_full"]["median"]), dfull_npos=cp["paired_full"]["n_pos"],
                                     mdd=pp(cp["paired_mdd"]["median"]), mdd_npos=cp["paired_mdd"]["n_pos"]),
                           researcher=RES["cont_paths_FZ"])
mm["cont_paths_LOW_2019"] = (pp(M["cont_paths"]["LOW"]["paired_2019"]["median"]), RES["cont_paths_LOW_2019"])
mm["max_abs_diff"] = dict(ann_pp=round(maxd["ann"], 3), mdd_pp=round(maxd["mdd"], 3), n_trades_rel=round(maxd["ntr_rel"], 4))
cmp["main"] = mm

# ---------------- 交易逐筆（trades.csv.gz）
rt = pd.read_csv(os.path.join(SPEC, "trades.csv.gz"), dtype={"code": str})
mt = pd.read_csv(os.path.join(HERE, "my_trades.csv.gz"), dtype={"code": str})
tc = {}
for run in ["CUR|cont", "FZ|cont", "LOW|cont", "CUR|fresh", "FZ|fresh", "LOW|fresh"]:
    rule, mode = run.split("|")
    a = rt[rt.run == run].copy()
    b = mt[(mt.run == mode) & (mt.rule == rule)].copy()
    a["k"] = a.code + "|" + a.ent_date
    b["k"] = b.code + "|" + b.buy_date
    ja = a.set_index("k"); jb = b.set_index("k")
    com = ja.index.intersection(jb.index)
    ex_a = ja.loc[com].exit_date.where(ja.loc[com].why != "open_end", "OPEN")
    same_exit = int((ex_a.values == jb.loc[com].sell_date.values).sum())
    dp = (jb.loc[com].pnl.values - ja.loc[com].pnl.values)
    tc[run] = dict(n_res=int(len(a)), n_mine=int(len(b)), n_common=int(len(com)), n_same_exit=same_exit,
                   only_res=sorted(set(ja.index) - set(jb.index))[:10], only_mine=sorted(set(jb.index) - set(ja.index))[:10],
                   max_abs_pnl_diff=float(np.max(np.abs(dp))) if len(dp) else None,
                   max_abs_ret_diff=float(np.max(np.abs(jb.loc[com].ret.values - ja.loc[com].ret.values))) if len(dp) else None)
cmp["trades"] = tc

# ---------------- 審查 A
A = M["reviewA"]; RA = RES["reviewA"]


def q3(o):
    return (pp(o["median"]), pp(o["p10"]), pp(o["p90"]))
cmp["reviewA"] = dict(
    rule_stats=dict(mine=dict(CUR_train=pp(A["keep_FZ"]["cur_stats"]["ann_train"]), CUR_valid=pp(A["keep_FZ"]["cur_stats"]["ann_valid"]),
                              FZ_train=pp(A["keep_FZ"]["rule_stats"]["ann_train"]), FZ_valid=pp(A["keep_FZ"]["rule_stats"]["ann_valid"]),
                              LOW_train=pp(A["keep_LOW"]["rule_stats"]["ann_train"]), LOW_valid=pp(A["keep_LOW"]["rule_stats"]["ann_valid"])),
                    researcher={k: RA[k] for k in ["CUR_train", "CUR_valid", "FZ_train", "FZ_valid", "LOW_train", "LOW_valid"]}),
    rnd67=dict(mine=dict(train=q3(A["keep_FZ"]["ann_train"]), valid=q3(A["keep_FZ"]["ann_valid"]), y2019=q3(A["keep_FZ"]["ann_2019"])),
               researcher=dict(train=RA["rnd67_train"], valid=RA["rnd67_valid"], y2019=RA["rnd67_2019"])),
    rnd46=dict(mine=dict(train=q3(A["keep_LOW"]["ann_train"]), valid=q3(A["keep_LOW"]["ann_valid"]), y2019=q3(A["keep_LOW"]["ann_2019"])),
               researcher=dict(train=RA["rnd46_train"], valid=RA["rnd46_valid"], y2019=RA["rnd46_2019"])))

# ---------------- 審查 B（含凍結後追加：留下組 − 當月全部的月份群聚 t）
B = M["reviewB"]; RB = RES["reviewB"]
cmp["reviewB"] = dict(
    mine=dict(all_train=(B["train"]["all"]["n"], pp(B["train"]["all"]["mean"])), all_valid=(B["valid"]["all"]["n"], pp(B["valid"]["all"]["mean"])),
              drop_train=(B["train"]["ms_lt_thr"]["n"], pp(B["train"]["ms_lt_thr"]["mean"])),
              drop_valid=(B["valid"]["ms_lt_thr"]["n"], pp(B["valid"]["ms_lt_thr"]["mean"])),
              drop_top5_train=pp(B["train"]["ms_lt_thr"]["mean_drop_top5"]), drop_top5_valid=pp(B["valid"]["ms_lt_thr"]["mean_drop_top5"]),
              mugt0_train=pp(B["train"]["mu_gt0"]["mean"]), mule0_train=pp(B["train"]["mu_le0"]["mean"]),
              mugt0_valid=pp(B["valid"]["mu_gt0"]["mean"]), mule0_valid=pp(B["valid"]["mu_le0"]["mean"])),
    researcher=RB)
if len(sys.argv) > 1:
    import rep_s20_lib as K
    S, Q, R, L = K.S, K.Q, K.R, K.L
    sc = sys.argv[1]
    P = R.build(sc); excl = S.innov_mask_fn(P, sc)
    Rv, mi0 = L.load_rev(P)
    Sstd, F = K.features(Rv, mi0)
    lists = Q.make_lists(P, Sstd, mi0, shift=0, excl=excl)
    buy, keep = Q.cell_plans(lists, "C10", x=0.10)
    K.attach_e50_fixed(P, sc)
    top20 = {s: set(int(i) for i in lists[s]["lma"][:20]) for s in lists}
    FZ = dict(feat="MS", op=">=", thr=json.load(open(os.path.join(SPEC, "FROZEN.json")))["frozen_rule"]["thr"])
    rows = []
    for s in sorted(buy):
        m = lists[s]["m"]
        for i in buy[s]:
            o = K.outcome_e0(P, None, top20, s, int(i), P.T - 1)
            if o is None:
                continue
            rows.append(dict(s=s, per="valid" if mi0 + m >= K.VALID_MI else "train", kept=K.rule_pass(F, FZ, m, int(i)),
                             MS=F["MS"][m, int(i)], exc=o["exc"]))
    d = pd.DataFrame(rows)
    ex = {}
    for per in ["train", "valid"]:
        x = d[d.per == per]
        allm = x.groupby("s").exc.mean()
        o = {}
        for lab, msk in [("kept_minus_all", x.kept), ("dropped_minus_all", ~x.kept)]:
            gm = x[msk].groupby("s").exc.mean()
            dd = gm - allm.reindex(gm.index)
            o[lab] = dict(n_months=int(len(dd)), mean=float(dd.mean()), t=float(dd.mean() / (dd.std(ddof=1) / np.sqrt(len(dd)))))
        # 研究員說「μ÷σ 最高三分之一對全部」：用該期 MS 三分位
        q = np.percentile(x.MS, [100 / 3, 200 / 3])
        o["terciles_mean"] = [float(x[x.MS < q[0]].exc.mean()), float(x[(x.MS >= q[0]) & (x.MS < q[1])].exc.mean()),
                              float(x[x.MS >= q[1]].exc.mean())]
        gm = x[x.MS >= q[1]].groupby("s").exc.mean(); dd = gm - allm.reindex(gm.index)
        o["top_tercile_minus_all"] = dict(n_months=int(len(dd)), mean=float(dd.mean()), t=float(dd.mean() / (dd.std(ddof=1) / np.sqrt(len(dd)))))
        ex[per] = o
    cmp["reviewB"]["postfreeze_extra"] = ex

json.dump(cmp, open(os.path.join(HERE, "compare_vs_researcher.json"), "w"), ensure_ascii=False, indent=1,
          default=lambda o: bool(o) if isinstance(o, np.bool_) else (int(o) if isinstance(o, np.integer) else float(o)))
print(json.dumps(cmp, ensure_ascii=False, default=str)[:12000])
