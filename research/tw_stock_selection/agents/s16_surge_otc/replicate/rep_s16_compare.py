# -*- coding: utf-8 -*-
"""S16 複製：凍結後與研究員比對（my_numbers.json 已凍結、sha256 已寫）。
讀：研究員 FINDINGS.md 全文（數字以下方 RS 字典手抄）、trades.csv.gz。不讀 results*.json、navs.csv.gz。
另做分歧診斷：SURGE 在 24 個月年增差標準差＝0 時，研究員實作會得到 ±inf（排第 1），複製員主設計當缺值；
用「標準差＝0 時照除（得 ±inf）」的變體重跑五格主跑與 20 條路徑，看能否完全重現研究員。
用法：python3 -I -B rep_s16_compare.py <scratch_dir>
輸出：compare_vs_researcher.json
"""
import sys, os, json, hashlib
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import rep_s16_lib as K
S, Q, R, L = K.S, K.Q, K.R, K.L
ST = K.ST
SD = "/home/user/-/research/tw_stock_selection/agents/s16_surge_otc"
CAPS = [600_000.0, 800_000.0, 1_000_000.0, 1_500_000.0, 2_000_000.0]
SHIFTS = [0, 1, 2, 3]

# 研究員數字（FINDINGS.md 手抄；% 或個百分點）
RS = dict(
    main={"CUR": (48.13, 34.08, 37.01, 70.16, -33.40, 379), "ALL": (44.64, 31.82, 38.19, 56.87, -36.81, 454),
          "OTC": (23.20, 17.30, 20.40, 28.35, -38.39, 459), "ALL_O20": (44.96, 32.09, 34.73, 65.10, -33.42, 434),
          "ALL_RAW": (42.94, 30.69, 35.65, 56.87, -36.85, 438)},
    bench=(28.91, 21.14, 16.43, 54.48, -33.96),
    p20={"CUR": (43.24, 46.23, 48.51, 20, -34.01, 35.69, 65.98, 24.4), "ALL": (37.53, 40.72, 45.63, 20, -38.47, 35.26, 51.96, 35.4),
         "OTC": (21.94, 24.26, 28.23, 0, -37.97, 22.17, 29.08, 40.2), "ALL_O20": (38.09, 41.90, 45.02, 20, -34.65, 33.80, 56.30, 34.9),
         "ALL_RAW": (36.44, 39.16, 43.62, 20, -38.72, 32.64, 51.92, 35.6)},
    paired={"ALL-CUR": (0, -3.71, 10, 0.15, 0, -13.22, 5, -3.99, 0, 11.3), "OTC-CUR": (0, -21.62, 0, -13.34, 0, -38.97, 5, -4.64, 0, 16.7),
            "ALL_O20-CUR": (0, -4.10, 2, -2.36, 0, -8.24, 10, 0.01, 0, 11.2), "ALL_O20-ALL": (9, -0.04, 5, -2.51, 20, 4.16, 20, 3.52, 10, 0.0),
            "ALL_RAW-ALL": (5, -2.20, 5, -3.26, 11, 0.00, 11, 0.01, 9, 0.0), "ALL_RAW-CUR": (0, -6.30, 0, -3.05, 0, -12.75, 2, -3.70, None, None)},
    bears={"2018": (-16.38, -19.82, -16.79, -19.65, 3, -3.01), "2020-03": (-26.92, -28.17, -26.47, -28.90, 0, -2.16),
           "2022": (-28.79, -25.25, -29.69, -25.87, 18, 3.41), "2024-07~2025-04": (-33.40, -36.81, -34.01, -38.47, 5, -3.99)},
    trades={"CUR": (379, 51.2, 9.39), "ALL": (454, 45.4, 7.42), "ALL_twse": (236, 52.5, 11.76), "ALL_tpex": (218, 37.6, 2.73),
            "OTC": (459, 39.0, 3.47), "ALL_O20": (434, 47.2, 8.04), "CUR_2019": (300, None, 11.08), "ALL_2019": (360, None, 8.73)},
    cand={"CUR": (4.07, 4, 278, 4.73), "ALL": (4.95, 0, 420, 5.19), "OTC": (5.21, 0, 142, 5.94), "ALL_O20": (4.78, 0, 388, 5.01)},
    cost={"CUR": (50, 4, 1, 1, 3, 0, 3.18, 3.97), "ALL": (72, 11, 3, 2, 24, 0, 3.71, 4.60), "OTC": (78, 13, 4, 2, 13, 5, 3.28, 4.09),
          "ALL_O20": (76, 8, 2, 1, 10, 4, 3.67, 4.55)},
    eight={"CUR": (42.28, 29.70, 43.42, 31.12, 26.52, 42.05), "ALL": (38.95, 27.28, 37.50, 27.29, 28.24, 35.48),
           "OTC": (17.99, 13.36, 19.14, 14.61, 18.18, 16.44), "ALL_O20": (38.85, 27.80, 36.65, 26.81, 27.98, 36.31)},
    contrib_tpex_share={"full": 18.2, "from2019": 16.6, "2019_2023": 24.0, "from2024": 5.6},
    decisions=dict(main="加上櫃較差", concentration="較集中", bears=("較深", "較深", "較淺", "較深"), otc="在上櫃無效",
                   o20_cur="較差", o20_all="差不多", xq="現行不改"),
)


def surge_inf(Rm, mi0):
    """rep_s1_lib.surge 的同一算法，但標準差＝0 時照除（±inf／nan），模擬研究員的處理。"""
    nm, N = Rm.shape
    D = np.full((nm, N), np.nan)
    D[12:] = Rm[12:] - Rm[:-12]
    a = 2013 * 12 - mi0
    D[a:a + 12] = np.nan
    Sx = np.full((nm, N), np.nan)
    for m in range(24, nm):
        w = D[m - 24:m]
        ok = np.isfinite(w).all(axis=0) & np.isfinite(D[m])
        mu = np.where(ok, np.nan_to_num(w).mean(axis=0), np.nan)
        sd = np.where(ok, np.nan_to_num(w).std(axis=0, ddof=1), np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            Sx[m] = np.where(ok, (D[m] - mu) / sd, np.nan)
    # ±inf 視為「有值」且排最前／最後（pandas notna 會把 inf 當有值）；用 ±1e12 代表，np.isfinite 才不會把它丟掉
    Sx[np.isposinf(Sx)] = 1e12
    Sx[np.isneginf(Sx)] = -1e12
    return Sx


def main(sc):
    my = json.load(open(os.path.join(HERE, "my_numbers.json")))
    h = hashlib.sha256(open(os.path.join(HERE, "my_numbers.json"), "rb").read()).hexdigest()
    assert open(os.path.join(HERE, "my_numbers.sha256")).read().split()[0] == h
    out = dict(my_numbers_sha256=h)
    pct = lambda x: None if x is None else round(100 * x, 2)

    # ---------------- 主跑對照
    tab = {}
    for cc, r in RS["main"].items():
        c = my["cells"][cc]
        m = (pct(c["ann_2019"]), pct(c["ann_full"]), pct(c["ann_1923"]), pct(c["ann_2024"]), pct(c["mdd_full"]), c["n_trades"])
        d = [round(a - b, 2) for a, b in zip(m[:5], r[:5])]
        tab[cc] = dict(me=m, researcher=r, diff=d, trade_diff_pct=round(100 * (m[5] - r[5]) / r[5], 2),
                       pass_ann=bool(max(abs(x) for x in d[:4]) <= 0.5), pass_mdd=bool(abs(d[4]) <= 0.5),
                       pass_n=bool(abs(m[5] - r[5]) / r[5] <= 0.02))
    out["main"] = tab
    b = my["bench_0050"]
    out["bench"] = dict(me=(pct(b["ann_2019"]), pct(b["ann_full"]), pct(b["ann_1923"]), pct(b["ann_2024"]), pct(b["mdd_full"])),
                        researcher=RS["bench"])
    p20 = {}
    for cc, r in RS["p20"].items():
        p = my["paths20"][cc]
        m = (pct(p["ann_2019_min"]), pct(p["ann_2019_median"]), pct(p["ann_2019_max"]), p["n_beat_0050"], pct(p["mdd_median"]),
             pct(p["ann_1923_median"]), pct(p["ann_2024_median"]), round(100 * p["top5_share_median"], 1))
        p20[cc] = dict(me=m, researcher=r, diff=[None if a is None else round(a - bb, 2) for a, bb in zip(m, r)])
    out["paths20"] = p20
    pr = {}
    for k, r in RS["paired"].items():
        p = my["paired"][k]
        nlow = None
        m = (p["ann_2019"]["n_pos"], pct(p["ann_2019"]["median"]), p["ann_1923"]["n_pos"], pct(p["ann_1923"]["median"]),
             p["ann_2024"]["n_pos"], pct(p["ann_2024"]["median"]), p["mdd_full"]["n_pos"], pct(p["mdd_full"]["median"]),
             20 - p["top5_share"]["n_pos"], pct(p["top5_share"]["median"]))
        pr[k] = dict(me=m, researcher=r, note="第 9 欄＝最賺 5 筆占比較低條數（20 − 較高條數，未扣相等）")
    out["paired"] = pr
    bb = {}
    for k, r in RS["bears"].items():
        m = (pct(my["cells"]["CUR"]["bears"][k]), pct(my["cells"]["ALL"]["bears"][k]), pct(my["paths20"]["CUR"]["bears_median"][k]),
             pct(my["paths20"]["ALL"]["bears_median"][k]), my["paired"]["ALL-CUR"]["bear_" + k]["n_pos"],
             pct(my["paired"]["ALL-CUR"]["bear_" + k]["median"]))
        bb[k] = dict(me=m, researcher=r)
    out["bears"] = bb
    ms = my["market_split"]["ALL"]
    trd = {}
    for k, r in RS["trades"].items():
        if k in ("ALL_twse", "ALL_tpex"):
            v = ms["full"][k.split("_")[1]]
            m = (v["n"], round(100 * v["win"], 1), pct(v["exp"]))
        elif k.endswith("_2019"):
            v = my["cells"][k[:-5]]["trade_stats_2019"]
            m = (v["n"], round(100 * v["win"], 1), pct(v["exp"]))
        else:
            v = my["cells"][k]["trade_stats_full"]
            m = (v["n"], round(100 * v["win"], 1), pct(v["exp"]))
        trd[k] = dict(me=m, researcher=r)
    out["trades_stats"] = trd
    out["candidates"] = {cc: dict(me=(round(my["cells"][cc]["candidates"]["buy_mean"], 2), my["cells"][cc]["candidates"]["buy_empty_months"],
                                      round(my["cells"][cc]["candidates"]["lma_mean"]), round(my["cells"][cc]["avg_npos"], 2)),
                                  researcher=r) for cc, r in RS["cand"].items()}
    out["cost_blocked"] = {}
    for cc, r in RS["cost"].items():
        c = my["cells"][cc]; bl = c["blocked"]
        out["cost_blocked"][cc] = dict(me=(bl["blk_lot"], bl["blk_limup"], bl["blk_cap"], bl["blk_min"], bl["sell_partial_cap"],
                                           bl["sell_defer_limdn"], round(c["turnover_oneway_ann"], 2), pct(c["cost_over_nav_ann"])),
                                       researcher=r)
    out["eight"] = {cc: dict(me=(pct(my["cells"][cc]["x2"]["ann_2019"]), pct(my["cells"][cc]["x2"]["ann_full"]),
                                 pct(my["cells"][cc]["block5"]["ann_2019"]), pct(my["cells"][cc]["block5"]["ann_full"]),
                                 pct(my["cells"][cc]["ann_h1"]), pct(my["cells"][cc]["ann_h2"]), my["cells"][cc]["n_checks_pass"]),
                             researcher=r) for cc, r in RS["eight"].items()}
    out["contrib_tpex_share"] = {k: dict(me=round(100 * ms["contribution"][k]["contrib_share"]["tpex"], 1), researcher=r)
                                 for k, r in RS["contrib_tpex_share"].items()}
    dm = my["decisions"]
    out["decisions"] = dict(me=dict(main=dm["main_ALL_vs_CUR"]["cls"], concentration=dm["concentration"]["cls"],
                                    bears=[dm["bears_ALL_vs_CUR"][k]["cls"] for k in RS["bears"]], otc=dm["OTC_alone"]["cls"],
                                    o20_cur=dm["sens_ALL_O20_vs_CUR"]["cls"], o20_all=dm["sens_ALL_O20_vs_ALL"]["cls"], xq=dm["xq"]),
                            researcher=RS["decisions"])

    # ---------------- 交易逐筆
    t = pd.read_csv(os.path.join(SD, "trades.csv.gz"), dtype={"code": str})
    mt = pd.read_csv(os.path.join(HERE, "my_trades.csv.gz"), dtype={"code": str})
    tcmp = {}
    for cc in ["CUR", "ALL", "OTC", "ALL_O20", "ALL_RAW"]:
        a = t[(t.run == cc) & (t.why == "rule")]; bq = mt[mt.cell == cc]
        x = a.merge(bq, left_on=["code", "ent_date"], right_on=["code", "buy_date"], how="outer", indicator=True)
        bo = x[x._merge == "both"]
        dret = (bo.ret_x - bo.ret_y).abs()
        tcmp[cc] = dict(researcher_closed=int(len(a)), mine=int(len(bq)), both=int(len(bo)),
                        only_researcher=x[x._merge == "left_only"][["code", "ent_date", "exit_date", "ret_x"]].values.tolist(),
                        only_me=x[x._merge == "right_only"][["code", "buy_date", "sell_date", "ret_y"]].values.tolist(),
                        same_exit=int((bo.exit_date == bo.sell_date).sum()), max_abs_ret_diff=float(dret.max()),
                        n_ret_diff_gt_1e4=int((dret > 1e-4).sum()),
                        market_same=int((bo.market.map({"twse": 1, "tpex": 2}) == bo.mkt_entry).sum()))
    out["trades_match"] = tcmp
    print(json.dumps(tcmp, ensure_ascii=False, default=str)[:3000], flush=True)

    # ---------------- 分歧診斷：標準差＝0 → ±inf
    P = R.build(sc)
    K.attach_e50_fixed(P, sc)
    S.innov_mask_fn(P, sc)
    U = K.univ_mats(P, P.innov_M)
    raw = K.read_rev_csv()
    patch = pd.read_csv(K.RPATCH, dtype={"code": str})
    Rp, mi0 = K.load_rev_df(P, raw, patch)
    Rr, _ = K.load_rev_df(P, raw)
    Sp, _ = L.surge(Rp, mi0, null2013=True)
    Spi = surge_inf(Rp, mi0); Sri = surge_inf(Rr, mi0)
    infc = np.abs(Spi) >= 1e12
    e = P.T - 1; t0 = P.t0
    i19 = L.didx(P, "2018-12-28"); i24 = L.didx(P, "2023-12-29")
    lists0 = K.make_lists_u(P, Spi, mi0, U["ALL"])
    inf_hits = []
    for s, q in lists0.items():
        sv = Spi[q["m"]]
        for rk, i in enumerate(q["lma"][:20]):
            if abs(sv[i]) >= 1e12:
                inf_hits.append((P.dstr[s], P.codes[i], int(P.M[s, i]), rk + 1, float(q["r20"][i])))
    diag = dict(n_inf_cells_any=int(infc.sum()), inf_codes=sorted(set(P.codes[i] for i in np.where(infc.any(axis=0))[0])),
                ALL_lma_top20_inf_hits_shift0=inf_hits)
    nonzero = np.isfinite(Sp) != np.isfinite(Spi)
    diag["cells_finite_in_mine_vs_variant_differ"] = int((np.isfinite(Sp) != np.isfinite(Spi)).sum())
    rows = []; mains = {}
    for cc in ["CUR", "ALL", "OTC", "ALL_O20", "ALL_RAW"]:
        Ux = U["ALL" if cc == "ALL_RAW" else cc]; Sx = Sri if cc == "ALL_RAW" else Spi
        for sh in SHIFTS:
            b, k = Q.cell_plans(K.make_lists_u(P, Sx, mi0, Ux, shift=sh), "C10", x=0.10)
            for cp in CAPS:
                x = K.simulate(P, b, k, sat_w=0.7, sizing="W10", capital=cp, **ST)
                nav = x["nav"]
                rows.append(dict(cell=cc, shift=sh, capital=cp, ann_2019=L.ann(P, nav, i19, e), ann_full=L.ann(P, nav, t0, e),
                                 ann_1923=L.ann(P, nav, i19, i24), ann_2024=L.ann(P, nav, i24, e), mdd=L.maxdd(P, nav, t0, e)[0]))
                if sh == 0 and cp == 1_000_000.0:
                    mains[cc] = x
    pv = pd.DataFrame(rows)
    vm = {}
    for cc, x in mains.items():
        nav = x["nav"]; tr = x["trades"]
        m = (pct(L.ann(P, nav, i19, e)), pct(L.ann(P, nav, t0, e)), pct(L.ann(P, nav, i19, i24)), pct(L.ann(P, nav, i24, e)),
             pct(L.maxdd(P, nav, t0, e)[0]), int(len(tr)))
        mk = P.M[tr.tb.values, tr.i.values]
        a = t[(t.run == cc) & (t.why == "rule")]
        xx = a.merge(tr.assign(buy_date=tr.buy_date), left_on=["code", "ent_date"], right_on=["code", "buy_date"], how="outer",
                     indicator=True)
        vm[cc] = dict(main=m, researcher=RS["main"][cc], diff=[round(p - q, 2) for p, q in zip(m[:5], RS["main"][cc][:5])],
                      exp_twse=pct(tr.ret[mk == 1].mean()) if (mk == 1).any() else None,
                      exp_tpex=pct(tr.ret[mk == 2].mean()) if (mk == 2).any() else None,
                      trades_both=int((xx._merge == "both").sum()), only_researcher=int((xx._merge == "left_only").sum()),
                      only_me=int((xx._merge == "right_only").sum()))
    g = pv.groupby("cell")
    vp = {cc: dict(min=pct(h.ann_2019.min()), median=pct(h.ann_2019.median()), max=pct(h.ann_2019.max()), mdd_median=pct(h.mdd.median()),
                   researcher=RS["p20"][cc][:3] + (RS["p20"][cc][4],)) for cc, h in g}
    pn = pv.drop(columns="cell")
    xa = pn[pv.cell == "ALL"].set_index(["shift", "capital"]); xc = pn[pv.cell == "CUR"].set_index(["shift", "capital"]).loc[xa.index]
    xr = pn[pv.cell == "ALL_RAW"].set_index(["shift", "capital"]).loc[xa.index]
    d = xa - xc
    diag["variant_main"] = vm
    diag["variant_p20"] = vp
    diag["variant_paired_ALL_CUR"] = dict(n_pos=int((d.ann_2019 > 0).sum()), median=pct(d.ann_2019.median()),
                                          min=pct(d.ann_2019.min()), max=pct(d.ann_2019.max()),
                                          n_pos_1923=int((d.ann_1923 > 0).sum()), median_1923=pct(d.ann_1923.median()),
                                          n_pos_2024=int((d.ann_2024 > 0).sum()), median_2024=pct(d.ann_2024.median()),
                                          n_shallower=int((d.mdd > 0).sum()), mdd_median=pct(d.mdd.median()),
                                          researcher=(0, -3.71, -9.58, -2.50, 10, 0.15, 0, -13.22, 5, -3.99))
    dr = xr - xa
    diag["variant_paired_RAW_ALL"] = dict(n_pos=int((dr.ann_2019 > 0).sum()), median=pct(dr.ann_2019.median()), researcher=(5, -2.20))
    # 審查 A 的交易層拆解，用研究員口徑（變體）的 ALL 主跑重算（對應研究員的上櫃 +2.73%）
    AO, Gf = K.adj_open_matrix(P)
    op_ok = np.isfinite(P.O)
    trA = mains["ALL"]["trades"].copy()
    trA["mk"] = P.M[trA.tb.values, trA.i.values]

    def pool_ew(Um, s_, a_, b_):
        idx = np.where(Um[s_] & op_ok[a_])[0]
        return float(np.nanmean(AO[b_, idx] / AO[a_, idx] - 1.0))
    gr = []; pw = []; pwo = []; r5 = []
    for r_ in trA.itertuples():
        a_, b_ = int(r_.tb), int(r_.ts)
        gr.append(float(AO[b_, r_.i] / AO[a_, r_.i] - 1.0))
        own, oth = (U["CUR"], U["OTC"]) if r_.mk == 1 else (U["OTC"], U["CUR"])
        pw.append(pool_ew(own, a_ - 1, a_, b_)); pwo.append(pool_ew(oth, a_ - 1, a_, b_))
        r5.append(float(P.e50o[b_] / P.e50o[a_] - 1.0))
    trA["gross"] = gr; trA["pool_own"] = pw; trA["pool_other"] = pwo; trA["r50"] = r5
    ra = {}
    for nm_, g_ in (("full", trA), ("from2019", trA[trA.tb > i19]), ("2019_2023", trA[(trA.tb > i19) & (trA.tb <= i24)]),
                    ("from2024", trA[trA.tb > i24])):
        ra[nm_] = {("twse" if k_ == 1 else "tpex"): dict(n=int(len(h_)), exp_net=float(h_.ret.mean()), gross=float(h_.gross.mean()),
                                                         cost_drag=float((h_.gross - h_.ret).mean()), pool_own_ew=float(h_.pool_own.mean()),
                                                         pool_other_ew=float(h_.pool_other.mean()), r50=float(h_.r50.mean()),
                                                         selection=float((h_.gross - h_.pool_own).mean()),
                                                         own_vs_other_pool=float((h_.pool_own - h_.pool_other).mean()))
                   for k_, h_ in g_.groupby("mk")}
    diag["reviewA_trade_level_variant_ALL"] = ra
    print("reviewA variant", json.dumps({k: {m: (v["n"], round(v["exp_net"], 4), round(v["pool_own_ew"], 4), round(v["selection"], 4),
                                                 round(v["cost_drag"], 4), round(v["own_vs_other_pool"], 4)) for m, v in d_.items()}
                                         for k, d_ in ra.items()}), flush=True)
    out["divergence_diag_sd0_inf"] = diag
    print(json.dumps(diag, ensure_ascii=False, default=str)[:4000], flush=True)
    js = json.dumps(out, ensure_ascii=False, indent=1, default=lambda o: (bool(o) if isinstance(o, np.bool_) else
                                                                          (int(o) if isinstance(o, np.integer) else float(o))))
    open(os.path.join(HERE, "compare_vs_researcher.json"), "w").write(js)
    print("written")


if __name__ == "__main__":
    main(sys.argv[1])
