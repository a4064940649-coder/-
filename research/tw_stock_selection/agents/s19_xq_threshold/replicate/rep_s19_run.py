# -*- coding: utf-8 -*-
"""S19 獨立複製：驗收（P0、A0、A1、A2、A3）、7 策略 × 2 資金格主跑＋八項、20 條路徑、配對、拆解、四段空頭、事前判定 J1～J7、
逐年表、名單差異、S5 多買／多續抱／少買、審查 A（月份聚類 t）、審查 B（XQ 不能設上限／固定金額）、審查 C（腳本細節計數），然後凍結。
用法：python3 -I -B rep_s19_run.py <scratch_dir>
需先跑 s1 的 rep_s1_panel.py <scratch_dir> 與 s3 的 rep_s3_names.py <scratch_dir>/innov_names.json。
輸出（本資料夾）：my_numbers.json、my_numbers.sha256、my_trades.csv.gz、my_paths.csv.gz
凍結前不讀研究員 FINDINGS 開頭註記與第 3 節以後、results*.json、navs.csv.gz、trades.csv.gz。
"""
import sys, os, json, hashlib, resource, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import rep_s19_lib as K
S, Q, R, L = K.S, K.Q, K.R, K.L
ST, X2 = K.ST, K.X2

STRATS = ["E0", "XR", "XU", "XT", "XT0", "XDYN", "XS3"]
THRC = ["XU", "XT", "XT0", "XDYN", "XS3"]
FUNDS = {"W10_70": dict(sat_w=0.7, e50_min=20_000.0, caps=[600_000.0, 800_000.0, 1_000_000.0, 1_500_000.0, 2_000_000.0],
                        main=1_000_000.0),
         "SAT": dict(sat_w=1.0, e50_min=float("inf"), caps=[420_000.0, 560_000.0, 700_000.0, 1_050_000.0, 1_400_000.0],
                     main=700_000.0)}
SHIFTS = [0, 1, 2, 3]
BEARS = {"2018": ("2018-01-01", "2018-12-31"), "2020-03": ("2020-01-01", "2020-06-30"),
         "2022": ("2022-01-01", "2022-12-31"), "2024-07~2025-04": ("2024-07-01", "2025-04-30")}
DYN = {2016: (2.5, 2.0), 2017: (2.06, 1.71), 2018: (2.51, 1.98), 2019: (2.39, 1.93), 2020: (2.33, 1.83),
       2021: (3.16, 2.54), 2022: (3.43, 2.97), 2023: (2.29, 1.77), 2024: (2.36, 1.81), 2025: (2.6, 2.19),
       2026: (2.59, 2.11)}


def rss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024


def jdef(o):
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, set):
        return sorted(o)
    return str(o)


def main(sc, odir=HERE):
    t_start = time.time()
    P = R.build(sc)
    excl = S.innov_mask_fn(P, sc)
    T = P.T; e = T - 1; t0 = P.t0
    i19 = L.didx(P, "2018-12-28"); i24 = L.didx(P, "2023-12-29"); ih = L.didx(P, "2021-02-19")
    assert P.dstr[i19] == "2018-12-28" and P.dstr[i24] == "2023-12-29"
    da = np.array(P.dstr); codes = np.array(P.codes)
    Rv, mi0 = L.load_rev(P)
    F, _aux = K.K17.family(Rv, mi0)
    SS = F["STD"]; nm = SS.shape[0]
    K.K14.attach_e50_fixed(P, sc)
    e50 = P.e50.values.astype(float)
    K.script_panel(P)
    mav = R.ma(P, 60)
    print("panel", round(time.time() - t_start), "rss", rss(), flush=True)

    def win(a, b):
        return int(np.searchsorted(da, a, side="left")), int(np.searchsorted(da, b, side="right")) - 1
    bw = {k: win(a, b) for k, (a, b) in BEARS.items()}

    def nav_stats(nav):
        md = L.maxdd(P, nav, t0, e)
        return dict(ann_2019=L.ann(P, nav, i19, e), ann_full=L.ann(P, nav, t0, e), ann_2024=L.ann(P, nav, i24, e),
                    ann_1923=L.ann(P, nav, i19, i24), ann_pre19=L.ann(P, nav, t0, i19), mdd_full=md[0],
                    mdd_peak=md[1], mdd_trough=md[2], ann_h1=L.ann(P, nav, t0, ih), ann_h2=L.ann(P, nav, ih, e),
                    **{"bear_" + k: L.maxdd(P, nav, a, b)[0] for k, (a, b) in bw.items()})

    out = dict(meta=dict(T=T, N=P.N, first=P.dstr[0], last=P.dstr[-1], t0=P.dstr[t0], i19=P.dstr[i19],
                         i24=P.dstr[i24], ih=P.dstr[ih], bear_windows={k: (P.dstr[a], P.dstr[b]) for k, (a, b) in bw.items()},
                         note="面板為複製員自建（N＝2130；研究員 2142，差 12 檔 91xx，股池本來就排除）。SURGE＝rep_s17 STD（σ≈0 當缺值）；"
                              "0050 停牌日修正（rep_s14）。門檻版買進掃完整份候選、買到 10 檔滿（listmax＝9999）。"))

    # ================================================================ 訊號
    schA = K.K9.schedule(P, nm, mi0, "A10")
    schB = K.K9.schedule(P, nm, mi0, "B11opt")
    lists_E0 = {}; lists_XR = {}; plans = {}
    for sh in SHIFTS:
        lists_E0[sh] = Q.make_lists(P, SS, mi0, shift=sh, excl=excl)
        b, k = Q.cell_plans(lists_E0[sh], "C10", x=0.10)
        plans[("E0", sh)] = (b, k)
        lists_XR[sh] = K.K9.make_lists_sched(P, SS, schB, shift=sh, excl=excl)
        b, k, _ = K.K9.plans_from_lists(lists_XR[sh])
        plans[("XR", sh)] = (b, k)

    def thr_const(ms, xs):
        return lambda s0, s: (ms, xs)

    def thr_dyn(s0, s):
        return DYN[int(P.dstr[s0][:4])]
    CFG = {"XU": ("res", True, thr_const(2.5, 2.0)), "XT": ("scr", False, thr_const(2.5, 2.0)),
           "XT0": ("scr", False, thr_const(0.0, 2.0)), "XDYN": ("scr", False, thr_dyn),
           "XS3": ("scr", False, thr_const(3.0, 2.5))}
    infos = {}
    for c, (u, xu, fn) in CFG.items():
        for sh in SHIFTS:
            b, k, inf = K.make_thr(P, SS, schB, excl, mav, u, xu, fn, shift=sh)
            plans[(c, sh)] = (b, k); infos[(c, sh)] = inf
    print("signals", round(time.time() - t_start), "rss", rss(), flush=True)

    # ================================================================ 驗收
    acc = dict(P0=dict(T=T, N=P.N, last=P.dstr[-1], ok=bool(T == 2876 and P.dstr[-1] == "2026-10-22"),
                       note="研究員 N＝2142；複製員面板 2130（已知差異，91xx）"))
    LM = {"E0": None, "XR": None}

    def sim(c, fund, sh=0, cap=None, cost=None, block_span=None, rec_pos=False, pl=None, **kw):
        f = FUNDS[fund]
        b, k = pl if pl is not None else plans[(c, sh)]
        lm = kw.pop("listmax") if "listmax" in kw else (9999 if c in THRC else None)
        return K.simulate_s19(P, b, k, sat_w=f["sat_w"], e50_min=f["e50_min"], capital=cap or f["main"],
                              block_span=block_span, rec_pos=rec_pos, listmax=lm,
                              eod=K.make_eod_keep(k), **kw, **(cost or ST))
    x = sim("E0", "W10_70"); n_ = x["nav"]
    acc["A0"] = dict(ann_2019=L.ann(P, n_, i19, e), ann_full=L.ann(P, n_, t0, e), mdd=L.maxdd(P, n_, t0, e)[0],
                     known=(0.48126, 0.34082, -0.33404))
    acc["A0"]["ok_0.025pp"] = bool(abs(acc["A0"]["ann_2019"] - 0.48126) < 0.00025 and abs(acc["A0"]["ann_full"] - 0.34082) < 0.00025
                                   and abs(acc["A0"]["mdd"] + 0.33404) < 0.00025)
    y = K.K14.simulate_s14(P, *plans[("E0", 0)], sat_w=0.7, **ST)
    acc["A0"]["eod_wrapper_identical_to_engine_keep"] = bool(np.array_equal(y["nav"][t0:], n_[t0:]))
    x = sim("E0", "SAT", cap=1_000_000.0); n_ = x["nav"]
    acc["A0_SAT_1m"] = dict(ann_2019=L.ann(P, n_, i19, e), ann_full=L.ann(P, n_, t0, e), known=(0.4192, 0.2945))
    x = sim("XR", "W10_70"); n_ = x["nav"]
    acc["A1"] = dict(ann_2019=L.ann(P, n_, i19, e), ann_full=L.ann(P, n_, t0, e), ann_2024=L.ann(P, n_, i24, e),
                     mdd=L.maxdd(P, n_, t0, e)[0], known_researcher=(0.4948, 0.3524, -0.3006), known_rep_s9=(0.4953, 0.3527))
    acc["A1"]["ok_0.10pp_vs_researcher"] = bool(abs(acc["A1"]["ann_2019"] - 0.4948) < 0.001 and abs(acc["A1"]["ann_full"] - 0.3524) < 0.001)
    # A2：門檻＝當月第 10／20 名分數 → 等於 XR
    a2 = {}
    for sh in SHIFTS:
        lx = lists_XR[sh]

        def fn(s0, s, lx=lx):
            sv = SS[lx[s]["m_rev"]]; a = lx[s]["lma"]
            return (sv[a[9]] if len(a) >= 10 else -np.inf, sv[a[19]] if len(a) >= 20 else -np.inf)
        b, k, _ = K.make_thr(P, SS, schB, excl, mav, "res", True, fn, shift=sh)
        bx, kx = plans[("XR", sh)]
        nb = sum(1 for s in bx if not np.array_equal(np.asarray(bx[s], int), np.asarray(b.get(s, []), int)))
        nk = sum(1 for s in kx if kx[s] != k.get(s))
        a2["shift%d" % sh] = dict(n_rows=len(bx), rows_equal=sorted(bx) == sorted(b), buy_diff_rows=nb, keep_diff_rows=nk)
    acc["A2"] = a2
    print("acceptance", json.dumps(acc, default=jdef), flush=True)
    assert acc["A0"]["ok_0.025pp"] and acc["A1"]["ok_0.10pp_vs_researcher"]

    # ================================================================ 主跑＋八項
    b50 = nav_stats(e50); b50["yearly"] = L.yearly(P, e50)
    out["bench_0050"] = b50
    main_res = {}; cells = {}; trade_rows = []
    for c in STRATS:
        for fund in FUNDS:
            x = sim(c, fund, rec_pos=(c in ("XT", "XR", "E0")))
            main_res[(c, fund)] = x
            nav = x["nav"]; tr = x["trades"]
            o = nav_stats(nav)
            o["yearly"] = L.yearly(P, nav)
            o["n_trades"] = int(len(tr)); o["n_open"] = len(x["open_pos"])
            o["n_buys"] = int(len(tr) + len(x["open_pos"]))
            sp, top = R.span_of(tr, 5)
            x2 = sim(c, fund, cost=X2); bl = sim(c, fund, block_span=sp)
            o["x2"] = dict(ann_2019=L.ann(P, x2["nav"], i19, e), ann_full=L.ann(P, x2["nav"], t0, e))
            o["block5"] = dict(ann_2019=L.ann(P, bl["nav"], i19, e), ann_full=L.ann(P, bl["nav"], t0, e))
            o["top5"] = top
            ck = dict(a2019=o["ann_2019"] > b50["ann_2019"], afull=o["ann_full"] > b50["ann_full"],
                      x2_2019=o["x2"]["ann_2019"] > b50["ann_2019"], x2_full=o["x2"]["ann_full"] > b50["ann_full"],
                      bl_2019=o["block5"]["ann_2019"] > b50["ann_2019"], bl_full=o["block5"]["ann_full"] > b50["ann_full"],
                      h1=o["ann_h1"] > b50["ann_h1"], h2=o["ann_h2"] > b50["ann_h2"])
            o["checks"] = ck; o["n_checks_pass"] = int(sum(ck.values()))
            o["J7"] = "值得上 XQ 測" if all(ck.values()) else ("不值得" if (not ck["a2019"] and not ck["afull"]) else "有條件")
            exec_days = [s + 1 for s in plans[(c, 0)][0]]
            npx = x["npos"][exec_days]
            o["npos_exec_dist"] = {str(int(k_)): int(v_) for k_, v_ in zip(*np.unique(npx, return_counts=True))}
            o["npos_max"] = int(x["npos"][t0:].max()); o["npos_mean"] = float(x["npos"][t0 + 1:].mean())
            o["cnt"] = {k_: (float(v_) if isinstance(v_, float) else int(v_)) for k_, v_ in x["cnt"].items()}
            cells["%s|%s" % (c, fund)] = o
            for r_ in tr.itertuples():
                trade_rows.append((c, fund, r_.code, r_.buy_date, r_.sell_date, round(r_.ret, 6), round(r_.pnl, 1),
                                   int(r_.hold), r_.why))
            for cc_, bd, cst, val, i_ in x["open_pos"]:
                trade_rows.append((c, fund, cc_, bd, "", np.nan, round(val - cst, 1), -1, "open"))
            print(c, fund, round(o["ann_2019"], 4), round(o["ann_full"], 4), round(o["ann_2024"], 4),
                  round(o["mdd_full"], 4), o["n_trades"], o["n_checks_pass"], o["J7"], flush=True)
    out["main"] = cells
    pd.DataFrame(trade_rows, columns=["strat", "fund", "code", "buy_date", "sell_date", "ret", "pnl", "hold", "why"]).to_csv(
        os.path.join(odir, "my_trades.csv.gz"), index=False, compression="gzip")

    # A3：XT 主跑
    a3 = {}
    for fund in FUNDS:
        x = main_res[("XT", fund)]; tr = x["trades"]
        bad_buy = []; bad_exit = []
        allbuys = [(int(r_.i), int(r_.tb)) for r_ in tr.itertuples()] + [(int(i_), P.dstr.index(bd)) for _, bd, _, _, i_ in x["open_pos"]]
        for i, tb in allbuys:
            s = tb - 1; m = infos[("XT", 0)][s]["m_rev"]; sv = SS[m, i]
            ok = (np.isfinite(sv) and sv >= 2.5 and P.G[s, i] > mav[s, i] and P.G[s, i] / P.G[s - 20, i] - 1 > 0.10
                  and bool(K.script_universe(P, s, excl)[i]))
            if not ok:
                bad_buy.append((codes[i], P.dstr[tb]))
        nrule = 0
        for r_ in tr.itertuples():
            if r_.why != "sig":
                continue
            nrule += 1
            s = int(r_.mk); i = int(r_.i); m = infos[("XT", 0)][s]["m_rev"]; sv = SS[m, i]
            if not ((not np.isfinite(sv)) or sv < 2.0 or (np.isfinite(mav[s, i]) and P.G[s, i] <= mav[s, i])):
                bad_exit.append((codes[i], P.dstr[s]))
        a3[fund] = dict(npos_max=int(x["npos"][t0:].max()), n_buys=len(allbuys), bad_buy=bad_buy, n_rule_exits=nrule,
                        bad_exit=bad_exit, why_counts=tr.why.value_counts().to_dict())
    acc["A3"] = a3
    out["acceptance"] = acc
    print("A3", json.dumps(a3, default=jdef), flush=True)

    # ================================================================ 20 條路徑
    rows = []; pnav = {}
    for c in STRATS:
        for fund, f in FUNDS.items():
            for sh in SHIFTS:
                for cp in f["caps"]:
                    x = main_res[(c, fund)] if (sh == 0 and cp == f["main"]) else sim(c, fund, sh, cp)
                    nav = x["nav"]
                    rows.append(dict(strat=c, fund=fund, shift=sh, capital=cp, n_trades=int(len(x["trades"])),
                                     **{k_: v_ for k_, v_ in nav_stats(nav).items() if k_ not in ("mdd_peak", "mdd_trough")}))
                    pnav[(c, fund, sh, cp)] = nav
        print("paths", c, round(time.time() - t_start), "rss", rss(), flush=True)
    pr = pd.DataFrame(rows)
    pr.to_csv(os.path.join(odir, "my_paths.csv.gz"), index=False, compression="gzip", float_format="%.6f")
    PC = ["ann_2019", "ann_full", "ann_1923", "ann_2024", "ann_pre19", "mdd_full"] + ["bear_" + k for k in BEARS]
    p20 = {}
    for (c, fund), g in pr.groupby(["strat", "fund"]):
        p20["%s|%s" % (c, fund)] = dict(
            ann_2019_median=float(g.ann_2019.median()), ann_2019_min=float(g.ann_2019.min()), ann_2019_max=float(g.ann_2019.max()),
            ann_full_median=float(g.ann_full.median()), ann_2024_median=float(g.ann_2024.median()),
            ann_1923_median=float(g.ann_1923.median()), mdd_median=float(g.mdd_full.median()),
            mdd_min=float(g.mdd_full.min()), mdd_max=float(g.mdd_full.max()),
            n_beat_0050_2019=int((g.ann_2019 > b50["ann_2019"]).sum()), n_trades_median=float(g.n_trades.median()))
    out["paths20"] = p20

    def paired(a, b, fund):
        xa = pr[(pr.strat == a) & (pr.fund == fund)].set_index(["shift", "capital"])
        xb = pr[(pr.strat == b) & (pr.fund == fund)].set_index(["shift", "capital"]).loc[xa.index]
        o = {}
        for col in PC:
            dd = (xa[col] - xb[col]).values
            o[col] = dict(n_pos=int((dd > 0).sum()), n_neg=int((dd < 0).sum()), median=float(np.median(dd)),
                          min=float(dd.min()), max=float(dd.max()))
        return o
    PAIRS = [("XT", "E0"), ("XR", "E0"), ("XU", "XR"), ("XT", "XU"), ("XT0", "E0"), ("XDYN", "E0"), ("XS3", "E0"),
             ("XU", "E0"), ("XT0", "XT"), ("XDYN", "XT"), ("XS3", "XT")]
    pair = {"%s-%s|%s" % (a, b, f): paired(a, b, f) for f in FUNDS for a, b in PAIRS}
    out["paired"] = pair

    # ================================================================ 名單差異（位移 0）
    bXR = plans[("XR", 0)][0]
    ld = {}
    for c in THRC:
        bc = plans[(c, 0)][0]
        rows_ = sorted(bc)
        assert rows_ == sorted(bXR)
        ncand = np.array([len(bc[s]) for s in rows_]); nres = np.array([len(bXR[s]) for s in rows_])
        jac = [K.jaccard(bc[s], bXR[s]) for s in rows_]
        jv = np.array([j for j in jac if j is not None])
        inter = np.array([len(set(map(int, bc[s])) & set(map(int, bXR[s]))) for s in rows_])
        yr = {}
        for s, nc, j in zip(rows_, ncand, jac):
            yr.setdefault(P.dstr[s][:4], []).append((nc, j))
        ld[c] = dict(n_months=len(rows_), cand_mean=float(ncand.mean()), cand_median=float(np.median(ncand)),
                     cand_max=int(ncand.max()), months_cand_ge10=int((ncand >= 10).sum()), months_cand_gt10=int((ncand > 10).sum()),
                     months_cand0=int((ncand == 0).sum()),
                     xr_mean=float(nres.mean()), share_xr_in_cand=float(inter.sum() / nres.sum()),
                     jaccard_mean=float(jv.mean()), jaccard_n=int(len(jv)), n_both_empty=int(sum(j is None for j in jac)),
                     jaccard_mean_empty_as1=float(np.mean([1.0 if j is None else j for j in jac])),
                     by_year={y: dict(cand_mean=float(np.mean([a for a, _ in v])),
                                      jaccard_mean=float(np.mean([b for _, b in v if b is not None])) if any(b is not None for _, b in v) else None)
                              for y, v in sorted(yr.items())},
                     monthly=[(P.dstr[s], int(nc), None if j is None else round(j, 4)) for s, nc, j in zip(rows_, ncand, jac)])
        if c == "XT":
            nfull = {}
            for fund in FUNDS:
                npx = main_res[("XT", fund)]["npos"]
                nfull[fund] = dict(exec_days_full10=int(sum(1 for s in rows_ if npx[s + 1] >= 10)),
                                   signal_close_full10=int(sum(1 for s in rows_ if npx[s] >= 10)))
            ld[c]["portfolio_full"] = nfull
    out["list_diff"] = ld

    # ================================================================ 事前判定
    dec = {}

    def j1(pc):
        md = pc["median"]; ns = max(pc["n_pos"], pc["n_neg"])
        if abs(md) >= 0.03:
            return ("明顯不同（%s）" % ("高" if md > 0 else "低")) if ns >= 15 else "明顯不同但方向不穩"
        if abs(md) >= 0.01:
            return "略有不同（%s）" % ("高" if md > 0 else "低")
        return "接近"
    for fund in FUNDS:
        o = {}
        pc = pair["XT-E0|%s" % fund]
        o["J1"] = {w: dict(median=pc[w]["median"], n_pos=pc[w]["n_pos"], n_neg=pc[w]["n_neg"], min=pc[w]["min"], max=pc[w]["max"],
                           cls=j1(pc[w])) for w in ("ann_2019", "ann_1923", "ann_2024")}
        comp = dict(signal_day=pair["XR-E0|%s" % fund]["ann_2019"]["median"], threshold=pair["XU-XR|%s" % fund]["ann_2019"]["median"],
                    script_detail=pair["XT-XU|%s" % fund]["ann_2019"]["median"], total=pc["ann_2019"]["median"])
        big = max(("signal_day", "threshold", "script_detail"), key=lambda k_: abs(comp[k_]))
        comp["J2"] = "三者都小" if all(abs(comp[k_]) < 0.01 for k_ in ("signal_day", "threshold", "script_detail")) else big
        o["J2"] = comp
        o["J3"] = {}
        for bk in BEARS:
            q = pc["bear_" + bk]
            cls = "較淺" if (q["median"] >= 0.01 and q["n_pos"] >= 15) else ("較深" if (q["median"] <= -0.01 and q["n_neg"] >= 15) else "差不多")
            o["J3"][bk] = dict(median=q["median"], n_shallower=q["n_pos"], n_deeper=q["n_neg"], cls=cls)
        o["J6"] = dict(main=cells["XT|%s" % fund], paths=p20["XT|%s" % fund])
        o["J6"] = dict(main_2019=cells["XT|%s" % fund]["ann_2019"], main_full=cells["XT|%s" % fund]["ann_full"],
                       main_2024=cells["XT|%s" % fund]["ann_2024"], main_mdd=cells["XT|%s" % fund]["mdd_full"],
                       paths_2019_median=p20["XT|%s" % fund]["ann_2019_median"], paths_2019_min=p20["XT|%s" % fund]["ann_2019_min"],
                       paths_2019_max=p20["XT|%s" % fund]["ann_2019_max"], paths_mdd_median=p20["XT|%s" % fund]["mdd_median"])
        dec[fund] = o
    # J4、J5（W10_70）
    fW = "W10_70"
    mxt = pair["XT-E0|%s" % fW]; mdy = pair["XDYN-E0|%s" % fW]
    j4c1 = abs(mdy["ann_2019"]["median"]) <= abs(mxt["ann_2019"]["median"]) - 0.01
    j4c2 = (abs(mdy["ann_1923"]["median"]) <= abs(mxt["ann_1923"]["median"])) and (abs(mdy["ann_2024"]["median"]) <= abs(mxt["ann_2024"]["median"]))
    j4c3 = ld["XDYN"]["jaccard_mean"] >= ld["XT"]["jaccard_mean"]
    dec["J4"] = dict(c1=bool(j4c1), c2=bool(j4c2), c3=bool(j4c3), xdyn_med=mdy["ann_2019"]["median"], xt_med=mxt["ann_2019"]["median"],
                     xdyn_1923=mdy["ann_1923"]["median"], xt_1923=mxt["ann_1923"]["median"],
                     xdyn_2024=mdy["ann_2024"]["median"], xt_2024=mxt["ann_2024"]["median"],
                     jac_xdyn=ld["XDYN"]["jaccard_mean"], jac_xt=ld["XT"]["jaccard_mean"],
                     cls="建議改動態" if (j4c1 and j4c2 and j4c3) else "不必改動態")
    clo = {c: abs(pair["%s-E0|%s" % (c, fW)]["ann_2019"]["median"]) for c in ("XT", "XT0", "XDYN", "XS3")}
    best = min(clo, key=lambda c: clo[c])
    rec = best if (best != "XT" and clo[best] <= clo["XT"] - 0.01 and ld[best]["jaccard_mean"] >= ld["XT"]["jaccard_mean"] - 0.05) else "XT"
    dec["J5"] = dict(closeness=clo, jaccard={c: ld[c]["jaccard_mean"] for c in clo}, best=best, recommend=rec,
                     warn_xt0=bool(clo["XT0"] >= 0.03))
    dec["J7"] = {k_: (v_["n_checks_pass"], v_["J7"]) for k_, v_ in cells.items()}
    out["decisions"] = dec
    print("decisions", json.dumps(dec, ensure_ascii=False, default=jdef)[:3000], flush=True)

    # ================================================================ 逐年表（事後）
    yt = {}
    for c in ("XT", "E0", "XR"):
        for fund, f in FUNDS.items():
            yrs = {}
            for y in range(2016, 2027):
                vals = [L.yearly(P, pnav[(c, fund, sh, cp)])[y] for sh in SHIFTS for cp in f["caps"]]
                yrs[y] = dict(main=cells["%s|%s" % (c, fund)]["yearly"][y], paths_median=float(np.median(vals)),
                              paths_min=float(np.min(vals)), paths_max=float(np.max(vals)))
            yt["%s|%s" % (c, fund)] = yrs
    for fund, f in FUNDS.items():
        yrs = {}
        for y in range(2016, 2027):
            dd = [L.yearly(P, pnav[("XT", fund, sh, cp)])[y] - L.yearly(P, pnav[("E0", fund, sh, cp)])[y] for sh in SHIFTS for cp in f["caps"]]
            yrs[y] = dict(median=float(np.median(dd)), n_pos=int((np.array(dd) > 0).sum()))
        yt["XT-E0|%s" % fund] = yrs
    yt["0050"] = {y: b50["yearly"][y] for y in range(2016, 2027)}
    out["yearly"] = yt

    # ================================================================ S5 與審查 A（XT｜W10_70 主跑）
    def adjo(t, i):
        if t >= T:
            return float(P.G[e, i]) if np.isfinite(P.G[e, i]) else K.K10.adj_open(P, e, i)
        return K.K10.adj_open(P, t, i)

    def e50o(t):
        return float(P.e50o[t]) if t < T else float(e50[e])
    s5 = {}
    for fund in FUNDS:
        x = main_res[("XT", fund)]; snap = x["snap"]; tr = x["trades"]
        bXT, kXT = plans[("XT", 0)]; bR, kR = plans[("XR", 0)]
        rows_ = sorted(bXT)
        nxt = {s: (rows_[j + 1] + 1 if j + 1 < len(rows_) else T) for j, s in enumerate(rows_)}
        skip = {}
        for t_, i_, why in x["skiplog"]:
            skip[(t_ - 1, int(i_))] = why
        A = []; B = []; Cc = []
        for s in rows_:
            t = s + 1; tn = nxt[s]
            held_s = set(snap.get(s, {}).keys()); held_t = set(snap.get(t, {}).keys())
            bought = held_t - held_s
            xr = set(map(int, bR[s]))
            base = e50o(tn) / e50o(t) - 1.0

            def ex(i):
                return adjo(tn, i) / adjo(t, i) - 1.0 - base
            for i in bought:
                if i not in xr:
                    A.append((s, i, ex(i)))
            for i in held_s:
                if i in kXT[s] and i not in kR[s]:
                    B.append((s, i, ex(i)))
            full_t = len(held_t) >= 10
            candXT = set(map(int, bXT[s]))
            for i in xr:
                if i in held_t:
                    continue
                reason = skip.get((s, i), "in_cand_not_reached" if i in candXT else "not_in_cand")
                if i in held_s:
                    reason = "sold_today"
                Cc.append((s, i, ex(i), full_t, reason))

        def summ(v, g=None):
            v = np.asarray(v, float)
            if len(v) == 0:
                return dict(n=0)
            o = dict(n=int(len(v)), mean=float(v.mean()), median=float(np.median(v)), t_naive=K.tstat(v))
            if g is not None and len(set(g)) > 1:
                ct, G = K.cluster_t(v, g); mm, mt, nmn = K.month_mean_t(v, g)
                o.update(t_cluster_month=ct, n_months=G, month_mean=mm, t_month_mean=mt)
            return o
        bset = {}
        for r_ in tr.itertuples():
            bset[(int(r_.tb), int(r_.i))] = float(r_.ret)
        whole = [bset[(s + 1, i)] for s, i, _ in A if (s + 1, i) in bset]
        o = dict(a_extra_buys=summ([v for _, _, v in A], [s for s, _, _ in A]),
                 a_whole_trade=summ(whole),
                 b_extra_holds=summ([v for _, _, v in B], [s for s, _, _ in B]),
                 c_missed_all=summ([v for _, _, v, _, _ in Cc], [s for s, _, _, _, _ in Cc]))
        o["c_missed_full10"] = summ([v for _, _, v, f_, _ in Cc if f_], [s for s, _, _, f_, _ in Cc if f_])
        o["c_missed_noslot"] = summ([v for _, _, v, _, r_ in Cc if r_ == "no_slot"], [s for s, _, _, _, r_ in Cc if r_ == "no_slot"])
        o["c_missed_full10_in_cand"] = summ([v for s, i, v, f_, r_ in Cc if f_ and r_ != "not_in_cand"],
                                            [s for s, i, v, f_, r_ in Cc if f_ and r_ != "not_in_cand"])
        o["c_missed_full10_not_in_cand"] = summ([v for s, i, v, f_, r_ in Cc if f_ and r_ == "not_in_cand"],
                                                [s for s, i, v, f_, r_ in Cc if f_ and r_ == "not_in_cand"])
        o["c_missed_notfull"] = summ([v for _, _, v, f_, _ in Cc if not f_], [s for s, _, _, f_, _ in Cc if not f_])
        o["c_reason_counts"] = pd.Series([r_ for *_, r_ in Cc]).value_counts().to_dict()
        # 前 3 大月份貢獻（滿 10 檔錯過）
        dfm = pd.DataFrame([(P.dstr[s], v) for s, i, v, f_, r_ in Cc if f_], columns=["d", "v"])
        if len(dfm):
            gm = dfm.groupby("d").v.agg(["sum", "count", "mean"]).sort_values("sum", ascending=False)
            o["c_full10_top_months"] = [(d_, int(r_["count"]), float(r_["mean"])) for d_, r_ in gm.head(5).iterrows()]
            v_ = dfm.v.values
            o["c_full10_trimmed"] = dict(drop_top5_obs_mean=float(np.sort(v_)[:-5].mean()) if len(v_) > 5 else None,
                                         median=float(np.median(v_)))
        s5[fund] = o
    out["s5_reviewA"] = s5
    print("s5", json.dumps(s5, default=jdef)[:3000], flush=True)

    # ================================================================ 審查 B（SAT：XQ 不能設上限／固定金額）
    rB = {}
    fund = "SAT"; f = FUNDS[fund]
    VAR = {"B1_nav10_nolimit": dict(slot_max=9999), "B2_fixed_nolimit": dict(slot_max=9999, fixed=True),
           "B3_fixed_max10": dict(fixed=True)}
    for vn, vk in VAR.items():
        vals = []; dd = []; ddm = []; main_v = None
        for sh in SHIFTS:
            for cp in f["caps"]:
                kw = dict(slot_max=vk.get("slot_max"))
                if vk.get("fixed"):
                    kw["fixed_amt"] = cp / 10.0
                x = sim("XT", fund, sh, cp, **kw)
                nav = x["nav"]; st = nav_stats(nav)
                base = pnav[("XT", fund, sh, cp)]
                vals.append(st["ann_2019"]); dd.append(st["ann_2019"] - L.ann(P, base, i19, e))
                ddm.append(st["mdd_full"] - L.maxdd(P, base, t0, e)[0])
                if sh == 0 and cp == f["main"]:
                    main_v = dict(ann_2019=st["ann_2019"], ann_full=st["ann_full"], ann_2024=st["ann_2024"], mdd=st["mdd_full"],
                                  n_trades=int(len(x["trades"])), npos_max=int(x["npos"][t0:].max()),
                                  npos_mean=float(x["npos"][t0 + 1:].mean()), stock_w_mean=float(x["stkw"][t0 + 1:].mean()),
                                  stock_w_2024_mean=float(x["stkw"][i24:].mean()),
                                  yearly=L.yearly(P, nav))
        dd = np.array(dd); ddm = np.array(ddm)
        rB[vn] = dict(main=main_v, paths_2019_median=float(np.median(vals)), paths_2019_min=float(np.min(vals)),
                      paths_2019_max=float(np.max(vals)), diff_vs_XT_2019_median=float(np.median(dd)),
                      diff_vs_XT_n_pos=int((dd > 0).sum()), diff_vs_XT_min=float(dd.min()), diff_vs_XT_max=float(dd.max()),
                      mdd_diff_vs_XT_median=float(np.median(ddm)))
        print("reviewB", vn, json.dumps(rB[vn]["main"], default=jdef)[:400], rB[vn]["diff_vs_XT_2019_median"], flush=True)
    xb = main_res[("XT", "SAT")]
    rB["XT_SAT_base"] = dict(ann_2019=cells["XT|SAT"]["ann_2019"], npos_mean=float(xb["npos"][t0 + 1:].mean()),
                             stock_w_mean=float(xb["stkw"][t0 + 1:].mean()))
    # listmax 敏感度：引擎原本只掃名單前 10 個
    for fund_ in FUNDS:
        vals = []; dd = []
        for sh in SHIFTS:
            for cp in FUNDS[fund_]["caps"]:
                x = sim("XT", fund_, sh, cp, listmax=None)
                a_ = L.ann(P, x["nav"], i19, e); vals.append(a_); dd.append(a_ - L.ann(P, pnav[("XT", fund_, sh, cp)], i19, e))
                if sh == 0 and cp == FUNDS[fund_]["main"]:
                    mv = dict(ann_2019=a_, ann_full=L.ann(P, x["nav"], t0, e), mdd=L.maxdd(P, x["nav"], t0, e)[0], n_trades=int(len(x["trades"])))
        rB["listmax10_%s" % fund_] = dict(main=mv, paths_2019_median=float(np.median(vals)), diff_median=float(np.median(dd)),
                                          diff_n_nonzero=int((np.abs(dd) > 1e-12).sum()))
    out["reviewB"] = rB

    # ================================================================ 審查 C（腳本細節計數）
    rC = {}
    # C1：IsRebal（市場日曆）＝B11
    cal_ok = 0; cal_bad = []
    for e_ in schB:
        k_ = e_["k"]
        if k_ >= T or k_ < 1:
            continue
        dn = int(da[k_][8:10]); dp = int(da[k_ - 1][8:10])
        isr = dn >= 11 and (dp < 11 or dp > dn)
        if isr:
            cal_ok += 1
        else:
            cal_bad.append(da[k_])
    # 同月內不應有第二個 IsRebal
    extra = 0
    for t in range(1, T):
        dn = int(da[t][8:10]); dp = int(da[t - 1][8:10])
        if dn >= 11 and (dp < 11 or dp > dn):
            extra += 1
    rC["C1_isrebal_market_calendar"] = dict(n_sched=len(schB), isrebal_on_B11=cal_ok, not_isrebal=cal_bad, n_isrebal_days_total=extra)
    # C2：個股 K 棒版 IsRebal（Date[1] 是該股前一根 K 棒）
    tradedT = P.traded
    ym = np.array([d[:7] for d in da])
    sigrows = sorted(plans[("XT", 0)][0])
    sig_by_ym = {da[s][:7]: s for s in sigrows}

    def stock_isrebal(i, ym_):
        rows = np.where(tradedT[:, i] & (ym == ym_))[0]
        for t in rows:
            pr_ = np.where(tradedT[:t, i])[0]
            if len(pr_) == 0:
                return None
            dp = int(da[pr_[-1]][8:10]); dn = int(da[t][8:10])
            if dn >= 11 and (dp < 11 or dp > dn):
                return int(t)
        return -1
    held_events = dict(same=0, later=0, missed=0, list=[])
    for fund in FUNDS:
        snap = main_res[("XT", fund)]["snap"]
        ev = dict(same=0, later=0, missed=0, not_traded_on_signal=0, list=[])
        for s in sigrows:
            for i in snap.get(s, {}):
                if tradedT[s, i] and int(da[s][8:10]) >= 11:
                    pr_ = np.where(tradedT[:s, i])[0]
                    dp = int(da[pr_[-1]][8:10]) if len(pr_) else 0
                    dn = int(da[s][8:10])
                    if dn >= 11 and (dp < 11 or dp > dn):
                        ev["same"] += 1; continue
                r_ = stock_isrebal(i, da[s][:7])
                if not tradedT[s, i]:
                    ev["not_traded_on_signal"] += 1
                if r_ is None or r_ == -1:
                    ev["missed"] += 1; ev["list"].append((da[s], codes[i], "missed"))
                elif r_ == s:
                    ev["same"] += 1
                else:
                    ev["later"] += 1; ev["list"].append((da[s], codes[i], da[r_]))
        rC["C2_held_stock_isrebal_%s" % fund] = ev
    # C3：B11 當天沒成交、但同月稍後自己的 IsRebal 那天符合 XT 買進條件的（XQ 會在非換股日買）
    late = []
    for s in sigrows:
        m = infos[("XT", 0)][s]["m_rev"]; sv = SS[m]
        cand = np.where(np.isfinite(sv) & (sv >= 2.5) & P.inbar[s] & ~tradedT[s] & P.is_twse[s] & ~excl(s))[0]
        for i in cand:
            r_ = stock_isrebal(int(i), da[s][:7])
            if r_ is None or r_ == -1 or r_ <= s:
                continue
            u = K.script_universe(P, r_, excl)[i]
            ok = bool(u and P.G[r_, i] > mav[r_, i] and P.G[r_, i] / P.G[r_ - 20, i] - 1 > 0.10)
            if ok:
                late.append((da[s], codes[i], da[r_], float(sv[i])))
    rC["C3_late_entry_events"] = dict(n=len(late), list=late)
    # C4：XQ K 棒版（只用有成交的日子）的成交值、均線、20 根報酬 → 候選名單差異（位移 0）
    advb, mab, r20b = K.script_panel_bars(P)
    nd = 0; nadd = 0; nrem = 0; nrows = 0; ex_ = []
    kd = 0
    for s in sigrows:
        m = infos[("XT", 0)][s]["m_rev"]; sv = SS[m]
        with np.errstate(invalid="ignore"):
            u = P.is_twse[s] & ~excl(s) & tradedT[s] & (P.Cadj[s] >= 10.0) & (advb[s] >= 1e7)
            cand = u & np.isfinite(sv) & (sv >= 2.5) & (P.Cadj[s] > mab[s]) & (r20b[s] > 0.10)
        a_ = set(np.where(cand)[0].tolist()); b_ = set(map(int, plans[("XT", 0)][0][s]))
        nrows += 1
        if a_ != b_:
            nd += 1; nadd += len(a_ - b_); nrem += len(b_ - a_)
            if len(ex_) < 15:
                ex_.append((da[s], [codes[i] for i in sorted(a_ - b_)], [codes[i] for i in sorted(b_ - a_)]))
    rC["C4_bar_based_vs_row_based"] = dict(n_rows=nrows, rows_diff=nd, added=nadd, removed=nrem, examples=ex_)
    del advb, mab, r20b
    # C5：XT 候選中「研究股池外」的檔數（腳本股池才有）與 XT 持股在換股日已離開研究股池的次數
    n_out = 0; n_tot = 0
    for s in sigrows:
        ures = K.research_universe(P, s, excl)
        b_ = plans[("XT", 0)][0][s]
        n_tot += len(b_); n_out += int((~ures[np.asarray(b_, int)]).sum()) if len(b_) else 0
    held_out = 0; held_tot = 0
    snap = main_res[("XT", "W10_70")]["snap"]
    for s in sigrows:
        ures = K.research_universe(P, s, excl)
        for i in snap.get(s, {}):
            held_tot += 1; held_out += int(not ures[i])
    rC["C5_universe"] = dict(cand_total=n_tot, cand_outside_research_univ=n_out, held_total_W10=held_tot,
                             held_outside_research_univ_W10=held_out)
    out["reviewC"] = rC
    print("reviewC", json.dumps(rC, default=jdef, ensure_ascii=False)[:2500], flush=True)

    out["resources"] = dict(runtime_s=round(time.time() - t_start, 1), maxrss_mb=rss())
    js = json.dumps(out, ensure_ascii=False, indent=1, default=jdef)
    fp = os.path.join(odir, "my_numbers.json")
    open(fp, "w").write(js)
    h = hashlib.sha256(js.encode("utf-8")).hexdigest()
    open(os.path.join(odir, "my_numbers.sha256"), "w").write(h + "  my_numbers.json\n")
    print("done", h, out["resources"], flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
