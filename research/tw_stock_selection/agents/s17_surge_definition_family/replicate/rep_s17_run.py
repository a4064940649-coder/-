# -*- coding: utf-8 -*-
"""S17 獨立複製：驗收（P0、A0、A1、A2）、資料檢查（σ≈0 格數、股池內有值檔數、截面 Spearman）、
7 定義 × W10_70／WN_70 主跑＋八項、20 條路徑、逐條配對、四段空頭、min_frac 0.45／0.55、事前判定、
事後（重疊率、逐年、集中度、大贏家）、審查 A（NOMU 結構性或時期效果）、B（NOMU 機制：規模／持續成長）、C（XQ 門檻），然後凍結。
用法：python3 -I -B rep_s17_run.py <scratch_dir>
需先跑 s1 的 rep_s1_panel.py <scratch_dir> 與 s3 的 rep_s3_names.py <scratch_dir>/innov_names.json。
輸出（本資料夾）：my_numbers.json、my_numbers.sha256、my_trades.csv.gz、my_paths.csv.gz
凍結前不讀研究員 FINDINGS 第 2、8 節以外、results*.json、navs.csv.gz、trades.csv.gz。
"""
import sys, os, json, hashlib, resource, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import rep_s17_lib as K
S, Q, R, L = K.S, K.Q, K.R, K.L
ST, X2 = K.ST, K.X2
DEFS = K.DEFS
NEW = DEFS[1:]

CAPS = [600_000.0, 800_000.0, 1_000_000.0, 1_500_000.0, 2_000_000.0]
SHIFTS = [0, 1, 2, 3]
CELLS = {"W10_70": "W10", "WN_70": "WN"}
KNOWN = {"W10_70": (0.4813, 0.3408, -0.3340), "WN_70": (0.6224, 0.4455, -0.3950)}
BEARS = {"2018": ("2018-01-01", "2018-12-31"), "2020-03": ("2020-01-01", "2020-06-30"),
         "2022": ("2022-01-01", "2022-12-31"), "2024-07~2025-04": ("2024-07-01", "2025-04-30")}
BIG = ["2408", "3653", "1325", "2603"]
DC_SIGMA0 = dict(STD=3, W12=15, W36=0, RATE=3, SM3=1, NOMU=3, ROB=20)


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
    assert P.dstr[i19] == "2018-12-28" and P.dstr[i24] == "2023-12-29" and P.dstr[ih] == "2021-02-19"
    da = np.array(P.dstr)
    codes = np.array(P.codes)

    def win(a, b):
        return int(np.searchsorted(da, a, side="left")), int(np.searchsorted(da, b, side="right")) - 1
    bw = {k: win(a, b) for k, (a, b) in BEARS.items()}

    def bears(nav):
        return {k: L.maxdd(P, nav, a, b)[0] for k, (a, b) in bw.items()}

    def nav_stats(nav):
        md = L.maxdd(P, nav, t0, e)
        return dict(ann_2019=L.ann(P, nav, i19, e), ann_full=L.ann(P, nav, t0, e), ann_2024=L.ann(P, nav, i24, e),
                    ann_1923=L.ann(P, nav, i19, i24), ann_pre19=L.ann(P, nav, t0, i19), mdd_full=md[0],
                    mdd_peak=md[1], mdd_trough=md[2], ann_h1=L.ann(P, nav, t0, ih), ann_h2=L.ann(P, nav, ih, e))

    out = dict(meta=dict(T=T, N=P.N, first=P.dstr[0], last=P.dstr[-1], t0=P.dstr[t0], i19=P.dstr[i19],
                         i24=P.dstr[i24], ih=P.dstr[ih],
                         bear_windows={k: (P.dstr[a], P.dstr[b]) for k, (a, b) in bw.items()},
                         note="面板為複製員自建（N＝2130；研究員 2142，差 12 檔為 91xx，股池本來就排除）。"
                              "0050 停牌日用 S14 複製員的修正（還原開盤＝前一日還原收盤、原始開盤＝前一日原始收盤）。"))

    # ================================================================ SURGE 家族
    Rv, mi0 = L.load_rev(P)
    F, aux = K.family(Rv, mi0)
    S_l, _ = L.surge(Rv, mi0, null2013=True)        # S1 複製員的 SURGE（pandas 以外的 numpy std），A1 對照
    print("family built", round(time.time() - t_start), "rss", rss(), flush=True)

    acc = dict(P0=dict(T=T, N=P.N, last=P.dstr[-1], ok=bool(T == 2876 and P.dstr[-1] == "2026-10-22"),
                       note="研究員 N＝2142；複製員面板 2130（已知差異，91xx）"))

    # ---------------------------------------------------------------- A0（原 0050 與修正 0050）
    def lists_of(Sx, sh, Scnt=None):
        if Scnt is None:
            return Q.make_lists(P, Sx, mi0, shift=sh, excl=excl)
        return K.make_lists_cnt(P, Sx, Scnt, mi0, shift=sh, excl=excl)

    lists = {}; plans = {}
    for d in DEFS:
        for sh in SHIFTS:
            lists[(d, sh)] = lists_of(F[d], sh)
            plans[(d, sh)] = Q.cell_plans(lists[(d, sh)], "C10", x=0.10)
    print("lists", round(time.time() - t_start), "rss", rss(), flush=True)
    plans_l = {}
    for sh in SHIFTS:
        ll = Q.make_lists(P, S_l, mi0, shift=sh, excl=excl)
        plans_l[sh] = (ll, Q.cell_plans(ll, "C10", x=0.10))

    K.attach_e50_orig(P, sc)
    a0o = {}
    nav_orig = {}
    for cell, sz in CELLS.items():
        x = K.simulate(P, *plans[("STD", 0)], sat_w=0.7, sizing=sz, **ST)
        y = K.K9.simulate_s9(P, *plans_l[0][1], sat_w=0.7, sizing=sz, **ST)
        same = bool(np.array_equal(x["nav"][t0:], y["nav"][t0:]) and len(x["trades"]) == len(y["trades"]) and
                    (x["trades"][["code", "tb", "ts", "sh"]].values == y["trades"][["code", "tb", "ts", "sh"]].values).all())
        nav_orig[cell] = x["nav"]
        a0o[cell] = dict(ann_2019=L.ann(P, x["nav"], i19, e), ann_full=L.ann(P, x["nav"], t0, e),
                         mdd_full=L.maxdd(P, x["nav"], t0, e)[0], n_trades=int(len(x["trades"])),
                         identical_to_rep_s9_with_rep_s1_surge=same)
    acc["A0_orig_e50"] = a0o
    fixed = K.attach_e50_fixed(P, sc)
    e50 = P.e50.values.astype(float)
    a0 = {}
    for cell, sz in CELLS.items():
        x = K.simulate(P, *plans[("STD", 0)], sat_w=0.7, sizing=sz, **ST)
        nav = x["nav"]
        a0[cell] = dict(ann_2019=L.ann(P, nav, i19, e), ann_full=L.ann(P, nav, t0, e), ann_2024=L.ann(P, nav, i24, e),
                        mdd_full=L.maxdd(P, nav, t0, e)[0], n_trades=int(len(x["trades"])), known_researcher=KNOWN[cell],
                        max_abs_nav_diff_vs_orig_e50=float(np.max(np.abs(nav[t0:] - nav_orig[cell][t0:]))),
                        first_diff_date=(P.dstr[t0 + int(np.argmax(np.abs(nav[t0:] - nav_orig[cell][t0:]) > 1e-9))]
                                         if np.any(np.abs(nav[t0:] - nav_orig[cell][t0:]) > 1e-9) else None),
                        ann_2019_diff_vs_orig=L.ann(P, nav, i19, e) - a0o[cell]["ann_2019"],
                        ann_full_diff_vs_orig=L.ann(P, nav, t0, e) - a0o[cell]["ann_full"])
    a0["W10_70"]["ok_vs_researcher_0.025pp"] = bool(abs(a0["W10_70"]["ann_2019"] - 0.4813) < 0.00025 and
                                                    abs(a0["W10_70"]["ann_full"] - 0.3408) < 0.00025 and
                                                    abs(a0["W10_70"]["mdd_full"] + 0.3340) < 0.00025)
    a0["WN_70"]["ok_task_62.2x_44.5x"] = bool(0.6215 <= a0["WN_70"]["ann_2019"] < 0.6230 and
                                             0.4450 <= a0["WN_70"]["ann_full"] < 0.4460)
    a0["fixed_days"] = [x["date"] for x in fixed]
    acc["A0_fixed_e50"] = a0
    print("A0", json.dumps(acc["A0_orig_e50"], default=jdef), json.dumps(a0, default=jdef), flush=True)
    assert a0["W10_70"]["ok_vs_researcher_0.025pp"] and a0["WN_70"]["ok_task_62.2x_44.5x"]

    # ---------------------------------------------------------------- A1：STD 名單＝rep_s1 surge 名單
    a1 = dict(identical=True)
    for sh in SHIFTS:
        ll, (bq, kq) = plans_l[sh]
        b_, k_ = plans[("STD", sh)]
        ok = sorted(bq) == sorted(b_) and all(np.array_equal(bq[s], b_[s]) and kq[s] == k_[s] for s in bq)
        ok &= sorted(ll) == sorted(lists[("STD", sh)]) and all(np.array_equal(ll[s]["lma"], lists[("STD", sh)][s]["lma"]) for s in ll)
        a1["shift%d" % sh] = bool(ok); a1["identical"] &= bool(ok)
    rows16 = np.array([mi0 + m >= K.FIRST_SIG for m in range(Rv.shape[0])])
    A = F["STD"][rows16]; B = S_l[rows16]
    both = np.isfinite(A) & np.isfinite(B)
    a1["max_abs_score_diff"] = float(np.max(np.abs(A[both] - B[both])))
    a1["n_both"] = int(both.sum())
    only_l = np.argwhere(np.isfinite(B) & ~np.isfinite(A)); only_me = np.argwhere(np.isfinite(A) & ~np.isfinite(B))
    m16 = np.where(rows16)[0]
    a1["only_rep_s1_surge"] = [("%04d-%02d" % ((mi0 + m16[a]) // 12, (mi0 + m16[a]) % 12 + 1), codes[b], float(B[a, b]))
                               for a, b in only_l]
    a1["only_mine"] = len(only_me)
    acc["A1"] = a1
    print("A1", json.dumps(a1, default=jdef, ensure_ascii=False), flush=True)

    # ---------------------------------------------------------------- A2：純 Python 逐格重算
    rng = np.random.default_rng(17)
    a2 = {}
    for d in DEFS:
        Sx = F[d]
        fin = np.argwhere(np.isfinite(Sx))
        nan_ = np.argwhere(~np.isfinite(Sx) & np.isfinite(Rv) & rows16[:, None])
        pick_f = fin[rng.choice(len(fin), 30, replace=False)]
        pick_n = nan_[rng.choice(len(nan_), 10, replace=False)]
        mx = 0.0; bad = 0
        for m, j in pick_f:
            v = K.py_score(Rv, mi0, d, int(m), int(j))
            if v is None:
                bad += 1; continue
            mx = max(mx, abs(v - Sx[m, j]) / max(1.0, abs(v)))
        for m, j in pick_n:
            if K.py_score(Rv, mi0, d, int(m), int(j)) is not None:
                bad += 1
        a2[d] = dict(max_rel_diff=mx, n_mismatch_missing=bad, ok=bool(mx < 1e-8 and bad == 0))
    acc["A2"] = a2
    print("A2", a2, flush=True)
    assert all(v["ok"] for v in a2.values())
    out["acceptance"] = acc

    # ================================================================ 資料檢查（不含報酬）
    dc = {}
    s0c = K.sigma0_count(Rv, mi0)
    dc["sigma0_cells_all_market"] = {d: dict(mine=s0c[d][0], data_check=DC_SIGMA0[d],
                                             cells=[("%04d-%02d" % ((mi0 + m) // 12, (mi0 + m) % 12 + 1), codes[j])
                                                    for m, j in s0c[d][1]]) for d in DEFS}
    cov = {}; spr = {}
    sigrows = {}
    for m in range(Rv.shape[0]):
        if mi0 + m < K.FIRST_SIG:
            continue
        d10, j, s0 = L.rev_avail(P, mi0 + m)
        if s0 < t0 or s0 >= T - 1:
            continue
        sigrows[m] = s0
    for d in DEFS:
        ns = []; first = None; rhos = []
        for m, s0 in sigrows.items():
            u = R.universe(P, s0) & ~excl(s0)
            ok = u & np.isfinite(F[d][m])
            n = int(ok.sum())
            if n == 0:
                continue
            if n >= 50:
                ns.append(n)
                first = first or "%04d-%02d" % ((mi0 + m) // 12, (mi0 + m) % 12 + 1)
            if d != "STD":
                okb = ok & np.isfinite(F["STD"][m])
                if okb.sum() >= 50:
                    rhos.append(K.spearman(F[d][m][okb], F["STD"][m][okb]))
        cov[d] = dict(n_signal_months=len(ns), first_month=first, median_n=float(np.median(ns)), min_n=int(min(ns)),
                      max_n=int(max(ns)))
        if d != "STD":
            spr[d] = dict(median=float(np.median(rhos)), min=float(np.min(rhos)), n=len(rhos))
    dc["coverage_in_universe"] = cov
    dc["spearman_vs_STD"] = spr
    out["data_check"] = dc
    print("coverage", json.dumps(cov), "\nspearman", json.dumps(spr), flush=True)

    # ================================================================ 模擬
    def sim(d, cell, sh=0, cap=1_000_000.0, cost=None, block_span=None, min_frac=0.5, pl=None, rec_pos=False):
        b, k = pl if pl is not None else plans[(d, sh)]
        return K.simulate(P, b, k, sat_w=0.7, sizing=CELLS[cell], capital=cap, block_span=block_span,
                          min_frac=min_frac, rec_pos=rec_pos, **(cost or ST))

    b50 = dict(nav_stats(e50), bears=bears(e50), yearly=L.yearly(P, e50))
    out["bench_0050"] = b50

    def top5_share(tr, nav):
        return float(tr.sort_values("pnl", ascending=False).head(5).pnl.sum() / float(nav[e] - nav[t0]))

    def tstats(tr):
        w = tr.ret > 0
        return dict(n=int(len(tr)), win=float(w.mean()), exp=float(tr.ret.mean()), hold_mean=float(tr.hold.mean()),
                    avg_win=float(tr.ret[w].mean()), avg_loss=float(-tr.ret[~w].mean()))

    main_res = {}; cells = {}; trade_rows = []
    for d in DEFS:
        for cell in CELLS:
            x = sim(d, cell)
            main_res[(d, cell)] = x
            nav = x["nav"]; tr = x["trades"]
            o = nav_stats(nav)
            o["bears"] = bears(nav); o["yearly"] = L.yearly(P, nav)
            o["n_trades"] = int(len(tr)); o["n_open"] = len(x["open_pos"])
            o["trade_stats"] = tstats(tr)
            sp, top = R.span_of(tr, 5)
            x2 = sim(d, cell, cost=X2); bl = sim(d, cell, block_span=sp)
            o["x2"] = dict(ann_2019=L.ann(P, x2["nav"], i19, e), ann_full=L.ann(P, x2["nav"], t0, e))
            o["block5"] = dict(ann_2019=L.ann(P, bl["nav"], i19, e), ann_full=L.ann(P, bl["nav"], t0, e))
            o["top5"] = top; o["top5_share_of_gain"] = top5_share(tr, nav)
            ck = dict(a2019=o["ann_2019"] > b50["ann_2019"], afull=o["ann_full"] > b50["ann_full"],
                      x2_2019=o["x2"]["ann_2019"] > b50["ann_2019"], x2_full=o["x2"]["ann_full"] > b50["ann_full"],
                      bl_2019=o["block5"]["ann_2019"] > b50["ann_2019"], bl_full=o["block5"]["ann_full"] > b50["ann_full"],
                      h1=o["ann_h1"] > b50["ann_h1"], h2=o["ann_h2"] > b50["ann_h2"])
            o["checks"] = ck; o["n_checks_pass"] = int(sum(ck.values()))
            o["eight_cls"] = "值得上 XQ 測" if all(ck.values()) else (
                "不值得" if (not ck["a2019"] and not ck["afull"]) else "有條件")
            # 大贏家
            bw_ = {}
            for c in BIG:
                g = tr[tr.code == c]
                bw_[c] = [dict(buy=r_.buy_date, sell=r_.sell_date, ret=float(r_.ret), pnl=float(r_.pnl)) for r_ in g.itertuples()]
                bw_[c] += [dict(buy=bd, sell=None, open=True, cost=cst, val=val) for cc_, bd, cst, val, i_ in x["open_pos"] if cc_ == c]
            o["big_winners"] = bw_
            cells["%s|%s" % (d, cell)] = o
            for r_ in tr.itertuples():
                trade_rows.append((d, cell, r_.code, r_.buy_date, r_.sell_date, round(r_.ret, 6), round(r_.pnl, 1),
                                   int(r_.hold), r_.why))
            print(d, cell, round(o["ann_2019"], 4), round(o["ann_full"], 4), round(o["ann_2024"], 4),
                  round(o["mdd_full"], 4), o["n_trades"], o["n_checks_pass"], o["eight_cls"], flush=True)
    out["main"] = cells
    pd.DataFrame(trade_rows, columns=["def", "cell", "code", "buy_date", "sell_date", "ret", "pnl", "hold", "why"]).to_csv(
        os.path.join(odir, "my_trades.csv.gz"), index=False, compression="gzip")

    # ---------------------------------------------------------------- 20 條路徑（＋min_frac 0.45／0.55）
    rows = []; pnav = {}
    for d in DEFS:
        for cell in CELLS:
            for mf in (0.5, 0.45, 0.55):
                for sh in SHIFTS:
                    for cp in CAPS:
                        x = main_res[(d, cell)] if (mf == 0.5 and sh == 0 and cp == 1_000_000.0) else sim(d, cell, sh, cp, min_frac=mf)
                        nav = x["nav"]
                        r_ = dict(defn=d, cell=cell, min_frac=mf, shift=sh, capital=cp, **nav_stats(nav),
                                  n_trades=int(len(x["trades"])), top5_share=top5_share(x["trades"], nav),
                                  **{"bear_" + k: v for k, v in bears(nav).items()})
                        rows.append(r_)
                        if mf == 0.5:
                            pnav[(d, cell, sh, cp)] = nav
        print("paths", d, round(time.time() - t_start), "rss", rss(), flush=True)
    pr = pd.DataFrame(rows)
    pr.drop(columns=["mdd_peak", "mdd_trough"]).to_csv(os.path.join(odir, "my_paths.csv.gz"), index=False,
                                                       compression="gzip", float_format="%.6f")
    p5 = pr[pr.min_frac == 0.5]

    def p20_of(g):
        return dict(ann_2019_median=float(g.ann_2019.median()), ann_2019_min=float(g.ann_2019.min()),
                    ann_2019_max=float(g.ann_2019.max()), n_beat_0050=int((g.ann_2019 > b50["ann_2019"]).sum()),
                    mdd_median=float(g.mdd_full.median()), ann_full_median=float(g.ann_full.median()),
                    ann_2024_median=float(g.ann_2024.median()), ann_1923_median=float(g.ann_1923.median()),
                    ann_pre19_median=float(g.ann_pre19.median()),
                    top5_share_median=float(g.top5_share.median()), n_trades_median=float(g.n_trades.median()),
                    bears_median={k: float(g["bear_" + k].median()) for k in BEARS})
    p20 = {"%s|%s" % (d, c): p20_of(g) for (d, c), g in p5.groupby(["defn", "cell"])}
    p20mf = {"%s|%s|%.2f" % (d, c, mf): dict(ann_2019_median=float(g.ann_2019.median()), mdd_median=float(g.mdd_full.median()))
             for (d, c, mf), g in pr[pr.min_frac != 0.5].groupby(["defn", "cell", "min_frac"])}
    PCOLS = ["ann_2019", "ann_full", "ann_1923", "ann_2024", "ann_pre19", "mdd_full", "top5_share"] + ["bear_" + k for k in BEARS]

    def paired(prx, a, b, cell):
        o = {}
        xa = prx[(prx.defn == a) & (prx.cell == cell)].set_index(["shift", "capital"])
        xb = prx[(prx.defn == b) & (prx.cell == cell)].set_index(["shift", "capital"]).loc[xa.index]
        for col in PCOLS:
            dd = (xa[col] - xb[col]).values
            o[col] = dict(n_pos=int((dd > 0).sum()), median=float(np.median(dd)), min=float(dd.min()), max=float(dd.max()))
        return o
    pair = {}; pair_mf = {}
    for cell in CELLS:
        for d in NEW:
            pair["%s-STD|%s" % (d, cell)] = paired(p5, d, "STD", cell)
            for mf in (0.45, 0.55):
                pq = pr[pr.min_frac == mf]
                pp = paired(pq, d, "STD", cell)["ann_2019"]
                pair_mf["%s-STD|%s|%.2f" % (d, cell, mf)] = pp

    # ---------------------------------------------------------------- 事前判定
    dec = {}

    def better(pc):
        n, md = pc["n_pos"], pc["median"]
        if n >= 15 and md >= 0.02:
            return "比標準好"
        if n <= 5 and md <= -0.02:
            return "比標準差"
        return "差不多"
    for cell in CELLS:
        meds = {d: p20["%s|%s" % (d, cell)]["ann_2019_median"] for d in DEFS}
        mdds = {d: p20["%s|%s" % (d, cell)]["mdd_median"] for d in DEFS}
        allp = p5[p5.cell == cell].ann_2019.values
        fam_med = float(np.median(list(meds.values())))
        rank_r = 1 + sum(1 for d in DEFS if meds[d] > meds["STD"])
        rank_d = 1 + sum(1 for d in DEFS if mdds[d] > mdds["STD"])
        p3 = meds["STD"] - fam_med
        o = dict(P1=dict(path_medians=meds, family_median=fam_med, family_min=float(min(meds.values())),
                         family_max=float(max(meds.values())), all140_median=float(np.median(allp)),
                         all140_p10=float(np.percentile(allp, 10)), n140=int(len(allp)),
                         main_1m={d: cells["%s|%s" % (d, cell)]["ann_2019"] for d in DEFS}),
                 P2=dict(rank_return=rank_r, rank_mdd=rank_d, mdd_medians=mdds,
                         order_return=sorted(DEFS, key=lambda d: -meds[d])),
                 P3=p3, optimism=("標準定義偏樂觀" if p3 >= 0.03 else "不必因定義打折"))
        o["vs_std"] = {}
        for d in NEW:
            pc = pair["%s-STD|%s" % (d, cell)]
            o["vs_std"][d] = dict(n_pos=pc["ann_2019"]["n_pos"], median=pc["ann_2019"]["median"],
                                  seg_1923=pc["ann_1923"]["median"], seg_2024=pc["ann_2024"]["median"],
                                  mdd_n_shallower=pc["mdd_full"]["n_pos"], mdd_median=pc["mdd_full"]["median"],
                                  cls=better(pc["ann_2019"]),
                                  cls_mf045=better(pair_mf["%s-STD|%s|0.45" % (d, cell)]),
                                  cls_mf055=better(pair_mf["%s-STD|%s|0.55" % (d, cell)]),
                                  mdd_cls=("能壓回撤" if (pc["mdd_full"]["n_pos"] >= 15 and pc["mdd_full"]["median"] >= 0.02) else "否"),
                                  bears={k: dict(n_shallower=pc["bear_" + k]["n_pos"], median=pc["bear_" + k]["median"],
                                                 cls=("較淺" if (pc["bear_" + k]["n_pos"] >= 15 and pc["bear_" + k]["median"] >= 0.01) else
                                                      ("較深" if (pc["bear_" + k]["n_pos"] <= 5 and pc["bear_" + k]["median"] <= -0.01) else "差不多")))
                                         for k in BEARS})
        dec[cell] = o
    rk = [dec[c]["P2"]["rank_return"] for c in CELLS]
    dec["std_is_best"] = "是" if all(r == 1 for r in rk) else ("前段" if all(r <= 3 for r in rk) else "不是")
    stab = {}
    for d in NEW:
        v = [dec[c]["vs_std"][d] for c in CELLS]
        nb = sum(1 for x in v if x["cls"] == "比標準好")
        if nb == 2:
            segs = all(x["seg_1923"] > 0 and x["seg_2024"] > 0 for x in v)
            mfs = all(x["cls_mf045"] == "比標準好" and x["cls_mf055"] == "比標準好" for x in v)
            stab[d] = "穩定地比較好" if (segs and mfs) else ("比標準好但只靠一段" if not segs else "比標準好但對最小買進門檻敏感")
        elif nb == 1:
            stab[d] = "只有一格較好"
        else:
            stab[d] = "否"
    dec["stable"] = stab
    dec["xq"] = ("寫成 20／21 號腳本修改建議：" + "、".join(d for d in NEW if stab[d] == "穩定地比較好")
                 if any(stab[d] == "穩定地比較好" for d in NEW) else "維持現行 SURGE 算法")
    dec["eight_checks"] = {k: (v["n_checks_pass"], v["eight_cls"]) for k, v in cells.items()}
    out["paths20"] = p20; out["paths20_minfrac"] = p20mf; out["paired"] = pair; out["paired_minfrac"] = pair_mf
    out["decisions"] = dec
    print("decisions", json.dumps(dec, ensure_ascii=False, default=jdef)[:4000], flush=True)

    # ================================================================ 事後：重疊率、集中度
    ph = {}
    ov = {}
    for d in NEW:
        bS, kS = plans[("STD", 0)]; bD, kD = plans[(d, 0)]
        lS = lists[("STD", 0)]; lD = lists[(d, 0)]
        com = sorted(set(bS) & set(bD))
        ib = sum(len(set(map(int, bS[s])) & set(map(int, bD[s]))) for s in com)
        nbS = sum(len(bS[s]) for s in com); nbD = sum(len(bD[s]) for s in com)
        it = sum(len(set(map(int, lS[s]["lma"][:20])) & set(map(int, lD[s]["lma"][:20]))) for s in com)
        ntS = sum(min(20, len(lS[s]["lma"])) for s in com)
        mon = [len(set(map(int, lS[s]["lma"][:20])) & set(map(int, lD[s]["lma"][:20]))) / max(1, min(20, len(lS[s]["lma"]))) for s in com]
        ov[d] = dict(n_months=len(com), buy_overlap_of_std=ib / nbS, buy_overlap_of_def=ib / nbD if nbD else None,
                     top20_overlap=it / ntS, top20_overlap_month_median=float(np.median(mon)),
                     n_buy_std=nbS, n_buy_def=nbD)
    ph["overlap_vs_STD_shift0"] = ov
    ph["top5_share"] = {k: dict(main=v["top5_share_of_gain"], paths_median=p20[k]["top5_share_median"]) for k, v in cells.items()}
    ph["mdd_dates"] = {k: (v["mdd_peak"], v["mdd_trough"]) for k, v in cells.items()}
    out["posthoc"] = ph

    # ================================================================ 審查 A：NOMU 結構性還是時期效果
    rA = {}
    for cell in CELLS:
        o = {}
        pc = pair["NOMU-STD|%s" % cell]
        o["pre2019_paths"] = dict(nomu_median=p20["NOMU|%s" % cell]["ann_pre19_median"],
                                  std_median=p20["STD|%s" % cell]["ann_pre19_median"],
                                  paired_n_pos=pc["ann_pre19"]["n_pos"], paired_median=pc["ann_pre19"]["median"],
                                  paired_min=pc["ann_pre19"]["min"], paired_max=pc["ann_pre19"]["max"])
        yr = {}
        for y in range(2015, 2027):
            dd = []
            for sh in SHIFTS:
                for cp in CAPS:
                    yn = L.yearly(P, pnav[("NOMU", cell, sh, cp)]); ys = L.yearly(P, pnav[("STD", cell, sh, cp)])
                    dd.append(yn[y] - ys[y])
            dd = np.array(dd)
            yr[y] = dict(main_diff=cells["NOMU|%s" % cell]["yearly"][y] - cells["STD|%s" % cell]["yearly"][y],
                         main_nomu=cells["NOMU|%s" % cell]["yearly"][y], main_std=cells["STD|%s" % cell]["yearly"][y],
                         paths_n_pos=int((dd > 0).sum()), paths_median=float(np.median(dd)))
        o["yearly"] = yr
        o["yearly_main_wins"] = int(sum(1 for y in yr if yr[y]["main_diff"] > 0))
        o["yearly_paths_majority_wins"] = int(sum(1 for y in yr if yr[y]["paths_n_pos"] >= 11))
        mt = {}
        for nm, a, b in (("from2019", "2019-01", None), ("2019_2023", "2019-01", "2023-12"), ("from2024", "2024-01", None),
                         ("2015H2_2018", "2015-08", "2018-12"), ("full", "2015-08", None)):
            mm = K.month_paired(P, pnav[("NOMU", cell, 0, 1_000_000.0)], pnav[("STD", cell, 0, 1_000_000.0)], a, b)
            ts = [K.month_paired(P, pnav[("NOMU", cell, sh, cp)], pnav[("STD", cell, sh, cp)], a, b)["t"]
                  for sh in SHIFTS for cp in CAPS]
            mm["paths_t_median"] = float(np.median(ts)); mm["paths_t_min"] = float(np.min(ts)); mm["paths_t_max"] = float(np.max(ts))
            mt[nm] = mm
        o["monthly_t"] = mt
        # 禁買雙方最賺 5 筆聯集、整段不准買 2408
        spS, topS = R.span_of(main_res[("STD", cell)]["trades"], 5)
        spN, topN = R.span_of(main_res[("NOMU", cell)]["trades"], 5)
        uni = {}
        for sp in (spS, spN):
            for c, v in sp.items():
                uni.setdefault(c, []).extend(v)
        blk = {"top5_union": uni, "no_2408": {"2408": [(0, T)]},
               "top5_union_plus_2408": dict(uni, **{"2408": uni.get("2408", []) + [(0, T)]})}
        o["top5_std"] = topS; o["top5_nomu"] = topN
        bres = {}
        for bn, sp in blk.items():
            dd19 = []; ddf = []; vals = {}
            for sh in SHIFTS:
                for cp in CAPS:
                    xn = sim("NOMU", cell, sh, cp, block_span=sp)["nav"]; xs = sim("STD", cell, sh, cp, block_span=sp)["nav"]
                    a_ = L.ann(P, xn, i19, e) - L.ann(P, xs, i19, e); dd19.append(a_)
                    ddf.append(L.ann(P, xn, t0, e) - L.ann(P, xs, t0, e))
                    if sh == 0 and cp == 1_000_000.0:
                        vals = dict(nomu_2019=L.ann(P, xn, i19, e), std_2019=L.ann(P, xs, i19, e), diff_2019=a_,
                                    nomu_full=L.ann(P, xn, t0, e), std_full=L.ann(P, xs, t0, e),
                                    diff_full=L.ann(P, xn, t0, e) - L.ann(P, xs, t0, e),
                                    nomu_mdd=L.maxdd(P, xn, t0, e)[0], std_mdd=L.maxdd(P, xs, t0, e)[0])
            dd19 = np.array(dd19); ddf = np.array(ddf)
            bres[bn] = dict(main=vals, paths_2019_n_pos=int((dd19 > 0).sum()), paths_2019_median=float(np.median(dd19)),
                            paths_full_median=float(np.median(ddf)))
        o["blocks"] = bres
        o["unblocked_main_diff_2019"] = cells["NOMU|%s" % cell]["ann_2019"] - cells["STD|%s" % cell]["ann_2019"]
        o["unblocked_main_diff_full"] = cells["NOMU|%s" % cell]["ann_full"] - cells["STD|%s" % cell]["ann_full"]
        rA[cell] = o
    # NOMU 交易中 2408 的貢獻
    rA["trades_2408"] = {cell: dict(nomu=cells["NOMU|%s" % cell]["big_winners"]["2408"], std=cells["STD|%s" % cell]["big_winners"]["2408"])
                         for cell in CELLS}
    out["reviewA"] = rA
    print("reviewA", json.dumps(rA, ensure_ascii=False, default=jdef)[:3000], flush=True)

    # ================================================================ 審查 B：NOMU 機制（μ÷σ、近 12 月營收、股價）
    mu, sd = aux["mu"], aux["sd"]
    with np.errstate(invalid="ignore", divide="ignore"):
        mus = mu / sd
    R12 = np.full_like(Rv, np.nan)
    cs_ = np.nancumsum(np.nan_to_num(Rv), axis=0)
    okc = pd.DataFrame(np.isfinite(Rv).astype(float)).rolling(12, min_periods=12).sum().values >= 12
    R12[11:] = cs_[11:] - np.vstack([np.zeros((1, Rv.shape[1])), cs_[:-12]])
    R12[~okc] = np.nan
    rB = {}

    def feats(lst_plans, lsts, which):
        mv = []; rv = []; pv = []; dv = []
        for s, b in lst_plans[0].items():
            m = lsts[s]["m"]
            ii = np.asarray(b if which == "buy" else lsts[s]["lma"][:10], int)
            if len(ii) == 0:
                continue
            mv.extend(mus[m, ii]); rv.extend(R12[m, ii]); pv.extend(P.C[s, ii]); dv.extend(aux["D_amt"][m, ii])
        mv, rv, pv = np.array(mv), np.array(rv), np.array(pv)
        return dict(n=len(mv), mu_over_sd_median=float(np.nanmedian(mv)), mu_over_sd_q25=float(np.nanpercentile(mv, 25)),
                    mu_over_sd_q75=float(np.nanpercentile(mv, 75)), share_mu_pos=float(np.nanmean(mv > 0)),
                    rev12m_median_kTWD=float(np.nanmedian(rv)), price_median=float(np.nanmedian(pv)),
                    D_median_kTWD=float(np.nanmedian(dv)))
    for which in ("buy", "lma10"):
        rB[which] = {d: feats(plans[(d, 0)], lists[(d, 0)], which) for d in ("STD", "NOMU")}
    thr = rB["buy"]["STD"]["mu_over_sd_median"]
    rB["threshold_mu_over_sd"] = thr
    with np.errstate(invalid="ignore"):
        Sr = np.where(mus <= thr, F["NOMU"], np.nan)
        Ssr = np.where(mus <= thr, F["STD"], np.nan)
    pl_r = {}; pl_sr = {}
    for sh in SHIFTS:
        lr = lists_of(Sr, sh, Scnt=F["NOMU"])
        pl_r[sh] = (lr, Q.cell_plans(lr, "C10", x=0.10))
        lr = lists_of(Ssr, sh, Scnt=F["STD"])
        pl_sr[sh] = (lr, Q.cell_plans(lr, "C10", x=0.10))
    rB["restricted_feats_buy"] = feats(pl_r[0][1], pl_r[0][0], "buy")
    rB["restricted_std_feats_buy"] = feats(pl_sr[0][1], pl_sr[0][0], "buy")
    res_cells = {}
    for cell in CELLS:
        ddc = []; valc = []
        for sh in SHIFTS:
            for cp in CAPS:
                xr = sim("NOMU_R", cell, sh, cp, pl=pl_r[sh][1])["nav"]
                xs = sim("STD_R", cell, sh, cp, pl=pl_sr[sh][1])["nav"]
                valc.append(L.ann(P, xs, i19, e))
                ddc.append(L.ann(P, xr, i19, e) - L.ann(P, xs, i19, e))
        ddc = np.array(ddc)
        ctrl = dict(std_r_paths_median=float(np.median(valc)), nomu_r_minus_std_r_n_pos=int((ddc > 0).sum()),
                    nomu_r_minus_std_r_median=float(np.median(ddc)))
        dd = []; ddN = []; vals = []
        for sh in SHIFTS:
            for cp in CAPS:
                xr = sim("NOMU_R", cell, sh, cp, pl=pl_r[sh][1])["nav"]
                a_ = L.ann(P, xr, i19, e)
                vals.append(a_)
                dd.append(a_ - L.ann(P, pnav[("STD", cell, sh, cp)], i19, e))
                ddN.append(a_ - L.ann(P, pnav[("NOMU", cell, sh, cp)], i19, e))
                if sh == 0 and cp == 1_000_000.0:
                    main_r = dict(ann_2019=a_, ann_full=L.ann(P, xr, t0, e), mdd=L.maxdd(P, xr, t0, e)[0])
        pcN = pair["NOMU-STD|%s" % cell]["ann_2019"]["median"]
        dd = np.array(dd)
        res_cells[cell] = dict(main=main_r, paths_median=float(np.median(vals)), vs_std_n_pos=int((dd > 0).sum()),
                               vs_std_median=float(np.median(dd)), vs_nomu_median=float(np.median(ddN)),
                               nomu_vs_std_median=pcN, share_of_edge_kept=float(np.median(dd) / pcN) if pcN != 0 else None,
                               control_std_restricted_same_subset=ctrl,
                               within_subset_edge_share=float(ctrl["nomu_r_minus_std_r_median"] / pcN) if pcN != 0 else None)
    rB["restricted_runs"] = res_cells
    out["reviewB"] = rB
    print("reviewB", json.dumps(rB, ensure_ascii=False, default=jdef), flush=True)

    # ================================================================ 審查 C：NOMU L_MA 第 10、20 名分數
    rC = {}
    for d in ("NOMU", "STD"):
        v10 = []; v20 = []; vbmin = []
        for s, q in lists[(d, 0)].items():
            a = q["lma"]; m = q["m"]
            if len(a) >= 10:
                v10.append(F[d][m, a[9]])
            if len(a) >= 20:
                v20.append(F[d][m, a[19]])
            b = plans[(d, 0)][0][s]
            if len(b):
                vbmin.append(F[d][m, b[-1]])
        v10, v20 = np.array(v10), np.array(v20)

        def qs(v):
            return dict(n=len(v), q25=float(np.percentile(v, 25)), median=float(np.median(v)), q75=float(np.percentile(v, 75)),
                        min=float(v.min()), max=float(v.max()))
        rC[d] = dict(rank10=qs(v10), rank20=qs(v20), buy_list_lowest=qs(np.array(vbmin)),
                     share_rank10_ge_3p2=float((v10 >= 3.2).mean()), share_rank20_ge_2p5=float((v20 >= 2.5).mean()))
        # 年度中位數（看門檻是否隨時間漂移）
        yrs = {}
        for s, q in lists[(d, 0)].items():
            a = q["lma"]
            if len(a) >= 20:
                yrs.setdefault(P.dstr[s][:4], []).append((F[d][q["m"], a[9]], F[d][q["m"], a[19]]))
        rC[d]["by_year_median"] = {y: (float(np.median([x[0] for x in v])), float(np.median([x[1] for x in v]))) for y, v in sorted(yrs.items())}
    out["reviewC"] = rC
    print("reviewC", json.dumps(rC, ensure_ascii=False, default=jdef), flush=True)

    out["resources"] = dict(runtime_s=round(time.time() - t_start, 1), maxrss_mb=rss())
    js = json.dumps(out, ensure_ascii=False, indent=1, default=jdef)
    fp = os.path.join(odir, "my_numbers.json")
    open(fp, "w").write(js)
    h = hashlib.sha256(js.encode("utf-8")).hexdigest()
    open(os.path.join(odir, "my_numbers.sha256"), "w").write(h + "  my_numbers.json\n")
    print("done", h, out["resources"], flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
