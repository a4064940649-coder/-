# -*- coding: utf-8 -*-
"""S14 獨立複製：驗收、D／DA／C／CA／C0（＋E0）主跑＋八項、20 條路徑、逐條配對、四段空頭、相關與重疊、事前判定、
審查 A（延長鏈 vs 非延長事件、延長的 PIT）、B（預收款券：單筆 ≥ 30 張改買 29 張）、C（2019 名單與 2019 年 D 衛星換 0050），然後凍結。
用法：python3 -I -B rep_s14_run.py <scratch_dir>
需先跑 s1 的 rep_s1_panel.py <scratch_dir> 與 s3 的 rep_s3_names.py <scratch_dir>/innov_names.json。
輸出（本資料夾）：my_numbers.json、my_numbers.sha256、my_trades.csv.gz、my_paths.csv.gz
凍結前不讀研究員的 FINDINGS 第 2 節以外、results*.json、navs.csv.gz、trades.csv.gz。
"""
import sys, os, json, hashlib, resource, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import rep_s14_lib as K
S, Q, R, L = K.S, K.Q, K.R, K.L
ST, X2 = K.ST, K.X2

CAPS = [600_000.0, 800_000.0, 1_000_000.0, 1_500_000.0, 2_000_000.0]
SHIFTS = [0, 1, 2, 3]
KNOWN_E0 = (0.4814, 0.3409)
BEARS = {"2018": ("2018-01-01", "2018-12-31"), "2020-03": ("2020-01-01", "2020-06-30"),
         "2022": ("2022-01-01", "2022-12-31"), "2024-07~2025-04": ("2024-07-01", "2025-04-30")}
SINGLE = ["E0", "D", "DA"]
COMBO = {"C": ("E0", "D"), "CA": ("E0", "DA"), "C0": ("E0", "0050")}
ALL = ["E0", "D", "DA", "C", "CA", "C0"]
LISTMAX = 9999
CAP29 = 29_000


def rss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024


def fl(x):
    if x is None:
        return None
    x = float(x)
    return x if np.isfinite(x) else None


def main(sc, odir=HERE):
    t_start = time.time()
    P = R.build(sc)
    K.attach_e50_orig(P, sc)
    excl = K.S.innov_mask_fn(P, sc)
    Rv, mi0 = L.load_rev(P)
    S_, _ = L.surge(Rv, mi0, null2013=True)
    e = P.T - 1
    i19 = L.didx(P, "2018-12-28"); i24 = L.didx(P, "2023-12-29"); ih = L.didx(P, "2021-02-19")
    da = np.array(P.dstr)

    def win(a, b):
        return int(np.searchsorted(da, a, side="left")), int(np.searchsorted(da, b, side="right")) - 1
    bw = {k: win(a, b) for k, (a, b) in BEARS.items()}
    out = dict(meta=dict(T=P.T, N=P.N, first=P.dstr[0], last=P.dstr[-1], t0=P.dstr[P.t0], i19=P.dstr[i19],
                         i24=P.dstr[i24], ih=P.dstr[ih],
                         bear_windows={k: (P.dstr[a], P.dstr[b]) for k, (a, b) in bw.items()},
                         note="面板為複製員自建（N＝2130，研究員 2142；差的是從未進股池的代號）"))

    # ---------------- 名單（SURGE）與事件鏈
    plansE = {sh: Q.cell_plans(Q.make_lists(P, S_, mi0, shift=sh, excl=excl), "C10", x=0.10) for sh in SHIFTS}
    disp, sha = K.load_disp()
    cs = K.adv20(P)
    chains = {"D": K.build_chains(P, disp, (1,)), "DA": K.build_chains(P, disp, (1, 2))}
    plansD = {}
    infos = {}
    for c in ("D", "DA"):
        for sh in SHIFTS:
            b, mk, info = K.disp_plans(P, chains[c], sh, excl, cs)
            plansD[(c, sh)] = (b, mk)
            infos[(c, sh)] = info
    print("plans", "rss", rss(), round(time.time() - t_start, 1), flush=True)

    # ---------------- 驗收 A0（rep_s4 原始 0050 開盤 → 修正後）
    acc = {}
    acc["P0"] = dict(T=P.T, N=P.N, last=P.dstr[-1], note="研究員 N＝2142；複製員面板 2130（已知差異，不影響股池）")
    b0, k0 = plansE[0]
    a9 = K.K9.simulate_s9(P, b0, k0, sat_w=0.7, sizing="W10", **ST)
    a14 = K.simulate_s14(P, b0, k0, sat_w=0.7, sizing="W10", **ST)
    ta = a9["trades"].sort_values(["ts", "code"])[["code", "tb", "ts", "sh"]].values
    tc = a14["trades"].sort_values(["ts", "code"])[["code", "tb", "ts", "sh"]].values
    y19, yf = L.ann(P, a14["nav"], i19, e), L.ann(P, a14["nav"], P.t0, e)
    acc["A0_orig_e50"] = dict(ann_2019=y19, ann_full=yf, mdd_full=L.maxdd(P, a14["nav"], P.t0, e)[0],
                              n_trades=int(len(a14["trades"])),
                              nav_identical_rep_s9=bool(np.nanmax(np.abs(a9["nav"] - a14["nav"])) < 1e-6),
                              trades_identical_rep_s9=bool(len(ta) == len(tc) and (ta == tc).all()),
                              known=KNOWN_E0, ok=bool(abs(y19 - KNOWN_E0[0]) < 0.00006 and abs(yf - KNOWN_E0[1]) < 0.00006))
    print("A0 orig", acc["A0_orig_e50"], flush=True)
    assert acc["A0_orig_e50"]["ok"]
    # D 主跑（修正前）留著比較
    bD, mD = plansD[("D", 0)]
    d_orig = K.simulate_s14(P, bD, {}, sat_w=0.7, sizing="W10", eod=K.make_eod_disp(mD), listmax=LISTMAX, **ST)
    nav_e0_orig = a14["nav"]; nav_d_orig = d_orig["nav"]
    fixed = K.attach_e50_fixed(P, sc)
    a14f = K.simulate_s14(P, b0, k0, sat_w=0.7, sizing="W10", **ST)
    acc["A0_fixed_e50"] = dict(ann_2019=L.ann(P, a14f["nav"], i19, e), ann_full=L.ann(P, a14f["nav"], P.t0, e),
                               mdd_full=L.maxdd(P, a14f["nav"], P.t0, e)[0], n_trades=int(len(a14f["trades"])),
                               fixed_days=fixed,
                               max_abs_nav_diff_vs_orig=float(np.nanmax(np.abs(a14f["nav"] - nav_e0_orig))),
                               note="0050 停牌日（2025-06-11～17）還原開盤改為前一日還原收盤、原始開盤改為前一日原始收盤")
    print("A0 fixed", {k: v for k, v in acc["A0_fixed_e50"].items() if k != "fixed_days"}, flush=True)

    e50 = P.e50.values.astype(float)

    def e50nav(capital):
        return capital * e50 / e50[P.t0]

    def bears(nav):
        return {k: L.maxdd(P, nav, a, b)[0] for k, (a, b) in bw.items()}

    def nav_stats(nav, ntr=None):
        md = L.maxdd(P, nav, P.t0, e)
        return dict(ann_2019=L.ann(P, nav, i19, e), ann_full=L.ann(P, nav, P.t0, e), ann_2024=L.ann(P, nav, i24, e),
                    ann_1923=L.ann(P, nav, i19, i24), mdd_full=md[0], mdd_peak=md[1], mdd_trough=md[2],
                    ann_h1=L.ann(P, nav, P.t0, ih), ann_h2=L.ann(P, nav, ih, e), n_trades=ntr)

    b50 = dict(nav_stats(e50), bears=bears(e50), yearly=L.yearly(P, e50))
    out["bench_0050"] = b50
    print("0050", {k: round(v, 4) for k, v in b50.items() if isinstance(v, float)}, flush=True)

    # ---------------- 引擎包裝
    def run(cell, sh=0, capital=1_000_000.0, cost=None, block_span=None, rec_pos=False, max_sh=None, listmax=LISTMAX,
            plans_override=None):
        if cell == "E0":
            b, k = plansE[sh]
            return K.simulate_s14(P, b, k, sat_w=0.7, sizing="W10", capital=capital, block_span=block_span,
                                  rec_pos=rec_pos, **(cost or ST))
        b, mk = plans_override if plans_override is not None else plansD[(cell, sh)]
        return K.simulate_s14(P, b, {}, sat_w=0.7, sizing="W10", capital=capital, block_span=block_span,
                              rec_pos=rec_pos, eod=K.make_eod_disp(mk), listmax=listmax, max_sh=max_sh,
                              **(cost or ST))

    # ---------------- 主跑
    res = {c: run(c) for c in ("D", "DA")}
    res["E0"] = a14f
    half = {c: run(c, capital=500_000.0, rec_pos=(c in ("E0", "D"))) for c in ("E0", "D", "DA")}
    navs = {c: res[c]["nav"] for c in SINGLE}
    navs["0050"] = e50nav(1_000_000.0)
    for cc, (a, b) in COMBO.items():
        navs[cc] = half[a]["nav"] + (half[b]["nav"] if b != "0050" else e50nav(500_000.0))
    acc["A2"] = dict(max_abs_diff_C_vs_halves=float(np.nanmax(np.abs(navs["C"] - (half["E0"]["nav"] + half["D"]["nav"])))),
                     C_start=float(navs["C"][P.t0]), ok=True)

    def trades_of(cc):
        if cc in SINGLE:
            return res[cc]["trades"]
        a, b = COMBO[cc]
        t1 = half[a]["trades"].assign(half=a)
        if b == "0050":
            return t1
        return pd.concat([t1, half[b]["trades"].assign(half=b)], ignore_index=True)

    def open_pos(cc):
        if cc in SINGLE:
            return res[cc]["open_pos"]
        a, b = COMBO[cc]
        return half[a]["open_pos"] + (half[b]["open_pos"] if b != "0050" else [])

    def run_cell(cc, sh=0, capital=1_000_000.0, cost=None, block_span=None):
        if cc in SINGLE:
            x = run(cc, sh, capital, cost, block_span)
            return x["nav"], len(x["trades"])
        a, b = COMBO[cc]
        xa = run(a, sh, capital / 2, cost, block_span)
        if b == "0050":
            return xa["nav"] + e50nav(capital / 2), len(xa["trades"])
        xb = run(b, sh, capital / 2, cost, block_span)
        return xa["nav"] + xb["nav"], len(xa["trades"]) + len(xb["trades"])

    # ---------------- A1：訊號與交易對鏈
    a1 = dict(n_sig=0, sig_in_chain_and_univ=True, ann_before_entry=True, trades_checked=0, trade_entry_ok=True,
              exit_on_time=0, exit_delayed=0, exit_delist=0, exit_reduce=0, exit_other=0, open_end=0)
    for c in ("D", "DA"):
        ch = chains[c]
        k0set = {(int(r.k0), int(r.i)): r for r in ch.itertuples(index=False)}
        a1["ann_before_entry"] &= bool(ch.ann_le_k0.all())
        for sh in SHIFTS:
            b, mk = plansD[(c, sh)]
            for s, ii in b.items():
                u = R.universe(P, s) & ~excl(s)
                for i in ii:
                    a1["n_sig"] += 1
                    if (s - sh, int(i)) not in k0set or not u[i]:
                        a1["sig_in_chain_and_univ"] = False
    for c in ("D", "DA"):
        for sh in SHIFTS:
            for cap in CAPS:
                x = run(c, sh, cap)
                b, mk = plansD[(c, sh)]
                for r_ in x["trades"].itertuples():
                    a1["trades_checked"] += 1
                    s = int(r_.tb) - 1
                    if (s, int(r_.i)) not in mk:
                        a1["trade_entry_ok"] = False; continue
                    m = mk[(s, int(r_.i))]
                    if r_.why == "end":
                        if int(r_.mk) != m:
                            a1["exit_other"] += 1
                        elif int(r_.ts) == m + 1:
                            a1["exit_on_time"] += 1
                        else:
                            a1["exit_delayed"] += 1
                    elif r_.why == "delist":
                        a1["exit_delist"] += 1
                    elif r_.why == "reduce":
                        a1["exit_reduce"] += 1
                    else:
                        a1["exit_other"] += 1
                a1["open_end"] += len(x["open_pos"])
    td = res["D"]["trades"]
    a1["D_main_delayed"] = int(sum(1 for r_ in td.itertuples() if r_.why == "end" and int(r_.ts) != int(r_.mk) + 1))
    acc["A1"] = a1
    out["acceptance"] = acc
    print("A1", a1, flush=True)

    # ---------------- 主跑統計＋八項
    cells = {}
    trade_rows = []
    for cc in ALL:
        nav = navs[cc]; tr = trades_of(cc)
        o = nav_stats(nav, int(len(tr)))
        o["bears"] = bears(nav)
        o["yearly"] = L.yearly(P, nav)
        x2nav, _ = run_cell(cc, cost=X2)
        sp, top = R.span_of(tr, 5)
        blnav, _ = run_cell(cc, block_span=sp)
        o["x2"] = dict(ann_2019=L.ann(P, x2nav, i19, e), ann_full=L.ann(P, x2nav, P.t0, e))
        o["block5"] = dict(ann_2019=L.ann(P, blnav, i19, e), ann_full=L.ann(P, blnav, P.t0, e))
        o["block5_top"] = top
        inc = float(nav[e] - nav[P.t0])
        o["top5_share_of_gain"] = float(sum(t["pnl"] for t in top) / inc)
        ops = open_pos(cc)
        o["n_open_end"] = len(ops)
        o["final_nav"] = float(nav[e])
        if len(tr):
            o["trade_stats"] = dict(exp=float(tr.ret.mean()), win=float((tr.ret > 0).mean()),
                                    hold_mean=float(tr.hold.mean()), hold_median=float(tr.hold.median()),
                                    n_delist=int(tr.delist.sum()))
        ck = dict(a2019=o["ann_2019"] > b50["ann_2019"], afull=o["ann_full"] > b50["ann_full"],
                  x2_2019=o["x2"]["ann_2019"] > b50["ann_2019"], x2_full=o["x2"]["ann_full"] > b50["ann_full"],
                  bl_2019=o["block5"]["ann_2019"] > b50["ann_2019"], bl_full=o["block5"]["ann_full"] > b50["ann_full"],
                  h1=o["ann_h1"] > b50["ann_h1"], h2=o["ann_h2"] > b50["ann_h2"])
        o["checks"] = ck
        o["n_checks_pass"] = int(sum(ck.values()))
        o["eight_cls"] = "值得上 XQ 測" if all(ck.values()) else (
            "不值得" if (not ck["a2019"] and not ck["afull"]) else "有條件")
        cells[cc] = o
        if cc in ("E0", "D", "DA", "C"):
            for r_ in tr.itertuples():
                trade_rows.append((cc, getattr(r_, "half", cc), r_.code, r_.buy_date, r_.sell_date, round(r_.ret, 6),
                                   round(r_.pnl, 1), int(r_.hold), int(r_.sh), str(getattr(r_, "why", "") or "")))
        print(cc, round(o["ann_2019"], 4), round(o["ann_full"], 4), round(o["ann_1923"], 4), round(o["ann_2024"], 4),
              round(o["mdd_full"], 4), o["n_trades"], o["n_checks_pass"], o["eight_cls"], "rss", rss(),
              round(time.time() - t_start, 1), flush=True)
    out["cells"] = cells
    pd.DataFrame(trade_rows, columns=["cell", "half", "code", "buy_date", "sell_date", "ret", "pnl", "hold", "sh",
                                      "why"]).to_csv(os.path.join(odir, "my_trades.csv.gz"), index=False,
                                                     compression="gzip")
    out["e50_fix_effect_main"] = dict(
        E0=dict(orig=nav_stats(nav_e0_orig), fixed=nav_stats(navs["E0"])),
        D=dict(orig=nav_stats(nav_d_orig, int(len(d_orig["trades"]))), fixed=nav_stats(navs["D"], int(len(res["D"]["trades"])))))

    # ---------------- 20 條路徑
    pnav = {}

    def single_nav(cell, sh, cap, **kw):
        key = (cell, sh, cap, tuple(sorted(kw.items())))
        if key not in pnav:
            x = run(cell, sh, cap, **kw)
            pnav[key] = (x["nav"], len(x["trades"]), x)
        return pnav[key]

    rows = []
    path_nav = {}
    for cc in ALL:
        for sh in SHIFTS:
            for cp in CAPS:
                if cc in SINGLE:
                    nav, n, _ = single_nav(cc, sh, cp)
                else:
                    a, b = COMBO[cc]
                    na, n1, _ = single_nav(a, sh, cp / 2)
                    if b == "0050":
                        nav, n = na + e50nav(cp / 2), n1
                    else:
                        nb, n2, _ = single_nav(b, sh, cp / 2)
                        nav, n = na + nb, n1 + n2
                path_nav[(cc, sh, cp)] = nav
                st = nav_stats(nav, n)
                rows.append(dict(cell=cc, shift=sh, capital=cp, **st, **{"bear_" + k: v for k, v in bears(nav).items()}))
        print("paths", cc, "rss", rss(), round(time.time() - t_start, 1), flush=True)
    pr = pd.DataFrame(rows)
    pr.to_csv(os.path.join(odir, "my_paths.csv.gz"), index=False, float_format="%.6f", compression="gzip")
    p20 = {}
    for cc, g in pr.groupby("cell"):
        p20[cc] = dict(ann_2019_median=float(g.ann_2019.median()), ann_2019_min=float(g.ann_2019.min()),
                       ann_2019_max=float(g.ann_2019.max()), n_beat_0050=int((g.ann_2019 > b50["ann_2019"]).sum()),
                       mdd_median=float(g.mdd_full.median()), mdd_deepest=float(g.mdd_full.min()),
                       ann_full_median=float(g.ann_full.median()), ann_2024_median=float(g.ann_2024.median()),
                       ann_1923_median=float(g.ann_1923.median()), n_trades_median=float(g.n_trades.median()),
                       bears_median={k: float(g["bear_" + k].median()) for k in BEARS})
    out["paths20"] = p20

    cols = ["ann_2019", "ann_full", "ann_2024", "ann_1923", "mdd_full"] + ["bear_" + k for k in BEARS]

    def paired_df(prx, a, b):
        x = prx[prx.cell == a].set_index(["shift", "capital"])
        if b == "0050":
            ref = dict(ann_2019=b50["ann_2019"], ann_full=b50["ann_full"], ann_2024=b50["ann_2024"],
                       ann_1923=b50["ann_1923"], mdd_full=b50["mdd_full"],
                       **{"bear_" + k: b50["bears"][k] for k in BEARS})
            return {c: (x[c] - ref[c]).values for c in cols}
        y = prx[prx.cell == b].set_index(["shift", "capital"]).loc[x.index]
        return {c: (x[c] - y[c]).values for c in cols}

    def summ(dd):
        return {c: dict(n_pos=int((v > 0).sum()), median=float(np.median(v)), min=float(v.min()), max=float(v.max()))
                for c, v in dd.items()}
    pair = {}
    for a, b in (("C", "C0"), ("D", "0050"), ("C", "E0"), ("CA", "C0"), ("DA", "0050"), ("DA", "D"), ("CA", "C"),
                 ("C0", "E0")):
        pair["%s-%s" % (a, b)] = summ(paired_df(pr, a, b))
    out["paired"] = pair

    # ---------------- 相關與重疊
    m50 = L.monthly(P, e50, P.t0)

    def corr(na, nb, start=None):
        a = L.monthly(P, na, P.t0) - m50; b = L.monthly(P, nb, P.t0) - m50
        df = pd.concat([a, b], axis=1).dropna()
        if start:
            df = df[df.index >= pd.Period(start, "M")]
        return float(df.iloc[:, 0].corr(df.iloc[:, 1])), int(len(df))
    c1 = corr(navs["D"], navs["E0"]); c2 = corr(navs["D"], navs["E0"], "2019-01")
    pc = [corr(path_nav[("D", sh, cp)], path_nav[("E0", sh, cp)])[0] for sh in SHIFTS for cp in CAPS]
    cA = corr(navs["DA"], navs["E0"])
    sE = half["E0"]["snap"]; sD = half["D"]["snap"]
    ov = []; ovn = []; ndays_d = 0
    for t in range(P.t0 + 1, P.T):
        dh = set(sD.get(t, {})); eh = set(sE.get(t, {}))
        if dh:
            ndays_d += 1
            ov.append(len(dh & eh) / len(dh)); ovn.append(len(dh & eh))
    npos_d = res["D"]["npos"][P.t0 + 1:]
    out["corr_overlap"] = dict(
        D_E0_excess_full=c1[0], n_months=c1[1], D_E0_excess_2019=c2[0], n_months_2019=c2[1],
        paths20_median=float(np.median(pc)), paths20_min=float(np.min(pc)), paths20_max=float(np.max(pc)),
        DA_E0_excess_full=cA[0],
        C_overlap_mean_share_of_D_half=float(np.mean(ov)), C_overlap_days_any=int(np.sum(np.array(ovn) > 0)),
        C_overlap_days_D_has_pos=ndays_d, C_overlap_mean_names=float(np.mean(ovn)),
        D_main_mean_npos=float(npos_d.mean()), D_main_share_days_empty=float((npos_d == 0).mean()),
        D_main_mean_stock_weight=float(res["D"]["stkw"][P.t0 + 1:].mean()))
    print("corr/overlap", out["corr_overlap"], flush=True)

    # ---------------- 事前判定
    def d_rule(cc):
        pq = p20[cc]
        if pq["ann_2019_median"] > b50["ann_2019"] and pq["n_beat_0050"] >= 15 and cells[cc]["ann_full"] > b50["ann_full"]:
            return "贏 0050"
        if pq["ann_2019_median"] < b50["ann_2019"] and pq["n_beat_0050"] <= 5:
            return "輸 0050"
        return "不確定"

    def c_rule(key):
        pc_ = pair[key]
        n = pc_["ann_2019"]["n_pos"]
        m1 = pc_["ann_1923"]["median"]; m2 = pc_["ann_2024"]["median"]
        if n >= 15:
            k = (m1 > 0) + (m2 > 0)
            v = "有補充價值" if k == 2 else ("有補充價值，但只靠一段" if k == 1 else "通過 15／20，但兩段中位數都不正")
        elif n <= 5:
            v = "沒有補充價值"
        else:
            v = "不確定"
        return dict(n_pos_2019=n, median_2019=pc_["ann_2019"]["median"], median_1923=m1, median_2024=m2, cls=v)

    def bear_rule(key):
        pc_ = pair[key]
        o = {}
        for k in BEARS:
            x = pc_["bear_" + k]
            o[k] = dict(n_shallower=x["n_pos"], median=x["median"],
                        cls="較抗跌" if (x["n_pos"] >= 15 and x["median"] >= 0.005) else (
                            "較深" if x["n_pos"] <= 5 else "差不多"))
        return o
    dec = dict(D_vs_0050=d_rule("D"), DA_vs_0050=d_rule("DA"), C_supplement=c_rule("C-C0"),
               CA_supplement=c_rule("CA-C0"), bears_C_vs_C0=bear_rule("C-C0"), bears_D_vs_0050=bear_rule("D-0050"),
               eight_checks={k: cells[k]["eight_cls"] for k in cells})
    dec["xq"] = "可以加（紙上追蹤＋XQ 測）" if dec["C_supplement"]["cls"] == "有補充價值" else "不建議加進現行策略"
    out["decisions"] = dec
    print("decisions", json.dumps(dec, ensure_ascii=False), flush=True)

    # ---------------- listmax 敏感度（買進名單只看前 10 名）
    x10 = run("D", listmax=None)
    out["sens_listmax10_D_main"] = dict(nav_stats(x10["nav"], int(len(x10["trades"]))),
                                        rows_gt10=int(sum(len(v) > 10 for v in plansD[("D", 0)][0].values())))

    # ---------------- 審查 A：延長鏈 vs 非延長事件
    revA = {}
    chD = chains["D"]
    chm = {(int(r.k0), int(r.i)): r for r in chD.itertuples(index=False)}

    def tr_ext(tr, sh=0):
        rows_ = []
        for r_ in tr.itertuples():
            c_ = chm[(int(r_.tb) - 1 - sh, int(r_.i))]
            r50 = P.e50o[int(r_.ts)] / P.e50o[int(r_.tb)] - 1.0
            rows_.append(dict(code=r_.code, tb=int(r_.tb), ts=int(r_.ts), ret=float(r_.ret), exc=float(r_.ret) - r50,
                              pnl=float(r_.pnl), ext=bool(c_.extended), hold=int(r_.hold), why=r_.why))
        return pd.DataFrame(rows_)

    def grp(df, col="exc"):
        o = {}
        for nm, g in (("extended", df[df.ext]), ("not_extended", df[~df.ext]), ("all", df)):
            o[nm] = dict(n=int(len(g)), mean=fl(g[col].mean()), median=fl(g[col].median()), t=K.tstat(g[col].values),
                         pnl_sum=fl(g.pnl.sum()) if "pnl" in g else None,
                         hold_mean=fl(g.hold.mean()) if "hold" in g else None)
        return o
    tD = tr_ext(res["D"]["trades"])
    revA["D_main_trades"] = grp(tD)
    revA["D_main_trades_pnl_share_extended"] = fl(tD[tD.ext].pnl.sum() / tD.pnl.sum())
    revA["D_main_trades_by_period"] = {}
    for nm, (a0, a1_) in (("2015_2018", (P.t0, i19)), ("2019_2023", (i19 + 1, i24)), ("2024_", (i24 + 1, e))):
        revA["D_main_trades_by_period"][nm] = grp(tD[(tD.tb >= a0) & (tD.tb <= a1_)])
    # 事件層（位移 0、所有可買事件不論名額；起日開盤漲停或沒開盤不算）
    AO, Gf = K.adj_open_matrix(P)
    ev = []
    for r in chD.itertuples(index=False):
        s = int(r.k0)
        if s < P.t0 or s >= P.T - 1:
            continue
        u = R.universe(P, s) & ~excl(s)
        if not u[r.i]:
            continue
        a = s + 1; i = int(r.i)
        o = P.O[a, i]
        if not (o == o) or o >= S.limit_up(P.ref[a, i]) - 1e-9:
            continue
        if r.L is None or pd.isna(r.L) or int(r.L) + 1 > e:
            continue
        x = int(r.L) + 1
        ox = P.O[x, i] if P.O[x, i] == P.O[x, i] else P.Cf[x, i]
        g = AO[x, i] / AO[a, i] - 1.0
        net = K.strict_net(AO[a, i], AO[x, i], o, ox)
        r50 = P.e50o[x] / P.e50o[a] - 1.0
        ev.append(dict(code=r.code, a=a, x=x, ext=bool(r.extended), gross=g, net=net, exc_gross=g - r50,
                       exc=net - r50, hold=x - a, n_extend=int(r.n_extend), year=P.dstr[a][:4]))
    evd = pd.DataFrame(ev)
    revA["event_level_net_excess"] = grp(evd, "exc")
    revA["event_level_gross_excess"] = grp(evd, "exc_gross")
    revA["event_level_by_period"] = {nm: grp(evd[(evd.a > a0) & (evd.a <= a1_)], "exc") for nm, (a0, a1_) in
                                     (("2015_2018", (P.t0, i19)), ("2019_2023", (i19, i24)), ("2024_", (i24, e)))}
    # 延長鏈：只算到原本的最後一天（不延長）時的事件層超額
    orig_end = []
    for r in chD[chD.extended].itertuples(index=False):
        s = int(r.k0)
        if s < P.t0:
            continue
        u = R.universe(P, s) & ~excl(s)
        if not u[r.i]:
            continue
        a = s + 1; i = int(r.i); o = P.O[a, i]
        if not (o == o) or o >= S.limit_up(P.ref[a, i]) - 1e-9:
            continue
        L0 = int(np.searchsorted(da, r.end0, side="right")) - 1
        if r.L is None or pd.isna(r.L) or int(r.L) + 1 > e:
            continue
        x0 = L0 + 1; x1 = int(r.L) + 1
        ox0 = P.O[x0, i] if P.O[x0, i] == P.O[x0, i] else P.Cf[x0, i]
        ox1 = P.O[x1, i] if P.O[x1, i] == P.O[x1, i] else P.Cf[x1, i]
        n0 = K.strict_net(AO[a, i], AO[x0, i], o, ox0) - (P.e50o[x0] / P.e50o[a] - 1.0)
        n1 = K.strict_net(AO[a, i], AO[x1, i], o, ox1) - (P.e50o[x1] / P.e50o[a] - 1.0)
        tail = AO[x1, i] / AO[x0, i] - 1.0 - (P.e50o[x1] / P.e50o[x0] - 1.0)
        orig_end.append(dict(n0=n0, n1=n1, tail=tail))
    oe = pd.DataFrame(orig_end)
    revA["extended_chain_split"] = dict(
        n=int(len(oe)), net_exc_to_original_end_mean=fl(oe.n0.mean()), t0=K.tstat(oe.n0.values),
        net_exc_full_chain_mean=fl(oe.n1.mean()), t1=K.tstat(oe.n1.values),
        gross_exc_extension_tail_mean=fl(oe["tail"].mean()), t_tail=K.tstat(oe["tail"].values),
        note="延長鏈拆成「起日～原最後一天＋1 開盤」與「原最後一天＋1 開盤～延長後最後一天＋1 開盤」兩段；"
             "前段可視為『事前就能知道的第一次處置部分』，後段才是延長帶來的部分")
    # PIT 檢查
    lagx = []
    nonpit = []
    for r in chD.itertuples(index=False):
        for (ann, ldate, newl, kind, nth) in r.merged:
            if ldate is None:
                continue
            ia = int(np.searchsorted(da, ann, side="left")); il = int(np.searchsorted(da, ldate, side="left"))
            lagx.append(il - ia)
    # 公告日在原最後一天之後（隔一個交易日內）、起日緊接著的同檔記錄＝若放寬就會被誤併的
    dd = disp[disp.code.isin(P.cidx)]
    for r in chD[chD.start0 >= "2015-07-01"].itertuples(index=False):
        if r.L is None or pd.isna(r.L):
            continue
        Ld = P.dstr[int(r.L)]
        nxt = P.dstr[min(int(r.L) + 2, e)]
        g = dd[(dd.code == r.code) & (dd.announce > Ld) & (dd.announce <= nxt)]
        for x in g.itertuples(index=False):
            nonpit.append(dict(code=r.code, L=Ld, announce=x.announce, start=x.start, kind=x.kind, nth=int(x.nth)))
    revA["pit"] = dict(all_merges_announce_le_current_last_day=bool(chD.pit_ok.all()),
                       n_merges=int(len(lagx)),
                       lag_td_last_day_minus_announce=pd.Series(lagx).value_counts().sort_index().to_dict(),
                       n_records_announced_1_2_td_after_last_day=len(nonpit), examples=nonpit[:10])
    revA["chains_since_2015_07"] = dict(n=int((chD.start0 >= "2015-07-01").sum()),
                                        n_extended=int(chD[chD.start0 >= "2015-07-01"].extended.sum()),
                                        n_eligible=int(sum(1 for x in infos[("D", 0)] if x[2])),
                                        n_eligible_extended=int(sum(1 for x in infos[("D", 0)] if x[2] and
                                                                    chm[(x[1], P.cidx[x[0]])].extended)),
                                        n_event_level=int(len(evd)), n_event_level_extended=int(evd.ext.sum()),
                                        n_D_main_trades_extended=int(tD.ext.sum()))
    # 不延長版本（D 主跑：只持有到第一筆記錄的最後一天）
    ch_noext = chD.copy()
    ch_noext["L"] = [int(np.searchsorted(da, x, side="right")) - 1 if x <= P.dstr[-1] else np.nan for x in ch_noext.end0]
    noext_plans = {}
    rows_ne = {}
    for sh in SHIFTS:
        b, mk, _ = K.disp_plans(P, ch_noext, sh, excl, cs)
        noext_plans[sh] = (b, mk)
    xne = run("D", plans_override=noext_plans[0])
    revA["D_main_no_extension"] = nav_stats(xne["nav"], int(len(xne["trades"])))
    ne_d = []
    ne_c = []
    for sh in SHIFTS:
        for cp in CAPS:
            nd = run("D", sh, cp, plans_override=noext_plans[sh])["nav"]
            ndh = run("D", sh, cp / 2, plans_override=noext_plans[sh])["nav"]
            ne_d.append(L.ann(P, nd, i19, e))
            ne_c.append(L.ann(P, single_nav("E0", sh, cp / 2)[0] + ndh, i19, e) - L.ann(P, path_nav[("C0", sh, cp)], i19, e))
    revA["no_extension_paths20"] = dict(D_ann_2019_median=float(np.median(ne_d)),
                                        D_n_beat_0050=int((np.array(ne_d) > b50["ann_2019"]).sum()),
                                        C_minus_C0_n_pos=int((np.array(ne_c) > 0).sum()),
                                        C_minus_C0_median=float(np.median(ne_c)),
                                        note="事後描述：不延長（只持有到第一筆處置記錄的最後一天）")
    out["reviewA"] = revA
    print("reviewA", json.dumps({k: v for k, v in revA.items() if k in ("D_main_trades", "event_level_net_excess",
                                                                         "chains_since_2015_07", "pit")},
                                ensure_ascii=False, default=str), flush=True)

    # ---------------- 審查 B：預收款券
    def lots_count(x):
        tr = x["trades"]
        sh = list(tr.sh.values) if len(tr) else []
        for c_, bd, cost, val, i_ in x["open_pos"]:
            tb = int(np.searchsorted(da, bd))
            o_ = P.O[tb, i_]
            sh.append(cost / (o_ * (1.0 + S.tick(o_) / 2.0 / o_ + 0.003) * (1.0 + S.FEE)))
        sh = np.array(sh, float)
        return dict(n=int(len(sh)), ge10=int((sh >= 10_000 - 0.5).sum()), ge30=int((sh >= 30_000 - 0.5).sum()))
    revB = dict(rule="證交所處置內容（PREREG design.pre_collection 記載、研究員在 punish 年檔處置內容文字看到）："
                     "第一次處置只有單筆達 10 交易單位或多筆累積達 30 交易單位以上才要預收款券；第二次處置所有委託都要預收。"
                     "審查版：處置股單筆買進上限 29 張（分多筆 < 10 張、當日累積 < 30 張即不需預收）；"
                     "被截的那筆不再套用最小買進金額門檻。", paths={})
    bpaths_d = []; bpaths_c = []; bcnt = []
    for sh in SHIFTS:
        for cp in CAPS:
            for nm, capx in (("D", cp), ("Dh", cp / 2)):
                base = single_nav("D", sh, capx)[2]
                cnt0 = lots_count(base)
                xb = run("D", sh, capx, max_sh=CAP29)
                bcnt.append(dict(shift=sh, capital=capx, cell=nm, **cnt0, capped=int(xb["cnt"].get("capped", 0))))
                if nm == "D":
                    navb_d = xb["nav"]
                else:
                    navb_dh = xb["nav"]
            nE, _, _ = single_nav("E0", sh, cp / 2)
            bpaths_d.append(dict(shift=sh, capital=cp, ann_2019=L.ann(P, navb_d, i19, e), ann_full=L.ann(P, navb_d, P.t0, e),
                                 base_2019=L.ann(P, path_nav[("D", sh, cp)], i19, e)))
            navc = nE + navb_dh
            bpaths_c.append(dict(shift=sh, capital=cp, c_c0_2019=L.ann(P, navc, i19, e) - L.ann(P, path_nav[("C0", sh, cp)], i19, e),
                                 c_c0_1923=L.ann(P, navc, i19, i24) - L.ann(P, path_nav[("C0", sh, cp)], i19, i24),
                                 c_c0_2024=L.ann(P, navc, i24, e) - L.ann(P, path_nav[("C0", sh, cp)], i24, e)))
    bd = pd.DataFrame(bpaths_d); bc = pd.DataFrame(bpaths_c); bn = pd.DataFrame(bcnt)
    xbm = run("D", max_sh=CAP29)
    revB["D_main_cap29"] = nav_stats(xbm["nav"], int(len(xbm["trades"])))
    revB["D_main_lots"] = lots_count(res["D"])
    revB["counts_by_path"] = bn.to_dict("records")
    revB["counts_D_by_capital"] = {str(int(k)): dict(ge10=int(g.ge10.sum()), ge30=int(g.ge30.sum()), n=int(g.n.sum()))
                                   for k, g in bn[bn.cell == "D"].groupby("capital")}
    revB["paths20_D_cap29"] = dict(ann_2019_median=float(bd.ann_2019.median()),
                                   n_beat_0050=int((bd.ann_2019 > b50["ann_2019"]).sum()),
                                   max_abs_change_2019=float((bd.ann_2019 - bd.base_2019).abs().max()),
                                   median_change_2019=float((bd.ann_2019 - bd.base_2019).median()))
    revB["paths20_C_minus_C0_cap29"] = dict(n_pos=int((bc.c_c0_2019 > 0).sum()), median=float(bc.c_c0_2019.median()),
                                            median_1923=float(bc.c_c0_1923.median()),
                                            median_2024=float(bc.c_c0_2024.median()))
    out["reviewB"] = revB
    print("reviewB", {k: v for k, v in revB.items() if k not in ("counts_by_path",)}, flush=True)

    # ---------------- 審查 C：2019 年名單
    dd = disp.copy()
    dd["ym"] = dd.announce.str[:7]
    rn1 = dd[(dd.kind == "regular") & (dd.nth == 1)]
    by_year = dd.groupby(dd.announce.str[:4]).size().to_dict()
    by_year_r1 = rn1.groupby(rn1.announce.str[:4]).size().to_dict()
    months = pd.period_range("2015-01", "2026-10", freq="M").astype(str)
    mcnt = rn1.groupby("ym").size().reindex(months, fill_value=0)
    mall = dd.groupby("ym").size().reindex(months, fill_value=0)
    revC = dict(by_year_all=by_year, by_year_regular_nth1=by_year_r1,
                months_2019_regular_nth1=mcnt[[m for m in months if m.startswith("2019")]].to_dict(),
                months_2019_all=mall[[m for m in months if m.startswith("2019")]].to_dict(),
                zero_months_by_year_regular_nth1={y: int((mcnt[[m for m in months if m.startswith(y)]] == 0).sum())
                                                  for y in [str(z) for z in range(2015, 2027)]},
                zero_months_by_year_all={y: int((mall[[m for m in months if m.startswith(y)]] == 0).sum())
                                         for y in [str(z) for z in range(2015, 2027)]})
    # 價格面近似：上市、6 日還原漲幅 > 32% 連 3 日的新段數（每年），與官方第一次處置筆數比
    with np.errstate(invalid="ignore", divide="ignore"):
        r6 = np.full((P.T, P.N), np.nan)
        r6[6:] = P.G[6:] / P.G[:-6] - 1.0
    fl6 = np.nan_to_num(r6, nan=0.0) > 0.32
    fl6 &= P.is_twse & P.traded
    c3 = fl6.copy()
    c3[2:] = fl6[2:] & fl6[1:-1] & fl6[:-2]
    c3[:2] = False
    prev = np.zeros_like(c3)
    for k_ in range(1, 11):
        prev[k_:] |= c3[:-k_]
    newep = c3 & ~prev
    yrs = np.array([d[:4] for d in P.dstr])
    prox = {y: int(newep[yrs == y].sum()) for y in sorted(set(yrs))}
    revC["proxy_6d32_3days_new_episodes_twse"] = prox
    revC["ratio_official_r1_to_proxy"] = {y: (by_year_r1.get(y, 0) / prox[y] if prox[y] else None) for y in prox}
    elig_year = pd.Series([P.dstr[x[1] + 1][:4] for x in infos[("D", 0)] if x[2]]).value_counts().sort_index().to_dict()
    revC["eligible_D_by_year"] = elig_year
    del r6, fl6, c3, prev, newep
    # 2019 年 D 衛星換 0050：(a) 進場在 2019 的事件全部不買（衛星閒錢掃進 0050，含成本）；(b) 淨值拼接（2019 日報酬換 0050）
    y0 = L.didx(P, "2019-01-01"); y1 = L.didx(P, "2020-01-01") - 1
    drop = (lambda s: P.dstr[s + 1][:4] == "2019")
    d19 = []; c19 = []; sp19d = []; sp19c = []
    for sh in SHIFTS:
        b, mk, _ = K.disp_plans(P, chD, sh, excl, cs, drop=drop)
        for cp in CAPS:
            nd = run("D", sh, cp, plans_override=(b, mk))["nav"]
            ndh = run("D", sh, cp / 2, plans_override=(b, mk))["nav"]
            nE = single_nav("E0", sh, cp / 2)[0]
            c0n = path_nav[("C0", sh, cp)]
            d19.append(dict(ann_2019=L.ann(P, nd, i19, e), ann_full=L.ann(P, nd, P.t0, e)))
            cn = nE + ndh
            c19.append(dict(d2019=L.ann(P, cn, i19, e) - L.ann(P, c0n, i19, e),
                            d1923=L.ann(P, cn, i19, i24) - L.ann(P, c0n, i19, i24),
                            d2024=L.ann(P, cn, i24, e) - L.ann(P, c0n, i24, e)))

            def splice(nav):
                x = nav.copy()
                rr = nav[1:] / nav[:-1]
                r5 = e50[1:] / e50[:-1]
                rr[y0 - 1:y1] = r5[y0 - 1:y1]
                x[P.t0 + 1:] = nav[P.t0] * np.cumprod(rr[P.t0:])
                return x
            ns = splice(path_nav[("D", sh, cp)])
            nsh = splice(single_nav("D", sh, cp / 2)[0])
            sp19d.append(dict(ann_2019=L.ann(P, ns, i19, e), ann_full=L.ann(P, ns, P.t0, e)))
            cns = nE + nsh
            sp19c.append(dict(d2019=L.ann(P, cns, i19, e) - L.ann(P, c0n, i19, e),
                              d1923=L.ann(P, cns, i19, i24) - L.ann(P, c0n, i19, i24),
                              d2024=L.ann(P, cns, i24, e) - L.ann(P, c0n, i24, e)))
    def d_cls(df, full_main):
        med = float(df.ann_2019.median()); n = int((df.ann_2019 > b50["ann_2019"]).sum())
        if med > b50["ann_2019"] and n >= 15 and full_main > b50["ann_full"]:
            c_ = "贏 0050"
        elif med < b50["ann_2019"] and n <= 5:
            c_ = "輸 0050"
        else:
            c_ = "不確定"
        return dict(ann_2019_median=med, n_beat=n, main_full=full_main, cls=c_)

    def c_cls(df):
        n = int((df.d2019 > 0).sum()); m1 = float(df.d1923.median()); m2 = float(df.d2024.median())
        if n >= 15:
            k = (m1 > 0) + (m2 > 0)
            v = "有補充價值" if k == 2 else ("有補充價值，但只靠一段" if k == 1 else "通過 15／20，但兩段中位數都不正")
        elif n <= 5:
            v = "沒有補充價值"
        else:
            v = "不確定"
        return dict(n_pos=n, median_2019=float(df.d2019.median()), median_1923=m1, median_2024=m2, cls=v)
    d19 = pd.DataFrame(d19); c19 = pd.DataFrame(c19); sp19d = pd.DataFrame(sp19d); sp19c = pd.DataFrame(sp19c)
    revC["drop2019_events"] = dict(D=d_cls(d19, float(d19.ann_full.iloc[[i for i, (sh, cp) in enumerate(
        [(a, b) for a in SHIFTS for b in CAPS]) if sh == 0 and cp == 1_000_000.0][0]])), C_minus_C0=c_cls(c19))
    revC["splice2019_0050"] = dict(D=d_cls(sp19d, float(sp19d.ann_full.iloc[[i for i, (sh, cp) in enumerate(
        [(a, b) for a in SHIFTS for b in CAPS]) if sh == 0 and cp == 1_000_000.0][0]])), C_minus_C0=c_cls(sp19c))
    revC["main_D_2019_return"] = cells["D"]["yearly"].get(2019); revC["main_0050_2019_return"] = b50["yearly"].get(2019)
    out["reviewC"] = revC
    print("reviewC", json.dumps({k: v for k, v in revC.items() if k.startswith(("drop", "splice", "ratio", "zero"))},
                                ensure_ascii=False, default=str), flush=True)

    # ---------------- 事件密度、逐年
    out["posthoc"] = dict(yearly={k: cells[k]["yearly"] for k in cells}, yearly_0050=b50["yearly"],
                          event_level_by_year={y: dict(n=int(len(g)), mean_exc=float(g.exc.mean()))
                                               for y, g in evd.groupby("year")},
                          event_level_hold_median=float(evd.hold.median()))

    out["resources"] = dict(runtime_s=round(time.time() - t_start, 1), maxrss_mb=int(rss()), disp_sha256=sha)
    s = json.dumps(out, ensure_ascii=False, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x))
    with open(os.path.join(odir, "my_numbers.json"), "w") as f:
        f.write(s)
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    with open(os.path.join(odir, "my_numbers.sha256"), "w") as f:
        f.write(h + "  my_numbers.json\n")
    print("frozen", h, "rss", rss(), round(time.time() - t_start, 1), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
