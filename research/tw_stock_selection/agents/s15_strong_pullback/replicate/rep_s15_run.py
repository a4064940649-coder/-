# -*- coding: utf-8 -*-
"""S15 獨立複製：驗收、E0／P／C／C0／P40 主跑＋八項、20 條路徑、逐條配對、四段空頭、相關與重疊、事前判定、
事件層級（扣成本前、獨立事件）、審查 A（排名池只含上市滿 120 日）、B（進場改訊號日收盤／次日收盤）、
C（同日隨機排序 5 組種子），然後凍結。
用法：python3 -I -B rep_s15_run.py <scratch_dir>
需先跑 s1 的 rep_s1_panel.py <scratch_dir> 與 s3 的 rep_s3_names.py <scratch_dir>/innov_names.json。
輸出（本資料夾）：my_numbers.json、my_numbers.sha256、my_trades.csv.gz、my_paths.csv.gz
凍結前不讀研究員的 FINDINGS 第 2 節以外、results*.json、navs.csv.gz、trades.csv.gz。
"""
import sys, os, json, hashlib, resource, time, math
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import rep_s15_lib as K
S, Q, R, L = K.S, K.Q, K.R, K.L
ST, X2 = K.ST, K.X2

CAPS = [600_000.0, 800_000.0, 1_000_000.0, 1_500_000.0, 2_000_000.0]
SHIFTS = [0, 1, 2, 3]
BEARS = {"2018": ("2018-01-01", "2018-12-31"), "2020-03": ("2020-01-01", "2020-06-30"),
         "2022": ("2022-01-01", "2022-12-31"), "2024-07~2025-04": ("2024-07-01", "2025-04-30")}
SINGLE = ["E0", "P", "P40"]
COMBO = {"C": ("E0", "P"), "C0": ("E0", "0050")}
ALL = ["E0", "P", "C", "C0", "P40"]
HOLD = {"P": 20, "P40": 40}
LISTMAX = 9999
HORIZ = [1, 5, 10, 20, 40, 60]


def rss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024


def fl(x):
    if x is None:
        return None
    x = float(x)
    return x if np.isfinite(x) else None


def dsum(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) == 0:
        return dict(n=0)
    return dict(n=int(len(x)), mean=float(x.mean()), median=float(np.median(x)), t=K.tstat(x),
                pos_share=float((x > 0).mean()))


def main(sc, odir=HERE):
    t_start = time.time()
    P = R.build(sc)
    fixed = K.attach_e50_fixed(P, sc)
    excl = K.S.innov_mask_fn(P, sc)
    U = K.univ_matrix(P, excl)
    Rv, mi0 = L.load_rev(P)
    S_, _ = L.surge(Rv, mi0, null2013=True)
    e = P.T - 1
    i19 = L.didx(P, "2018-12-28"); i24 = L.didx(P, "2023-12-29"); ih = L.didx(P, "2021-02-19")
    da = np.array(P.dstr)
    t0 = P.t0

    def win(a, b):
        return int(np.searchsorted(da, a, side="left")), int(np.searchsorted(da, b, side="right")) - 1
    bw = {k: win(a, b) for k, (a, b) in BEARS.items()}
    out = dict(meta=dict(T=P.T, N=P.N, first=P.dstr[0], last=P.dstr[-1], t0=P.dstr[t0], i19=P.dstr[i19],
                         i24=P.dstr[i24], ih=P.dstr[ih],
                         bear_windows={k: (P.dstr[a], P.dstr[b]) for k, (a, b) in bw.items()},
                         note="面板為複製員自建（N＝2130，研究員 2142；差的是從未進股池的代號）。"
                              "0050 停牌日用 S14 複製員的修正（還原開盤＝前一日還原收盤、原始開盤＝前一日原始收盤）。",
                         e50_fixed_days=[x["date"] for x in fixed]))

    # ---------------- 訊號
    M = K.signal_mats(P, U)
    sig, rs = M["sig"], M["rs"]
    base = M["base"][t0:]
    nb = int(base.sum())
    dc = {}
    for k in ["strong", "near", "pb1", "vd1", "turn"]:
        dc[k] = dict(alone=float((M["alone"][k][t0:] & base).sum() / nb), cumulative=float(M["cum"][k][t0:].sum() / nb),
                     cumulative_n=int(M["cum"][k][t0:].sum()))
    yrs = np.array([d[:4] for d in P.dstr])
    nd = sig[t0:].sum(1)
    # 獨立事件：同檔前 19 列內沒有訊號（＝同檔 20 日內只算第一次；與 data_check 5,124 筆相同的定義）
    indep = np.zeros_like(sig)
    for i in np.where(sig[t0:].any(0))[0]:
        prev = -10 ** 9
        for t in np.where(sig[:, i])[0]:
            if t < t0:
                prev = t; continue
            if t - prev >= 20:
                indep[t, i] = True
            prev = t
    out["signal_check"] = dict(
        base_n=nb, condition_pass_rate=dc, signals_total=int(sig[t0:].sum()),
        signals_per_year=pd.Series(nd, index=yrs[t0:]).groupby(level=0).sum().astype(int).to_dict(),
        days_with_signal_share=float((nd > 0).mean()), per_day_median=float(np.median(nd)),
        per_day_p90=float(np.percentile(nd, 90)), per_day_max=int(nd.max()), days_ge_10=int((nd >= 10).sum()),
        independent_events=int(indep[t0:].sum()),
        independent_per_year=pd.Series(indep[t0:].sum(1), index=yrs[t0:]).groupby(level=0).sum().astype(int).to_dict(),
        distinct_stocks=int(sig[t0:].any(0).sum()),
        strong_new_listing_days=int((M["cum"]["strong"][t0:] & (P.barno[t0:] < 121)).sum()))
    print("signals", out["signal_check"]["signals_total"], out["signal_check"]["independent_events"], "rss", rss(),
          flush=True)

    # 審查 A 的訊號（排名池只含上市滿 121 根＝G[t−120] 在上市後）
    MA = K.signal_mats(P, U, pool_filter=(P.barno >= 121))
    sigA = MA["sig"]
    out["reviewA_signals"] = dict(signals_total=int(sigA[t0:].sum()), only_main=int((sig & ~sigA)[t0:].sum()),
                                  only_A=int((sigA & ~sig)[t0:].sum()),
                                  strong_cum_n=int(MA["cum"]["strong"][t0:].sum()))
    del MA
    tw = K.twse_tenure(P)
    out["signal_check"]["signals_twse_tenure_lt120"] = int((sig[t0:] & (tw[t0:] < 120)).sum())
    del tw

    # ---------------- 名單
    plansE = {sh: Q.cell_plans(Q.make_lists(P, S_, mi0, shift=sh, excl=excl), "C10", x=0.10) for sh in SHIFTS}
    plansP = {sh: K.pb_plans(P, sig, rs, U, shift=sh) for sh in SHIFTS}
    plansPA = {sh: K.pb_plans(P, sigA, rs, U, shift=sh) for sh in SHIFTS}
    print("plans", "rss", rss(), round(time.time() - t_start, 1), flush=True)

    e50 = P.e50.values.astype(float)

    def e50nav(capital):
        return capital * e50 / e50[t0]

    def bears(nav):
        return {k: L.maxdd(P, nav, a, b)[0] for k, (a, b) in bw.items()}

    def nav_stats(nav, ntr=None):
        md = L.maxdd(P, nav, t0, e)
        return dict(ann_2019=L.ann(P, nav, i19, e), ann_full=L.ann(P, nav, t0, e), ann_2024=L.ann(P, nav, i24, e),
                    ann_1923=L.ann(P, nav, i19, i24), mdd_full=md[0], mdd_peak=md[1], mdd_trough=md[2],
                    ann_h1=L.ann(P, nav, t0, ih), ann_h2=L.ann(P, nav, ih, e), n_trades=ntr)

    b50 = dict(nav_stats(e50), bears=bears(e50), yearly=L.yearly(P, e50))
    out["bench_0050"] = b50

    # ---------------- 引擎包裝
    def run(cell, sh=0, capital=1_000_000.0, cost=None, block_span=None, rec_pos=False, listmax=LISTMAX,
            plans_override=None):
        if cell == "E0":
            b, k = plansE[sh]
            return K.simulate(P, b, k, sat_w=0.7, sizing="W10", capital=capital, block_span=block_span,
                              rec_pos=rec_pos, **(cost or ST))
        b = plans_override if plans_override is not None else plansP[sh][0]
        return K.simulate(P, b, {}, sat_w=0.7, sizing="W10", capital=capital, block_span=block_span,
                          rec_pos=rec_pos, eod=K.make_eod_hold(HOLD[cell]), listmax=listmax, **(cost or ST))

    # ---------------- 驗收 A0
    acc = {}
    acc["P0"] = dict(T=P.T, N=P.N, last=P.dstr[-1], note="研究員 N＝2142；複製員面板 2130（已知差異）")
    a14f = run("E0")
    acc["A0"] = dict(ann_2019=L.ann(P, a14f["nav"], i19, e), ann_full=L.ann(P, a14f["nav"], t0, e),
                     mdd_full=L.maxdd(P, a14f["nav"], t0, e)[0], n_trades_closed=int(len(a14f["trades"])),
                     n_open=len(a14f["open_pos"]), known=(0.4813, 0.3408, -0.3340))
    acc["A0"]["ok"] = bool(abs(acc["A0"]["ann_2019"] - 0.4813) < 0.00025 and abs(acc["A0"]["ann_full"] - 0.3408) < 0.00025)
    print("A0", acc["A0"], flush=True)
    assert acc["A0"]["ok"]

    # ---------------- 主跑
    res = {"E0": a14f, "P": run("P", rec_pos=False), "P40": run("P40")}
    half = {c: run(c, capital=500_000.0, rec_pos=True) for c in ("E0", "P")}
    navs = {c: res[c]["nav"] for c in SINGLE}
    navs["0050"] = e50nav(1_000_000.0)
    navs["C"] = half["E0"]["nav"] + half["P"]["nav"]
    navs["C0"] = half["E0"]["nav"] + e50nav(500_000.0)
    acc["A2"] = dict(max_abs_diff_C_vs_halves=float(np.nanmax(np.abs(navs["C"] - (half["E0"]["nav"] + half["P"]["nav"])))),
                     C_start=float(navs["C"][t0]))

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

    # ---------------- A1
    a1 = {}
    # (i)
    ok_i = True; nsig = 0
    for sh in SHIFTS:
        b, info = plansP[sh]
        for s, ii in b.items():
            for i in ii:
                nsig += 1
                t = info[(s, int(i))]
                if not (U[s, i] and U[t, i] and np.isfinite(rs[t, i]) and sig[t, i] and t == s - sh):
                    ok_i = False
    a1["i_all_sig_in_U_rank_valid"] = ok_i; a1["n_plan_entries"] = nsig
    # (ii) 300 訊號＋300 差一條件
    rng = np.random.default_rng(15)
    allsig = np.argwhere(sig[t0:]) + [t0, 0]
    cond = [M["base"], M["cum"]["strong"] | False, None]
    al = M["alone"]
    k5 = [M["cum"]["strong"], al["near"], al["pb1"], al["vd1"], al["turn"]]
    nfail = (~k5[0]).astype(np.int8)
    for x in k5[1:]:
        nfail += (~x).astype(np.int8)
    miss1 = M["base"] & (nfail == 1)
    miss1[:t0] = False
    allmiss = np.argwhere(miss1)
    del nfail, miss1
    pick_s = allsig[rng.choice(len(allsig), 300, replace=False)]
    pick_m = allmiss[rng.choice(len(allmiss), 300, replace=False)]
    r0 = np.where(np.isfinite(P.ret), P.ret, 0.0)
    rank_cache = {}

    def strong_py(t):
        if t in rank_cache:
            return rank_cache[t]
        if t < 120:
            rank_cache[t] = set()
            return rank_cache[t]
        e1 = float(e50[t]) / float(e50[t - 120]) - 1.0
        vals = []
        for i in np.where(U[t])[0]:
            i = int(i)
            if not P.inbar[t, i]:
                continue
            col = r0[t - 119:t + 1, i].tolist()
            g = math.prod([1.0 + v for v in col]) - 1.0 - e1
            vals.append((g, i))
        vals.sort(key=lambda z: -z[0])
        k = int(math.ceil(0.2 * len(vals)))
        st = set(i for _, i in vals[:k])
        rank_cache[t] = st
        return st
    mism = 0
    for kind, pk in (("sig", pick_s), ("miss1", pick_m)):
        for t, i in pk:
            t = int(t); i = int(i)
            c = K.check_conditions_py(P, U, i in strong_py(t), t, i)
            vec = [bool(M["base"][t, i]), bool(M["cum"]["strong"][t, i]), bool(al["near"][t, i]),
                   bool(al["pb1"][t, i]), bool(al["vd1"][t, i]), bool(al["turn"][t, i])]
            if c != vec:
                mism += 1
            if kind == "sig" and not all(c):
                mism += 1
            if kind == "miss1" and sum(not x for x in c) != 1:
                mism += 1
    del r0
    a1["ii_mismatch"] = mism; a1["ii_n"] = 600
    print("A1 ii", mism, round(time.time() - t_start, 1), flush=True)

    # ---------------- 主跑統計＋八項
    cells = {}
    trade_rows = []
    for cc in ALL:
        nav = navs[cc]; tr = trades_of(cc)
        ops = open_pos(cc)
        o = nav_stats(nav, int(len(tr)))
        o["n_trades_incl_open"] = int(len(tr) + len(ops))
        o["bears"] = bears(nav)
        o["yearly"] = L.yearly(P, nav)
        x2nav, _ = run_cell(cc, cost=X2)
        sp, top = R.span_of(tr, 5)
        blnav, _ = run_cell(cc, block_span=sp)
        o["x2"] = dict(ann_2019=L.ann(P, x2nav, i19, e), ann_full=L.ann(P, x2nav, t0, e))
        o["block5"] = dict(ann_2019=L.ann(P, blnav, i19, e), ann_full=L.ann(P, blnav, t0, e))
        o["block5_top"] = top
        inc = float(nav[e] - nav[t0])
        o["top5_share_of_gain"] = float(sum(t["pnl"] for t in top) / inc)
        o["n_open_end"] = len(ops)
        o["final_nav"] = float(nav[e])
        if len(tr):
            o["trade_stats"] = dict(exp=float(tr.ret.mean()), win=float((tr.ret > 0).mean()),
                                    avg_win=float(tr.ret[tr.ret > 0].mean()), avg_loss=float(tr.ret[tr.ret <= 0].mean()),
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
        if cc in ("E0", "P", "P40", "C"):
            for r_ in tr.itertuples():
                trade_rows.append((cc, getattr(r_, "half", cc), r_.code, r_.buy_date, r_.sell_date, round(r_.ret, 6),
                                   round(r_.pnl, 1), int(r_.hold), int(r_.sh), str(getattr(r_, "why", "") or "")))
        print(cc, round(o["ann_2019"], 4), round(o["ann_full"], 4), round(o["ann_1923"], 4), round(o["ann_2024"], 4),
              round(o["mdd_full"], 4), o["n_trades"], o["n_trades_incl_open"], o["n_checks_pass"], o["eight_cls"],
              "rss", rss(), round(time.time() - t_start, 1), flush=True)
    out["cells"] = cells
    pd.DataFrame(trade_rows, columns=["cell", "half", "code", "buy_date", "sell_date", "ret", "pnl", "hold", "sh",
                                      "why"]).to_csv(os.path.join(odir, "my_trades.csv.gz"), index=False,
                                                     compression="gzip")
    pP = res["P"]
    out["P_main_engine"] = dict(cnt={k: (float(v) if isinstance(v, float) else int(v)) for k, v in pP["cnt"].items()},
                                mean_npos=float(pP["npos"][t0 + 1:].mean()),
                                share_days_full=float((pP["npos"][t0 + 1:] >= 10).mean()),
                                mean_stock_weight=float(pP["stkw"][t0 + 1:].mean()),
                                cost_per_year_share_of_nav=None)
    yrs_n = (P.dates[e] - P.dates[t0]).days / 365.25
    out["P_main_engine"]["cost_total_per_year_over_mean_nav"] = float(
        pP["cnt"]["cost_total"] / yrs_n / np.nanmean(pP["nav"][t0:]))
    out["P_main_engine"]["turnover_buy_per_year_over_mean_nav"] = float(
        pP["cnt"]["buy_amt"] / yrs_n / np.nanmean(pP["nav"][t0:]))

    # ---------------- A1 (iii)：交易進出場
    a1iii = dict(trades_checked=0, entry_ok=0, entry_bad=0, exit_on_time=0, exit_delayed=0, exit_delist=0,
                 exit_reduce=0, exit_other=0, open_end=0)
    pcache = {}

    def single_nav(cell, sh, cap):
        key = (cell, sh, cap)
        if key not in pcache:
            x = run(cell, sh, cap)
            pcache[key] = (x["nav"], len(x["trades"]), len(x["open_pos"]), x)
        return pcache[key]
    for cell in ("P", "P40"):
        H = HOLD[cell]
        for sh in SHIFTS:
            b, info = plansP[sh]
            for cap in CAPS:
                x = single_nav(cell, sh, cap)[3]
                for r_ in x["trades"].itertuples():
                    a1iii["trades_checked"] += 1
                    s = int(r_.tb) - 1
                    if (s, int(r_.i)) in info and info[(s, int(r_.i))] == s - sh:
                        a1iii["entry_ok"] += 1
                    else:
                        a1iii["entry_bad"] += 1
                    if r_.why == "end":
                        if int(r_.ts) - int(r_.tb) == H:
                            a1iii["exit_on_time"] += 1
                        else:
                            a1iii["exit_delayed"] += 1
                    elif r_.why == "delist":
                        a1iii["exit_delist"] += 1
                    elif r_.why == "reduce":
                        a1iii["exit_reduce"] += 1
                    else:
                        a1iii["exit_other"] += 1
                a1iii["open_end"] += len(x["open_pos"])
        # 只保留 nav 以省記憶體
        for key in list(pcache):
            v = pcache[key]
            pcache[key] = (v[0], v[1], v[2], None)
    a1["iii"] = a1iii
    acc["A1"] = a1
    out["acceptance"] = acc
    print("A1", a1, flush=True)

    # ---------------- 20 條路徑
    rows = []
    path_nav = {}
    for cc in ALL:
        for sh in SHIFTS:
            for cp in CAPS:
                if cc in SINGLE:
                    nav, n, no, _ = single_nav(cc, sh, cp)
                else:
                    a, b = COMBO[cc]
                    na, n1, no1, _ = single_nav(a, sh, cp / 2)
                    if b == "0050":
                        nav, n, no = na + e50nav(cp / 2), n1, no1
                    else:
                        nb_, n2, no2, _ = single_nav(b, sh, cp / 2)
                        nav, n, no = na + nb_, n1 + n2, no1 + no2
                path_nav[(cc, sh, cp)] = nav
                st = nav_stats(nav, n)
                rows.append(dict(cell=cc, shift=sh, capital=cp, n_incl_open=n + no, **st,
                                 **{"bear_" + k: v for k, v in bears(nav).items()}))
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
    for a, b in (("P", "0050"), ("C", "C0"), ("C", "E0"), ("P40", "P"), ("P40", "0050"), ("C0", "E0"), ("E0", "0050")):
        pair["%s-%s" % (a, b)] = summ(paired_df(pr, a, b))
    out["paired"] = pair

    # ---------------- 相關與重疊
    m50 = L.monthly(P, e50, t0)

    def corr(na, nb_, start=None):
        a = L.monthly(P, na, t0) - m50; b = L.monthly(P, nb_, t0) - m50
        df = pd.concat([a, b], axis=1).dropna()
        if start:
            df = df[df.index >= pd.Period(start, "M")]
        return float(df.iloc[:, 0].corr(df.iloc[:, 1])), int(len(df))
    c1 = corr(navs["P"], navs["E0"]); c2 = corr(navs["P"], navs["E0"], "2019-01")
    pc = [corr(path_nav[("P", sh, cp)], path_nav[("E0", sh, cp)])[0] for sh in SHIFTS for cp in CAPS]
    sE = half["E0"]["snap"]; sP = half["P"]["snap"]
    ov = []; ovn = []
    for t in range(t0 + 1, P.T):
        ph = set(sP.get(t, {})); eh = set(sE.get(t, {}))
        if ph:
            ov.append(len(ph & eh) / len(ph)); ovn.append(len(ph & eh))
    # P 交易與 E0 交易：同股票、持有期重疊
    tP = res["P"]["trades"]; tE = res["E0"]["trades"]
    n_ovl = 0
    eg = {c: list(zip(g.tb, g.ts)) for c, g in tE.groupby("code")}
    for r_ in tP.itertuples():
        for a, b in eg.get(r_.code, []):
            if r_.tb < b and a < r_.ts:
                n_ovl += 1; break
    out["corr_overlap"] = dict(
        P_E0_excess_full=c1[0], n_months=c1[1], P_E0_excess_2019=c2[0], n_months_2019=c2[1],
        paths20_median=float(np.median(pc)), paths20_min=float(np.min(pc)), paths20_max=float(np.max(pc)),
        C_overlap_mean_share_of_P_half=float(np.mean(ov)), C_overlap_days_any=int(np.sum(np.array(ovn) > 0)),
        C_P_half_days_with_pos=int(len(ov)), C_overlap_mean_names=float(np.mean(ovn)),
        P_main_trades_overlapping_E0_trades=int(n_ovl), P_main_n_trades=int(len(tP)))
    del half["E0"]["snap"], half["P"]["snap"]
    print("corr/overlap", out["corr_overlap"], flush=True)

    # ---------------- 事前判定
    def p_rule(cc):
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
    dec = dict(P_vs_0050=p_rule("P"), P40_vs_0050_desc=p_rule("P40"), C_supplement=c_rule("C-C0"),
               bears_C_vs_C0=bear_rule("C-C0"), bears_P_vs_0050=bear_rule("P-0050"),
               eight_checks={k: cells[k]["eight_cls"] for k in cells})
    dec["xq"] = "可以加（紙上追蹤＋XQ 測）" if dec["C_supplement"]["cls"] == "有補充價值" else "不建議加進現行策略"
    out["decisions"] = dec
    print("decisions", json.dumps(dec, ensure_ascii=False), flush=True)

    # ---------------- 名單掃描長度敏感度（只看名單前 10 名）
    x10 = run("P", listmax=None)
    out["sens_listmax10_P_main"] = dict(nav_stats(x10["nav"], int(len(x10["trades"]))),
                                        rows_gt10=int(sum(len(v) > 10 for v in plansP[0][0].values())))
    del x10

    # ---------------- 事件層級（扣成本前）
    AO, Gf = K.adj_open_matrix(P)
    e50o = P.e50o
    ev = []
    for t, i in np.argwhere(sig[t0:]) + [t0, 0]:
        t = int(t); i = int(i)
        a = t + 1
        if a > e:
            continue
        o = P.O[a, i]
        buyable = bool(o == o and o < S.limit_up(P.ref[a, i]) - 1e-9)
        d = dict(code=P.codes[i], t=t, a=a, i=i, indep=bool(indep[t, i]), buyable=buyable, year=P.dstr[a][:4],
                 rs=float(rs[t, i]))
        d["gap"] = AO[a, i] / Gf[t, i] - 1.0
        d["gap_exc"] = d["gap"] - (e50o[a] / e50[t] - 1.0)
        for h in HORIZ:
            k = a + h - 1
            if k <= e:
                d["cum%d" % h] = Gf[k, i] / AO[a, i] - 1.0
                d["exc%d" % h] = d["cum%d" % h] - (e50[k] / e50o[a] - 1.0)
        if a + 20 <= e:
            x = a + 20
            d["oo20"] = AO[x, i] / AO[a, i] - 1.0
            d["oo20_exc"] = d["oo20"] - (e50o[x] / e50o[a] - 1.0)
            # 審查 B：同一出場（a＋20 開盤），進場改訊號日收盤、次日收盤
            d["B_close_t_exc"] = AO[x, i] / Gf[t, i] - 1.0 - (e50o[x] / e50[t] - 1.0)
            d["B_close_a_exc"] = AO[x, i] / Gf[a, i] - 1.0 - (e50o[x] / e50[a] - 1.0)
        if t + 20 <= e:
            d["B_cc_t_exc"] = Gf[t + 20, i] / Gf[t, i] - 1.0 - (e50[t + 20] / e50[t] - 1.0)
        if a + 20 <= e:
            d["B_cc_a_exc"] = Gf[a + 20, i] / Gf[a, i] - 1.0 - (e50[a + 20] / e50[a] - 1.0)
        ev.append(d)
    evd = pd.DataFrame(ev)
    del AO, Gf

    def evsum(df):
        o = dict(n=int(len(df)))
        for h in HORIZ:
            o["exc%d" % h] = dsum(df["exc%d" % h].values)
            o["cum%d" % h] = dsum(df["cum%d" % h].values)
        o["oo20"] = dsum(df.oo20.values); o["oo20_exc"] = dsum(df.oo20_exc.values)
        o["gap"] = dsum(df.gap.values); o["gap_exc"] = dsum(df.gap_exc.values)
        return o
    ind = evd[evd.indep & evd.buyable]
    allb = evd[evd.buyable]
    evo = dict(independent_buyable=evsum(ind), all_buyable=evsum(allb),
               n_indep_total=int(evd.indep.sum()), n_indep_not_buyable=int((evd.indep & ~evd.buyable).sum()),
               independent_by_year={y: dict(n=int(len(g)), exc20_mean=fl(g.exc20.mean()), oo20_exc_mean=fl(g.oo20_exc.mean()))
                                    for y, g in ind.groupby("year")},
               independent_by_period={nm: evsum(ind[(ind.a > a0) & (ind.a <= a1_)]) for nm, (a0, a1_) in
                                      (("2015_2018", (t0, i19)), ("2019_2023", (i19, i24)), ("2024_", (i24, e)))})
    # 同日 RS 分 3 組（事件層，全部可買訊號）
    def terc(df):
        g = df.groupby("t")
        r = g.rs.rank(ascending=False, method="first"); n = g.rs.transform("count")
        grp = np.where(n < 3, -1, np.minimum((r - 1) * 3 // n, 2))
        return {str(k): dsum(df.exc20.values[grp == k]) for k in (0, 1, 2)}
    evo["rs_tercile_same_day_exc20_all_buyable"] = terc(allb)
    out["event_level"] = evo
    print("event", {k: (v["mean"], v["t"]) for k, v in evo["independent_buyable"].items() if isinstance(v, dict) and "mean" in v},
          flush=True)

    # ---------------- 審查 B
    revB = {}
    for nm, df in (("independent_buyable", ind), ("all_buyable", allb)):
        revB[nm] = dict(
            entry_open_next_exit_open20=dsum(df.oo20_exc.values),
            entry_close_signal_exit_open20=dsum(df.B_close_t_exc.values),
            entry_close_next_exit_open20=dsum(df.B_close_a_exc.values),
            close_to_close20_from_signal_close=dsum(df.B_cc_t_exc.values),
            close_to_close20_from_next_close=dsum(df.B_cc_a_exc.values),
            gap_exc=dsum(df.gap_exc.values),
            day1_open_to_close_exc=dsum(df.exc1.values))
        sub = df.dropna(subset=["oo20_exc", "B_close_t_exc"])
        revB[nm]["gap_cost_same_exit_mean"] = fl((sub.B_close_t_exc - sub.oo20_exc).mean())
        revB[nm]["gap_cost_same_exit_t"] = K.tstat((sub.B_close_t_exc - sub.oo20_exc).values)
    revB["note"] = ("事後描述：同一出場（進場後第 20 個交易日開盤，與 P 的出場相同），進場價分別用訊號日收盤（不可能成交，上界）、"
                    "次日開盤（實際）、次日收盤；0050 用同一時點。另報收盤到第 20 個交易日收盤的 20 日窗。扣成本前。")
    out["reviewB"] = revB
    print("reviewB", {k: (v["entry_open_next_exit_open20"]["mean"], v["entry_close_signal_exit_open20"]["mean"],
                          v["entry_close_next_exit_open20"]["mean"]) for k, v in revB.items() if isinstance(v, dict)},
          flush=True)
    evd[["code", "t", "a", "indep", "buyable", "rs", "gap_exc", "exc1", "exc20", "oo20_exc"]].assign(
        sig_date=lambda d: [P.dstr[x] for x in d.t]).drop(columns=["t", "a"]).to_csv(
        os.path.join(odir, "my_events.csv.gz"), index=False, float_format="%.5f", compression="gzip")
    del evd, ev

    # ---------------- 審查 A：排名池只含上市滿 121 根
    revA = dict(signals=out["reviewA_signals"])
    xa = run("P", plans_override=plansPA[0][0])
    revA["P_main"] = nav_stats(xa["nav"], int(len(xa["trades"])))
    pa = []; ca = []
    for sh in SHIFTS:
        for cp in CAPS:
            nP = run("P", sh, cp, plans_override=plansPA[sh][0])["nav"]
            nPh = run("P", sh, cp / 2, plans_override=plansPA[sh][0])["nav"]
            pa.append(dict(ann_2019=L.ann(P, nP, i19, e), mdd=L.maxdd(P, nP, t0, e)[0],
                           base_2019=L.ann(P, path_nav[("P", sh, cp)], i19, e)))
            nc = single_nav("E0", sh, cp / 2)[0] + nPh
            c0 = path_nav[("C0", sh, cp)]
            ca.append(dict(d2019=L.ann(P, nc, i19, e) - L.ann(P, c0, i19, e),
                           d1923=L.ann(P, nc, i19, i24) - L.ann(P, c0, i19, i24),
                           d2024=L.ann(P, nc, i24, e) - L.ann(P, c0, i24, e)))
    pa = pd.DataFrame(pa); ca = pd.DataFrame(ca)
    revA["paths20_P"] = dict(ann_2019_median=float(pa.ann_2019.median()), n_beat_0050=int((pa.ann_2019 > b50["ann_2019"]).sum()),
                             mdd_median=float(pa.mdd.median()),
                             median_change_vs_main=float((pa.ann_2019 - pa.base_2019).median()),
                             max_abs_change=float((pa.ann_2019 - pa.base_2019).abs().max()))
    revA["paths20_C_minus_C0"] = dict(n_pos=int((ca.d2019 > 0).sum()), median=float(ca.d2019.median()),
                                      median_1923=float(ca.d1923.median()), median_2024=float(ca.d2024.median()))
    out["reviewA"] = revA
    print("reviewA", revA, flush=True)

    # ---------------- 審查 C：同日隨機排序
    revC = dict(seeds=[1, 2, 3, 4, 5], runs=[])
    for sd in revC["seeds"]:
        b, _ = K.pb_plans(P, sig, rs, U, shift=0, order="random", seed=sd)
        x = run("P", plans_override=b)
        revC["runs"].append(dict(seed=sd, **nav_stats(x["nav"], int(len(x["trades"])))))
    df = pd.DataFrame(revC["runs"])
    revC["summary"] = dict(ann_2019_mean=float(df.ann_2019.mean()), ann_2019_min=float(df.ann_2019.min()),
                           ann_2019_max=float(df.ann_2019.max()), ann_full_mean=float(df.ann_full.mean()),
                           mdd_mean=float(df.mdd_full.mean()),
                           main_minus_random_mean_2019=float(cells["P"]["ann_2019"] - df.ann_2019.mean()),
                           main_minus_random_mean_full=float(cells["P"]["ann_full"] - df.ann_full.mean()))
    out["reviewC"] = revC
    print("reviewC", revC["summary"], flush=True)

    out["posthoc"] = dict(yearly={k: cells[k]["yearly"] for k in cells}, yearly_0050=b50["yearly"])
    out["resources"] = dict(runtime_s=round(time.time() - t_start, 1), maxrss_mb=int(rss()))
    s = json.dumps(out, ensure_ascii=False, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x))
    with open(os.path.join(odir, "my_numbers.json"), "w") as f:
        f.write(s)
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    with open(os.path.join(odir, "my_numbers.sha256"), "w") as f:
        f.write(h + "  my_numbers.json\n")
    print("frozen", h, "rss", rss(), round(time.time() - t_start, 1), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
