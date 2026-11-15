# -*- coding: utf-8 -*-
"""S9 獨立複製：驗收、36 個組合（9 定義 × 開盤／收盤 × 兩格）主跑與 20 條路徑、家族統計、審查 A／B／C，然後凍結。
用法：python3 -I -B rep_s9_run.py <scratch_dir>
需先跑 s1 的 rep_s1_panel.py <scratch_dir> 與 s3 的 rep_s3_names.py <scratch_dir>/innov_names.json。
輸出（本資料夾）：my_numbers.json、my_numbers.sha256、my_trades.csv.gz
凍結前不讀研究員的任何輸出（FINDINGS 第 3、11 節以外、results*.json、trades.csv.gz、navs.csv.gz）。
"""
import sys, os, json, hashlib, resource, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import rep_s9_lib as K
C, S, Q, R, L = K.C, K.S, K.Q, K.R, K.L
ST, X2 = K.ST, K.X2

CELLS = {"W10_70": "W10", "WN_70": "WN"}
AFAM = ["A08", "A09", "A10", "A11", "A12", "A13", "A15"]
A2FAM = ["A10", "A11", "A12", "A13", "A15"]
CLOSE_REQ = ["A10", "A11", "A13", "A15"]
DEFS = list(K.DEFS)
CAPS = [600_000.0, 800_000.0, 1_000_000.0, 1_500_000.0, 2_000_000.0]
SHIFTS = [0, 1, 2, 3]
KNOWN = {"W10_70": (0.4814, 0.3409), "WN_70": (0.6216, 0.4450)}   # rep_s6 複製員已知值（2019 起／全期）


def rss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024


def ols_slope(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    xm = x - x.mean()
    return float((xm * (y - y.mean())).sum() / (xm ** 2).sum())


def main(sc, odir=HERE):
    t_start = time.time()
    P = R.build(sc)
    S.attach_e50(P, sc)
    K.attach_e50_close(P, sc)
    excl = S.innov_mask_fn(P, sc)
    Rv, mi0 = L.load_rev(P)
    S_, _ = L.surge(Rv, mi0, null2013=True)
    nm = S_.shape[0]
    e = P.T - 1; i19 = L.didx(P, "2018-12-28"); i24 = L.didx(P, "2023-12-29"); ih = L.didx(P, "2021-02-19")
    e50 = P.e50.values
    out = dict(meta=dict(T=P.T, N=P.N, first=P.dstr[0], last=P.dstr[-1], t0=P.dstr[P.t0], i19=P.dstr[i19],
                         i24=P.dstr[i24], ih=P.dstr[ih],
                         note="面板為複製員自建（N＝2130，研究員 2142；S6～S8 複製時已知，差的是從未進股池的代號）"))
    b50 = dict(ann_2019=L.ann(P, e50, i19, e), ann_full=L.ann(P, e50, P.t0, e), ann_2024=L.ann(P, e50, i24, e),
               mdd_full=L.maxdd(P, e50, P.t0, e), ann_h1=L.ann(P, e50, P.t0, ih), ann_h2=L.ann(P, e50, ih, e),
               yearly=L.yearly(P, e50))
    out["bench_0050"] = b50

    # ---------------- 訊號日日曆
    scheds = {d: K.schedule(P, nm, mi0, d) for d in DEFS}
    _a10l = K.make_lists_sched(P, S_, scheds["A10"], excl=excl)
    _a10m = set(v["m"] for v in _a10l.values())
    kA10 = {x["m"]: x["k"] for x in scheds["A10"] if x["m"] in _a10m}
    cal = {}
    for d in DEFS:
        lag = []; months = []
        for x in scheds[d]:
            if x["m"] in kA10 and P.t0 <= kA10[x["m"]] < P.T - 1 and x["k"] < P.T - 1:
                lag.append(x["k"] - kA10[x["m"]]); months.append(x["ym_pub"])
        lag = np.array(lag)
        cal[d] = dict(n_months=int(len(lag)), same_as_A10=int((lag == 0).sum()), mean_lag=float(lag.mean()),
                      dist={str(int(k)): int(v) for k, v in zip(*np.unique(lag, return_counts=True))},
                      n_rev_M2=int(sum(1 for x in scheds[d] if x["m_rev"] != x["m"] and x["m"] in kA10)))
    early_months = [x["ym_pub"] for x in scheds["B11pes"] if x["m_rev"] != x["m"] and x["m"] in kA10]
    cal["B11_early_months"] = early_months
    out["calendar"] = cal
    print("calendar", {d: (cal[d]["n_months"], cal[d]["same_as_A10"], round(cal[d]["mean_lag"], 2), cal[d]["n_rev_M2"])
                       for d in DEFS}, flush=True)

    # ---------------- 名單（定義 × 成交時點 × 位移）
    plans = {}
    for d in DEFS:
        for ex in ("open", "close"):
            for sh in SHIFTS:
                lst = K.make_lists_sched(P, S_, scheds[d], shift=sh, list_off=(0 if ex == "open" else -1), excl=excl)
                plans[(d, ex, sh)] = K.plans_from_lists(lst)[:2]
    print("plans built", len(plans), "rss", rss(), round(time.time() - t_start), flush=True)

    def run(d, ex, cell, sh=0, capital=1_000_000.0, pl=None, **kw):
        b, k = plans[(d, ex, sh)] if pl is None else pl
        return K.simulate_s9(P, b, k, sat_w=0.7, sizing=CELLS[cell], exec_mode=ex, capital=capital, **{**ST, **kw})

    def brief(res, full=True):
        nav = res["nav"]; tr = res["trades"]
        md = L.maxdd(P, nav, P.t0, e)
        o = dict(ann_2019=L.ann(P, nav, i19, e), ann_full=L.ann(P, nav, P.t0, e), ann_2024=L.ann(P, nav, i24, e),
                 mdd_full=md[0], mdd_peak=md[1], mdd_trough=md[2], n_trades=int(len(tr)))
        if full:
            r = nav[P.t0 + 1:e + 1] / nav[P.t0:e] - 1.0
            o.update(sharpe=float(r.mean() / r.std(ddof=1) * np.sqrt(252)), exp_per_trade=float(tr.ret.mean()),
                     ann_h1=L.ann(P, nav, P.t0, ih), ann_h2=L.ann(P, nav, ih, e), yearly=L.yearly(P, nav),
                     limit_up_skips=int(res["cnt"]["limit_up"]), buys=int(res["cnt"]["buy"]),
                     sell_defer=int(res["cnt"]["sell_defer"]), sell_defer_limdn=int(res["cnt"]["sell_defer_limdn"]),
                     blk_min=int(res["cnt"]["blk_min"]), a3close_viol=len(res["a3close"]))
        return o

    # ---------------- 驗收
    acc = {}
    base_lists = Q.make_lists(P, S_, mi0, excl=excl)
    b0, k0 = Q.cell_plans(base_lists, "C10", x=0.10)
    acc["A1_sched_eq_rev_avail"] = bool(all(L.rev_avail(P, mi0 + x["m"])[2] == x["k"] for x in scheds["A10"]))
    acc["A1_lists_eq"] = bool(sorted(plans[("A10", "open", 0)][0]) == sorted(b0) and all(
        (plans[("A10", "open", 0)][0][s] == b0[s]).all() and plans[("A10", "open", 0)][1][s] == k0[s] for s in b0))
    res_main = {}
    for cell, sz in CELLS.items():
        a = K.K6.simulate_s6(P, b0, k0, sat_w=0.7, sizing=sz, **ST)
        c = run("A10", "open", cell)
        ta = a["trades"].sort_values(["ts", "code"])[["code", "tb", "ts", "sh"]].values
        tc = c["trades"].sort_values(["ts", "code"])[["code", "tb", "ts", "sh"]].values
        y19, yf = L.ann(P, c["nav"], i19, e), L.ann(P, c["nav"], P.t0, e)
        acc["A0A2_" + cell] = dict(max_abs_nav_diff_vs_rep_s6=float(np.nanmax(np.abs(a["nav"] - c["nav"]))),
                                   trades_identical=bool(len(ta) == len(tc) and (ta == tc).all()),
                                   ann_2019=y19, ann_full=yf, mdd_full=L.maxdd(P, c["nav"], P.t0, e)[0],
                                   known=KNOWN[cell],
                                   ok=bool(abs(y19 - KNOWN[cell][0]) < 0.00006 and abs(yf - KNOWN[cell][1]) < 0.00006))
        res_main[("A10", "open", cell)] = c
    # A3：收盤版名單列＝成交列−1、成交價＝收盤×(1＋成本)
    a3 = {}
    for d in CLOSE_REQ:
        bo = plans[(d, "open", 0)][0]; bc = plans[(d, "close", 0)][0]
        so = sorted(bo); scl = sorted(bc)
        a3[d] = dict(n_open=len(so), n_close=len(scl),
                     list_row_eq_k_minus_1=bool(set(scl) == set(s - 1 for s in so)))
    acc["A3_close_rows"] = a3
    out["acceptance"] = acc
    print("acc", json.dumps(acc, ensure_ascii=False), flush=True)
    assert acc["A0A2_W10_70"]["ok"] and acc["A0A2_WN_70"]["ok"] and acc["A1_lists_eq"]

    # ---------------- 36 個組合主跑＋八項檢查
    combos = {}
    for d in DEFS:
        for ex in ("open", "close"):
            for cell in CELLS:
                key = (d, ex, cell)
                if key not in res_main:
                    res_main[key] = run(d, ex, cell)
                o = brief(res_main[key])
                assert o["a3close_viol"] == 0
                x2 = brief(run(d, ex, cell, **X2), full=False)
                sp, top = R.span_of(res_main[key]["trades"], 5)
                bl = brief(run(d, ex, cell, block_span=sp), full=False)
                checks = dict(a2019=o["ann_2019"] > b50["ann_2019"], afull=o["ann_full"] > b50["ann_full"],
                              x2_2019=x2["ann_2019"] > b50["ann_2019"], x2_full=x2["ann_full"] > b50["ann_full"],
                              bl_2019=bl["ann_2019"] > b50["ann_2019"], bl_full=bl["ann_full"] > b50["ann_full"],
                              h1=o["ann_h1"] > b50["ann_h1"], h2=o["ann_h2"] > b50["ann_h2"])
                o["x2"] = {k: x2[k] for k in ("ann_2019", "ann_full", "mdd_full")}
                o["block5"] = {k: bl[k] for k in ("ann_2019", "ann_full", "mdd_full")}
                o["block5_top"] = top
                o["checks"] = checks
                o["cls"] = "值得上 XQ 測" if all(checks.values()) else (
                    "不值得" if (not checks["a2019"] and not checks["afull"]) else "有條件")
                combos["%s|%s|%s" % key] = o
                print(key, round(o["ann_2019"], 4), round(o["ann_full"], 4), round(o["ann_2024"], 4),
                      round(o["mdd_full"], 4), o["n_trades"], o["limit_up_skips"], o["cls"], flush=True)
    out["combos"] = combos
    print("main done rss", rss(), round(time.time() - t_start), flush=True)

    # ---------------- 20 條路徑
    rows = []
    for d in DEFS:
        for ex in ("open", "close"):
            for cell in CELLS:
                for sh in SHIFTS:
                    for cp in CAPS:
                        if sh == 0 and cp == 1_000_000.0:
                            x = res_main[(d, ex, cell)]
                        else:
                            x = run(d, ex, cell, sh=sh, capital=cp)
                        md = L.maxdd(P, x["nav"], P.t0, e)
                        yr = L.yearly(P, x["nav"])
                        rows.append(dict(d=d, ex=ex, cell=cell, shift=sh, capital=cp,
                                         ann_2019=L.ann(P, x["nav"], i19, e), ann_full=L.ann(P, x["nav"], P.t0, e),
                                         mdd_full=md[0], mdd_peak=md[1], mdd_trough=md[2],
                                         n_trades=int(len(x["trades"])), limit_up=int(x["cnt"]["limit_up"]),
                                         **{"y%d" % k: v for k, v in yr.items()}))
        print("paths", d, "rss", rss(), round(time.time() - t_start), flush=True)
    pr = pd.DataFrame(rows)
    pr["dd_2024_25"] = (pr.mdd_peak >= "2024-07-01") & (pr.mdd_peak <= "2024-07-31") & \
                       (pr.mdd_trough >= "2025-04-01") & (pr.mdd_trough <= "2025-04-30")
    p20 = {}
    for (d, ex, cell), g in pr.groupby(["d", "ex", "cell"]):
        p20["%s|%s|%s" % (d, ex, cell)] = dict(
            ann_2019_min=float(g.ann_2019.min()), ann_2019_median=float(g.ann_2019.median()),
            ann_2019_max=float(g.ann_2019.max()), n_beat_0050=int((g.ann_2019 > b50["ann_2019"]).sum()),
            mdd_median=float(g.mdd_full.median()), mdd_min=float(g.mdd_full.min()), mdd_max=float(g.mdd_full.max()),
            n_dd_2024_25=int(g.dd_2024_25.sum()),
            ret_median_by_shift={int(s): float(v.ann_2019.median()) for s, v in g.groupby("shift")},
            mdd_median_by_shift={int(s): float(v.mdd_full.median()) for s, v in g.groupby("shift")},
            ann_full_median=float(g.ann_full.median()))
    out["paths20"] = p20

    # ---------------- 家族統計
    fam = {}
    for cell in CELLS:
        for ex in ("open", "close"):
            m = {d: combos["%s|%s|%s" % (d, ex, cell)] for d in AFAM}
            r19 = pd.Series({d: m[d]["ann_2019"] for d in AFAM})
            dd = pd.Series({d: m[d]["mdd_full"] for d in AFAM})
            g140 = pr[(pr.cell == cell) & (pr.ex == ex) & pr.d.isin(AFAM)]
            g100 = pr[(pr.cell == cell) & (pr.ex == ex) & pr.d.isin(A2FAM)]
            a10 = r19["A10"]
            fam["%s|%s" % (cell, ex)] = dict(
                ann_2019=r19.to_dict(), mdd_full=dd.to_dict(),
                A10_rank_ret_7=int((r19 > a10).sum() + 1), A10_rank_dd_7=int((dd > dd["A10"]).sum() + 1),
                A10_rank_ret_5=int((r19[A2FAM] > a10).sum() + 1), A10_rank_dd_5=int((dd[A2FAM] > dd["A10"]).sum() + 1),
                fam_median=float(r19.median()), fam_min=float(r19.min()),
                A10_minus_median_pp=float((a10 - r19.median()) * 100), A10_minus_min_pp=float((a10 - r19.min()) * 100),
                fam5_median=float(r19[A2FAM].median()), A10_minus_fam5_median_pp=float((a10 - r19[A2FAM].median()) * 100),
                dd_median=float(dd.median()), dd_deepest=float(dd.min()),
                p140_ret_median=float(g140.ann_2019.median()), p140_ret_p10=float(g140.ann_2019.quantile(0.10)),
                p140_dd_median=float(g140.mdd_full.median()), p140_dd_p10=float(g140.mdd_full.quantile(0.10)),
                p140_n_beat_0050=int((g140.ann_2019 > b50["ann_2019"]).sum()), p140_n=int(len(g140)),
                p100_ret_median=float(g100.ann_2019.median()), p100_ret_p10=float(g100.ann_2019.quantile(0.10)),
                p100_dd_median=float(g100.mdd_full.median()), p100_dd_p10=float(g100.mdd_full.quantile(0.10)),
                p100_n=int(len(g100)),
                close_minus_open_pp=None)
        # 成交時點：close − open
        co = {d: (combos["%s|close|%s" % (d, cell)]["ann_2019"] - combos["%s|open|%s" % (d, cell)]["ann_2019"]) * 100
              for d in DEFS}
        cop = {d: (p20["%s|close|%s" % (d, cell)]["ann_2019_median"] - p20["%s|open|%s" % (d, cell)]["ann_2019_median"]) * 100
               for d in DEFS}
        fam["%s|open" % cell]["close_minus_open_pp"] = co
        fam["%s|open" % cell]["close_minus_open_path_median_pp"] = cop
        fam["%s|open" % cell]["close_minus_open_A7_median_pp"] = float(np.median([co[d] for d in AFAM]))
        fam["%s|open" % cell]["close_minus_open_A7_npos"] = int(sum(co[d] > 0 for d in AFAM))
        fam["%s|open" % cell]["close_minus_open_req4"] = {d: co[d] for d in CLOSE_REQ}
    out["family"] = fam

    # ---------------- 審查 A：B11pes vs B11opt；「10 日假日月份不換股」在 A10 上（事後）
    rva = {}
    early_set = set(early_months)
    for cell in CELLS:
        rp, ro = res_main[("B11pes", "open", cell)], res_main[("B11opt", "open", cell)]
        ra10 = res_main[("A10", "open", cell)]
        o = dict(pes_minus_opt_2019_pp=(L.ann(P, rp["nav"], i19, e) - L.ann(P, ro["nav"], i19, e)) * 100,
                 pes_minus_opt_full_pp=(L.ann(P, rp["nav"], P.t0, e) - L.ann(P, ro["nav"], P.t0, e)) * 100,
                 opt_minus_A10_2019_pp=(L.ann(P, ro["nav"], i19, e) - L.ann(P, ra10["nav"], i19, e)) * 100)
        # 期間拆解：以 B11 換股執行日切段（兩者執行日相同），每段 log 報酬差
        bpes = plans[("B11pes", "open", 0)][0]; bopt = plans[("B11opt", "open", 0)][0]
        o["signals_only_in_opt"] = [P.dstr[s] for s in sorted(set(bopt) - set(bpes))]
        assert set(bpes) <= set(bopt)
        sched_b = {x["k"]: x for x in scheds["B11pes"]}
        ex_days = sorted(s + 1 for s in bopt)
        seg = []
        for a_, b_ in zip(ex_days, ex_days[1:] + [e + 1]):
            s_ = a_ - 1
            lp = np.log(rp["nav"][b_ - 1] / rp["nav"][a_ - 1]); lo = np.log(ro["nav"][b_ - 1] / ro["nav"][a_ - 1])
            seg.append(dict(signal=P.dstr[s_], ym=sched_b[s_]["ym_pub"], early=bool(sched_b[s_]["ym_pub"] in early_set),
                            dlog=float(lp - lo), after2019=bool(s_ >= i19)))
        sg = pd.DataFrame(seg)
        g19 = sg[sg.after2019]
        o["seg_dlog_sum_2019"] = float(g19.dlog.sum())
        o["seg_dlog_early_2019"] = float(g19[g19.early].dlog.sum())
        o["seg_dlog_nonearly_2019"] = float(g19[~g19.early].dlog.sum())
        o["seg_n_early_2019"] = int(g19.early.sum())
        o["seg_early_pos"] = int((g19[g19.early].dlog > 0).sum())
        o["seg_top10"] = sg.reindex(sg.dlog.abs().sort_values(ascending=False).index).head(10).to_dict("records")
        o["seg_dlog_by_year"] = {int(y): float(v) for y, v in sg.groupby(sg.signal.str[:4].astype(int)).dlog.sum().items()}
        # 個股
        spp, spo = K.stock_pnl(rp), K.stock_pnl(ro)
        dif = {c: spp.get(c, 0.0) - spo.get(c, 0.0) for c in set(spp) | set(spo)}
        o["stock_diff_top"] = sorted(dif.items(), key=lambda kv: -kv[1])[:8]
        o["stock_diff_bottom"] = sorted(dif.items(), key=lambda kv: kv[1])[:8]
        o["stock_diff_sum"] = float(sum(dif.values()))
        # 早月：名單與上月重疊、換手
        ov = []
        sl = sorted(bpes)
        for n_, s_ in enumerate(sl):
            if sched_b[s_]["ym_pub"] in early_set and n_ > 0:
                prv = set(int(i) for i in bpes[sl[n_ - 1]])
                ov.append(dict(pes_overlap_prev=len(set(int(i) for i in bpes[s_]) & prv), pes_n=len(bpes[s_]),
                               opt_overlap_prev=len(set(int(i) for i in bopt[s_]) & prv), opt_n=len(bopt[s_])))
        ov = pd.DataFrame(ov)
        o["early_list_overlap_prev"] = dict(pes_mean_overlap=float(ov.pes_overlap_prev.mean()),
                                            pes_mean_n=float(ov.pes_n.mean()),
                                            opt_mean_overlap=float(ov.opt_overlap_prev.mean()),
                                            opt_mean_n=float(ov.opt_n.mean()), n=int(len(ov)))
        tp, to = rp["trades"], ro["trades"]
        exd_early = set(s + 1 for s in bpes if sched_b[s]["ym_pub"] in early_set)
        o["buys_on_early_exec"] = dict(pes=int(tp.tb.isin(exd_early).sum()), opt=int(to.tb.isin(exd_early).sum()))
        o["sells_after_early_signal"] = dict(pes=int(tp.ts.isin(exd_early).sum()), opt=int(to.ts.isin(exd_early).sum()))
        o["n_trades"] = dict(pes=int(len(tp)), opt=int(len(to)))
        rva[cell] = o

    # 變體（事後）：B11skip（早月不換股、不出場）、A10skip（同 49 個月不換股）、A10pes（同 49 個月用 M−2）
    def drop_months(pl, months_sched, ymset):
        b, k = pl
        drop = set(s for s in b if months_sched[s] in ymset)
        return {s: v for s, v in b.items() if s not in drop}, {s: v for s, v in k.items() if s not in drop}, len(drop)

    sched_A10 = scheds["A10"]
    ten_holiday = set()  # 10 日不是交易日的公布月（≡ B11 早月）
    da = np.array(P.dstr)
    for x in sched_A10:
        y, mo = map(int, x["ym_pub"].split("-"))
        if P.dstr[x["k"] - 1] != "%04d-%02d-10" % (y, mo):
            ten_holiday.add(x["ym_pub"])
    rva["ten_holiday_eq_B11_early"] = bool(ten_holiday & set(early_months) == set(early_months))
    a10pes_sched = [dict(x, m_rev=(x["m"] - 1 if x["ym_pub"] in ten_holiday else x["m"])) for x in sched_A10]
    var = {}
    for sh in SHIFTS:
        for nm_, base_d in (("A10skip", "A10"), ("B11skip", "B11opt")):
            lst = K.make_lists_sched(P, S_, scheds[base_d], shift=sh, excl=excl)
            msch = {s: v["ym_pub"] for s, v in lst.items()}
            b, k, nd = drop_months(K.plans_from_lists(lst)[:2], msch, early_set)
            plans[(nm_, "open", sh)] = (b, k)
            var.setdefault(nm_, {})["n_dropped_s%d" % sh] = nd
        lst = K.make_lists_sched(P, S_, a10pes_sched, shift=sh, excl=excl)
        plans[("A10pes", "open", sh)] = K.plans_from_lists(lst)[:2]
    vrows = []
    for nm_ in ("A10skip", "B11skip", "A10pes", "A10", "B11opt", "B11pes"):
        for cell in CELLS:
            for sh in SHIFTS:
                for cp in CAPS:
                    x = run(nm_, "open", cell, sh=sh, capital=cp)
                    vrows.append(dict(v=nm_, cell=cell, shift=sh, capital=cp, ann_2019=L.ann(P, x["nav"], i19, e),
                                      ann_full=L.ann(P, x["nav"], P.t0, e), mdd_full=L.maxdd(P, x["nav"], P.t0, e)[0]))
                    if sh == 0 and cp == 1_000_000.0:
                        var.setdefault(nm_, {})[cell] = dict(ann_2019=vrows[-1]["ann_2019"], ann_full=vrows[-1]["ann_full"],
                                                             mdd_full=vrows[-1]["mdd_full"], n_trades=int(len(x["trades"])),
                                                             ann_2024=L.ann(P, x["nav"], i24, e))
    vr = pd.DataFrame(vrows)
    for nm_ in ("A10skip", "B11skip", "A10pes", "A10", "B11opt", "B11pes"):
        for cell in CELLS:
            g = vr[(vr.v == nm_) & (vr.cell == cell)]
            var[nm_][cell].update(path_ret_median=float(g.ann_2019.median()), path_dd_median=float(g.mdd_full.median()),
                                  path_ret_min=float(g.ann_2019.min()), path_ret_max=float(g.ann_2019.max()))
    # 逐條路徑配對：A10skip − A10、B11pes − B11opt
    pairs = {}
    for a_, b_ in (("A10skip", "A10"), ("B11pes", "B11opt"), ("B11skip", "B11pes"), ("A10pes", "A10")):
        for cell in CELLS:
            ga = vr[(vr.v == a_) & (vr.cell == cell)].sort_values(["shift", "capital"]).ann_2019.values
            gb = vr[(vr.v == b_) & (vr.cell == cell)].sort_values(["shift", "capital"]).ann_2019.values
            dd_ = (ga - gb) * 100
            pairs["%s-%s|%s" % (a_, b_, cell)] = dict(median_pp=float(np.median(dd_)), min_pp=float(dd_.min()),
                                                      max_pp=float(dd_.max()), n_pos=int((dd_ > 0).sum()),
                                                      by_shift_median_pp={int(s): float(np.median(dd_[s * 5:(s + 1) * 5]))
                                                                          for s in SHIFTS})
    var["pairs_20paths"] = pairs
    # 事後：隨機挑 49 個月不換股（30 組亂數，種子固定）對照「10 日假日的 49 個月」；零成本下 A10skip − A10
    rng = np.random.default_rng(20261115)
    bA, kA = plans[("A10", "open", 0)]
    lstA = K.make_lists_sched(P, S_, scheds["A10"], excl=excl)
    mA = {s_: v["ym_pub"] for s_, v in lstA.items()}
    allm = sorted(set(mA.values()))
    n_early = len(early_set & set(allm))
    rnd = {c: [] for c in CELLS}
    for r_ in range(30):
        pick = set(rng.choice(allm, size=n_early, replace=False).tolist())
        b_, k_, _ = drop_months((bA, kA), mA, pick)
        for cell in CELLS:
            x = run("A10", "open", cell, pl=(b_, k_))
            rnd[cell].append(L.ann(P, x["nav"], i19, e))
    zc = {}
    for cell in CELLS:
        a0 = combos["A10|open|%s" % cell]["ann_2019"]
        hs_ = var["A10skip"][cell]["ann_2019"]
        v_ = np.array(rnd[cell])
        x0 = run("A10", "open", cell, cmult=0.0, extra=0.0); x1 = run("A10skip", "open", cell, cmult=0.0, extra=0.0)
        zc[cell] = dict(A10=L.ann(P, x0["nav"], i19, e), A10skip=L.ann(P, x1["nav"], i19, e),
                        diff_pp=(L.ann(P, x1["nav"], i19, e) - L.ann(P, x0["nav"], i19, e)) * 100,
                        strict_diff_pp=(hs_ - a0) * 100)
        var.setdefault("random_skip49", {})[cell] = dict(
            n_skip=n_early, draws=30, median_minus_A10_pp=float((np.median(v_) - a0) * 100),
            p10_minus_A10_pp=float((np.quantile(v_, 0.1) - a0) * 100), p90_minus_A10_pp=float((np.quantile(v_, 0.9) - a0) * 100),
            n_beat_A10=int((v_ > a0).sum()), n_beat_holiday_skip=int((v_ >= hs_).sum()),
            holiday_skip_minus_A10_pp=float((hs_ - a0) * 100))
    var["zero_cost_A10skip"] = zc
    # 事後：49 個月的月份分布；只跳 10 月（國慶）／只跳其他月；逐月單獨跳過的效果
    em = sorted(early_set & set(allm))
    var["early_by_calmonth"] = {str(k_): int(v_) for k_, v_ in pd.Series([x[5:] for x in em]).value_counts().sort_index().items()}
    var["early_months_list"] = em
    sub = {}
    for nm_, ms in (("skip_oct_only", set(x for x in em if x.endswith("-10"))),
                    ("skip_non_oct", set(x for x in em if not x.endswith("-10"))),
                    ("skip_2019on", set(x for x in em if x >= "2019"))):
        b_, k_, nd = drop_months((bA, kA), mA, ms)
        sub[nm_] = {"n": nd}
        for cell in CELLS:
            sub[nm_][cell] = (L.ann(P, run("A10", "open", cell, pl=(b_, k_))["nav"], i19, e) -
                              combos["A10|open|%s" % cell]["ann_2019"]) * 100
    var["skip_subsets_pp"] = sub
    one = {c: {} for c in CELLS}
    for ym in em:
        b_, k_, _ = drop_months((bA, kA), mA, {ym})
        for cell in CELLS:
            one[cell][ym] = (L.ann(P, run("A10", "open", cell, pl=(b_, k_))["nav"], i19, e) -
                             combos["A10|open|%s" % cell]["ann_2019"]) * 100
    var["skip_one_month_pp"] = {c: dict(sum=float(sum(v_.values())), n_pos=int(sum(x > 0 for x in v_.values())),
                                        top=sorted(v_.items(), key=lambda kv: -kv[1])[:8],
                                        bottom=sorted(v_.items(), key=lambda kv: kv[1])[:5]) for c, v_ in one.items()}
    rva["variants_posthoc"] = var
    out["reviewA"] = rva
    print("reviewA", json.dumps({c: {k: rva[c][k] for k in ("pes_minus_opt_2019_pp", "seg_dlog_early_2019",
                                                             "seg_dlog_nonearly_2019")} for c in CELLS}), flush=True)
    print("variants", json.dumps({k: v for k, v in var.items() if k != "pairs_20paths"}, ensure_ascii=False)[:2000], flush=True)
    print("pairs", json.dumps(pairs), flush=True)

    # ---------------- 審查 B：時效遞減斜率（20 條路徑）、A10→A11 損失的年份
    lagm = {d: cal[d]["mean_lag"] for d in A2FAM}
    rvb = {}
    for cell in CELLS:
        g = pr[(pr.cell == cell) & (pr.ex == "open") & pr.d.isin(A2FAM)]
        slopes = []
        for (sh, cp), h in g.groupby(["shift", "capital"]):
            h = h.set_index("d").loc[A2FAM]
            slopes.append(dict(shift=int(sh), capital=cp, slope_pp=ols_slope([lagm[d] for d in A2FAM], h.ann_2019) * 100,
                               dd_slope_pp=ols_slope([lagm[d] for d in A2FAM], h.mdd_full) * 100,
                               a10_minus_a11_pp=float((h.loc["A10", "ann_2019"] - h.loc["A11", "ann_2019"]) * 100)))
        sl = pd.DataFrame(slopes)
        main_sl = ols_slope([lagm[d] for d in A2FAM], [combos["%s|open|%s" % (d, cell)]["ann_2019"] for d in A2FAM]) * 100
        # 全部 100 條：有效延遲＝定義平均延遲＋位移
        effl = g.d.map(lagm).values + g["shift"].values
        all_sl = ols_slope(effl, g.ann_2019.values) * 100
        # 年份拆解：A10 vs A11（主跑與 20 條配對）
        def yr_share(ya, yb):
            yrs = [y for y in range(2019, 2027)]
            dl = {y: np.log1p(ya["y%d" % y]) - np.log1p(yb["y%d" % y]) for y in yrs}
            tot = sum(dl.values())
            big = dl[2020] + dl[2024] + dl[2025]
            return dl, tot, big
        ga = g[g.d == "A10"].sort_values(["shift", "capital"]).reset_index(drop=True)
        gb = g[g.d == "A11"].sort_values(["shift", "capital"]).reset_index(drop=True)
        shares = []; dls = []
        for n_ in range(len(ga)):
            dl, tot, big = yr_share(ga.loc[n_], gb.loc[n_])
            shares.append(dict(shift=int(ga.loc[n_, "shift"]), capital=ga.loc[n_, "capital"], tot_dlog=tot, big3_dlog=big,
                               share=(big / tot if abs(tot) > 1e-12 else None)))
            dls.append(dl)
        shd = pd.DataFrame(shares)
        m0 = shd[(shd["shift"] == 0) & (shd.capital == 1_000_000.0)].iloc[0]
        yrs_span = (P.dates[e] - P.dates[i19]).days / 365.25
        rvb[cell] = dict(
            mean_lag=lagm, main_slope_pp_per_day=main_sl, all100_slope_pp_per_effday=all_sl,
            path_slopes=dict(median=float(sl.slope_pp.median()), min=float(sl.slope_pp.min()), max=float(sl.slope_pp.max()),
                             n_neg=int((sl.slope_pp < 0).sum()), n=int(len(sl)),
                             by_shift_median={int(s): float(v.slope_pp.median()) for s, v in sl.groupby("shift")}),
            path_dd_slopes=dict(median=float(sl.dd_slope_pp.median()), min=float(sl.dd_slope_pp.min()),
                                max=float(sl.dd_slope_pp.max())),
            a10_minus_a11=dict(main_pp=float(sl[(sl["shift"] == 0) & (sl.capital == 1_000_000.0)].a10_minus_a11_pp.iloc[0]),
                               path_median_pp=float(sl.a10_minus_a11_pp.median()), path_min_pp=float(sl.a10_minus_a11_pp.min()),
                               path_max_pp=float(sl.a10_minus_a11_pp.max()), n_pos=int((sl.a10_minus_a11_pp > 0).sum())),
            a10_a11_year_dlog_main={int(y): float(v) for y, v in dls[int(m0.name)].items()},
            big3_share_main=float(m0["share"]), big3_share_path_median=float(shd.share.median()),
            big3_share_path_min=float(shd.share.min()), big3_share_path_max=float(shd.share.max()),
            ann_gap_main_excl_big3_pp=float((np.exp((m0["tot_dlog"] - m0["big3_dlog"]) / (yrs_span - 3.0)) - 1.0) * 100),
            note="share＝2020＋2024＋2025 三年 log 報酬差 ÷ 2019～2026 全部 log 報酬差；ann_gap_main_excl_big3＝扣掉三年後其餘年份每年平均差（約略）")
    out["reviewB"] = rvb
    print("reviewB", json.dumps({c: {k: rvb[c][k] for k in ("main_slope_pp_per_day", "path_slopes", "a10_minus_a11",
                                                             "big3_share_main", "big3_share_path_median")} for c in CELLS}), flush=True)

    # ---------------- 審查 C：家族重估值拆成「選股（同一訊號日、不同資金／位移）」與「時效（不同訊號日）」
    rvc = {}
    for cell in CELLS:
        g = pr[(pr.cell == cell) & (pr.ex == "open")]
        a10main = combos["A10|open|%s" % cell]["ann_2019"]
        a10_s0 = g[(g.d == "A10") & (g["shift"] == 0)].ann_2019.median()
        a10_20 = g[g.d == "A10"].ann_2019.median()
        f100 = g[g.d.isin(A2FAM)].ann_2019.median()
        f140 = g[g.d.isin(AFAM)].ann_2019.median()
        later80 = g[g.d.isin(["A11", "A12", "A13", "A15"])].ann_2019.median()
        tab = []
        for d in A2FAM:
            for sh in SHIFTS:
                h = g[(g.d == d) & (g["shift"] == sh)]
                tab.append(dict(d=d, shift=sh, eff_lag=lagm[d] + sh, ret_median=float(h.ann_2019.median()),
                                dd_median=float(h.mdd_full.median())))
        tb = pd.DataFrame(tab)
        # A10 位移 s 對照「平均延遲最接近 s 的定義」位移 0
        match = {}
        for sh in (1, 2, 3):
            dnear = min(["A11", "A12", "A13", "A15"], key=lambda d: abs(lagm[d] - sh))
            match[sh] = dict(A10_shift=float(tb[(tb.d == "A10") & (tb["shift"] == sh)].ret_median.iloc[0]),
                             nearest_def=dnear, nearest_def_lag=lagm[dnear],
                             nearest_def_shift0=float(tb[(tb.d == dnear) & (tb["shift"] == 0)].ret_median.iloc[0]))
        # 迴歸：100 條 ann_2019 ~ 有效延遲；殘差依定義
        effl = g[g.d.isin(A2FAM)].d.map(lagm).values + g[g.d.isin(A2FAM)]["shift"].values
        y = g[g.d.isin(A2FAM)].ann_2019.values
        b1 = ols_slope(effl, y); b0_ = y.mean() - b1 * effl.mean()
        resid = y - (b0_ + b1 * effl)
        rd = pd.Series(resid, index=g[g.d.isin(A2FAM)].d.values).groupby(level=0).mean()
        r2 = 1 - (resid ** 2).sum() / ((y - y.mean()) ** 2).sum()
        # 只用 A10 的 20 條估「同訊號日、位移」斜率，再預測 A11～A15
        ga = g[g.d == "A10"]
        bA = ols_slope(ga["shift"].values, ga.ann_2019.values); aA = ga.ann_2019.mean() - bA * ga["shift"].mean()
        pred = {d: float(aA + bA * lagm[d]) for d in A2FAM}
        actual = {d: float(g[(g.d == d) & (g["shift"] == 0)].ann_2019.median()) for d in A2FAM}
        rvc[cell] = dict(
            A10_main=a10main, A10_shift0_5caps_median=float(a10_s0), A10_20paths_median=float(a10_20),
            fam100_median=float(f100), fam140_median=float(f140), later4_80paths_median=float(later80),
            gap_total_pp=float((a10main - f100) * 100),
            gap_capital_pp=float((a10main - a10_s0) * 100),
            gap_shift_same_day_pp=float((a10_s0 - a10_20) * 100),
            gap_other_defs_pp=float((a10_20 - f100) * 100),
            table=tb.to_dict("records"), A10shift_vs_def=match,
            reg_all100=dict(slope_pp_per_day=b1 * 100, r2=float(r2), resid_by_def_pp={k: float(v * 100) for k, v in rd.items()}),
            A10_only_shift_slope_pp=bA * 100,
            A10_shift_model_pred_vs_actual_shift0_pp={d: dict(pred=pred[d], actual=actual[d],
                                                               diff_pp=(actual[d] - pred[d]) * 100) for d in A2FAM})
    out["reviewC"] = rvc
    print("reviewC", json.dumps({c: {k: rvc[c][k] for k in ("A10_main", "A10_shift0_5caps_median", "A10_20paths_median",
                                                             "fam100_median", "gap_total_pp", "gap_capital_pp",
                                                             "gap_shift_same_day_pp", "gap_other_defs_pp", "reg_all100",
                                                             "A10_only_shift_slope_pp")} for c in CELLS}), flush=True)

    # ---------------- 收盤漲停（主跑）與開盤對照
    out["limit_up_main"] = {k: v["limit_up_skips"] for k, v in combos.items()}

    out["resources"] = dict(peak_rss_mb=rss(), seconds=round(time.time() - t_start, 1))
    trs = []
    for (d, ex, cell), x in res_main.items():
        t_ = x["trades"].drop(columns=["i"]).copy()
        t_.insert(0, "combo", "%s|%s|%s" % (d, ex, cell))
        trs.append(t_[["combo", "code", "buy_date", "sell_date", "sh", "cost", "proceeds", "pnl", "ret", "hold", "delist"]])
    pd.concat(trs, ignore_index=True).round(4).to_csv(os.path.join(odir, "my_trades.csv.gz"), index=False,
                                                      compression="gzip")
    pr.round(6).to_csv(os.path.join(odir, "my_paths.csv.gz"), index=False, compression="gzip")
    js = json.dumps(out, ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    open(os.path.join(odir, "my_numbers.json"), "w").write(js)
    h = hashlib.sha256(open(os.path.join(odir, "my_numbers.json"), "rb").read()).hexdigest()
    open(os.path.join(odir, "my_numbers.sha256"), "w").write(h + "  my_numbers.json\n")
    print("frozen", h, "rss", rss(), round(time.time() - t_start), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
