# -*- coding: utf-8 -*-
"""S10 獨立複製：驗收、22 個組合（11 出場 × 兩格）主跑＋八項檢查、20 條路徑 × 最小買進門檻 0.5／0.45／0.55、
事前判定、審查 A（MANO）、B（移動停損與各段空頭回撤）、C（最小買進門檻），然後凍結。
用法：python3 -I -B rep_s10_run.py <scratch_dir>
需先跑 s1 的 rep_s1_panel.py <scratch_dir> 與 s3 的 rep_s3_names.py <scratch_dir>/innov_names.json。
輸出（本資料夾）：my_numbers.json、my_numbers.sha256、my_trades.csv.gz、my_paths.csv.gz
凍結前不讀研究員的任何輸出（FINDINGS 第 2、9 節以外、results*.json、trades.csv.gz、navs.csv.gz）。
"""
import sys, os, json, hashlib, resource, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import rep_s10_lib as K
C, S, Q, R, L = K.C, K.S, K.Q, K.R, K.L
ST, X2 = K.ST, K.X2

CELLS = {"W10_70": "W10", "WN_70": "WN"}
VARS = K.VARS
CAPS = [600_000.0, 800_000.0, 1_000_000.0, 1_500_000.0, 2_000_000.0]
SHIFTS = [0, 1, 2, 3]
MFS = [0.5, 0.45, 0.55]
KNOWN = {"W10_70": (0.4814, 0.3409), "WN_70": (0.6216, 0.4450)}
BEARS = {"2018": ("2018-01-01", "2018-12-31"), "2020-03": ("2020-01-01", "2020-04-30"),
         "2022": ("2022-01-01", "2022-12-31"), "2024-07~2025-04": ("2024-07-01", "2025-04-30"),
         "pre_2024-07": ("2015-07-01", "2024-06-30")}


def rss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024


def tst(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return None
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))


def main(sc, odir=HERE):
    t_start = time.time()
    P = R.build(sc)
    S.attach_e50(P, sc)
    excl = S.innov_mask_fn(P, sc)
    Rv, mi0 = L.load_rev(P)
    S_, _ = L.surge(Rv, mi0, null2013=True)
    e = P.T - 1
    i19 = L.didx(P, "2018-12-28"); i24 = L.didx(P, "2023-12-29"); ih = L.didx(P, "2021-02-19")
    bw = {k: (L.didx(P, a), L.didx(P, b) - (0 if (L.didx(P, b) < P.T and P.dstr[L.didx(P, b)] == b) else 1))
          for k, (a, b) in BEARS.items()}
    bw["pre_2024-07"] = (P.t0, bw["pre_2024-07"][1])
    e50 = P.e50.values
    out = dict(meta=dict(T=P.T, N=P.N, first=P.dstr[0], last=P.dstr[-1], t0=P.dstr[P.t0], i19=P.dstr[i19],
                         i24=P.dstr[i24], ih=P.dstr[ih],
                         bear_windows={k: (P.dstr[a], P.dstr[b]) for k, (a, b) in bw.items()},
                         note="面板為複製員自建（N＝2130，研究員 2142；S6～S9 複製時已知，差的是從未進股池的代號）"))
    b50 = dict(ann_2019=L.ann(P, e50, i19, e), ann_full=L.ann(P, e50, P.t0, e), ann_2024=L.ann(P, e50, i24, e),
               mdd_full=L.maxdd(P, e50, P.t0, e), ann_h1=L.ann(P, e50, P.t0, ih), ann_h2=L.ann(P, e50, ih, e),
               bears={k: L.maxdd(P, e50, a, b)[0] for k, (a, b) in bw.items()})
    out["bench_0050"] = b50
    ma60 = R.ma(P, 60); ma20 = R.ma(P, 20)
    mavs = {20: ma20, 60: ma60}

    # ---------------- 名單與續抱集合（位移 0～3）
    lists = {sh: Q.make_lists(P, S_, mi0, shift=sh, excl=excl) for sh in SHIFTS}
    plans = {}
    ks_cache = {}
    for sh in SHIFTS:
        b, k20 = Q.cell_plans(lists[sh], "C10", x=0.10)
        for kind in ("top10", "top15", "top20", "top30", "ma20", "mano", "ma60split"):
            ks_cache[(kind, sh)] = K.keep_sets(P, S_, lists[sh], kind, excl, mavs)
        ks_cache[("cell", sh)] = k20
        for v in VARS:
            plans[(v, sh)] = (b, ks_cache[(K.SPEC[v]["keep"], sh)])
    print("plans", len(plans), "rss", rss(), round(time.time() - t_start), flush=True)

    def run(v, cell, sh=0, capital=1_000_000.0, cost=None, log=None, ts_override=None, rq_override=None, **kw):
        b, kp = plans[(v, sh)]
        sp = dict(K.SPEC[v])
        ts = sp.get("ts"); rq = sp.get("rq")
        if ts_override is not None:
            ts = ts_override
        if rq_override is not None:
            rq = rq_override
        eod = K.make_eod(P, kp, b, ma60, ts=ts, ts_daily=sp.get("ts_daily", True), rq=rq, dma=sp.get("dma", False),
                         log=log)
        return K.simulate_s10(P, b, kp, sat_w=0.7, sizing=CELLS[cell], capital=capital, eod=eod,
                              **{**(cost or ST), **kw})

    def wins(nav):
        return {k: L.maxdd(P, nav, a, b)[0] for k, (a, b) in bw.items()}

    def brief(res, full=True):
        nav = res["nav"]; tr = res["trades"]
        md = L.maxdd(P, nav, P.t0, e)
        o = dict(ann_2019=L.ann(P, nav, i19, e), ann_full=L.ann(P, nav, P.t0, e), ann_2024=L.ann(P, nav, i24, e),
                 mdd_full=md[0], mdd_peak=md[1], mdd_trough=md[2], n_trades=int(len(tr)))
        if full:
            r = nav[P.t0 + 1:e + 1] / nav[P.t0:e] - 1.0
            yrs = (P.dates[e] - P.dates[P.t0]).days / 365.25
            mnav = float(np.mean(nav[P.t0:e + 1]))
            cn = res["cnt"]
            tr19 = tr[tr.tb > i19]
            o.update(sharpe=float(r.mean() / r.std(ddof=1) * np.sqrt(252)), exp_per_trade=float(tr.ret.mean()),
                     hold_mean=float(tr.hold.mean()), hold_median=float(tr.hold.median()),
                     n_trades_2019=int(len(tr19)), exp_2019=float(tr19.ret.mean()),
                     hold_mean_2019=float(tr19.hold.mean()),
                     why={str(k): int(x) for k, x in tr.why.value_counts().items()},
                     turnover_1way=float((cn["buy_amt"] + cn["sell_amt"]) / 2.0 / mnav / yrs),
                     cost_stock_pct=float(cn["cost_total"] / mnav / yrs),
                     cost_all_pct=float((cn["cost_total"] + cn["e50_cost"]) / mnav / yrs),
                     ann_h1=L.ann(P, nav, P.t0, ih), ann_h2=L.ann(P, nav, ih, e), yearly=L.yearly(P, nav),
                     buys=int(cn["buy"]), blk_min=int(cn["blk_min"]), limit_up=int(cn["limit_up"]),
                     sell_defer=int(cn["sell_defer"]), bears=wins(nav), n_open=len(res["open_pos"]))
        return o

    # ---------------- 驗收
    acc = {}
    acc["P0"] = dict(T=P.T, N=P.N, last=P.dstr[-1], note="研究員 N＝2142；複製員面板 2130（已知差異，不影響股池）")
    b0, k0 = Q.cell_plans(Q.make_lists(P, S_, mi0, excl=excl), "C10", x=0.10)
    acc["A1_E0_eq_cell_plans_all_shifts"] = bool(all(
        ks_cache[("top20", sh)] == ks_cache[("cell", sh)] for sh in SHIFTS) and all(
        (plans[("E0", 0)][0][s] == b0[s]).all() for s in b0) and sorted(plans[("E0", 0)][0]) == sorted(b0))
    nd = 0; nrows = 0
    for sh in SHIFTS:
        for s in ks_cache[("top20", sh)]:
            nrows += 1
            if ks_cache[("ma60split", sh)][s] != ks_cache[("top20", sh)][s]:
                nd += 1
    acc["A2_ma60split_eq_top20"] = dict(rows=nrows, rows_differ=nd, ok=bool(nd == 0))
    res_main = {}
    for cell, sz in CELLS.items():
        a = K.K9.simulate_s9(P, b0, k0, sat_w=0.7, sizing=sz, **ST)
        c = run("E0", cell)
        ta = a["trades"].sort_values(["ts", "code"])[["code", "tb", "ts", "sh"]].values
        tc = c["trades"].sort_values(["ts", "code"])[["code", "tb", "ts", "sh"]].values
        y19, yf = L.ann(P, c["nav"], i19, e), L.ann(P, c["nav"], P.t0, e)
        acc["A0_" + cell] = dict(max_abs_nav_diff_vs_rep_s9=float(np.nanmax(np.abs(a["nav"] - c["nav"]))),
                                 trades_identical=bool(len(ta) == len(tc) and (ta == tc).all()),
                                 ann_2019=y19, ann_full=yf, mdd_full=L.maxdd(P, c["nav"], P.t0, e)[0],
                                 known_rep_s6=KNOWN[cell],
                                 ok=bool(abs(y19 - KNOWN[cell][0]) < 0.00006 and abs(yf - KNOWN[cell][1]) < 0.00006))
        res_main[("E0", cell)] = c
        a3 = {}
        for nm_, kw in (("TS15_d1", dict(v="TS15", ts_override=1.0)), ("TS25_d1", dict(v="TS25", ts_override=1.0)),
                        ("TS15M_d1", dict(v="TS15M", ts_override=1.0)), ("RQ6_n1e4", dict(v="RQ6", rq_override=10_000))):
            v = kw.pop("v")
            x = run(v, cell, **kw)
            tx = x["trades"].sort_values(["ts", "code"])[["code", "tb", "ts", "sh"]].values
            a3[nm_] = bool(np.nanmax(np.abs(x["nav"] - c["nav"])) < 1e-6 and len(tx) == len(tc) and (tx == tc).all())
        acc["A3_" + cell] = a3
    print("acc", json.dumps(acc, ensure_ascii=False), flush=True)
    assert acc["A0_W10_70"]["ok"] and acc["A0_WN_70"]["ok"] and acc["A1_E0_eq_cell_plans_all_shifts"]
    assert all(acc["A3_W10_70"].values()) and all(acc["A3_WN_70"].values())

    # ---------------- 22 個組合主跑＋八項檢查
    combos = {}
    trade_rows = []
    for v in VARS:
        for cell in CELLS:
            key = (v, cell)
            if key not in res_main:
                res_main[key] = run(v, cell)
            rm = res_main[key]
            o = brief(rm)
            x2 = brief(run(v, cell, cost=X2), full=False)
            sp, top = R.span_of(rm["trades"], 5)
            bl = brief(run(v, cell, block_span=sp), full=False)
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
            combos["%s|%s" % key] = o
            tr = rm["trades"]
            for r_ in tr.itertuples():
                trade_rows.append((v, cell, r_.code, r_.buy_date, r_.sell_date, round(r_.ret, 6), round(r_.pnl, 1),
                                   r_.why, int(r_.hold)))
            print(key, round(o["ann_2019"], 4), round(o["ann_full"], 4), round(o["ann_2024"], 4),
                  round(o["mdd_full"], 4), o["n_trades"], o["why"], o["cls"], flush=True)
    out["combos"] = combos
    print("main done rss", rss(), round(time.time() - t_start), flush=True)

    # ---------------- A4：移動停損出場用 G 獨立重算
    a4 = {}
    for v in ("TS15", "TS25"):
        d = K.SPEC[v]["ts"]
        for cell in CELLS:
            tr = res_main[(v, cell)]["trades"]
            tr = tr[tr.why == "ts"]
            bad = 0
            for r_ in tr.itertuples():
                i = int(r_.i); tb = int(r_.tb); mk = int(r_.mk)
                g = P.G[tb:mk + 1, i]
                okd = np.isfinite(g)
                runmax = np.maximum.accumulate(np.where(okd, g, -np.inf))
                cond = okd & (g <= runmax * (1.0 - d) * (1 + 1e-6))
                cond_strict = okd & (g <= runmax * (1.0 - d) * (1 - 1e-6))
                if not cond[-1] or cond_strict[:-1].any():
                    bad += 1
            a4["%s|%s" % (v, cell)] = dict(n_ts_exits=int(len(tr)), mismatches=bad)
    acc["A4"] = a4
    out["acceptance"] = acc
    print("A4", a4, flush=True)

    # ---------------- 20 條路徑 × min_frac
    rows = []
    for v in VARS:
        for cell in CELLS:
            for mf in MFS:
                for sh in SHIFTS:
                    for cp in CAPS:
                        if sh == 0 and cp == 1_000_000.0 and mf == 0.5:
                            x = res_main[(v, cell)]
                        else:
                            x = run(v, cell, sh=sh, capital=cp, min_frac=mf)
                        nav = x["nav"]
                        md = L.maxdd(P, nav, P.t0, e)
                        rows.append(dict(v=v, cell=cell, mf=mf, shift=sh, capital=cp,
                                         ann_2019=L.ann(P, nav, i19, e), ann_full=L.ann(P, nav, P.t0, e),
                                         ann_2024=L.ann(P, nav, i24, e), ann_1923=L.ann(P, nav, i19, i24),
                                         mdd_full=md[0], mdd_peak=md[1], mdd_trough=md[2],
                                         n_trades=int(len(x["trades"])),
                                         **{"bear_" + k: w for k, w in wins(nav).items()}))
        print("paths", v, "rss", rss(), round(time.time() - t_start), flush=True)
    pr = pd.DataFrame(rows)
    pr["dd_2024_25"] = (pr.mdd_peak >= "2024-07-01") & (pr.mdd_peak <= "2024-07-31") & \
                       (pr.mdd_trough >= "2025-04-01") & (pr.mdd_trough <= "2025-04-30")
    pr.to_csv(os.path.join(odir, "my_paths.csv.gz"), index=False, float_format="%.6f", compression="gzip")
    p20 = {}
    for (v, cell, mf), g in pr.groupby(["v", "cell", "mf"]):
        p20["%s|%s|%s" % (v, cell, mf)] = dict(
            ann_2019_min=float(g.ann_2019.min()), ann_2019_median=float(g.ann_2019.median()),
            ann_2019_max=float(g.ann_2019.max()), n_beat_0050=int((g.ann_2019 > b50["ann_2019"]).sum()),
            mdd_median=float(g.mdd_full.median()), mdd_deepest=float(g.mdd_full.min()),
            mdd_shallowest=float(g.mdd_full.max()), n_dd_2024_25=int(g.dd_2024_25.sum()),
            ann_full_median=float(g.ann_full.median()), ann_2024_median=float(g.ann_2024.median()),
            ann_1923_median=float(g.ann_1923.median()), n_trades_median=float(g.n_trades.median()),
            bears_median={k: float(g["bear_" + k].median()) for k in BEARS})
    out["paths20"] = p20

    # ---------------- 逐條配對、事前判定
    def paired(v, cell, mf, col):
        a = pr[(pr.v == v) & (pr.cell == cell) & (pr.mf == mf)].set_index(["shift", "capital"])[col]
        b = pr[(pr.v == "E0") & (pr.cell == cell) & (pr.mf == mf)].set_index(["shift", "capital"])[col]
        d = (a - b.loc[a.index]).values
        return d

    def cls_ret(d):
        n, m = int((d > 0).sum()), float(np.median(d))
        return "比標準好" if (n >= 15 and m >= 0.02) else ("比標準差" if (n <= 5 and m <= -0.02) else "差不多")

    pair = {}; dec = {}
    for v in VARS:
        if v == "E0":
            continue
        for cell in CELLS:
            for mf in MFS:
                da = paired(v, cell, mf, "ann_2019"); dd = paired(v, cell, mf, "mdd_full")
                n_dd = int((dd > 0).sum()); m_dd = float(np.median(dd))
                can = bool(n_dd >= 15 and m_dd >= 0.02)
                pair["%s|%s|%s" % (v, cell, mf)] = dict(
                    ann_n_pos=int((da > 0).sum()), ann_median=float(np.median(da)), ann_min=float(da.min()),
                    ann_max=float(da.max()), mdd_n_shallower=n_dd, mdd_median=m_dd, cls=cls_ret(da),
                    dd_cls=("有效率地壓回撤" if (can and float(np.median(da)) >= -0.02) else ("能壓回撤" if can else "不能")))
        cl = {cell: pair["%s|%s|0.5" % (v, cell)]["cls"] for cell in CELLS}
        good = [c for c in CELLS if cl[c] == "比標準好"]
        rob = len(good) == 2 and all(pair["%s|%s|%s" % (v, c, mf)]["cls"] == "比標準好" for c in CELLS for mf in (0.45, 0.55))
        if rob:
            overall = "穩定地比較好"
        elif len(good) == 2:
            overall = "比標準好但對最小買進門檻敏感"
        elif len(good) == 1:
            overall = "只有一格較好"
        else:
            bad = [c for c in CELLS if cl[c] == "比標準差"]
            overall = "兩格都比標準差" if len(bad) == 2 else ("一格比標準差" if len(bad) == 1 else "差不多")
        ddc = {cell: pair["%s|%s|0.5" % (v, cell)]["dd_cls"] for cell in CELLS}
        dec[v] = dict(ret_cls=cl, overall=overall, dd_cls=ddc,
                      xq=bool(rob or all(ddc[c] == "有效率地壓回撤" for c in CELLS)))
    tsd = [v for v in ("TS15", "TS25") if all(dec[v]["dd_cls"][c] != "不能" for c in CELLS)]
    tsp = [v for v in ("TS15", "TS25") if any(dec[v]["dd_cls"][c] != "不能" for c in CELLS)]
    ranks = {}
    for cell in CELLS:
        med = {v: p20["%s|%s|0.5" % (v, cell)]["ann_2019_median"] for v in VARS}
        mdm = {v: p20["%s|%s|0.5" % (v, cell)]["mdd_median"] for v in VARS}
        mr = {v: combos["%s|%s" % (v, cell)]["ann_2019"] for v in VARS}
        mm = {v: combos["%s|%s" % (v, cell)]["mdd_full"] for v in VARS}
        rk = lambda d_: {v: 1 + sum(1 for w in d_ if d_[w] > d_[v]) for v in d_}
        ranks[cell] = dict(path_ann=rk(med), path_mdd=rk(mdm), main_ann=rk(mr), main_mdd=rk(mm),
                           family_path_ann=dict(median=float(np.median(list(med.values()))), min=min(med.values()),
                                                max=max(med.values())),
                           family_path_mdd=dict(median=float(np.median(list(mdm.values()))), min=min(mdm.values()),
                                                max=max(mdm.values())),
                           family_main_ann=dict(median=float(np.median(list(mr.values()))), min=min(mr.values()),
                                                max=max(mr.values())),
                           family_main_mdd=dict(median=float(np.median(list(mm.values()))), min=min(mm.values()),
                                                max=max(mm.values())))
    e0r = [ranks[c]["path_ann"]["E0"] for c in CELLS]
    out["paired"] = pair
    out["decisions"] = dict(
        per_variant=dec, ts_can_reduce_dd=("能" if tsd else ("部分" if tsp else "不能")),
        e0_best=("是" if all(r_ == 1 for r_ in e0r) else ("前段" if all(r_ <= 3 for r_ in e0r) else "不是")),
        e0_rank_path_ann=e0r, ranks=ranks,
        xq=[v for v in dec if dec[v]["xq"]] or "維持現行參數",
        eight_checks={k: o["cls"] for k, o in combos.items()})
    # 審查 C：min_frac 下判定是否改變
    chg = {}
    for v in dec:
        for cell in CELLS:
            base = pair["%s|%s|0.5" % (v, cell)]
            for mf in (0.45, 0.55):
                x = pair["%s|%s|%s" % (v, cell, mf)]
                if x["cls"] != base["cls"] or x["dd_cls"] != base["dd_cls"]:
                    chg["%s|%s|%s" % (v, cell, mf)] = dict(base=(base["cls"], base["dd_cls"]), now=(x["cls"], x["dd_cls"]))
    out["review_C"] = dict(n_changed=len(chg), changed=chg)
    print("decisions", json.dumps(out["decisions"]["per_variant"], ensure_ascii=False), flush=True)
    print("reviewC", len(chg), chg, flush=True)

    # ---------------- 審查 A：MANO
    RA = {}
    seg = {}
    for v in VARS:
        if v == "E0":
            continue
        for cell in CELLS:
            o = {}
            for col in ("ann_1923", "ann_2024"):
                d = paired(v, cell, 0.5, col)
                o[col] = dict(n_pos=int((d > 0).sum()), median=float(np.median(d)), min=float(d.min()), max=float(d.max()))
            seg["%s|%s" % (v, cell)] = o
    RA["segments_all_variants"] = seg

    def event_stats(ev):
        if not ev:
            return dict(n=0)
        x = np.array([q["exc"] for q in ev])
        return dict(n=int(len(x)), mean=float(x.mean()), t=tst(x), median=float(np.median(x)),
                    win=float((x > 0).mean()))

    def nxt_sig(keys, k):
        j = np.searchsorted(keys, k, side="right")
        return int(keys[j]) if j < len(keys) else None

    keys0 = np.array(sorted(ks_cache[("top20", 0)]))
    t24 = L.didx(P, "2024-01-01")
    for cell in CELLS:
        ev_e0 = []
        tr = res_main[("E0", cell)]["trades"]
        nsig = 0; n_other = 0
        for r_ in tr[tr.why == "sig"].itertuples():
            nsig += 1
            k = int(r_.mk); i = int(r_.i)
            if not (i in ks_cache[("mano", 0)][k] and i not in ks_cache[("top20", 0)][k]):
                n_other += 1; continue
            k2 = nxt_sig(keys0, k)
            t1 = int(r_.ts); t2 = (k2 + 1) if k2 is not None else e
            if t2 > e:
                t2 = e
            rs = K.adj_open(P, t2, i) / K.adj_open(P, t1, i) - 1.0
            rb = P.e50o[t2] / P.e50o[t1] - 1.0
            ev_e0.append(dict(code=P.codes[i], k=P.dstr[k], t1=P.dstr[t1], t2=P.dstr[t2], ret=rs, b=rb, exc=rs - rb,
                              post24=bool(t1 >= t24)))
        log = []
        _ = run("MANO", cell, log=log)
        ev_m = []
        for (k, i) in log:
            if i in ks_cache[("top20", 0)][k] or i not in ks_cache[("mano", 0)][k]:
                continue
            k2 = nxt_sig(keys0, k)
            t1 = k + 1; t2 = (k2 + 1) if k2 is not None else e
            if t1 > e:
                continue
            t2 = min(t2, e)
            rs = K.adj_open(P, t2, i) / K.adj_open(P, t1, i) - 1.0
            rb = P.e50o[t2] / P.e50o[t1] - 1.0
            ev_m.append(dict(code=P.codes[i], k=P.dstr[k], t1=P.dstr[t1], t2=P.dstr[t2], ret=rs, b=rb, exc=rs - rb,
                             post24=bool(t1 >= t24)))
        RA["ma_only_events_" + cell] = dict(
            source_E0_run=dict(n_sig_exits=nsig, n_not_ma_only=n_other, all=event_stats(ev_e0),
                               pre2024=event_stats([q for q in ev_e0 if not q["post24"]]),
                               post2024=event_stats([q for q in ev_e0 if q["post24"]]),
                               top5=sorted(ev_e0, key=lambda q: -q["exc"])[:5]),
            source_MANO_run=dict(all=event_stats(ev_m), pre2024=event_stats([q for q in ev_m if not q["post24"]]),
                                 post2024=event_stats([q for q in ev_m if q["post24"]]),
                                 top5=sorted(ev_m, key=lambda q: -q["exc"])[:5],
                                 note="MANO 主跑：訊號列收盤持有、不在 L_MA 前 20 名（MANO 續抱、E0 會賣）的部位-月；"
                                      "同一部位連續幾個月會重複計"))
    # 去掉 2408、去掉最賺 5 筆（20 條路徑）
    cid2408 = P.cidx.get("2408")
    rows_b = []
    for cell in CELLS:
        for sh in SHIFTS:
            for cp in CAPS:
                m0 = run("MANO", cell, sh=sh, capital=cp); e0 = run("E0", cell, sh=sh, capital=cp)
                base = L.ann(P, m0["nav"], i19, e) - L.ann(P, e0["nav"], i19, e)
                tm = m0["trades"]; te = e0["trades"]
                spm, _ = R.span_of(tm, 5); spe, _ = R.span_of(te, 5)
                d_own5 = L.ann(P, run("MANO", cell, sh=sh, capital=cp, block_span=spm)["nav"], i19, e) - \
                    L.ann(P, run("E0", cell, sh=sh, capital=cp, block_span=spe)["nav"], i19, e)
                d_com5 = L.ann(P, run("MANO", cell, sh=sh, capital=cp, block_span=spm)["nav"], i19, e) - \
                    L.ann(P, run("E0", cell, sh=sh, capital=cp, block_span=spm)["nav"], i19, e)
                t8 = tm[tm.code == "2408"].sort_values("pnl", ascending=False)
                if len(t8):
                    sp8 = {"2408": [(int(t8.tb.iloc[0]), int(t8.ts.iloc[0]))]}
                    d_8span = L.ann(P, run("MANO", cell, sh=sh, capital=cp, block_span=sp8)["nav"], i19, e) - \
                        L.ann(P, run("E0", cell, sh=sh, capital=cp, block_span=sp8)["nav"], i19, e)
                    m8 = (str(t8.buy_date.iloc[0]), str(t8.sell_date.iloc[0]), float(t8.ret.iloc[0]))
                else:
                    d_8span = base; m8 = None
                spall = {"2408": [(0, P.T)]}
                d_8all = L.ann(P, run("MANO", cell, sh=sh, capital=cp, block_span=spall)["nav"], i19, e) - \
                    L.ann(P, run("E0", cell, sh=sh, capital=cp, block_span=spall)["nav"], i19, e)
                rows_b.append(dict(cell=cell, shift=sh, capital=cp, base=base, own5=d_own5, common5=d_com5,
                                   no2408_span=d_8span, no2408_all=d_8all, mano_2408_trade=m8))
    rb = pd.DataFrame(rows_b)
    blk = {}
    for cell, g in rb.groupby("cell"):
        blk[cell] = {c: dict(n_pos=int((g[c] > 0).sum()), median=float(g[c].median()), min=float(g[c].min()),
                             max=float(g[c].max())) for c in ("base", "own5", "common5", "no2408_span", "no2408_all")}
        blk[cell]["main_run_2408_trade"] = g[(g["shift"] == 0) & (g.capital == 1_000_000.0)].mano_2408_trade.iloc[0]
    RA["block_tests_MANO_minus_E0"] = blk
    out["review_A"] = RA
    print("reviewA", json.dumps({c: RA["ma_only_events_" + c]["source_E0_run"]["all"] for c in CELLS}), flush=True)
    print("reviewA blk", json.dumps(blk, ensure_ascii=False, default=str), flush=True)

    # ---------------- 審查 B：各段空頭最大跌幅
    RB = {}
    for v in VARS:
        for cell in CELLS:
            o = dict(main={k: combos["%s|%s" % (v, cell)]["bears"][k] for k in BEARS},
                     path_median={k: p20["%s|%s|0.5" % (v, cell)]["bears_median"][k] for k in BEARS})
            if v != "E0":
                pdiff = {}
                for k in BEARS:
                    d = paired(v, cell, 0.5, "bear_" + k)
                    pdiff[k] = dict(n_shallower=int((d > 0).sum()), median=float(np.median(d)))
                d = paired(v, cell, 0.5, "mdd_full")
                pdiff["full"] = dict(n_shallower=int((d > 0).sum()), median=float(np.median(d)))
                o["paired_vs_E0"] = pdiff
            RB["%s|%s" % (v, cell)] = o
    out["review_B"] = RB
    out["bench_0050_bears"] = b50["bears"]

    # ---------------- 凍結
    pd.DataFrame(trade_rows, columns=["v", "cell", "code", "buy_date", "sell_date", "ret", "pnl", "why", "hold"]).to_csv(
        os.path.join(odir, "my_trades.csv.gz"), index=False, compression="gzip")
    out["resources"] = dict(peak_rss_mb=rss(), seconds=round(time.time() - t_start, 1))
    js = json.dumps(out, ensure_ascii=False, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x))
    with open(os.path.join(odir, "my_numbers.json"), "w") as f:
        f.write(js)
    h = hashlib.sha256(js.encode("utf-8")).hexdigest()
    with open(os.path.join(odir, "my_numbers.sha256"), "w") as f:
        f.write(h + "  my_numbers.json\n")
    print("frozen", h, out["resources"], flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
