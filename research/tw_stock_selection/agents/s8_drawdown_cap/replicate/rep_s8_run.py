# -*- coding: utf-8 -*-
"""S8 獨立複製：驗收、7 格（5 新格＋2 對照）、主格穩健性、延遲、20 條路徑回撤（17 格）、效率前緣、審查 A／B，然後凍結。
用法：python3 -I -B rep_s8_run.py <scratch_dir>
需先跑 s1 的 rep_s1_panel.py <scratch_dir> 與 s3 的 rep_s3_names.py <scratch_dir>/innov_names.json。
輸出（本資料夾）：my_numbers.json、my_numbers.sha256、my_trades.csv.gz
凍結前不讀研究員的任何輸出檔（FINDINGS 第 2、10 節以外、results*.json、trades.csv.gz、navs.csv.gz）。
"""
import sys, os, json, hashlib, resource, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import rep_s8_lib as K
K7, C, S, Q, R, L = K.K7, K.C, K.S, K.Q, K.R, K.L
ST, X2 = S.STRICT, S.X2

# id: (sizing, sat_w, lock, cap)
S8CELLS = {
    "WN_70L_C12": ("WN", 0.7, True, 0.12), "WN_70L_C10": ("WN", 0.7, True, 0.10),
    "WN_70L_C08": ("WN", 0.7, True, 0.08), "WN_60L": ("WN", 0.6, True, None),
    "WN_60L_C10": ("WN", 0.6, True, 0.10), "W10_70L": ("W10", 0.7, True, None), "WN_70L": ("WN", 0.7, True, None),
}
# S6／S7 前緣格（研究員沒算路徑回撤）＋其餘 S6／S7 非名次加權格（補充）
EXTRA = {
    "W10_70": ("W10", 0.7, False, None), "WN_70": ("WN", 0.7, False, None), "WN_100": ("WN", 1.0, False, None),
    "W10_80L": ("W10", 0.8, True, None), "WN_80L": ("WN", 0.8, True, None),
}
SUPP = {
    "W7_70L": ("W7", 0.7, True, None), "W7_80L": ("W7", 0.8, True, None), "W7_70": ("W7", 0.7, False, None),
    "W10_100": ("W10", 1.0, False, None), "W7_100": ("W7", 1.0, False, None),
}
ALL = {**S8CELLS, **EXTRA, **SUPP}
REQ = list(S8CELLS) + list(EXTRA)          # 任務要求列表的 12 格
NEW = ["WN_70L_C12", "WN_70L_C10", "WN_70L_C08", "WN_60L", "WN_60L_C10"]
MAIN = "WN_70L_C12"
CAPS = [600_000.0, 800_000.0, 1_000_000.0, 1_500_000.0, 2_000_000.0]
LIM = -0.36


def rss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024


def frontier(pts):
    """pts: {id: (ret, dd)}，dd 為負數（越接近 0 越好）。被支配＝有別格 ret ≥ 且 dd ≥，且至少一項嚴格。"""
    on = []
    for a, (ra, da) in pts.items():
        dom = any((rb >= ra and db >= da and (rb > ra or db > da)) for b, (rb, db) in pts.items() if b != a)
        if not dom:
            on.append(a)
    return sorted(on, key=lambda k: pts[k][1])


def main(sc, odir=HERE):
    t_start = time.time()
    P = R.build(sc)
    S.attach_e50(P, sc)
    excl = S.innov_mask_fn(P, sc)
    Rv, mi0 = L.load_rev(P)
    S_, _ = L.surge(Rv, mi0, null2013=True)
    e = P.T - 1; i19 = L.didx(P, "2018-12-28"); i24 = L.didx(P, "2023-12-29"); ih = L.didx(P, "2021-02-19")
    e50 = P.e50.values
    lists = Q.make_lists(P, S_, mi0, excl=excl)
    buy, keep = Q.cell_plans(lists, "C10", x=0.10)
    ranks = K.c10_ranks(lists, buy)
    base_plans = (buy, keep, ranks)
    sig_exec = sorted(s + 1 for s in buy)
    out = dict(meta=dict(T=P.T, N=P.N, first=P.dstr[0], last=P.dstr[-1], t0=P.dstr[P.t0], n_signals=len(buy),
                         i19=P.dstr[i19], i24=P.dstr[i24], ih=P.dstr[ih],
                         note="面板為複製員自建（N＝2130，研究員 2142；S6／S7 複製時已知，差的是從未進股池的代號）"))
    b50 = dict(ann_2019=L.ann(P, e50, i19, e), ann_full=L.ann(P, e50, P.t0, e), ann_2024=L.ann(P, e50, i24, e),
               mdd_full=L.maxdd(P, e50, P.t0, e), mdd_2019=L.maxdd(P, e50, i19, e)[0],
               ann_h1=L.ann(P, e50, P.t0, ih), ann_h2=L.ann(P, e50, ih, e), yearly=L.yearly(P, e50))
    out["bench_0050"] = b50

    def run(cid, plans=base_plans, d=0, **kw):
        sz, w, lk, cp = ALL[cid]
        b, k, rk = K.delay_plans(plans, d, P.T)
        return K.simulate_s8(P, b, k, sat_w=w, sizing=sz, lock=lk, lranks=rk, cap=cp, **{**ST, **kw})

    def per_stock_w(res):
        nav = res["nav"]; f = int(res["trades"].tb.min())
        sw = res["stkw"][f:e + 1]; npz = res["npos"][f:e + 1]
        m = npz > 0
        tr = res["trades"]
        bw = (tr.cost / nav[tr.tb.values - 1]).values
        return float(np.mean(sw[m] / npz[m])), float(np.mean(bw)), float(np.median(bw))

    def brief(res, full=True):
        nav = res["nav"]; tr = res["trades"]
        f = int(tr.tb.min())
        md = L.maxdd(P, nav, P.t0, e)
        o = dict(ann_2019=L.ann(P, nav, i19, e), ann_full=L.ann(P, nav, P.t0, e), ann_2024=L.ann(P, nav, i24, e),
                 mdd_full=md[0], mdd_peak=md[1], mdd_trough=md[2], mdd_2019=L.maxdd(P, nav, i19, e)[0],
                 ann_h1=L.ann(P, nav, P.t0, ih), ann_h2=L.ann(P, nav, ih, e), n_trades=int(len(tr)))
        if full:
            c = res["cnt"]
            bw = res["B"][f:e + 1] / nav[f:e + 1]
            idle = (res["cashw"] + res["w50"]) - res["B"] / nav
            ip, it = L.didx(P, md[1]), L.didx(P, md[2])
            hits = res["cap_hits"]
            psw = per_stock_w(res)
            o.update(avg_stock_w=float(res["stkw"][f:e + 1].mean()), avg_npos=float(res["npos"][f:e + 1].mean()),
                     per_stock_w_daily=psw[0], per_stock_w_at_buy_mean=psw[1], per_stock_w_at_buy_median=psw[2],
                     base_w_mean=float(bw.mean()), base_w_min=float(bw.min()), base_lt10=float((bw < 0.10).mean()),
                     base_lt20=float((bw < 0.20).mean()), reb_up=c["reb_up"], reb_down=c["reb_down"],
                     reduce_legs=c["reb_reduce_legs"], a3_events=len(res["a3"]), a3_big=res["a3_big"],
                     a4_violations=len(res["a4"]),
                     cap_hits=len(hits), cap_saved_sum=float(sum(h["saved"] for h in hits)),
                     cap_saved_sum_frac_eq=float(sum(h["saved"] / h["eq"] for h in hits)),
                     cap_hit_days=len(set(h["date"] for h in hits)),
                     cap_hits_by_nlist={str(k): int(v) for k, v in
                                        pd.Series([min(h["n_list"], 6) for h in hits]).value_counts().sort_index().items()} if hits else {},
                     cap_hits_yearly={k: int(v) for k, v in pd.Series([h["date"][:4] for h in hits]).value_counts().sort_index().items()} if hits else {},
                     sat_idle_mean=float(idle[f:e + 1].mean()), sat_idle_dd=float(idle[ip:it + 1].mean()),
                     stock_w_dd=float(res["stkw"][ip:it + 1].mean()), stock_w_peak=float(res["stkw"][ip]),
                     npos_peak=int(res["npos"][ip]),
                     short_shrink=c["short_shrink"], short_skip=c["short_skip"], short_days=len(res["short_days"]),
                     blk_min=c["blk_min"], buys=c["buy"],
                     exp_per_trade=float(tr.ret.mean()), turnover_1side=float(c["buy_amt"] / np.nanmean(nav[f:e + 1]) /
                                                                             ((P.dates[e] - P.dates[f]).days / 365.25)),
                     yearly=L.yearly(P, nav))
        return o

    # ---------------- 驗收
    acc = {}
    res = {}
    for cid, cp in (("W10_70L", None), ("WN_70L", None), ("WN_70L", 1.0)):
        sz, w, lk, _ = ALL[cid]
        a = K7.simulate_s7(P, buy, keep, sat_w=w, sizing=sz, lock=lk, lranks=ranks, **ST)
        b = K.simulate_s8(P, buy, keep, sat_w=w, sizing=sz, lock=lk, lranks=ranks, cap=cp, **ST)
        ta, tb_ = a["trades"], b["trades"]
        acc["A1_%s_cap%s" % (cid, cp)] = dict(
            max_abs_nav_diff=float(np.nanmax(np.abs(a["nav"] - b["nav"]))),
            trades_identical=bool(len(ta) == len(tb_) and (ta[["code", "tb", "ts", "sh"]].values ==
                                                           tb_[["code", "tb", "ts", "sh"]].values).all()),
            ann_2019=L.ann(P, b["nav"], i19, e), ann_full=L.ann(P, b["nav"], P.t0, e),
            mdd_full=L.maxdd(P, b["nav"], P.t0, e)[0])
        if cp is None:
            res[cid] = b
    acc["A0_known"] = dict(WN_70L=(0.6026, 0.4309), W10_70L=(0.4751, 0.3369),
                           ok=bool(abs(acc["A1_WN_70L_capNone"]["ann_2019"] - 0.6026) < 0.0001 and
                                   abs(acc["A1_W10_70L_capNone"]["ann_full"] - 0.3369) < 0.0001))
    pl0 = tuple({s + 0: v for s, v in x.items()} for x in base_plans)
    a = run(MAIN, plans=pl0); b = run(MAIN)
    acc["A2_" + MAIN] = dict(max_abs_nav_diff=float(np.nanmax(np.abs(a["nav"] - b["nav"]))))
    res[MAIN] = b
    out["acceptance"] = acc
    print("acc", json.dumps(acc), "rss", rss(), flush=True)

    # ---------------- 7 格＋前緣格主跑
    for cid in list(S8CELLS) + list(EXTRA) + list(SUPP):
        if cid not in res:
            res[cid] = run(cid)
        out[cid] = brief(res[cid])
        assert out[cid]["a4_violations"] == 0, cid
        print(cid, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out[cid].items()
                    if k in ("ann_2019", "ann_full", "ann_2024", "mdd_full", "n_trades", "cap_hits", "short_skip",
                             "a3_big", "per_stock_w_daily")}, flush=True)
    base = out["WN_70L"]
    for cid in NEW:
        o = out[cid]
        o["keep_K"] = {k: (o[k] - b50[k]) / (base[k] - b50[k]) for k in ("ann_2019", "ann_full", "ann_2024")}
        o["less_than_WN70L_pp"] = {k: (base[k] - o[k]) * 100 for k in ("ann_2019", "ann_full", "ann_2024")}
        o["dd_saved_pp"] = (abs(base["mdd_full"]) - abs(o["mdd_full"])) * 100

    # ---------------- 穩健性（八項）：新格與兩個對照
    rob = {}
    for cid in NEW + ["WN_70L", "W10_70L"]:
        x2 = brief(run(cid, **X2), full=False)
        sp, top = R.span_of(res[cid]["trades"], 5)
        bl = brief(run(cid, block_span=sp), full=False)
        bl["blocked"] = top
        lt = brief(run(cid, extra=0.0), full=False)
        o = out[cid]
        checks = dict(a2019=o["ann_2019"] > b50["ann_2019"], afull=o["ann_full"] > b50["ann_full"],
                      x2_2019=x2["ann_2019"] > b50["ann_2019"], x2_full=x2["ann_full"] > b50["ann_full"],
                      bl_2019=bl["ann_2019"] > b50["ann_2019"], bl_full=bl["ann_full"] > b50["ann_full"],
                      h1=o["ann_h1"] > b50["ann_h1"], h2=o["ann_h2"] > b50["ann_h2"])
        cls = "值得上 XQ 測" if all(checks.values()) else (
            "不值得" if (not checks["a2019"] and not checks["afull"]) else "有條件")
        rob[cid] = dict(x2=x2, block5=bl, light=lt, checks=checks, cls=cls)
        print("rob", cid, round(x2["ann_2019"], 4), round(x2["ann_full"], 4), round(bl["ann_2019"], 4),
              round(bl["ann_full"], 4), round(o["ann_h1"], 4), round(o["ann_h2"], 4), cls, flush=True)
    out["robust"] = rob

    # ---------------- 延遲 d＝1／2：主格、W10_70L（WN_70L 一併跑供參考）
    dly = {}; dres = {}
    for cid in (MAIN, "W10_70L", "WN_70L"):
        dly[cid] = {"d0": {k: out[cid][k] for k in ("ann_2019", "ann_full", "ann_2024", "mdd_full", "n_trades")}}
        for d in (1, 2):
            x = run(cid, d=d); dres[(cid, d)] = x
            o = brief(x, full=False)
            o["cap_hits"] = len(x["cap_hits"]); o["short_skip"] = x["cnt"]["short_skip"]
            for k in ("ann_2019", "ann_full", "ann_2024"):
                o["loss_pp_" + k] = (dly[cid]["d0"][k] - o[k]) * 100
            o["caps_2019"] = [L.ann(P, run(cid, d=d, capital=cp)["nav"], i19, e) for cp in CAPS]
            o["caps_median_2019"] = float(np.median(o["caps_2019"]))
            dly[cid]["d%d" % d] = o
            print("delay", cid, d, round(o["ann_2019"], 4), round(o["ann_full"], 4), round(o["mdd_full"], 4), flush=True)
    out["delay"] = dly

    # ---------------- 20 條路徑（S6／S7 複製員定義：60/80/100/150/200 萬 × make_lists shift 0～3）
    shift_plans = {0: base_plans}
    for shf in range(1, 4):
        lk = Q.make_lists(P, S_, mi0, shift=shf, excl=excl)
        bb, kk = Q.cell_plans(lk, "C10", x=0.10)
        shift_plans[shf] = (bb, kk, K.c10_ranks(lk, bb))
    rows = []
    keep_nav = {}   # 100 萬各位移淨值（審查 A 用，WN_70L／主格／WN_60L／W10_70L／WN_70）
    for shf in range(4):
        for cp in CAPS:
            for cid in ALL:
                rp = (cp == 1_000_000.0 and cid in ("WN_70L", MAIN, "WN_60L", "W10_70L", "WN_70", "WN_80L"))
                x = run(cid, plans=shift_plans[shf], capital=cp, rec_pos=rp)
                md = L.maxdd(P, x["nav"], P.t0, e)
                psw = per_stock_w(x)
                rows.append(dict(cell=cid, shift=shf, capital=cp, ann_2019=L.ann(P, x["nav"], i19, e),
                                 ann_full=L.ann(P, x["nav"], P.t0, e), mdd_full=md[0], peak=md[1], trough=md[2],
                                 mdd_2019=L.maxdd(P, x["nav"], i19, e)[0], per_stock_w=psw[0],
                                 per_stock_w_buy=psw[1], n_trades=int(len(x["trades"])),
                                 cap_hits=len(x["cap_hits"]), short_skip=x["cnt"]["short_skip"]))
                if rp:
                    keep_nav[(cid, shf)] = x
                else:
                    del x
        print("paths shift", shf, "rss", rss(), round(time.time() - t_start), flush=True)
    pr = pd.DataFrame(rows)
    pz = {}
    for cid in ALL:
        g = pr[pr.cell == cid]
        dd = g.mdd_full.values; rr = g.ann_2019.values
        pz[cid] = dict(
            mdd_deepest=float(dd.min()), mdd_median=float(np.median(dd)), mdd_shallowest=float(dd.max()),
            n_within_36=int((dd >= LIM).sum()),
            mdd_median_by_shift={int(s): float(np.median(v.mdd_full)) for s, v in g.groupby("shift")},
            mdd_mean_by_shift={int(s): float(np.mean(v.mdd_full)) for s, v in g.groupby("shift")},
            ret_min=float(rr.min()), ret_median=float(np.median(rr)), ret_max=float(rr.max()),
            ret_median_by_shift={int(s): float(np.median(v.ann_2019)) for s, v in g.groupby("shift")},
            above_0050=int((rr > b50["ann_2019"]).sum()), full_median=float(np.median(g.ann_full)),
            mdd2019_median=float(np.median(g.mdd_2019)),
            per_stock_w_median=float(np.median(g.per_stock_w)), per_stock_w_buy_median=float(np.median(g.per_stock_w_buy)),
            peaks={str(k): int(v) for k, v in g.peak.str[:7].value_counts().items()},
            troughs={str(k): int(v) for k, v in g.trough.str[:7].value_counts().items()})
    # 主格 vs WN_70L、W10_70L 逐條相減
    def pair(a_, b_):
        A = pr[pr.cell == a_].sort_values(["shift", "capital"]); B = pr[pr.cell == b_].sort_values(["shift", "capital"])
        dr = A.ann_2019.values - B.ann_2019.values
        ddd = np.abs(B.mdd_full.values) - np.abs(A.mdd_full.values)     # 正＝a 比 b 淺
        return dict(ret_diff_median_pp=float(np.median(dr) * 100), ret_n_pos=int((dr > 0).sum()),
                    dd_shallower_median_pp=float(np.median(ddd) * 100), dd_n_shallower=int((ddd > 0).sum()),
                    per_path_dd_shallower_pp=[round(float(v) * 100, 3) for v in ddd])
    pz["pair_main_vs_WN_70L"] = pair(MAIN, "WN_70L")
    pz["pair_main_vs_W10_70L"] = pair(MAIN, "W10_70L")
    for c_ in NEW[1:]:
        pz["pair_%s_vs_WN_70L" % c_] = pair(c_, "WN_70L")
    pz["rows"] = pr.to_dict("records")
    out["paths20"] = pz
    out["success"] = dict(
        dd_ok=bool(out[MAIN]["mdd_full"] >= LIM),
        K_2019=out[MAIN]["keep_K"]["ann_2019"], K_full=out[MAIN]["keep_K"]["ann_full"],
        K_ok=bool(out[MAIN]["keep_K"]["ann_2019"] >= 0.75 and out[MAIN]["keep_K"]["ann_full"] >= 0.75),
        stable_n=pz["pair_main_vs_WN_70L"]["dd_n_shallower"],
        stable_ok=bool(pz["pair_main_vs_WN_70L"]["dd_n_shallower"] >= 15))
    s_ = out["success"]
    s_["verdict"] = ("能" if (s_["dd_ok"] and s_["K_ok"] and s_["stable_ok"]) else
                     ("能壓回但代價大" if s_["dd_ok"] and not s_["K_ok"] else
                      ("不能（主格）" if not s_["dd_ok"] else "壓回成立、保住率成立、但回撤不穩定")))

    # ---------------- 效率前緣（20 條路徑中位數回撤 × 中位數報酬；不含名次加權）
    fr = {}
    for name, cells in (("req12_plus0050", REQ), ("all17_plus0050", list(ALL))):
        pts = {c: (pz[c]["ret_median"], pz[c]["mdd_median"]) for c in cells}
        pts["0050"] = (b50["ann_2019"], b50["mdd_full"][0])
        pts_f = {c: (pz[c]["full_median"], pz[c]["mdd_median"]) for c in cells}
        pts_f["0050"] = (b50["ann_full"], b50["mdd_full"][0])
        fr[name] = dict(on_frontier_2019=frontier(pts), on_frontier_full=frontier(pts_f),
                        points={k: [round(v[0] * 100, 2), round(v[1] * 100, 2)] for k, v in pts.items()})
        # 單一路徑（100 萬、位移 0）前緣，對照用
        p1 = {c: (out[c]["ann_2019"], out[c]["mdd_full"]) for c in cells}
        p1["0050"] = (b50["ann_2019"], b50["mdd_full"][0])
        fr[name]["single_path_frontier_2019"] = frontier(p1)
    out["frontier"] = fr

    # ---------------- 審查 A：為什麼位移 0 的 WN 類回撤特別淺
    rva = {}
    cw_p, cw_t = L.didx(P, "2024-07-17"), L.didx(P, "2025-04-09")
    for cid in ("WN_70L", MAIN, "WN_60L", "W10_70L", "WN_70", "WN_80L"):
        per = {}
        for shf in range(4):
            x = keep_nav[(cid, shf)]; nav = x["nav"]
            md, pk, tr_ = L.maxdd(P, nav, P.t0, e)
            ip, it = L.didx(P, pk), L.didx(P, tr_)
            bw = x["B"] / nav
            hold = sorted(((P.codes[i], v / nav[ip]) for i, v in x["snap"].get(ip, {}).items()), key=lambda z: -z[1])
            wp = K.window_pnl(P, x, ip, it)
            los = sorted(((P.codes[i], v / nav[ip]) for i, v in wp.items()), key=lambda z: z[1])
            stock_sum = sum(wp.values()) / nav[ip]
            # 共同區間（位移 0 的高低點 2024-07-17 → 2025-04-09）
            cp_ = K.window_pnl(P, x, cw_p, cw_t)
            cl = sorted(((P.codes[i], v / nav[cw_p]) for i, v in cp_.items()), key=lambda z: z[1])
            # 2024-07-01 以前、以後的最大回撤
            pre = L.maxdd(P, nav, P.t0, L.didx(P, "2024-06-28"))
            # 前三個不重疊回撤
            v = nav[P.t0:e + 1]; pkv = np.maximum.accumulate(v); ddv = v / pkv - 1.0
            eps = []; inn = False
            for j in range(len(v)):
                if ddv[j] < 0 and not inn:
                    inn = True; st = j - 1; lo = j
                if inn:
                    if ddv[j] < ddv[lo]:
                        lo = j
                    if ddv[j] >= 0 or j == len(v) - 1:
                        eps.append((float(ddv[lo]), P.dstr[P.t0 + st], P.dstr[P.t0 + lo])); inn = False
            eps = sorted(eps)[:4]
            mr = R.month_ret(P, nav)
            mwin = {str(k): round(float(val) * 100, 2) for k, val in mr.items()
                    if pd.Period("2024-05", "M") <= k <= pd.Period("2025-05", "M")}
            # 高點日距離最近換股執行日（之前）幾個交易日
            prev_exec = max([t for t in range(ip + 1) if (t - 1) in shift_plans[shf][0]], default=None)
            per[shf] = dict(mdd=md, peak=pk, trough=tr_, days=it - ip,
                            peak_minus_last_exec_days=(ip - prev_exec) if prev_exec is not None else None,
                            last_exec_before_peak=P.dstr[prev_exec] if prev_exec is not None else None,
                            stock_w_peak=float(x["stkw"][ip]), base_w_peak=float(bw[ip]), npos_peak=int(x["npos"][ip]),
                            stock_w_dd=float(x["stkw"][ip:it + 1].mean()), e50_dd=float(e50[it] / e50[ip] - 1.0),
                            holdings_at_peak=[(c_, round(w_ * 100, 2)) for c_, w_ in hold],
                            top_losers_own_window=[(c_, round(w_ * 100, 2)) for c_, w_ in los[:8]],
                            stock_pnl_own_window_pct=round(stock_sum * 100, 2),
                            nonstock_pnl_own_window_pct=round(((nav[it] - nav[ip]) / nav[ip] - stock_sum) * 100, 2),
                            common_window_ret=float(nav[cw_t] / nav[cw_p] - 1.0),
                            common_window_dd_from_running_peak=float(nav[cw_t] / nav[P.t0:cw_t + 1].max() - 1.0),
                            common_top_losers=[(c_, round(w_ * 100, 2)) for c_, w_ in cl[:8]],
                            stock_w_at_common_peak=float(x["stkw"][cw_p]),
                            mdd_before_2024_07=pre, top_episodes=eps, monthly_2024_05_to_2025_05=mwin)
        rva[cid] = per
    # 細部：共同區間分段報酬、與位移 0 的逐檔損益差、區間內每次換股買了什麼、逐年最大回撤
    cuts = ["2024-07-17", "2024-08-30", "2024-09-30", "2024-10-31", "2024-12-31", "2025-02-27", "2025-03-31",
            "2025-04-09"]
    ci = [L.didx(P, c_) for c_ in cuts]
    det = {}
    for cid in ("WN_70L", "WN_60L", MAIN, "W10_70L"):
        dd_ = {}
        w0 = K.window_pnl(P, keep_nav[(cid, 0)], cw_p, cw_t)
        n0p = keep_nav[(cid, 0)]["nav"][cw_p]
        for shf in range(4):
            x = keep_nav[(cid, shf)]; nav = x["nav"]
            seg = {"%s→%s" % (cuts[j], cuts[j + 1]): round(float(nav[ci[j + 1]] / nav[ci[j]] - 1.0) * 100, 2)
                   for j in range(len(cuts) - 1)}
            wk = K.window_pnl(P, x, cw_p, cw_t)
            diff = {P.codes[i]: (wk.get(i, 0.0) / nav[cw_p] - w0.get(i, 0.0) / n0p) * 100 for i in set(wk) | set(w0)}
            tr = x["trades"]
            buys_w = {}
            for c_, tb_, cost_ in zip(tr.code, tr.tb, tr.cost):
                if L.didx(P, "2024-06-01") <= tb_ <= cw_t:
                    buys_w.setdefault(P.dstr[int(tb_)], []).append((c_, round(float(cost_ / nav[int(tb_) - 1]) * 100, 1)))
            ym = {}
            for y in range(2016, 2027):
                a_ = max(L.didx(P, "%d-01-01" % y) - 1, P.t0)
                b_ = min(L.didx(P, "%d-01-01" % (y + 1)) - 1, e)
                ym[y] = round(L.maxdd(P, nav, a_, b_)[0] * 100, 2)
            dd_[shf] = dict(segments=seg,
                            pnl_diff_vs_shift0_top=[(k_, round(v_, 2)) for k_, v_ in
                                                    sorted(diff.items(), key=lambda z: z[1])[:6]],
                            pnl_diff_vs_shift0_best=[(k_, round(v_, 2)) for k_, v_ in
                                                     sorted(diff.items(), key=lambda z: -z[1])[:4]],
                            pnl_diff_sum=round(sum(diff.values()), 2),
                            buys_2024_06_to_trough=buys_w, yearly_mdd=ym)
        det[cid] = dd_
    rva["detail"] = det
    # 位移 k 的換股執行日（相對位移 0 晚 k 個交易日）在 2024-06～2025-04 的日期
    rva["exec_dates_2024_06_2025_04"] = {shf: [P.dstr[s + 1] for s in sorted(shift_plans[shf][0])
                                               if "2024-06-01" <= P.dstr[s + 1] <= "2025-04-30"] for shf in range(4)}
    # 2024-07 以前的最大回撤（20 條路徑），WN 類與 W10 類各格，依位移的中位數
    pre_rows = []
    for cid in ("WN_70L", MAIN, "WN_60L", "WN_70", "WN_80L", "WN_100", "W10_70L", "W10_70", "W10_80L"):
        for shf in range(4):
            for cp in CAPS:
                x = run(cid, plans=shift_plans[shf], capital=cp)
                nav = x["nav"]
                pre = L.maxdd(P, nav, P.t0, L.didx(P, "2024-06-28"))
                post = L.maxdd(P, nav, L.didx(P, "2024-06-28"), e)
                # 2024-07-01 → 2025-04-30 區間內（以該區間開始前的歷史高點為準）的最深回撤
                j0, j1 = L.didx(P, "2024-07-01"), L.didx(P, "2025-04-30")
                rk_ = np.maximum.accumulate(nav[P.t0:j1 + 1])[j0 - P.t0:]
                ep = float((nav[j0:j1 + 1] / rk_ - 1.0).min())
                pre_rows.append(dict(cell=cid, shift=shf, capital=cp, mdd_pre=pre[0], pre_peak=pre[1],
                                     pre_trough=pre[2], mdd_post=post[0], dd_2024_25=ep))
                del x
    prd = pd.DataFrame(pre_rows)
    rs = {}
    for cid, g in prd.groupby("cell"):
        rs[cid] = dict(mdd_pre_median_by_shift={int(s): float(np.median(v.mdd_pre)) for s, v in g.groupby("shift")},
                       dd_2024_25_median_by_shift={int(s): float(np.median(v.dd_2024_25)) for s, v in g.groupby("shift")},
                       mdd_post_median_by_shift={int(s): float(np.median(v.mdd_post)) for s, v in g.groupby("shift")},
                       pre_peaks={str(k): int(v) for k, v in g.pre_peak.str[:7].value_counts().items()})
    rva["struct"] = rs
    rva["struct_rows"] = prd.to_dict("records")
    out["reviewA"] = rva
    print("reviewA done rss", rss(), round(time.time() - t_start), flush=True)

    # ---------------- 審查 B：每檔占總資產 vs 20 條路徑回撤中位數
    xs = np.array([out[c]["per_stock_w_daily"] for c in ALL]); ys = np.array([pz[c]["mdd_median"] for c in ALL])
    xb = np.array([out[c]["per_stock_w_at_buy_mean"] for c in ALL])
    xm = np.array([pz[c]["per_stock_w_median"] for c in ALL])

    def ols(x, y):
        A = np.vstack([np.ones_like(x), x]).T
        cf, *_ = np.linalg.lstsq(A, y, rcond=None)
        yh = A @ cf; r2 = 1 - ((y - yh) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        n = len(x); se = np.sqrt(((y - yh) ** 2).sum() / (n - 2) / ((x - x.mean()) ** 2).sum())
        # 二次項
        A2 = np.vstack([np.ones_like(x), x, x ** 2]).T
        cf2, *_ = np.linalg.lstsq(A2, y, rcond=None); yh2 = A2 @ cf2
        r22 = 1 - ((y - yh2) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        return dict(intercept=float(cf[0]), slope=float(cf[1]), slope_se=float(se), r2=float(r2), n=n,
                    pp_dd_per_1pp_weight=float(cf[1]), r2_quadratic=float(r22),
                    resid={c: round(float(v) * 100, 2) for c, v in zip(list(ALL), y - yh)} if n == len(ALL) else None)
    rvb = dict(
        daily_weight_vs_median_dd_all17=ols(xs, ys),
        daily_weight_vs_median_dd_req12=ols(np.array([out[c]["per_stock_w_daily"] for c in REQ]),
                                            np.array([pz[c]["mdd_median"] for c in REQ])),
        buy_weight_vs_median_dd_all17=ols(xb, ys),
        path_median_weight_vs_median_dd_all17=ols(xm, ys),
        per_path_weight_vs_dd_all340=ols(pr.per_stock_w.values, pr.mdd_full.values),
        table={c: dict(per_stock_w_daily=round(out[c]["per_stock_w_daily"] * 100, 2),
                       per_stock_w_buy=round(out[c]["per_stock_w_at_buy_mean"] * 100, 2),
                       avg_stock_w=round(out[c]["avg_stock_w"] * 100, 2),
                       mdd_median=round(pz[c]["mdd_median"] * 100, 2),
                       mdd_single=round(out[c]["mdd_full"] * 100, 2)) for c in ALL})
    # 只用總個股部位（不是每檔）解釋，對照
    rvb["stock_w_vs_median_dd_all17"] = ols(np.array([out[c]["avg_stock_w"] for c in ALL]), ys)
    out["reviewB"] = rvb
    out["runtime_s"] = time.time() - t_start; out["peak_rss_mb"] = rss()

    # ---------------- 存檔、凍結
    trs = []
    for k in S8CELLS:
        t = res[k]["trades"].copy(); t["run"] = k; trs.append(t)
    for (cid, d), x in dres.items():
        if cid == "WN_70L":
            continue
        t = x["trades"].copy(); t["run"] = "%s_d%d" % (cid, d); trs.append(t)
    pd.concat(trs, ignore_index=True).drop(columns=["i"]).to_csv(os.path.join(odir, "my_trades.csv.gz"), index=False,
                                                                 compression="gzip", float_format="%.2f")
    js = json.dumps(out, ensure_ascii=False, indent=1, default=float)
    open(os.path.join(odir, "my_numbers.json"), "w").write(js)
    h = hashlib.sha256(open(os.path.join(odir, "my_numbers.json"), "rb").read()).hexdigest()
    open(os.path.join(odir, "my_numbers.sha256"), "w").write(h + "  my_numbers.json\n")
    print("FROZEN", h, "rss", rss(), "s", round(time.time() - t_start))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else HERE)
