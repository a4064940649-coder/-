# -*- coding: utf-8 -*-
"""S12 獨立複製：驗收、CUR／TL／TM／TS／Q55／TL_ADV 主跑＋八項、20 條路徑、各層選股超額、拆解與 β、四段空頭、
事前判定、審查 A（CUR 2024 起損益的層別來源，金額 vs 報酬貢獻）、B（大型層選股超額：位移 1～3、前 5 名）、
C（Q55 每年再平衡），然後凍結。
用法：python3 -I -B rep_s12_run.py <scratch_dir>
需先跑 s1 的 rep_s1_panel.py <scratch_dir> 與 s3 的 rep_s3_names.py <scratch_dir>/innov_names.json。
輸出（本資料夾）：my_numbers.json、my_numbers.sha256、my_trades.csv.gz、my_paths.csv.gz
凍結前不讀研究員的任何輸出（FINDINGS 第 1、2 節以外、results*.json、trades.csv.gz、navs.csv.gz）。
"""
import sys, os, json, hashlib, resource, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import rep_s12_lib as K
S, Q, R, L = K.S, K.Q, K.R, K.L
ST, X2 = K.ST, K.X2

CAPS = [600_000.0, 800_000.0, 1_000_000.0, 1_500_000.0, 2_000_000.0]
SHIFTS = [0, 1, 2, 3]
KNOWN = (0.4814, 0.3409)
BEARS = {"2018": ("2018-01-01", "2018-12-31"), "2020-03": ("2020-01-01", "2020-06-30"),
         "2022": ("2022-01-01", "2022-12-31"), "2024-07~2025-04": ("2024-07-01", "2025-04-30")}
CELLS = ["CUR", "TL", "TM", "TS", "Q55", "TL_ADV"]
# 單帳戶格：(分層依據, 層集合)
SPEC = {"CUR": ("mc", None), "TL": ("mc", (1,)), "TM": ("mc", (2,)), "TS": ("mc", (3,)), "TL_ADV": ("adv", (1,)),
        "Q55L": ("mc", (1,)), "Q55S": ("mc", (2, 3))}
TIERNAME = {1: "L", 2: "M", 3: "S"}


def rss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024


def tst(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return None
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))


def ols(y, x):
    y = np.asarray(y, float); x = np.asarray(x, float)
    ok = np.isfinite(y) & np.isfinite(x)
    y = y[ok]; x = x[ok]
    X = np.column_stack([np.ones(len(x)), x])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ b
    s2 = res @ res / (len(y) - 2)
    cov = s2 * np.linalg.inv(X.T @ X)
    return dict(beta=float(b[1]), t=float(b[1] / np.sqrt(cov[1, 1])), alpha=float(b[0]), n=int(len(y)))


def main(sc, odir=HERE):
    t_start = time.time()
    P = R.build(sc)
    S.attach_e50(P, sc)
    excl = S.innov_mask_fn(P, sc)
    Rv, mi0 = L.load_rev(P)
    S_, _ = L.surge(Rv, mi0, null2013=True)
    K.load_shares(P)
    ma60 = R.ma(P, 60)
    AO, Gf = K.adj_open_matrix(P)
    e = P.T - 1
    i19 = L.didx(P, "2018-12-28"); i24 = L.didx(P, "2023-12-29"); ih = L.didx(P, "2021-02-19")
    da = np.array(P.dstr)

    def win(a, b):
        return int(np.searchsorted(da, a, side="left")), int(np.searchsorted(da, b, side="right")) - 1
    bw = {k: win(a, b) for k, (a, b) in BEARS.items()}
    e50 = P.e50.values.astype(float)

    def bears(nav):
        return {k: L.maxdd(P, nav, a, b)[0] for k, (a, b) in bw.items()}

    out = dict(meta=dict(T=P.T, N=P.N, first=P.dstr[0], last=P.dstr[-1], t0=P.dstr[P.t0], i19=P.dstr[i19],
                         i24=P.dstr[i24], ih=P.dstr[ih], share_cols_matched=int(P.sh_cols),
                         bear_windows={k: (P.dstr[a], P.dstr[b]) for k, (a, b) in bw.items()},
                         note="面板為複製員自建（N＝2130，研究員 2142；S6～S11 複製時已知，差的是從未進股池的代號）"))
    b50 = dict(ann_2019=L.ann(P, e50, i19, e), ann_full=L.ann(P, e50, P.t0, e), ann_2024=L.ann(P, e50, i24, e),
               ann_1923=L.ann(P, e50, i19, i24), mdd_full=L.maxdd(P, e50, P.t0, e)[0],
               ann_h1=L.ann(P, e50, P.t0, ih), ann_h2=L.ann(P, e50, ih, e), bears=bears(e50), yearly=L.yearly(P, e50))
    out["bench_0050"] = b50
    print("0050", {k: v for k, v in b50.items() if k != "yearly"}, flush=True)

    # ---------------- 名單、分層（位移 0～3）
    lists = {}; tier_of = {}; plans = {}; lmat = {}
    a2 = {}
    dck = []
    for sh in SHIFTS:
        lists[sh] = Q.make_lists(P, S_, mi0, shift=sh, excl=excl)
        tier_of[(sh, "mc")] = {s: K.tiers(P, s, excl, "mc") for s in lists[sh]}
        tier_of[(sh, "adv")] = {s: K.tiers(P, s, excl, "adv") for s in lists[sh]}
        for nm, (by, ts) in SPEC.items():
            ck = {} if ts is not None else None
            b, k, lt = K.tier_plans(P, S_, lists[sh], tier_of[(sh, by)], None if ts is None else set(ts), excl,
                                    ma60=ma60, check=ck)
            plans[(nm, sh)] = (b, k); lmat[(nm, sh)] = lt
            if ck is not None:
                a2["%s_sh%d" % (nm, sh)] = ck
    print("plans", len(plans), "rss", rss(), round(time.time() - t_start), flush=True)
    # 資料檢查重現（位移 0）
    for s in sorted(lists[0]):
        pool = R.universe(P, s) & ~excl(s)
        mc, okl, _ = K.mcap(P, s, detail=True)
        t = tier_of[(0, "mc")][s]
        dck.append(dict(n_pool=int(pool.sum()), cov_lag=float(okl[pool].mean()), cov_any=float(np.isfinite(mc[pool]).mean()),
                        n_nomc=int((~np.isfinite(mc[pool])).sum()), n_S=int((t == 3).sum()),
                        n_f1_L=len(plans[("TL", 0)][0][s]), n_f1_M=len(plans[("TM", 0)][0][s]),
                        n_f1_S=len(plans[("TS", 0)][0][s]), n_f1_cur=len(plans[("CUR", 0)][0][s]),
                        n_ma_L=len(lmat[("TL", 0)][s]), n_ma_M=len(lmat[("TM", 0)][s]), n_ma_S=len(lmat[("TS", 0)][s])))
    dd_ = pd.DataFrame(dck)
    out["data_check_repro"] = {c: [float(dd_[c].mean()), float(dd_[c].min()), float(dd_[c].max())] for c in dd_.columns}
    out["data_check_repro"]["n_months"] = int(len(dd_))

    # ---------------- 引擎包裝
    def sim(nm, sh, cap, cost=None, block_span=None, rec_pos=False):
        b, k = plans[(nm, sh)]
        if nm.startswith("Q55"):
            return K.simulate_s12(P, b, k, sat_w=0.7, sizing="W10", nmax=5, ntgt=5, capital=cap,
                                  block_span=block_span, rec_pos=rec_pos, **(cost or ST))
        return K.simulate_s12(P, b, k, sat_w=0.7, sizing="W10", capital=cap, block_span=block_span, rec_pos=rec_pos,
                              **(cost or ST))

    def run_cell(cc, sh=0, cap=1_000_000.0, cost=None, block_span=None, rec_pos=False):
        if cc == "Q55":
            xa = sim("Q55L", sh, cap / 2, cost, block_span, rec_pos)
            xb = sim("Q55S", sh, cap / 2, cost, block_span, rec_pos)
            tr = pd.concat([xa["trades"].assign(sub="L"), xb["trades"].assign(sub="MS")], ignore_index=True)
            return dict(nav=xa["nav"] + xb["nav"], trades=tr, open_pos=xa["open_pos"] + xb["open_pos"],
                        subs=(xa, xb))
        x = sim(cc, sh, cap, cost, block_span, rec_pos)
        x["trades"] = x["trades"].assign(sub=cc)
        return x

    def top5(tr, nav):
        inc = float(nav[e] - nav[P.t0])
        t5 = tr.sort_values("pnl", ascending=False).head(5)
        return float(t5.pnl.sum() / inc)

    def nav_stats(nav):
        md = L.maxdd(P, nav, P.t0, e)
        return dict(ann_2019=L.ann(P, nav, i19, e), ann_full=L.ann(P, nav, P.t0, e), ann_2024=L.ann(P, nav, i24, e),
                    ann_1923=L.ann(P, nav, i19, i24), mdd_full=md[0], mdd_peak=md[1], mdd_trough=md[2],
                    ann_h1=L.ann(P, nav, P.t0, ih), ann_h2=L.ann(P, nav, ih, e))

    # ---------------- 驗收
    acc = dict(P0=dict(T=P.T, N=P.N, last=P.dstr[-1], note="研究員 N＝2142；複製員面板 2130（已知差異）"))
    b0s = [Q.cell_plans(lists[sh], "C10", x=0.10) for sh in SHIFTS]
    acc["A1"] = dict(identical=bool(all(
        sorted(b0s[n][0]) == sorted(plans[("CUR", sh)][0]) and
        all((b0s[n][0][s] == plans[("CUR", sh)][0][s]).all() and b0s[n][1][s] == plans[("CUR", sh)][1][s]
            for s in b0s[n][0]) for n, sh in enumerate(SHIFTS))))
    main = {}
    main["CUR"] = run_cell("CUR", rec_pos=True)
    nav0 = main["CUR"]["nav"]
    x9 = K.K9.simulate_s9(P, *plans[("CUR", 0)], sat_w=0.7, sizing="W10", **ST)
    same9 = bool(np.allclose(x9["nav"][P.t0:], nav0[P.t0:], rtol=0, atol=1e-6) and
                 len(x9["trades"]) == len(main["CUR"]["trades"]) and
                 (x9["trades"][["code", "tb", "ts"]].values == main["CUR"]["trades"][["code", "tb", "ts"]].values).all())
    acc["A0"] = dict(ann_2019=L.ann(P, nav0, i19, e), ann_full=L.ann(P, nav0, P.t0, e),
                     mdd_full=L.maxdd(P, nav0, P.t0, e)[0], known=KNOWN, same_as_simulate_s9=same9,
                     ok=bool(abs(L.ann(P, nav0, i19, e) - KNOWN[0]) < 0.00006 and
                             abs(L.ann(P, nav0, P.t0, e) - KNOWN[1]) < 0.00006 and same9))
    print("A0", acc["A0"], flush=True)
    assert acc["A0"]["ok"]
    acc["A2"] = dict(per_plan=a2, total_diff_cells=int(sum(v["diff_cells"] for v in a2.values())),
                     total_buy_not_in_tier=int(sum(v["buy_not_in_tier"] for v in a2.values())))
    b, k = plans[("CUR", 0)]
    x3 = K.simulate_s12(P, b, k, sat_w=0.7, sizing="WN", nmax=10, ntgt=10, **ST)
    acc["A3"] = dict(same_trades=bool(len(x3["trades"]) == len(main["CUR"]["trades"]) and
                                      (x3["trades"][["code", "tb", "ts", "sh"]].values ==
                                       main["CUR"]["trades"][["code", "tb", "ts", "sh"]].values).all()),
                     max_nav_diff=float(np.nanmax(np.abs(x3["nav"] - nav0))))
    print("A1/A2/A3", acc["A1"], acc["A2"]["total_diff_cells"], acc["A2"]["total_buy_not_in_tier"], acc["A3"], flush=True)

    # ---------------- 主跑、成本加倍、禁買、八項
    for cc in CELLS[1:]:
        main[cc] = run_cell(cc, rec_pos=(cc == "Q55"))
    cells = {}
    trade_rows = []
    for cc in CELLS:
        x = main[cc]; nav = x["nav"]; tr = x["trades"]
        o = nav_stats(nav); o["n_trades"] = int(len(tr)); o["bears"] = bears(nav); o["yearly"] = L.yearly(P, nav)
        x2 = run_cell(cc, cost=X2)
        sp, top = R.span_of(tr, 5)
        bl = run_cell(cc, block_span=sp)
        o["x2"] = dict(ann_2019=L.ann(P, x2["nav"], i19, e), ann_full=L.ann(P, x2["nav"], P.t0, e))
        o["block5"] = dict(ann_2019=L.ann(P, bl["nav"], i19, e), ann_full=L.ann(P, bl["nav"], P.t0, e))
        o["block5"]["drop_2019"] = o["ann_2019"] - o["block5"]["ann_2019"]
        o["block5"]["drop_full"] = o["ann_full"] - o["block5"]["ann_full"]
        o["top5"] = top
        o["top5_share_of_gain"] = top5(tr, nav)
        o["final_nav"] = float(nav[e])
        o["trade_stats"] = dict(exp=float(tr.ret.mean()), win=float((tr.ret > 0).mean()), hold_mean=float(tr.hold.mean()),
                                n_delist=int(tr.delist.sum()))
        ck = dict(a2019=o["ann_2019"] > b50["ann_2019"], afull=o["ann_full"] > b50["ann_full"],
                  x2_2019=o["x2"]["ann_2019"] > b50["ann_2019"], x2_full=o["x2"]["ann_full"] > b50["ann_full"],
                  bl_2019=o["block5"]["ann_2019"] > b50["ann_2019"], bl_full=o["block5"]["ann_full"] > b50["ann_full"],
                  h1=o["ann_h1"] > b50["ann_h1"], h2=o["ann_h2"] > b50["ann_h2"])
        o["checks"] = ck; o["n_checks_pass"] = int(sum(ck.values()))
        o["eight_cls"] = "值得上 XQ 測" if all(ck.values()) else (
            "不值得" if (not ck["a2019"] and not ck["afull"]) else "有條件")
        if cc == "Q55":
            xa, xb = x["subs"]
            share = xa["nav"][P.t0:] / nav[P.t0:]
            o["q55_large_share"] = dict(min=float(share.min()), max=float(share.max()), end=float(share[-1]),
                                        at_2019=float(xa["nav"][i19] / nav[i19]), at_2024=float(xa["nav"][i24] / nav[i24]))
            o["sub_trades"] = dict(L=int(len(xa["trades"])), MS=int(len(xb["trades"])))
        cells[cc] = o
        for r_ in tr.itertuples():
            tier_e = int(tier_of[(0, "mc")][r_.tb - 1][r_.i]) if (r_.tb - 1) in tier_of[(0, "mc")] else -1
            trade_rows.append((cc, r_.sub, r_.code, r_.buy_date, r_.sell_date, round(r_.ret, 6), round(r_.pnl, 1),
                               int(r_.hold), int(r_.delist), int(r_.reduced), tier_e))
        print(cc, round(o["ann_2019"], 4), round(o["ann_full"], 4), round(o["ann_2024"], 4), round(o["ann_1923"], 4),
              round(o["mdd_full"], 4), o["n_trades"], round(o["top5_share_of_gain"], 3), o["n_checks_pass"],
              o["eight_cls"], "rss", rss(), round(time.time() - t_start), flush=True)
    out["acceptance"] = acc
    out["cells"] = cells
    tdf = pd.DataFrame(trade_rows, columns=["cell", "sub", "code", "buy_date", "sell_date", "ret", "pnl", "hold", "delist",
                                            "reduced", "entry_tier_mc"])
    tdf.to_csv(os.path.join(odir, "my_trades.csv.gz"), index=False, compression="gzip")

    # ---------------- 20 條路徑
    rows = []
    pnav = {}
    qsub = {}
    for cc in CELLS:
        for sh in SHIFTS:
            for cp in CAPS:
                if sh == 0 and cp == 1_000_000.0:
                    x = main[cc]
                else:
                    x = run_cell(cc, sh, cp)
                nav = x["nav"]
                pnav[(cc, sh, cp)] = nav
                if cc == "Q55":
                    xa, xb = x["subs"]
                    qsub[(sh, cp)] = (xa["nav"], xb["nav"], xa["trades"], xb["trades"])
                st = nav_stats(nav)
                rows.append(dict(cell=cc, shift=sh, capital=cp, **st, n_trades=int(len(x["trades"])),
                                 top5_share=top5(x["trades"], nav), **{"bear_" + k: v for k, v in bears(nav).items()}))
        print("paths", cc, "rss", rss(), round(time.time() - t_start), flush=True)
    pr = pd.DataFrame(rows)
    pr.to_csv(os.path.join(odir, "my_paths.csv.gz"), index=False, float_format="%.6f", compression="gzip")
    p20 = {}
    for cc, g in pr.groupby("cell"):
        p20[cc] = dict(ann_2019_median=float(g.ann_2019.median()), ann_2019_min=float(g.ann_2019.min()),
                       ann_2019_max=float(g.ann_2019.max()), n_beat_0050=int((g.ann_2019 > b50["ann_2019"]).sum()),
                       mdd_median=float(g.mdd_full.median()), ann_full_median=float(g.ann_full.median()),
                       ann_2024_median=float(g.ann_2024.median()), ann_1923_median=float(g.ann_1923.median()),
                       top5_share_median=float(g.top5_share.median()), n_trades_median=float(g.n_trades.median()),
                       bears_median={k: float(g["bear_" + k].median()) for k in BEARS})
    out["paths20"] = p20

    def paired(a, b, col):
        x = pr[pr.cell == a].set_index(["shift", "capital"])[col]
        y = pr[pr.cell == b].set_index(["shift", "capital"])[col]
        return (x - y.loc[x.index]).values

    pair = {}
    for a in CELLS[1:]:
        o = {}
        for col in ["ann_2019", "ann_full", "ann_2024", "ann_1923", "mdd_full", "top5_share"] + ["bear_" + k for k in BEARS]:
            d = paired(a, "CUR", col)
            o[col] = dict(n_pos=int((d > 0).sum()), median=float(np.median(d)), min=float(d.min()), max=float(d.max()))
        pair["%s-CUR" % a] = o
    out["paired_vs_CUR"] = pair

    # ---------------- 時窗：同層等權／名單等權／0050（扣成本前）
    def ew_table(sh, depth_plans=None):
        """回傳 DataFrame：每個時窗 × 層（ALL、L、M、S）的 EW、名單等權、名單檔數、0050。"""
        ws = K.windows(P, lists[sh])
        trow = []
        tmap = {"ALL": ("CUR", None), "L": ("TL", 1), "M": ("TM", 2), "S": ("TS", 3)}
        for s, ee, e2 in ws:
            tt = tier_of[(sh, "mc")][s]
            pool = R.universe(P, s) & ~excl(s)
            op = np.isfinite(P.O[ee])
            r50 = K.e50_win(P, ee, e2)
            r50c = (e50[e2 - 1] if e2 is not None else e50[e]) / e50[ee - 1] - 1.0
            for tn, (cnm, tv) in tmap.items():
                mem = pool & op if tv is None else pool & op & (tt == tv)
                idx = np.where(mem)[0]
                rew = K.win_ret(P, AO, Gf, idx, ee, e2)
                pl = (depth_plans or plans)[(cnm, sh)][0][s]
                pl = np.array([i for i in pl if op[i]], dtype=int)
                rl = K.win_ret(P, AO, Gf, pl, ee, e2) if len(pl) else rew
                trow.append(dict(s=s, e=ee, e2=-1 if e2 is None else e2, exec=P.dstr[ee],
                                 end=P.dstr[e2] if e2 is not None else P.dstr[e], year=int(P.dstr[ee][:4]), tier=tn,
                                 n_mem=len(idx), n_list=len(pl), r_ew=rew, r_list=rl, r50=r50, r50c=r50c))
        return pd.DataFrame(trow)

    def chain_ann(g, col):
        if len(g) == 0:
            return None
        v = float(np.prod(1.0 + g[col].values))
        yrs = (pd.Timestamp(g.end.iloc[-1]) - pd.Timestamp(g.exec.iloc[0])).days / 365.25
        return v ** (1.0 / yrs) - 1.0

    def sel_stats(g):
        g = g.sort_values("e")
        o = {}
        for nm_, sub in (("full", g), ("2019_2023", g[(g.year >= 2019) & (g.year <= 2023)]), ("from2019", g[g.year >= 2019]),
                         ("from2024", g[g.year >= 2024])):
            al = chain_ann(sub, "r_list"); ae = chain_ann(sub, "r_ew")
            o[nm_] = dict(ann_list=al, ann_ew=ae, diff=al - ae)
        ex = g.r_list - g.r_ew
        o["t_month"] = tst(ex); o["mean_month"] = float(ex.mean()); o["n_months"] = int(len(ex))
        o["n_months_list_empty"] = int((g.n_list == 0).sum())
        yy = pd.Series({y_: float(np.prod(1 + h.r_list.values) - np.prod(1 + h.r_ew.values)) for y_, h in g.groupby("year")})
        o["yearly_diff"] = {int(k): float(v) for k, v in yy.items()}
        o["n_years_pos"] = int((yy > 0).sum()); o["n_years"] = int(len(yy))
        d = o["full"]["diff"]
        o["cls"] = ("選股超額存在" if (d > 0 and o["n_years_pos"] >= 7 and (o["t_month"] or 0) >= 2) else
                    ("弱" if d > 0 else "不存在"))
        return o

    ew = {sh: ew_table(sh) for sh in SHIFTS}
    S1 = {}
    for sh in SHIFTS:
        S1["shift%d" % sh] = {tn: sel_stats(ew[sh][ew[sh].tier == tn]) for tn in ("ALL", "L", "M", "S")}
    out["S1_selection"] = S1
    print("S1", json.dumps({tn: (S1["shift0"][tn]["full"]["diff"], S1["shift0"][tn]["t_month"], S1["shift0"][tn]["n_years_pos"])
                            for tn in S1["shift0"]}), flush=True)
    # A4：0050 時窗串接 vs 官方 0050
    g0 = ew[0][ew[0].tier == "ALL"].sort_values("e")
    a4c = chain_ann(g0, "r50")
    a4o = L.ann(P, e50, int(g0.e.iloc[0]) - 1, e)
    acc["A4"] = dict(chain=a4c, official=a4o, diff=a4c - a4o, ok=bool(abs(a4c - a4o) <= 0.005),
                     start=g0.exec.iloc[0])

    # ---------------- 拆解（S7）與 β（S6）
    def nav_win(nav, g):
        a = nav[g.e.values - 1]
        b = np.where(g.e2.values >= 0, nav[np.maximum(g.e2.values, 1) - 1], nav[e])
        return b / a - 1.0

    def decomp(cc, tn, wser=None):
        g = ew[0][ew[0].tier == tn].sort_values("e").copy()
        nav = main[cc]["nav"]
        w = 0.7 if wser is None else wser(g)
        g["act"] = nav_win(nav, g)
        g["b_ew"] = (1 - w) * g.r50 + w * g.r_ew
        g["b_list"] = (1 - w) * g.r50 + w * g.r_list
        nl = np.minimum(g.n_list.values, 10)
        satd = np.where(nl > 0, nl / 10.0 * g.r_list + (1 - nl / 10.0) * g.r50, g.r50)
        g["b_dil"] = (1 - w) * g.r50 + w * satd
        o = {}
        for nm_, sub in (("full", g), ("from2019", g[g.year >= 2019]), ("2019_2023", g[(g.year >= 2019) & (g.year <= 2023)]),
                         ("from2024", g[g.year >= 2024])):
            a = {c: chain_ann(sub, c) for c in ("r50", "b_ew", "b_list", "b_dil", "act", "r50c")}
            o[nm_] = dict(ann=a, total_vs_0050=a["act"] - a["r50"], tier_lag=a["b_ew"] - a["r50"],
                          selection=a["b_list"] - a["b_ew"], exec_loss=a["act"] - a["b_list"],
                          w10_dilution=a["b_dil"] - a["b_list"], exec_other=a["act"] - a["b_dil"],
                          start=sub.exec.iloc[0], end=sub.end.iloc[-1])
        return o
    dec = {cc: decomp(cc, tn) for cc, tn in (("CUR", "ALL"), ("TL", "L"), ("TM", "M"), ("TS", "S"))}
    stkw = main["CUR"]["stkw"]
    dec["CUR_actual_weight_posthoc"] = decomp("CUR", "ALL", wser=lambda g: stkw[g.e.values])
    dec["CUR_actual_weight_posthoc"]["mean_w"] = float(np.mean(stkw[g0.e.values]))
    out["S7_decomposition"] = dec
    xfac = (g0.r_ew - g0.r50).values
    beta = {}
    for cc in CELLS:
        y = nav_win(main[cc]["nav"], g0) - g0.r50c.values
        beta[cc] = ols(y, xfac)
        beta[cc]["start"] = g0.exec.iloc[0]
    out["S6_beta"] = beta
    print("beta", {k: round(v["beta"], 3) for k, v in beta.items()}, flush=True)

    # ---------------- 事前判定
    dec_out = {}
    pt = p20["TL"]
    if pt["ann_2019_median"] > b50["ann_2019"] and pt["n_beat_0050"] >= 15:
        tl_cls = "SURGE 在大型股有效"
    elif pt["ann_2019_median"] < b50["ann_2019"] and pt["n_beat_0050"] <= 5:
        tl_cls = "在大型股無效"
    else:
        tl_cls = "不確定"
    dec_out["TL_surge_large"] = tl_cls
    dec_out["selection_by_tier"] = {tn: S1["shift0"][tn]["cls"] for tn in ("L", "M", "S", "ALL")}

    def dep_rule(cc):
        a_ = p20[cc]["top5_share_median"] < p20["CUR"]["top5_share_median"]
        pc = pair["%s-CUR" % cc]
        b_ = pc["ann_1923"]["median"] >= -0.03 and pc["ann_2024"]["median"] >= -0.03
        cls = "降低依賴" if (a_ and b_) else ("較分散，但代價超過 3 個百分點" if a_ else "沒有降低依賴")
        return dict(a_top5_lower=bool(a_), top5_median=p20[cc]["top5_share_median"],
                    cur_top5_median=p20["CUR"]["top5_share_median"], b_ok=bool(b_),
                    paired_1923_median=pc["ann_1923"]["median"], paired_2024_median=pc["ann_2024"]["median"],
                    cost_paired_2019_median=pc["ann_2019"]["median"], cost_paired_mdd_median=pc["mdd_full"]["median"],
                    cls=cls)
    dec_out["Q55_dependency"] = dep_rule("Q55")
    dec_out["TL_dependency_secondary"] = dep_rule("TL")
    bear_cls = {}
    for cc in CELLS[1:]:
        pc = pair["%s-CUR" % cc]; o = {}
        for k in BEARS:
            v = pc["bear_" + k]
            o[k] = dict(n_shallower=v["n_pos"], median=v["median"],
                        cls=("較淺" if (v["n_pos"] >= 15 and v["median"] >= 0.01) else
                             ("較深" if (v["n_pos"] <= 5 and v["median"] <= -0.01) else "差不多")))
        bear_cls[cc] = o
    dec_out["bears"] = bear_cls
    dec_out["beta"] = {cc: dict(beta=beta[cc]["beta"], cur=beta["CUR"]["beta"],
                                cls=("降低中小型股曝險" if beta[cc]["beta"] <= beta["CUR"]["beta"] - 0.10 else "沒有明顯降低"))
                       for cc in ("Q55", "TL")}
    dec_out["eight_checks"] = {cc: (cells[cc]["n_checks_pass"], cells[cc]["eight_cls"]) for cc in CELLS}
    xq_ok = ((dec_out["Q55_dependency"]["cls"] == "降低依賴" and cells["Q55"]["eight_cls"] == "值得上 XQ 測") or
             (tl_cls == "SURGE 在大型股有效" and cells["TL"]["eight_cls"] == "值得上 XQ 測"))
    dec_out["xq"] = "寫 20／21 號腳本要加的條件" if xq_ok else "不建議改現行"
    out["decisions"] = dec_out
    print("decisions", json.dumps(dec_out, ensure_ascii=False, default=str), flush=True)

    # ---------------- 事後：CUR 交易依進場層
    trc = tdf[tdf.cell == "CUR"]
    out["posthoc_cur_by_entry_tier"] = {TIERNAME.get(int(k), str(k)): dict(n=int(len(g)), exp=float(g.ret.mean()),
                                                                            win=float((g.ret > 0).mean()),
                                                                            hold_mean=float(g.hold.mean()),
                                                                            pnl=float(g.pnl.sum()))
                                        for k, g in trc.groupby("entry_tier_mc")}
    out["posthoc_yearly"] = {cc: cells[cc]["yearly"] for cc in CELLS}

    # ---------------- 審查 A：CUR 2024 起損益依進場層（金額 vs 報酬貢獻）
    xc = main["CUR"]; snap = xc["snap"]; flows = xc["flows"]
    trc_full = xc["trades"]
    ent = {}   # (i, tb) -> tier；逐日查詢用的區間
    spans = {}
    for r_ in trc_full.itertuples():
        spans.setdefault(r_.i, []).append((r_.tb, r_.ts, int(tier_of[(0, "mc")][r_.tb - 1][r_.i])))
    for code, bd, cost, val, i in xc["open_pos"]:
        tb = int(np.searchsorted(da, bd))
        spans.setdefault(i, []).append((tb, e + 1, int(tier_of[(0, "mc")][tb - 1][i])))

    def tier_at(i, t):
        for a, b_, tv in spans.get(i, ()):
            if a <= t <= b_:
                return tv
        return -1
    acc_m = {}; acc_c = {}
    navc = xc["nav"]
    per = {"from2024": (i24 + 1, e), "2019_2023": (i19 + 1, i24), "full": (P.t0 + 1, e)}
    resA = {}
    for pn, (a, b_) in per.items():
        mny = {1: 0.0, 2: 0.0, 3: 0.0, -1: 0.0}; con = {1: 0.0, 2: 0.0, 3: 0.0, -1: 0.0}
        for t in range(a, b_ + 1):
            prev = snap.get(t - 1, {}); cur = snap.get(t, {})
            for i in set(prev) | set(cur):
                pnl = cur.get(i, 0.0) - prev.get(i, 0.0) + flows.get((t, i), 0.0)
                tv = tier_at(i, t)
                mny[tv] += pnl; con[tv] += pnl / navc[t - 1]
        # 當天賣出（cur 沒有、prev 有）已含；當天買進（prev 沒有）已含
        tm = sum(mny.values()); tc = sum(con.values())
        resA[pn] = dict(money={TIERNAME.get(k, "?"): v for k, v in mny.items()},
                        money_share={TIERNAME.get(k, "?"): v / tm for k, v in mny.items()},
                        contrib={TIERNAME.get(k, "?"): v for k, v in con.items()},
                        contrib_share={TIERNAME.get(k, "?"): v / tc for k, v in con.items()},
                        stock_pnl_total=tm, nav_gain=float(navc[b_] - navc[a - 1]), contrib_total=tc)
    # 交易層級：2024 起進場（含期末未平倉）
    tlev = []
    for r_ in trc_full.itertuples():
        tlev.append((r_.tb, r_.i, r_.pnl, r_.ret, r_.cost))
    for code, bd, cost, val, i in xc["open_pos"]:
        tb = int(np.searchsorted(da, bd))
        tlev.append((tb, i, val - cost, val / cost - 1.0, cost))
    tl_ = pd.DataFrame(tlev, columns=["tb", "i", "pnl", "ret", "cost"])
    tl_["tier"] = [int(tier_of[(0, "mc")][tb - 1][i]) for tb, i in zip(tl_.tb, tl_.i)]
    tl_["w"] = tl_.cost / navc[tl_.tb.values - 1]
    tl_["wr"] = tl_.w * tl_.ret
    for pn, cond in (("entered_from2024", tl_.tb > i24), ("entered_2019_2023", (tl_.tb > i19) & (tl_.tb <= i24))):
        g = tl_[cond]
        sm = g.groupby("tier").pnl.sum(); sw = g.groupby("tier").wr.sum()
        resA[pn] = dict(n=int(len(g)), n_by_tier={TIERNAME[int(k)]: int(v) for k, v in g.groupby("tier").size().items()},
                        money_share={TIERNAME[int(k)]: float(v / sm.sum()) for k, v in sm.items()},
                        weight_x_ret_share={TIERNAME[int(k)]: float(v / sw.sum()) for k, v in sw.items()},
                        money={TIERNAME[int(k)]: float(v) for k, v in sm.items()},
                        weight_x_ret={TIERNAME[int(k)]: float(v) for k, v in sw.items()})
    resA["note"] = ("daily：逐日每檔損益＝收盤市值變動＋當日現金流（買進為負、賣出／下市為正），依該筆持股進場訊號日的市值層歸類；"
                    "money＝金額加總；contrib＝每日損益 ÷ 前一日總淨值後加總（權重 × 報酬，不受淨值放大）。"
                    "entered_*：依進場日歸期的交易（含期末未平倉），money＝損益金額，weight_x_ret＝進場成本 ÷ 前一日淨值 × 該筆報酬。"
                    "只含個股損益（不含 0050 底倉）。")
    out["review_A"] = resA
    print("reviewA", json.dumps({k: (v.get("money_share"), v.get("contrib_share") or v.get("weight_x_ret_share"))
                                 for k, v in resA.items() if k != "note"}, default=str), flush=True)

    # ---------------- 審查 B：大型層選股超額（位移 1～3、前 5 名）
    dp5 = {}
    for sh in SHIFTS:
        for nm, (by, ts) in SPEC.items():
            if nm in ("TL", "TM", "TS", "CUR"):
                b, k, _ = K.tier_plans(P, S_, lists[sh], tier_of[(sh, by)], None if ts is None else set(ts), excl,
                                       depth=5, ma60=ma60)
                dp5[(nm, sh)] = (b, k)
    revB = {}
    for sh in SHIFTS:
        g5 = ew_table(sh, depth_plans=dp5)
        revB["top5_shift%d" % sh] = {tn: sel_stats(g5[g5.tier == tn]) for tn in ("L", "M", "S", "ALL")}
        revB["top10_shift%d" % sh] = S1["shift%d" % sh]
    revB["note"] = ("top5＝層內 L_MA 前 5 名中 RET20 > 10%（其餘同 F1）；top10＝主設計。位移 s：訊號列往後移 s 天重算名單與分層。"
                    "只描述，不改判定。")
    out["review_B"] = revB
    print("reviewB", json.dumps({k: (v["L"]["full"]["diff"], v["L"]["2019_2023"]["diff"], v["L"]["from2024"]["diff"],
                                     v["L"]["t_month"], v["L"]["n_years_pos"]) for k, v in revB.items() if k != "note"}),
          flush=True)

    # ---------------- 審查 C：Q55 每年再平衡
    def annual_rebal(na, nb, ta, tb_):
        nav = np.full(P.T, np.nan)
        ye = [P.t0] + [int(np.searchsorted(da, "%d-01-01" % y)) - 1 for y in range(2016, 2027)]
        ye = sorted(set(i for i in ye if P.t0 <= i < e)) + [e]
        v = na[P.t0] + nb[P.t0]
        fa = np.ones(P.T); fb = np.ones(P.T)
        for a, b_ in zip(ye[:-1], ye[1:]):
            ka = 0.5 * v / na[a]; kb = 0.5 * v / nb[a]
            seg = ka * na[a:b_ + 1] + kb * nb[a:b_ + 1]
            nav[a:b_ + 1] = seg
            fa[a + 1:b_ + 1] = ka; fb[a + 1:b_ + 1] = kb
            v = seg[-1]
        pnl = np.r_[ta.pnl.values * fa[ta.tb.values], tb_.pnl.values * fb[tb_.tb.values]]
        t5 = float(np.sort(pnl)[::-1][:5].sum() / (nav[e] - nav[P.t0]))
        return nav, t5
    rowsC = []
    for sh in SHIFTS:
        for cp in CAPS:
            na, nb, ta, tb_ = qsub[(sh, cp)]
            nav_ar, t5 = annual_rebal(na, nb, ta, tb_)
            st = nav_stats(nav_ar)
            sh_ = na[P.t0:] / (na[P.t0:] + nb[P.t0:])
            rowsC.append(dict(shift=sh, capital=cp, **st, top5_share=t5, large_share_min=float(sh_.min()),
                              large_share_max=float(sh_.max()), **{"bear_" + k: v for k, v in bears(nav_ar).items()}))
    dc = pd.DataFrame(rowsC).set_index(["shift", "capital"])
    pq = pr[pr.cell == "Q55"].set_index(["shift", "capital"]).loc[dc.index]
    pc_ = pr[pr.cell == "CUR"].set_index(["shift", "capital"]).loc[dc.index]
    m = dc.loc[(0, 1_000_000.0)]
    revC = dict(main=dict(ann_2019=float(m.ann_2019), ann_full=float(m.ann_full), ann_2024=float(m.ann_2024),
                          ann_1923=float(m.ann_1923), mdd_full=float(m.mdd_full), top5_share=float(m.top5_share)),
                p20=dict(ann_2019_median=float(dc.ann_2019.median()), n_beat_0050=int((dc.ann_2019 > b50["ann_2019"]).sum()),
                         mdd_median=float(dc.mdd_full.median()), top5_share_median=float(dc.top5_share.median()),
                         ann_1923_median=float(dc.ann_1923.median()), ann_2024_median=float(dc.ann_2024.median())),
                q55_norebal_top5_median=float(pq.top5_share.median()), cur_top5_median=float(pc_.top5_share.median()),
                paired_vs_CUR={c: dict(median=float(np.median(dc[c] - pc_[c])), n_pos=int(((dc[c] - pc_[c]) > 0).sum()))
                               for c in ("ann_2019", "ann_1923", "ann_2024", "mdd_full", "top5_share")},
                paired_vs_Q55={c: dict(median=float(np.median(dc[c] - pq[c])), n_pos=int(((dc[c] - pq[c]) > 0).sum()))
                               for c in ("ann_2019", "ann_1923", "ann_2024", "mdd_full", "top5_share")},
                large_share_range_paths=dict(min_of_min=float(dc.large_share_min.min()),
                                             median_min=float(dc.large_share_min.median()),
                                             median_max=float(dc.large_share_max.median()),
                                             max_of_max=float(dc.large_share_max.max())))
    rc_a = revC["p20"]["top5_share_median"] < revC["cur_top5_median"]
    rc_b = revC["paired_vs_CUR"]["ann_1923"]["median"] >= -0.03 and revC["paired_vs_CUR"]["ann_2024"]["median"] >= -0.03
    revC["rule_applied_descriptive"] = ("降低依賴" if (rc_a and rc_b) else ("較分散，但代價超過 3 個百分點" if rc_a else "沒有降低依賴"))
    revC["note"] = ("每年最後一個交易日收盤把兩個子帳戶拉回各半（用各自日淨值比例縮放，不計移轉成本）；"
                    "最賺 5 筆以每筆交易損益 × 該子帳戶在進場日所屬年度的縮放倍數近似。只描述、不改判定。")
    out["review_C"] = revC
    print("reviewC", json.dumps(revC, default=str)[:1500], flush=True)

    out["resources"] = dict(peak_rss_mb=int(rss()), seconds=round(time.time() - t_start, 1))
    js = json.dumps(out, ensure_ascii=False, indent=1, default=lambda o: (bool(o) if isinstance(o, np.bool_) else
                                                                          (int(o) if isinstance(o, np.integer) else
                                                                           float(o))))
    open(os.path.join(odir, "my_numbers.json"), "w").write(js)
    h = hashlib.sha256(js.encode("utf-8")).hexdigest()
    open(os.path.join(odir, "my_numbers.sha256"), "w").write(h + "  my_numbers.json\n")
    print("frozen", h, "rss", rss(), round(time.time() - t_start), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
