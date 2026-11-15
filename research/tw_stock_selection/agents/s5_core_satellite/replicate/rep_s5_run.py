# -*- coding: utf-8 -*-
"""S5 獨立複製：凍結用主程式＋兩個審查。用法：python3 -I -B rep_s5_run.py <scratch_dir>
需先跑 rep_s1_panel.py <scratch_dir> 與 rep_s3_names.py <scratch_dir>/innov_names.json。
輸出（本資料夾）：my_numbers.json、my_numbers.sha256、my_trades.csv.gz、my_navs.csv.gz、review_numbers.json
凍結前不讀研究員的任何輸出檔（results*.json、trades.csv.gz、navs.csv.gz）。
"""
import sys, os, json, hashlib, resource, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import rep_s5_lib as C
S, Q, R, L = C.S, C.Q, C.R, C.L
ST, X2 = S.STRICT, S.X2


def rss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024


def short(s):
    keys = ("ann_2019", "ann_full", "ann_2024", "mdd_full", "mdd_2019", "n_trades", "ann_h1", "ann_h2",
            "avg_stock_w_from_first_buy", "avg_npos_from_first_buy", "clamp_days", "bind_days")
    d = {k: s.get(k) for k in keys}
    if "reb" in s:
        d["reb"] = {k: v for k, v in s["reb"].items() if k != "events"}
    return d


def main(sc):
    t_start = time.time()
    P = R.build(sc)
    S.attach_e50(P, sc)
    excl = S.innov_mask_fn(P, sc)
    Rv, mi0 = L.load_rev(P)
    S_, _ = L.surge(Rv, mi0, null2013=True)
    e = P.T - 1; i19 = L.didx(P, "2018-12-28"); i24 = L.didx(P, "2023-12-29"); ih = L.didx(P, "2021-02-19")
    e50 = P.e50.values
    lists = Q.make_lists(P, S_, mi0, excl=excl)
    out = dict(meta=dict(T=P.T, N=P.N, first=P.dstr[0], last=P.dstr[-1], t0=P.dstr[P.t0], n_signals=len(lists),
                         innov_codes=len(P.innov_check)))
    out["bench_0050"] = dict(ann_2019=L.ann(P, e50, i19, e), ann_full=L.ann(P, e50, P.t0, e),
                             ann_2024=L.ann(P, e50, i24, e), ann_h1=L.ann(P, e50, P.t0, ih), ann_h2=L.ann(P, e50, ih, e),
                             mdd_full=L.maxdd(P, e50, P.t0, e)[0])
    plans = {th: Q.cell_plans(lists, "C10", x=th) for th in (0.10, 0.15)}
    out["list_len_mean"] = {str(th): float(np.mean([len(v) for v in plans[th][0].values()])) for th in plans}
    # ---- 驗收：sat_w=1 與 rep_s4_lib.simulate(idle='e50') 相同（F1、E3'）
    acc = {}
    pE3 = Q.cell_plans(lists, "E3")
    for nm, pl in (("F1", plans[0.10]), ("E3p", pE3)):
        a = S.simulate(P, *pl, idle="e50", **ST); b = C.simulate_cs(P, *pl, sat_w=1.0, **ST)
        same_tr = len(a["trades"]) == len(b["trades"]) and bool(
            (a["trades"][["code", "tb", "ts"]].values == b["trades"][["code", "tb", "ts"]].values).all())
        acc[nm] = dict(max_abs_nav_diff=float(np.nanmax(np.abs(a["nav"] - b["nav"]))), trades_identical=same_tr,
                       rep_s4=S.brief(S.stats(P, a)))
    out["acceptance"] = acc
    del pE3
    # ---- 格子
    res = {}
    cells = {"W70_T10": (0.7, 0.10), "W50_T10": (0.5, 0.10), "W70_T15": (0.7, 0.15)}
    for k, (w, th) in cells.items():
        res[k] = C.simulate_cs(P, *plans[th], sat_w=w, **ST)
        out[k] = C.cell_stats(P, res[k], e50)
    pm = plans[0.10]
    m = res["W70_T10"]
    out["W70_T10_x2"] = short(C.cell_stats(P, C.simulate_cs(P, *pm, sat_w=0.7, **X2), e50))
    sp, top = R.span_of(m["trades"], 5)
    out["W70_T10_block5"] = short(C.cell_stats(P, C.simulate_cs(P, *pm, sat_w=0.7, block_span=sp, **ST), e50))
    out["W70_T10_block5"]["blocked"] = top
    tot = float(m["trades"].pnl.sum())
    out["W70_T10_block5"]["top5_share_closed_pnl"] = float(sum(x["pnl"] for x in top) / tot)
    # ---- 20 條路徑
    caps = [600_000.0, 800_000.0, 1_000_000.0, 1_500_000.0, 2_000_000.0]
    rows = []
    for shf in range(4):
        lk = lists if shf == 0 else Q.make_lists(P, S_, mi0, shift=shf, excl=excl)
        pl = Q.cell_plans(lk, "C10", x=0.10)
        for cap in caps:
            x = C.simulate_cs(P, *pl, sat_w=0.7, capital=cap, **ST)
            rows.append(dict(shift=shf, capital=cap, ann_2019=L.ann(P, x["nav"], i19, e),
                             ann_full=L.ann(P, x["nav"], P.t0, e), n_trades=int(len(x["trades"]))))
    a19 = np.array([r["ann_2019"] for r in rows])
    out["paths20"] = dict(min=float(a19.min()), median=float(np.median(a19)), max=float(a19.max()),
                          above_0050=int((a19 > out["bench_0050"]["ann_2019"]).sum()), rows=rows)
    # ---- D1：同一段持有期改抱 0050
    tr = m["trades"]
    ex = tr.ret.values - S.vs_e50(P, tr)
    h1 = tr.tb.values <= ih
    out["D1"] = dict(all=dict(S.tstat(ex), stock_mean=float(tr.ret.mean()), e50_mean=float(S.vs_e50(P, tr).mean()),
                              win_vs_e50=float((ex > 0).mean())),
                     h1=S.tstat(ex[h1]), h2=S.tstat(ex[~h1]))
    out["runtime_s"] = time.time() - t_start; out["peak_rss_mb_at_freeze"] = rss()
    # ---- 存檔、凍結
    trs = []
    for k in res:
        t = res[k]["trades"].copy(); t["run"] = k; trs.append(t)
    pd.concat(trs, ignore_index=True).drop(columns=["i"]).to_csv(os.path.join(HERE, "my_trades.csv.gz"), index=False,
                                                                 compression="gzip")
    nv = pd.DataFrame({"date": P.dstr[P.t0:], "e50": e50[P.t0:]})
    for k in res:
        nv[k] = res[k]["nav"][P.t0:]; nv[k + "_B"] = res[k]["B"][P.t0:]; nv[k + "_stkw"] = res[k]["stkw"][P.t0:]
    nv.to_csv(os.path.join(HERE, "my_navs.csv.gz"), index=False, compression="gzip")
    js = json.dumps(out, ensure_ascii=False, indent=1, default=float)
    open(os.path.join(HERE, "my_numbers.json"), "w").write(js)
    h = hashlib.sha256(open(os.path.join(HERE, "my_numbers.json"), "rb").read()).hexdigest()
    open(os.path.join(HERE, "my_numbers.sha256"), "w").write(h + "  my_numbers.json\n")
    print("FROZEN", h, "rss", rss(), "s", round(time.time() - t_start))
    print("acc", json.dumps(acc, default=float))
    for k in ("W70_T10", "W50_T10", "W70_T15"):
        print(k, json.dumps(short(out[k]), default=float))
        print("  ols", out[k]["ols_full"], out[k]["t_month_excess_full"]["t"])
    print("x2", json.dumps(out["W70_T10_x2"], default=float)); print("blk", json.dumps(out["W70_T10_block5"], default=float))
    print("paths", {k: v for k, v in out["paths20"].items() if k != "rows"}); print("D1", out["D1"])

    # ====================== 審查 ======================
    rv = {}
    xa = C.simulate_cs(P, *pm, sat_w=0.7, mode="fixed30", **ST)
    rv["A_fixed30"] = short(C.cell_stats(P, xa, e50))
    rv["A_fixed30"]["ols_full"] = C.cell_stats(P, xa, e50)["ols_full"]
    xa2 = C.simulate_cs(P, *pm, sat_w=0.7, mode="fixed30", **X2)
    rv["A_fixed30_x2"] = short(C.cell_stats(P, xa2, e50))
    xb = C.simulate_cs(P, *pm, sat_w=0.7, div_mode="list5", **ST)
    sb = C.cell_stats(P, xb, e50)
    rv["B_list5"] = short(sb)
    rv["B_list5"].update(ols_full=sb["ols_full"], t_month_excess_full=sb["t_month_excess_full"],
                         sat_ratio=sb.get("sat_ratio_from_first_buy"))
    sp2, top2 = R.span_of(xb["trades"], 5)
    rv["B_list5_block5"] = short(C.cell_stats(P, C.simulate_cs(P, *pm, sat_w=0.7, div_mode="list5", block_span=sp2, **ST), e50))
    rv["B_list5_x2"] = short(C.cell_stats(P, C.simulate_cs(P, *pm, sat_w=0.7, div_mode="list5", **X2), e50))
    # F1（衛星 100%）並列，看「用滿」與 F1 的距離
    rv["F1_ref"] = short(C.cell_stats(P, C.simulate_cs(P, *pm, sat_w=1.0, **ST), e50))
    rv["peak_rss_mb"] = rss(); rv["runtime_s"] = time.time() - t_start
    json.dump(rv, open(os.path.join(HERE, "review_numbers.json"), "w"), ensure_ascii=False, indent=1, default=float)
    for k, v in rv.items():
        print(k, json.dumps(v, default=float)[:700])
    print("ALL DONE rss", rss(), "s", round(time.time() - t_start))


if __name__ == "__main__":
    main(sys.argv[1])
