# -*- coding: utf-8 -*-
"""S3 E3 獨立複製：凍結用主程式＋三個審查。用法：python3 -I -B rep_s3_run.py <scratch_dir>
需先跑 rep_s1_panel.py <scratch_dir>（面板）與 rep_s3_names.py <scratch_dir>/innov_names.json（創新板名單）。
輸出（本資料夾）：my_numbers.json、my_numbers.sha256、my_trades.csv.gz、review_numbers.json
不讀研究員的任何輸出檔。
"""
import sys, os, json, hashlib, resource, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import rep_s3_lib as Q
R, L = Q.R, Q.L
ST, X2 = Q.STRICT, Q.X2


def rss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024


def main(sc):
    t_start = time.time()
    P = R.build(sc)
    Rv, mi0 = L.load_rev(P)
    S, _ = L.surge(Rv, mi0, null2013=True)
    del Rv
    e = P.T - 1; i19 = L.didx(P, "2018-12-28"); i24 = L.didx(P, "2023-12-29")
    lists = Q.make_lists(P, S, mi0)
    out = dict(meta=dict(T=P.T, N=P.N, first=P.dstr[0], last=P.dstr[-1], t0=P.dstr[P.t0], n_signals=len(lists),
                         first_signal=P.dstr[min(lists)], last_signal=P.dstr[max(lists)],
                         lma_n_median=float(np.median([len(q["lma"]) for q in lists.values()])),
                         lma_n_min=int(min(len(q["lma"]) for q in lists.values())),
                         lmom_n_median=float(np.median([len(Q.lmom(q, .1)) for q in lists.values()])),
                         lmom_n_min=int(min(len(Q.lmom(q, .1)) for q in lists.values())),
                         lmom_months_lt10=int(sum(len(Q.lmom(q, .1)) < 10 for q in lists.values()))))
    e50 = P.e50.values
    out["bench_0050"] = dict(ann_2019=L.ann(P, e50, i19, e), ann_full=L.ann(P, e50, P.t0, e), ann_2024=L.ann(P, e50, i24, e))
    plans = {c: Q.cell_plans(lists, c) for c in ("D1", "E1", "E2", "E3", "C10")}
    res = {}; trs = []
    for c in ("D1", "E1", "E2", "E3"):
        res[c] = Q.simulate(P, *plans[c], **ST)
        out[c] = Q.stats(P, res[c])
        t = res[c]["trades"].copy(); t["cell"] = c; trs.append(t)
        out[c]["trades_7610"] = t[t.code == "7610"][["buy_date", "sell_date", "ret", "pnl", "cost"]].to_dict("records")
        out[c]["open_7610"] = [x for x in res[c]["open_pos"] if x[0] == "7610"]
    out["E3"]["sold_while_in_buylist"] = int(res["E3"]["cnt"]["sold_in_buylist"])
    # 3：成本加倍、禁買最賺 5 筆（整段持有期）
    out["E3_x2"] = Q.stats(P, Q.simulate(P, *plans["E3"], **X2))
    sp, top = R.span_of(res["E3"]["trades"], 5)
    b = Q.simulate(P, *plans["E3"], block_span=sp, **ST)
    out["E3_block5"] = Q.stats(P, b); out["E3_block5"]["blocked"] = top
    # 4：整段不准買 7610
    ban = {"7610": [(0, P.T)]}
    for c in ("E3", "D1"):
        x = Q.simulate(P, *plans[c], block_span=ban, **ST)
        out[c + "_ban7610"] = Q.stats(P, x)
    # 5：配對差
    out["paired_E3_D1"] = dict(from_2016_03=R.paired(P, res["E3"]["nav"], res["D1"]["nav"], "2016-03"),
                               from_2019_01=R.paired(P, res["E3"]["nav"], res["D1"]["nav"], "2019-01"))
    # 6：D1 交易依進場訊號日的前 20 日漲幅
    tr = res["D1"]["trades"]
    rk, r20 = Q.lma_rank(lists, tr)
    out["D1_by_r20"] = dict(le10=Q.grp(tr, r20 <= 0.10), gt10=Q.grp(tr, r20 > 0.10), gt30=Q.grp(tr, r20 > 0.30),
                            nan=int(np.isnan(r20).sum()))
    out["runtime_s"] = time.time() - t_start; out["peak_rss_mb_at_freeze"] = rss()
    allt = pd.concat(trs, ignore_index=True)
    allt.drop(columns=["i"]).to_csv(os.path.join(HERE, "my_trades.csv.gz"), index=False, compression="gzip")
    js = json.dumps(out, ensure_ascii=False, indent=1, default=float)
    open(os.path.join(HERE, "my_numbers.json"), "w").write(js)
    h = hashlib.sha256(open(os.path.join(HERE, "my_numbers.json"), "rb").read()).hexdigest()
    open(os.path.join(HERE, "my_numbers.sha256"), "w").write(h + "  my_numbers.json\n")
    print("FROZEN", h, "rss", rss(), "s", round(time.time() - t_start))

    # ====================== 審查 ======================
    rv = {}
    # ---- C：E3 交易依進場當下 L_MA 名次分組；只買前 10 名且已在漲、不遞補
    t3 = res["E3"]["trades"]
    rk3, r3 = Q.lma_rank(lists, t3)
    no7610 = (t3.code != "7610").values
    rv["C_E3_by_rank"] = {"1-10": Q.grp(t3, (rk3 >= 1) & (rk3 <= 10)), "11-20": Q.grp(t3, (rk3 >= 11) & (rk3 <= 20)),
                          "21+": Q.grp(t3, rk3 >= 21), "not_in_lma": int((rk3 < 0).sum()),
                          "1-10_ex7610": Q.grp(t3, (rk3 >= 1) & (rk3 <= 10) & no7610)}
    rv["C_D1_by_rank"] = {"1-10": Q.grp(tr, (rk >= 1) & (rk <= 10)), "11+": Q.grp(tr, rk >= 11)}
    c10 = Q.simulate(P, *plans["C10"], **ST)
    rv["C_C10_posthoc"] = Q.stats(P, c10)
    rv["C_C10_posthoc"]["ban7610"] = Q.brief(Q.stats(P, Q.simulate(P, *plans["C10"], block_span=ban, **ST)))
    print("C done", rss())
    # ---- A：創新板
    inj = json.load(open(os.path.join(sc, "innov_names.json")))["innov"]
    inn = set(inj)
    bought = {}
    for c in ("D1", "E3"):
        tt = res[c]["trades"]
        for code, bd, sd, rr in zip(tt.code, tt.buy_date, tt.sell_date, tt.ret):
            if code in inn:
                bought.setdefault(code, dict(name=inj[code]["names"], innov_first=inj[code]["first"],
                                             innov_last=inj[code]["last"], trades=[]))["trades"].append(
                    dict(cell=c, buy=bd, sell=sd, ret=float(rr)))
        for code, bd, cost, val in res[c]["open_pos"]:
            if code in inn:
                bought.setdefault(code, dict(name=inj[code]["names"], innov_first=inj[code]["first"],
                                             innov_last=inj[code]["last"], trades=[]))["trades"].append(
                    dict(cell=c, buy=bd, sell="open", ret=val / cost - 1.0))
    rv["A_innov_codes_total"] = len(inn)
    rv["A_innov_bought"] = bought
    mask = np.array([c in inn for c in P.codes])
    lists_x = Q.make_lists(P, S, mi0, excl=lambda s: mask)
    rv["A_excl"] = {}
    for c in ("D1", "E3"):
        x = Q.simulate(P, *Q.cell_plans(lists_x, c), **ST)
        s = Q.stats(P, x)
        rv["A_excl"][c] = dict(Q.brief(s), top_trade=s.get("top_trade"),
                               x2=Q.brief(Q.stats(P, Q.simulate(P, *Q.cell_plans(lists_x, c), **X2))))
        rv["A_excl"][c]["nav"] = None
        if c == "E3":
            nE = x["nav"]
        else:
            nD = x["nav"]
    rv["A_excl"]["paired_E3_D1"] = R.paired(P, nE, nD, "2016-03")
    del lists_x
    print("A done", rss())
    # ---- B：20 條路徑
    caps = [600_000.0, 800_000.0, 1_000_000.0, 1_500_000.0, 2_000_000.0]
    paths = []
    for sh in (0, 1, 2, 3):
        lk = lists if sh == 0 else Q.make_lists(P, S, mi0, shift=sh)
        pl = {c: Q.cell_plans(lk, c) for c in ("D1", "E3")}
        for cap in caps:
            row = dict(shift=sh, capital=cap)
            for c in ("D1", "E3"):
                for tag, bs in (("", None), ("_ban7610", ban)):
                    x = Q.simulate(P, *pl[c], capital=cap, block_span=bs, **ST)
                    s = Q.stats(P, x)
                    mr = max([v for v in (s.get("max_ret_closed"), s.get("max_ret_open")) if v is not None])
                    tt = x["trades"]
                    big = tt[tt.ret >= 5.0]
                    bigl = [(a, b, round(float(r), 3)) for a, b, r in zip(big.code, big.buy_date, big.ret)]
                    bigl += [(a, b, round(v / co - 1, 3)) for a, b, co, v in x["open_pos"] if v / co - 1 >= 5.0]
                    row[c + tag] = dict(ann_2019=s["ann_2019"], ann_full=s["ann_full"], max_ret=mr, big500=bigl,
                                        n_trades=s["n_trades"])
            paths.append(row)
        print("B shift", sh, rss())
    rv["B_paths"] = paths
    summ = {}
    for k in ("D1", "E3", "D1_ban7610", "E3_ban7610"):
        a = np.array([p[k]["ann_2019"] for p in paths]); f = np.array([p[k]["ann_full"] for p in paths])
        summ[k] = dict(ann2019_min=float(a.min()), p25=float(np.percentile(a, 25)), median=float(np.median(a)),
                       p75=float(np.percentile(a, 75)), max=float(a.max()),
                       full_min=float(f.min()), full_median=float(np.median(f)), full_max=float(f.max()),
                       n_paths_big500=int(sum(len(p[k]["big500"]) > 0 for p in paths)))
    for tag in ("", "_ban7610"):
        d = np.array([p["E3" + tag]["ann_2019"] - p["D1" + tag]["ann_2019"] for p in paths])
        summ["E3_minus_D1" + tag] = dict(min=float(d.min()), median=float(np.median(d)), max=float(d.max()),
                                         n_pos=int((d > 0).sum()))
    rv["B_summary"] = summ
    rv["peak_rss_mb"] = rss(); rv["runtime_s"] = time.time() - t_start
    json.dump(rv, open(os.path.join(HERE, "review_numbers.json"), "w"), ensure_ascii=False, indent=1, default=float)
    print("ALL DONE rss", rss(), "s", round(time.time() - t_start))


if __name__ == "__main__":
    main(sys.argv[1])
