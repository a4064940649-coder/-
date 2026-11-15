# -*- coding: utf-8 -*-
"""S4 E3' 獨立複製：凍結用主程式＋兩個審查。用法：python3 -I -B rep_s4_run.py <scratch_dir>
需先跑 rep_s1_panel.py <scratch_dir>（面板）與 rep_s3_names.py <scratch_dir>/innov_names.json（創新板名單）。
輸出（本資料夾）：my_numbers.json、my_numbers.sha256、my_trades.csv.gz、review_numbers.json
凍結前不讀研究員的任何輸出檔（results*.json、trades.csv.gz、navs.csv.gz）。
"""
import sys, os, json, hashlib, resource, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import rep_s4_lib as S
Q, R, L = S.Q, S.R, S.L
ST, X2 = S.STRICT, S.X2


def rss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024


def main(sc):
    t_start = time.time()
    P = R.build(sc)
    S.attach_e50(P, sc)
    excl = S.innov_mask_fn(P, sc)
    Rv, mi0 = L.load_rev(P)
    S_, _ = L.surge(Rv, mi0, null2013=True)
    e = P.T - 1; i19 = L.didx(P, "2018-12-28"); i24 = L.didx(P, "2023-12-29"); ih = L.didx(P, "2021-02-19")
    lists = Q.make_lists(P, S_, mi0, excl=excl)
    e50 = P.e50.values
    out = dict(meta=dict(T=P.T, N=P.N, first=P.dstr[0], last=P.dstr[-1], t0=P.dstr[P.t0], n_signals=len(lists),
                         first_signal=P.dstr[min(lists)], last_signal=P.dstr[max(lists)],
                         e50_missing_open_close_days=P.e50_missing_oc,
                         innov_codes=len(P.innov_check),
                         innov_range_mismatch={k: v for k, v in P.innov_check.items() if v["n_days"] != v["present_in_range"]},
                         lmom_mean=float(np.mean([len(Q.lmom(q, .1)[:10]) for q in lists.values()])),
                         lmom_months_lt10=int(sum(len(Q.lmom(q, .1)) < 10 for q in lists.values()))))
    out["bench_0050"] = dict(ann_2019=L.ann(P, e50, i19, e), ann_full=L.ann(P, e50, P.t0, e),
                             ann_2024=L.ann(P, e50, i24, e), ann_h1=L.ann(P, e50, P.t0, ih), ann_h2=L.ann(P, e50, ih, e))
    # 0：引擎驗收（min_frac=0、空位現金應與 rep_s3_lib.simulate 逐日相同）
    lists_all = Q.make_lists(P, S_, mi0)
    pE3 = Q.cell_plans(lists_all, "E3")
    a = Q.simulate(P, *pE3, **ST); b = S.simulate(P, *pE3, min_frac=0.0, **ST)
    out["check_engine_vs_rep_s3"] = dict(max_abs_nav_diff=float(np.nanmax(np.abs(a["nav"] - b["nav"]))),
                                         rep_s3=Q.brief(Q.stats(P, a)), mine=S.brief(S.stats(P, b)))
    out["check_minbuy_only_incl_innov"] = S.brief(S.stats(P, S.simulate(P, *pE3, **ST)))
    pX = Q.cell_plans(lists, "E3")
    out["check_innov_only_minfrac0"] = S.brief(S.stats(P, S.simulate(P, *pX, min_frac=0.0, **ST)))
    del lists_all, a, b
    # 1：E3' 空位現金
    plans = {c: Q.cell_plans(lists, c) for c in ("E3", "C10")}
    res = {}
    res["E3p_cash"] = S.simulate(P, *plans["E3"], **ST)
    out["E3p_cash"] = S.stats(P, res["E3p_cash"])
    # 2：E3' 空位 0050
    res["E3p_e50"] = S.simulate(P, *plans["E3"], idle="e50", **ST)
    out["E3p_e50"] = S.stats(P, res["E3p_e50"])
    # 3：成本加倍、禁買最賺 5 筆、前後兩段（前後兩段在 stats 的 ann_h1／ann_h2）
    out["E3p_cash_x2"] = S.stats(P, S.simulate(P, *plans["E3"], **X2))
    sp, top = R.span_of(res["E3p_cash"]["trades"], 5)
    out["E3p_cash_block5"] = S.stats(P, S.simulate(P, *plans["E3"], block_span=sp, **ST))
    out["E3p_cash_block5"]["blocked"] = top
    # 4：F1（L_MA 前 10 名中 RET20 > 10%、不遞補；出場同 E3'；空位 0050）
    res["F1_e50"] = S.simulate(P, *plans["C10"], idle="e50", **ST)
    out["F1_e50"] = S.stats(P, res["F1_e50"])
    out["F1_cash"] = S.brief(S.stats(P, S.simulate(P, *plans["C10"], **ST)))
    out["F1_list_mean_len"] = float(np.mean([len(v) for v in plans["C10"][0].values()]))
    # 5：E3' 交易依進場名次對照同期 0050
    t3 = res["E3p_cash"]["trades"]
    rk, r20 = Q.lma_rank(lists, t3)
    ex = t3.ret.values - S.vs_e50(P, t3)
    h1 = t3.tb.values < ih
    g5 = {}
    for name, m in (("1-10", (rk >= 1) & (rk <= 10)), ("11-20", (rk >= 11) & (rk <= 20)), ("21+", rk >= 21),
                    ("11+", rk >= 11)):
        g5[name] = dict(S.tstat(ex[m]), stock_mean=float(t3.ret.values[m].mean()),
                        e50_mean=float(S.vs_e50(P, t3)[m].mean()),
                        h1=S.tstat(ex[m & h1]), h2=S.tstat(ex[m & ~h1]))
    g5["not_in_lma"] = int((rk < 0).sum())
    out["E3p_vs_e50_by_rank"] = g5
    # 6：位移 3／5／10
    out["E3p_shift"] = {}
    for sh in (3, 5, 10):
        lk = Q.make_lists(P, S_, mi0, shift=sh, excl=excl)
        out["E3p_shift"][str(sh)] = S.brief(S.stats(P, S.simulate(P, *Q.cell_plans(lk, "E3"), **ST)))
    out["runtime_s"] = time.time() - t_start; out["peak_rss_mb_at_freeze"] = rss()
    trs = []
    for k in ("E3p_cash", "E3p_e50", "F1_e50"):
        t = res[k]["trades"].copy(); t["run"] = k
        if k == "E3p_cash":
            t["lma_rank"] = rk; t["r20"] = r20
        trs.append(t)
    pd.concat(trs, ignore_index=True).drop(columns=["i"]).to_csv(os.path.join(HERE, "my_trades.csv.gz"), index=False,
                                                                 compression="gzip")
    js = json.dumps(out, ensure_ascii=False, indent=1, default=float)
    open(os.path.join(HERE, "my_numbers.json"), "w").write(js)
    h = hashlib.sha256(open(os.path.join(HERE, "my_numbers.json"), "rb").read()).hexdigest()
    open(os.path.join(HERE, "my_numbers.sha256"), "w").write(h + "  my_numbers.json\n")
    print("FROZEN", h, "rss", rss(), "s", round(time.time() - t_start))
    for k in ("E3p_cash", "E3p_e50", "E3p_cash_x2", "E3p_cash_block5", "F1_e50"):
        print(k, {kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in S.brief(out[k]).items()})
    print("shift", out["E3p_shift"]); print("rank", json.dumps(g5, default=float)[:600])

    # ====================== 審查 ======================
    rv = {}
    # ---- A：位移 0～10 × 起始資金 60／80／100／150／200 萬
    caps = [600_000.0, 800_000.0, 1_000_000.0, 1_500_000.0, 2_000_000.0]
    rows = []; keep_tr = {}
    for sh in range(0, 11):
        lk = lists if sh == 0 else Q.make_lists(P, S_, mi0, shift=sh, excl=excl)
        pl = Q.cell_plans(lk, "E3")
        for cap in caps:
            x = S.simulate(P, *pl, capital=cap, **ST)
            s = S.stats(P, x)
            rows.append(dict(shift=sh, capital=cap, ann_2019=s["ann_2019"], ann_full=s["ann_full"],
                             ann_2024=s["ann_2024"], n_trades=s["n_trades"], yearly=s["yearly"]))
            if cap == 1_000_000.0:
                keep_tr[sh] = x["trades"].assign(shift=sh)
    rv["A_rows"] = rows
    pd.concat(keep_tr.values(), ignore_index=True).drop(columns=["i"]).to_csv(
        os.path.join(sc, "shift_trades.csv.gz"), index=False, compression="gzip")
    print("A done", rss())
    # ---- B：營收年增率、股價三等分
    yoy = []; px = []
    for i, tb in zip(t3.i.values, t3.tb.values):
        q = lists[int(tb) - 1]; m = q["m"]
        with np.errstate(invalid="ignore", divide="ignore"):
            yoy.append(Rv[m, int(i)] / Rv[m - 12, int(i)] - 1.0)
        px.append(P.C[int(tb) - 1, int(i)])
    yoy = np.array(yoy); px = np.array(px); ret = t3.ret.values
    rv["B_nan"] = dict(yoy=int(np.isnan(yoy).sum()), px=int(np.isnan(px).sum()))
    for nm, v in (("yoy", yoy), ("price", px)):
        qs = np.nanpercentile(v, [100 / 3, 200 / 3])
        rv["B_" + nm + "_cuts"] = [float(x) for x in qs]
        lab = np.where(v <= qs[0], 0, np.where(v <= qs[1], 1, 2))
        d = {}
        for g in range(3):
            d[str(g + 1)] = dict(all=S.grp(ret, lab == g), h1=S.grp(ret, (lab == g) & h1), h2=S.grp(ret, (lab == g) & ~h1))
        # 半段內各自三等分（並列）
        for tag, hm in (("h1_own", h1), ("h2_own", ~h1)):
            q2 = np.nanpercentile(v[hm], [100 / 3, 200 / 3])
            lb2 = np.where(v <= q2[0], 0, np.where(v <= q2[1], 1, 2))
            d[tag] = {str(g + 1): S.grp(ret, (lb2 == g) & hm) for g in range(3)}
            d[tag]["cuts"] = [float(x) for x in q2]
        rv["B_" + nm] = d
    # 年增率 × 股價交叉（3×3，全期）
    qy = np.nanpercentile(yoy, [100 / 3, 200 / 3]); qp = np.nanpercentile(px, [100 / 3, 200 / 3])
    ly = np.where(yoy <= qy[0], 0, np.where(yoy <= qy[1], 1, 2)); lp = np.where(px <= qp[0], 0, np.where(px <= qp[1], 1, 2))
    rv["B_cross"] = {"y%d_p%d" % (a + 1, b + 1): S.grp(ret, (ly == a) & (lp == b)) for a in range(3) for b in range(3)}
    # 秩相關
    rv["B_spearman"] = dict(yoy=float(pd.Series(yoy).corr(pd.Series(ret), method="spearman")),
                            price=float(pd.Series(px).corr(pd.Series(ret), method="spearman")))
    rv["B_top_bottom20"] = {}
    o = np.argsort(ret)
    for tag, idx in (("top20", o[-20:]), ("bottom20", o[:20])):
        rv["B_top_bottom20"][tag] = dict(yoy_median=float(np.nanmedian(yoy[idx])), price_median=float(np.median(px[idx])),
                                         rank_median=float(np.median(rk[idx])))
    rv["peak_rss_mb"] = rss(); rv["runtime_s"] = time.time() - t_start
    json.dump(rv, open(os.path.join(HERE, "review_numbers.json"), "w"), ensure_ascii=False, indent=1, default=float)
    print("ALL DONE rss", rss(), "s", round(time.time() - t_start))


if __name__ == "__main__":
    main(sys.argv[1])
