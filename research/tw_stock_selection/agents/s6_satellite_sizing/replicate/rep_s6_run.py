# -*- coding: utf-8 -*-
"""S6 獨立複製：重算前緣關鍵格子＋審查 A、B，然後凍結。用法：python3 -I -B rep_s6_run.py <scratch_dir>
需先跑 s1 的 rep_s1_panel.py <scratch_dir> 與 s3 的 rep_s3_names.py <scratch_dir>/innov_names.json。
輸出（本資料夾）：my_numbers.json、my_numbers.sha256、my_trades.csv.gz、my_navs.csv.gz
凍結前不讀研究員的任何輸出檔（results*.json、trades.csv.gz、navs.csv.gz）。
"""
import sys, os, json, hashlib, resource, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import rep_s6_lib as K
C, S, Q, R, L = K.C, K.S, K.Q, K.R, K.L
ST, X2 = S.STRICT, S.X2

CELLS = {  # id: (sizing, sat_w, rank_w)
    "W10_70": ("W10", 0.7, None),
    "W7_70": ("W7", 0.7, None),
    "WN_70": ("WN", 0.7, None),
    "WN_100": ("WN", 1.0, None),
    "RW10_70": ("W10", 0.7, (1.5, 1.0)),
    "RWN_70": ("WN", 0.7, (1.5, 1.0)),
    "RW10_70_rev": ("W10", 0.7, (1.0, 1.5)),
    "RWN_70_rev": ("WN", 0.7, (1.0, 1.5)),
}


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
    e50 = P.e50.values
    lists = Q.make_lists(P, S_, mi0, excl=excl)
    buy, keep = Q.cell_plans(lists, "C10", x=0.10)
    ranks = K.c10_ranks(lists, buy)
    nl = np.array([len(buy[s]) for s in sorted(buy)])
    out = dict(meta=dict(T=P.T, first=P.dstr[0], last=P.dstr[-1], t0=P.dstr[P.t0], n_signals=len(buy),
                         list_len_hist={int(k): int((nl == k).sum()) for k in range(11)}, list_len_mean=float(nl.mean())))
    y50 = L.yearly(P, e50)
    out["bench_0050"] = dict(ann_2019=L.ann(P, e50, i19, e), ann_full=L.ann(P, e50, P.t0, e),
                             ann_2024=L.ann(P, e50, i24, e), mdd_full=L.maxdd(P, e50, P.t0, e)[0],
                             ann_h1=L.ann(P, e50, P.t0, ih), ann_h2=L.ann(P, e50, ih, e), yearly=y50)

    def run(cid, plans=(buy, keep, ranks), **kw):
        sz, w, rw = CELLS[cid]
        b, k, rk = plans
        return K.simulate_s6(P, b, k, sat_w=w, sizing=sz, rank_w=rw, ranks=rk, **{**ST, **kw})

    def brief(res):
        nav = res["nav"]; tr = res["trades"]
        f = int(tr.tb.min())
        return dict(ann_2019=L.ann(P, nav, i19, e), ann_full=L.ann(P, nav, P.t0, e), ann_2024=L.ann(P, nav, i24, e),
                    mdd_full=L.maxdd(P, nav, P.t0, e)[0], mdd_2019=L.maxdd(P, nav, i19, e)[0],
                    ann_h1=L.ann(P, nav, P.t0, ih), ann_h2=L.ann(P, nav, ih, e), n_trades=int(len(tr)),
                    avg_stock_w=float(res["stkw"][f:].mean()), reb_up=res["cnt"]["reb_up"],
                    reb_down=res["cnt"]["reb_down"], reduce_legs=res["cnt"]["reb_reduce_legs"],
                    clamp_days=res["clamp_days"], blk_min=res["cnt"]["blk_min"], yearly=L.yearly(P, nav))

    # ---- 驗收：W10 不加權＝rep_s5_lib.simulate_cs；WN_70＝rep_s5_lib 審查 B（list5）
    acc = {}
    a = C.simulate_cs(P, buy, keep, sat_w=0.7, **ST); b = run("W10_70")
    acc["W10_70_vs_rep_s5"] = dict(max_abs_nav_diff=float(np.nanmax(np.abs(a["nav"] - b["nav"]))),
                                   trades_identical=bool(len(a["trades"]) == len(b["trades"]) and (
                                       a["trades"][["code", "tb", "ts"]].values == b["trades"][["code", "tb", "ts"]].values).all()))
    a2 = C.simulate_cs(P, buy, keep, sat_w=0.7, div_mode="list5", **ST); b2 = run("WN_70")
    acc["WN_70_vs_rep_s5_list5"] = dict(max_abs_nav_diff=float(np.nanmax(np.abs(a2["nav"] - b2["nav"]))),
                                        trades_identical=bool(len(a2["trades"]) == len(b2["trades"]) and (
                                            a2["trades"][["code", "tb", "ts"]].values == b2["trades"][["code", "tb", "ts"]].values).all()))
    del a, a2
    out["acceptance"] = acc
    print("acc", acc, "rss", rss(), flush=True)

    res = {"W10_70": b, "WN_70": b2}
    for cid in CELLS:
        if cid not in res:
            res[cid] = run(cid)
    for cid in CELLS:
        out[cid] = brief(res[cid])
        print(cid, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out[cid].items() if k != "yearly"}, flush=True)

    # ---- RW10_70 穩健性
    m = res["RW10_70"]
    out["RW10_70_x2"] = brief(run("RW10_70", **X2))
    sp, top = R.span_of(m["trades"], 5)
    out["RW10_70_block5"] = brief(run("RW10_70", block_span=sp)); out["RW10_70_block5"]["blocked"] = top
    for k in ("RW10_70_x2", "RW10_70_block5"):
        out[k].pop("yearly")
    print("x2", out["RW10_70_x2"]["ann_2019"], out["RW10_70_x2"]["ann_full"], "blk", out["RW10_70_block5"]["ann_2019"],
          out["RW10_70_block5"]["ann_full"], [x["code"] for x in top], flush=True)

    # ---- 20 條路徑：RW10_70 與 W10_70
    caps = [600_000.0, 800_000.0, 1_000_000.0, 1_500_000.0, 2_000_000.0]
    rows = []
    for shf in range(4):
        if shf == 0:
            pl = (buy, keep, ranks)
        else:
            lk = Q.make_lists(P, S_, mi0, shift=shf, excl=excl)
            bb, kk = Q.cell_plans(lk, "C10", x=0.10)
            pl = (bb, kk, K.c10_ranks(lk, bb))
        for cap in caps:
            r = dict(shift=shf, capital=cap)
            for cid in ("RW10_70", "W10_70"):
                x = run(cid, plans=pl, capital=cap)
                r[cid] = L.ann(P, x["nav"], i19, e); r[cid + "_full"] = L.ann(P, x["nav"], P.t0, e)
                r[cid + "_n"] = int(len(x["trades"]))
            rows.append(r)
        print("paths shift", shf, "rss", rss(), flush=True)
    d = {}
    for cid in ("RW10_70", "W10_70"):
        v = np.array([r[cid] for r in rows])
        d[cid] = dict(min=float(v.min()), median=float(np.median(v)), max=float(v.max()),
                      above_0050=int((v > out["bench_0050"]["ann_2019"]).sum()))
    df = np.array([r["RW10_70"] - r["W10_70"] for r in rows])
    d["diff_RW10_minus_W10"] = dict(min=float(df.min()), median=float(np.median(df)), max=float(df.max()),
                                    n_pos=int((df > 0).sum()), n=len(df))
    d["rows"] = rows
    out["paths20"] = d

    # ---- 審查 A：逐年、2020 換 0050、逐年差
    yrs = list(range(2016, 2027))
    rva = {}
    for cid in ("RW10_70", "W10_70", "RWN_70", "WN_70", "W7_70", "WN_100"):
        nav = res[cid]["nav"]; y = out[cid]["yearly"]
        tot = nav[e] / nav[i19] * (1 + y50[2020]) / (1 + y[2020])
        span = (P.dates[e] - P.dates[i19]).days / 365.25
        rva[cid] = dict(yearly=y, ann_2019=out[cid]["ann_2019"], ann_2019_2020as0050=float(tot ** (1 / span) - 1),
                        years_beat_0050=int(sum(y[k] > y50[k] for k in yrs)),
                        years_lost=[k for k in yrs if y[k] <= y50[k]])
    for a_, b_ in (("RW10_70", "W10_70"), ("RWN_70", "WN_70")):
        dd = {k: out[a_]["yearly"][k] - out[b_]["yearly"][k] for k in yrs}
        rva[a_ + "_minus_" + b_] = dict(yearly_diff=dd, n_pos=int(sum(v > 0 for v in dd.values())), n=len(yrs),
                                        neg_years=[k for k, v in dd.items() if v <= 0],
                                        adv_2019=out[a_]["ann_2019"] - out[b_]["ann_2019"],
                                        adv_2019_2020as0050=rva[a_]["ann_2019_2020as0050"] - rva[b_]["ann_2019_2020as0050"])
    out["reviewA"] = rva

    # ---- 審查 B：W10_70 交易依 L_MA 名次分組
    def tag_rank(tr):
        pm = []
        for i, tb in zip(tr.i, tr.tb):
            s = int(tb) - 1
            w = np.where(buy[s] == int(i))[0]
            pm.append(int(ranks[s][w[0]]) if len(w) else -1)
        return np.array(pm)

    for cid in res:
        res[cid]["trades"]["pos_ma"] = tag_rank(res[cid]["trades"])
    tr = res["W10_70"]["trades"]
    ex = tr.ret.values - S.vs_e50(P, tr)
    top5 = tr.pos_ma.values <= 5
    h1 = tr.tb.values <= ih
    y2020 = np.array([str(x)[:4] == "2020" for x in tr.buy_date])

    def g(mask):
        r = tr.ret.values[mask]; x = ex[mask]
        t_ = S.tstat(x)
        return dict(n=int(mask.sum()), mean=float(r.mean()), median=float(np.median(r)), win=float((r > 0).mean()),
                    ex_mean=t_.get("mean"), ex_median=t_.get("median"), ex_t=t_.get("t"),
                    pnl_sum=float(tr.pnl.values[mask].sum()), cost_mean=float(tr.cost.values[mask].mean()))

    def welch(m1, m2, v):
        a_, b_ = v[m1], v[m2]
        se = np.sqrt(a_.var(ddof=1) / len(a_) + b_.var(ddof=1) / len(b_))
        return dict(diff=float(a_.mean() - b_.mean()), t=float((a_.mean() - b_.mean()) / se),
                    diff_median=float(np.median(a_) - np.median(b_)))

    rvb = {"n_unknown_rank": int((tr.pos_ma.values < 0).sum())}
    allm = np.ones(len(tr), bool)
    for nm, sub in (("all", allm), ("h1", h1), ("h2", ~h1), ("ex2020", ~y2020), ("only2020", y2020)):
        rvb[nm] = dict(top5=g(sub & top5), r6_10=g(sub & ~top5), diff_ret=welch(sub & top5, sub & ~top5, tr.ret.values),
                       diff_ex=welch(sub & top5, sub & ~top5, ex))
    # 名次逐名
    rvb["by_rank"] = {int(k): g(tr.pos_ma.values == k) for k in range(1, 11) if (tr.pos_ma.values == k).sum() > 1}
    # RW10_70 自己的交易同樣分組（參考）
    tr2 = res["RW10_70"]["trades"]; ex2 = tr2.ret.values - S.vs_e50(P, tr2); t5 = tr2.pos_ma.values <= 5
    rvb["RW10_70_trades"] = dict(top5=dict(n=int(t5.sum()), mean=float(tr2.ret[t5].mean()), ex=S.tstat(ex2[t5])),
                                 r6_10=dict(n=int((~t5).sum()), mean=float(tr2.ret[~t5].mean()), ex=S.tstat(ex2[~t5])))
    out["reviewB"] = rvb
    out["runtime_s"] = time.time() - t_start; out["peak_rss_mb"] = rss()

    # ---- 存檔、凍結
    trs = []
    for k in res:
        t = res[k]["trades"].copy(); t["run"] = k; trs.append(t)
    pd.concat(trs, ignore_index=True).drop(columns=["i"]).to_csv(os.path.join(HERE, "my_trades.csv.gz"), index=False,
                                                                 compression="gzip")
    nv = pd.DataFrame({"date": P.dstr[P.t0:], "e50": e50[P.t0:]})
    for k in res:
        nv[k] = res[k]["nav"][P.t0:]
    nv.to_csv(os.path.join(HERE, "my_navs.csv.gz"), index=False, compression="gzip")
    js = json.dumps(out, ensure_ascii=False, indent=1, default=float)
    open(os.path.join(HERE, "my_numbers.json"), "w").write(js)
    h = hashlib.sha256(open(os.path.join(HERE, "my_numbers.json"), "rb").read()).hexdigest()
    open(os.path.join(HERE, "my_numbers.sha256"), "w").write(h + "  my_numbers.json\n")
    print("FROZEN", h, "rss", rss(), "s", round(time.time() - t_start))


if __name__ == "__main__":
    main(sys.argv[1])
