# -*- coding: utf-8 -*-
"""S7 獨立複製：9 格（含 3 個不鎖定對照）、主格穩健性、執行延遲 d＝1／2、審查 A／B／C，然後凍結。
用法：python3 -I -B rep_s7_run.py <scratch_dir>
需先跑 s1 的 rep_s1_panel.py <scratch_dir> 與 s3 的 rep_s3_names.py <scratch_dir>/innov_names.json。
輸出（本資料夾）：my_numbers.json、my_numbers.sha256、my_trades.csv.gz、my_navs.csv.gz
凍結前不讀研究員的任何輸出檔（FINDINGS 第 2、10 節以外、results*.json、trades.csv.gz、navs.csv.gz）。
"""
import sys, os, json, hashlib, resource, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import rep_s7_lib as K
K6, C, S, Q, R, L = K.K6, K.C, K.S, K.Q, K.R, K.L
ST, X2 = S.STRICT, S.X2

CELLS = {  # id: (sizing, sat_w, lock)
    "W10_70L": ("W10", 0.7, True), "W7_70L": ("W7", 0.7, True), "WN_70L": ("WN", 0.7, True),
    "W10_80L": ("W10", 0.8, True), "W7_80L": ("W7", 0.8, True), "WN_80L": ("WN", 0.8, True),
    "W10_70": ("W10", 0.7, False), "WN_70": ("WN", 0.7, False), "WN_100": ("WN", 1.0, False),
}
CAPS = [600_000.0, 800_000.0, 1_000_000.0, 1_500_000.0, 2_000_000.0]


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
    base_plans = (buy, keep, ranks)
    out = dict(meta=dict(T=P.T, N=P.N, first=P.dstr[0], last=P.dstr[-1], t0=P.dstr[P.t0], n_signals=len(buy),
                         i19=P.dstr[i19], i24=P.dstr[i24], ih=P.dstr[ih]))
    y50 = L.yearly(P, e50)
    b50 = dict(ann_2019=L.ann(P, e50, i19, e), ann_full=L.ann(P, e50, P.t0, e), ann_2024=L.ann(P, e50, i24, e),
               mdd_full=L.maxdd(P, e50, P.t0, e)[0], mdd_2019=L.maxdd(P, e50, i19, e)[0],
               ann_h1=L.ann(P, e50, P.t0, ih), ann_h2=L.ann(P, e50, ih, e), yearly=y50)
    out["bench_0050"] = b50

    def run(cid, plans=base_plans, d=0, **kw):
        sz, w, lk = CELLS[cid]
        b, k, rk = K.delay_plans(plans, d, P.T)
        return K.simulate_s7(P, b, k, sat_w=w, sizing=sz, lock=lk, lranks=rk, **{**ST, **kw})

    def brief(res, full=True):
        nav = res["nav"]; tr = res["trades"]
        f = int(tr.tb.min())
        bw = res["B"][f:e + 1] / nav[f:e + 1]
        md = L.maxdd(P, nav, P.t0, e)
        o = dict(ann_2019=L.ann(P, nav, i19, e), ann_full=L.ann(P, nav, P.t0, e), ann_2024=L.ann(P, nav, i24, e),
                 mdd_full=md[0], mdd_peak=md[1], mdd_trough=md[2], mdd_2019=L.maxdd(P, nav, i19, e)[0],
                 ann_h1=L.ann(P, nav, P.t0, ih), ann_h2=L.ann(P, nav, ih, e), n_trades=int(len(tr)))
        if full:
            c = res["cnt"]
            o.update(avg_stock_w=float(res["stkw"][f:e + 1].mean()), base_w_mean=float(bw.mean()),
                     base_w_median=float(np.median(bw)), base_lt10=float((bw < 0.10).mean()),
                     base_lt20=float((bw < 0.20).mean()), reb_up=c["reb_up"], reb_down=c["reb_down"],
                     reb_total=c["reb_up"] + c["reb_down"], reduce_legs=c["reb_reduce_legs"],
                     reduce_amt=c["reb_reduce_amt"], clamp_days=res["clamp_days"],
                     a3_events=len(res["a3"]), a3_big=res["a3_big"],
                     short_shrink=c["short_shrink"], short_shrink_amt=c["short_shrink_amt"],
                     short_skip=c["short_skip"], short_skip_target_amt=c["short_skip_amt"],
                     short_days=len(res["short_days"]), n_exec_days=len(buy), blk_min=c["blk_min"],
                     buys=c["buy"], yearly=L.yearly(P, nav))
        return o

    # ---- 驗收 A0／A1：lock=False 與 rep_s6_lib.simulate_s6 相同
    acc = {}
    res = {}
    for cid in ("W10_70", "WN_70", "WN_100"):
        sz, w, _ = CELLS[cid]
        a = K6.simulate_s6(P, buy, keep, sat_w=w, sizing=sz, **ST)
        b = run(cid)
        ta, tb_ = a["trades"], b["trades"]
        acc[cid] = dict(max_abs_nav_diff=float(np.nanmax(np.abs(a["nav"] - b["nav"]))),
                        trades_identical=bool(len(ta) == len(tb_) and (
                            ta[["code", "tb", "ts"]].values == tb_[["code", "tb", "ts"]].values).all()),
                        ann_2019=L.ann(P, b["nav"], i19, e), ann_full=L.ann(P, b["nav"], P.t0, e),
                        mdd_full=L.maxdd(P, b["nav"], P.t0, e)[0])
        res[cid] = b
        del a
    # A2：延遲 d＝0（搬列 +0）與原訊號相同
    for cid in ("WN_70", "WN_70L"):
        pl0 = tuple({s + 0: v for s, v in x.items()} for x in base_plans)
        a = run(cid, plans=pl0); b = res.get(cid) or run(cid)
        res[cid] = b
        acc["A2_" + cid] = dict(max_abs_nav_diff=float(np.nanmax(np.abs(a["nav"] - b["nav"]))))
        del a
    out["acceptance"] = acc
    print("acc", json.dumps(acc), "rss", rss(), flush=True)

    for cid in CELLS:
        if cid not in res:
            res[cid] = run(cid)
        out[cid] = brief(res[cid])
        print(cid, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out[cid].items() if k != "yearly"},
              flush=True)
        out[cid]["a3_list"] = res[cid]["a3"][:20]
    # 鎖定的效果
    eff = {}
    for a_, b_ in (("WN_70L", "WN_70"), ("W10_70L", "W10_70"), ("WN_80L", "WN_70")):
        eff[a_ + "_vs_" + b_] = dict(
            keep_2019=(out[a_]["ann_2019"] - b50["ann_2019"]) / (out[b_]["ann_2019"] - b50["ann_2019"]),
            keep_full=(out[a_]["ann_full"] - b50["ann_full"]) / (out[b_]["ann_full"] - b50["ann_full"]),
            diff_2019=out[a_]["ann_2019"] - out[b_]["ann_2019"], diff_full=out[a_]["ann_full"] - out[b_]["ann_full"],
            dd_saved=abs(out[b_]["mdd_full"]) - abs(out[a_]["mdd_full"]))
    out["lock_effect"] = eff

    # ---- 主格穩健性：成本加倍、禁買最賺 5 筆、前後半（已在 brief）
    rob = {}
    for cid in ("WN_70L", "W10_70L", "WN_80L"):
        x2 = brief(run(cid, **X2), full=False)
        sp, top = R.span_of(res[cid]["trades"], 5)
        bl = brief(run(cid, block_span=sp), full=False)
        bl["blocked"] = top
        rob[cid] = dict(x2=x2, block5=bl)
        print("rob", cid, round(x2["ann_2019"], 4), round(x2["ann_full"], 4), round(bl["ann_2019"], 4),
              round(bl["ann_full"], 4), flush=True)
    out["robust"] = rob

    # ---- 20 條路徑（S6 複製員定義：60/80/100/150/200 萬 × make_lists shift 0～3），延遲 d＝1／2 的五種資金
    PATH_CELLS = ("WN_70L", "WN_70", "W10_70L", "W10_70", "WN_80L")
    rows = []
    shift_plans = {0: base_plans}
    for shf in range(1, 4):
        lk = Q.make_lists(P, S_, mi0, shift=shf, excl=excl)
        bb, kk = Q.cell_plans(lk, "C10", x=0.10)
        shift_plans[shf] = (bb, kk, K.c10_ranks(lk, bb))
    for shf in range(4):
        for cap in CAPS:
            r = dict(shift=shf, capital=cap)
            for cid in PATH_CELLS:
                x = run(cid, plans=shift_plans[shf], capital=cap)
                r[cid] = L.ann(P, x["nav"], i19, e); r[cid + "_full"] = L.ann(P, x["nav"], P.t0, e)
                r[cid + "_mdd"] = L.maxdd(P, x["nav"], P.t0, e)[0]
            rows.append(r)
        print("paths shift", shf, "rss", rss(), round(time.time() - t_start), flush=True)
    pz = {}
    for cid in PATH_CELLS:
        v = np.array([r[cid] for r in rows])
        pz[cid] = dict(min=float(v.min()), median=float(np.median(v)), max=float(v.max()),
                       above_0050=int((v > b50["ann_2019"]).sum()),
                       median_full=float(np.median([r[cid + "_full"] for r in rows])))
    for a_, b_ in (("WN_70L", "WN_70"), ("W10_70L", "W10_70"), ("WN_80L", "WN_70")):
        df = np.array([r[a_] - r[b_] for r in rows])
        dff = np.array([r[a_ + "_full"] - r[b_ + "_full"] for r in rows])
        dmd = np.array([abs(r[b_ + "_mdd"]) - abs(r[a_ + "_mdd"]) for r in rows])
        pz["diff_" + a_ + "_minus_" + b_] = dict(
            min=float(df.min()), median=float(np.median(df)), max=float(df.max()), n_pos=int((df > 0).sum()),
            n=len(df), median_full=float(np.median(dff)), n_pos_full=int((dff > 0).sum()),
            dd_saved_median=float(np.median(dmd)), dd_saved_n_pos=int((dmd > 0).sum()),
            per_path=[float(x) for x in df])
    pz["rows"] = rows
    out["paths20"] = pz

    # ---- 執行延遲
    DCELLS = ("WN_70L", "W10_70L", "WN_70")
    dly = {}
    dres = {}
    for cid in DCELLS:
        dly[cid] = {"d0": {k: out[cid][k] for k in ("ann_2019", "ann_full", "ann_2024", "mdd_full", "mdd_2019",
                                                    "n_trades")}}
        dly[cid]["d0"]["caps_median_2019"] = float(np.median([r[cid] for r in rows if r["shift"] == 0]))
        dly[cid]["d0"]["caps_2019"] = [r[cid] for r in rows if r["shift"] == 0]
        for d in (1, 2):
            x = run(cid, d=d)
            dres[(cid, d)] = x
            o = brief(x, full=False)
            o["yearly"] = L.yearly(P, x["nav"])
            cv = [L.ann(P, run(cid, d=d, capital=cap)["nav"], i19, e) if cap != 1_000_000.0 else o["ann_2019"]
                  for cap in CAPS]
            o["caps_2019"] = cv; o["caps_median_2019"] = float(np.median(cv))
            for k in ("ann_2019", "ann_full", "ann_2024"):
                o["loss_" + k] = dly[cid]["d0"][k] - o[k]
            o["loss_caps_median_2019"] = dly[cid]["d0"]["caps_median_2019"] - o["caps_median_2019"]
            dly[cid]["d%d" % d] = o
            print("delay", cid, d, round(o["ann_2019"], 4), round(o["ann_full"], 4), round(o["mdd_full"], 4),
                  round(o["caps_median_2019"], 4), flush=True)
    # 其餘 6 格 d＝1／2（敏感度，只報 2019 起／全期／最大回撤）
    for cid in CELLS:
        if cid in DCELLS:
            continue
        dly[cid] = {}
        for d in (1, 2):
            x = run(cid, d=d)
            dly[cid]["d%d" % d] = dict(ann_2019=L.ann(P, x["nav"], i19, e), ann_full=L.ann(P, x["nav"], P.t0, e),
                                       mdd_full=L.maxdd(P, x["nav"], P.t0, e)[0])
    out["delay"] = dly

    # ---- 審查 A：WN_70L 對 WN_70 逐年差、20 條路徑逐條相減
    yrs = list(range(2016, 2027))
    rva = {}
    for a_, b_ in (("WN_70L", "WN_70"), ("W10_70L", "W10_70")):
        ya, yb = out[a_]["yearly"], out[b_]["yearly"]
        dd = {k: ya[k] - yb[k] for k in yrs}
        # 去掉單一年份（該年用兩者相同報酬）後的 2019 起年化差
        span = (P.dates[e] - P.dates[i19]).days / 365.25
        tota = res[a_]["nav"][e] / res[a_]["nav"][i19]; totb = res[b_]["nav"][e] / res[b_]["nav"][i19]
        drop = {}
        for k in range(2019, 2027):
            drop[k] = float((tota / (1 + ya[k])) ** (1 / span) - (totb / (1 + yb[k])) ** (1 / span))
        # 逐月差
        ma = R.month_ret(P, res[a_]["nav"]); mb = R.month_ret(P, res[b_]["nav"])
        md = (ma - mb); md19 = md[md.index >= pd.Period("2019-01", "M")]
        rva[a_ + "_minus_" + b_] = dict(
            yearly_diff=dd, n_pos=int(sum(v > 0 for v in dd.values())), n=len(yrs),
            n_pos_2019on=int(sum(dd[k] > 0 for k in range(2019, 2027))),
            ann_2019_diff=out[a_]["ann_2019"] - out[b_]["ann_2019"],
            ann_2019_diff_if_year_neutral=drop,
            monthly_2019on=dict(n=int(len(md19)), mean=float(md19.mean()), n_pos=int((md19 > 0).sum()),
                                n_neg=int((md19 < 0).sum()), n_zero=int((md19.abs() < 1e-9).sum()),
                                t=float(md19.mean() / (md19.std(ddof=1) / np.sqrt(len(md19)))),
                                worst=[(str(i), float(v)) for i, v in md19.sort_values().head(6).items()],
                                best=[(str(i), float(v)) for i, v in md19.sort_values().tail(6).items()]),
            paths=pz["diff_" + a_ + "_minus_" + b_])
    out["reviewA"] = rva

    # ---- 審查 B：回撤期間部位；被跳過股票的事後表現
    rvb = {}
    for cid in ("WN_70L", "WN_70", "W10_70L", "W10_70", "WN_80L"):
        x = res[cid]; nav = x["nav"]
        md, pkd, trd = L.maxdd(P, nav, P.t0, e)
        ip, it = L.didx(P, pkd), L.didx(P, trd)
        bw = x["B"] / nav
        rvb[cid] = dict(mdd=md, peak=pkd, trough=trd, days=it - ip,
                        stock_w_dd=float(x["stkw"][ip:it + 1].mean()), base_w_dd=float(bw[ip:it + 1].mean()),
                        stock_w_at_peak=float(x["stkw"][ip]), base_w_at_peak=float(bw[ip]),
                        base_w_at_trough=float(bw[it]), e50_dd=float(e50[it] / e50[ip] - 1.0),
                        e50_mdd_full=b50["mdd_full"],
                        cash_w_dd=float(x["cashw"][ip:it + 1].mean()), w50_dd=float(x["w50"][ip:it + 1].mean()),
                        npos_dd=float(x["npos"][ip:it + 1].mean()))
        # 回撤期間逐日拆解：淨值日報酬 ≈ 0050 權重 × 0050 日報酬 ＋ 個股貢獻（現金近 0）
        rn = nav[ip + 1:it + 1] / nav[ip:it] - 1.0
        r5 = e50[ip + 1:it + 1] / e50[ip:it] - 1.0
        c5 = x["w50"][ip:it] * r5
        cs = rn - c5
        sw = x["stkw"][ip:it]
        rs = np.where(sw > 1e-6, cs / np.maximum(sw, 1e-6), 0.0)
        rvb[cid].update(contrib_stock_sum=float(cs.sum()), contrib_e50_sum=float(c5.sum()),
                        stock_sleeve_ret_dd=float(np.prod(1 + rs) - 1.0),
                        reb_in_dd=[r_ for r_ in x["reb"] if pkd <= r_["date"] <= trd],
                        n_reb_in_dd=int(sum(pkd <= r_["date"] <= trd for r_ in x["reb"])))
    # 被跳過的股票（因錢不夠）：照引擎規則假想買進的毛報酬，與同日實際買到的股票、0050 對照
    sk_all = {}
    for cid in ("WN_70L", "W10_70L", "WN_80L", "WN_70"):
        x = res[cid]
        sk = x["skips"]
        trd = x["trades"]
        rows_s = []
        for z in sk:
            hr, ts = K.hypo_ret(P, z["i"], z["t"], keep)
            rows_s.append(dict(code=z["code"], date=P.dstr[z["t"]], pos=z["pos"], lma=z["lma"], ret=hr,
                               e50=float(P.e50o[ts] / P.e50o[z["t"]] - 1.0), hold=ts - z["t"],
                               target=z["target"], avail=z["avail"]))
        ds = pd.DataFrame(rows_s)
        sdays = set(x["short_days"])
        rows_b = []
        for c_, i_, tb_ in zip(trd.code, trd.i, trd.tb):
            if int(tb_) in sdays:
                hr, ts = K.hypo_ret(P, int(i_), int(tb_), keep)
                rows_b.append(dict(code=c_, ret=hr, e50=float(P.e50o[ts] / P.e50o[int(tb_)] - 1.0)))
        # 還在手上的部位（未平倉）也算
        db = pd.DataFrame(rows_b)

        def g(df):
            if len(df) == 0:
                return dict(n=0)
            ex = df.ret - df.e50
            return dict(n=int(len(df)), mean=float(df.ret.mean()), median=float(df.ret.median()),
                        win=float((df.ret > 0).mean()), ex_mean=float(ex.mean()), ex_median=float(ex.median()),
                        ex_t=float(ex.mean() / (ex.std(ddof=1) / np.sqrt(len(ex)))) if len(ex) > 2 else None)
        wl = None
        if len(ds) > 2 and len(db) > 2:
            xa = (ds.ret - ds.e50).values; xb = (db.ret - db.e50).values
            se = np.sqrt(xa.var(ddof=1) / len(xa) + xb.var(ddof=1) / len(xb))
            wl = dict(diff_ex_mean=float(xa.mean() - xb.mean()), t=float((xa.mean() - xb.mean()) / se),
                      diff_ex_median=float(np.median(xa) - np.median(xb)))
        sk_all[cid] = dict(skipped=g(ds), bought_same_days=g(db), skipped_minus_bought=wl,
                           skipped_by_pos={int(k): g(v) for k, v in ds.groupby("pos")} if len(ds) else {},
                           skipped_years={str(k): int(v) for k, v in
                                          ds.date.str[:4].value_counts().sort_index().items()} if len(ds) else {},
                           top_skipped=ds.sort_values("ret", ascending=False).head(8).to_dict("records") if len(ds) else [],
                           worst_skipped=ds.sort_values("ret").head(5).to_dict("records") if len(ds) else [])
    # 全部訊號、全部名單：依名單位置（SURGE 順序）的假想毛報酬（不管錢夠不夠）——名次本身有沒有價值
    allr = []
    for s, b in buy.items():
        t = s + 1
        if t > e:
            continue
        for p_, i_ in enumerate(b):
            i_ = int(i_)
            o = P.O[t, i_]
            if not (o == o) or o >= K.limit_up(P.ref[t, i_]) - 1e-9:
                continue
            hr, ts = K.hypo_ret(P, i_, t, keep)
            allr.append(dict(pos=p_ + 1, n=len(b), ret=hr, e50=float(P.e50o[ts] / P.e50o[t] - 1.0), t=t))
    da = pd.DataFrame(allr)
    da["ex"] = da.ret - da.e50
    bypos = {}
    for k, v in da.groupby("pos"):
        bypos[int(k)] = dict(n=int(len(v)), mean=float(v.ret.mean()), median=float(v.ret.median()),
                             ex_mean=float(v.ex.mean()), ex_median=float(v.ex.median()))
    # 同一份名單內：前段（位置 ≤ 名單長度一半）對後段
    da["rel"] = (da.pos - 1) / da.n.clip(lower=1)
    front = da[da.rel < 0.5]; back = da[da.rel >= 0.5]
    w_ = da.groupby("t").apply(lambda v: v.ex[v.rel < 0.5].mean() - v.ex[v.rel >= 0.5].mean()).dropna()
    sk_all["all_list_by_pos"] = dict(by_pos=bypos, front=dict(n=int(len(front)), ex_mean=float(front.ex.mean()),
                                                               ex_median=float(front.ex.median())),
                                     back=dict(n=int(len(back)), ex_mean=float(back.ex.mean()),
                                               ex_median=float(back.ex.median())),
                                     within_month_front_minus_back=dict(n=int(len(w_)), mean=float(w_.mean()),
                                                                        median=float(w_.median()),
                                                                        t=float(w_.mean() / (w_.std(ddof=1) / np.sqrt(len(w_)))),
                                                                        n_pos=int((w_ > 0).sum())))
    rvb["skipped"] = sk_all
    out["reviewB"] = rvb

    # ---- 審查 C：延遲損失集中在哪幾年、哪幾筆、換股日後頭兩天
    rvc = {}
    sig = sorted(buy)
    for cid in DCELLS:
        n0 = res[cid]["nav"]
        lr0 = np.diff(np.log(n0[P.t0:e + 1]))
        days = np.arange(P.t0 + 1, e + 1)
        for d in (1, 2):
            x = dres[(cid, d)]; n1 = x["nav"]
            lr1 = np.diff(np.log(n1[P.t0:e + 1]))
            dif = lr0 - lr1
            o = {}
            o["yearly_loss"] = {k: dly[cid]["d0"].get("yearly", out[cid]["yearly"])[k] - dly[cid]["d%d" % d]["yearly"][k]
                                for k in yrs}
            for wname, wlen in (("W2", 2), ("Wd1", d + 1)):
                W = set()
                for s in sig:
                    for j in range(1, wlen + 1):
                        W.add(s + j)
                inW = np.array([t in W for t in days])
                for per, st in (("full", P.t0 + 1), ("2019", i19 + 1)):
                    m = days >= st
                    tot = float(dif[m].sum()); inw = float(dif[m & inW].sum())
                    yrs_span = (P.dates[e] - P.dates[st - 1]).days / 365.25
                    o[wname + "_" + per] = dict(total_log=tot, in_window_log=inw, out_window_log=tot - inw,
                                                share_in_window=inw / tot if tot != 0 else None,
                                                ann_gap=float(np.exp(tot / yrs_span) - 1),
                                                ann_gap_from_window=float(np.exp(inw / yrs_span) - 1),
                                                n_window_days=int((m & inW).sum()), n_days=int(m.sum()))
            # 逐筆：以 (代號, 訊號日) 配對 d＝0 與 d＝d 的交易
            t0_ = res[cid]["trades"].copy(); t1_ = x["trades"].copy()
            t0_["sig"] = t0_.tb - 1; t1_["sig"] = t1_.tb - 1 - d
            mg = t0_.merge(t1_, on=["code", "sig"], how="outer", suffixes=("_0", "_d"), indicator=True)
            both = mg[mg._merge == "both"].copy()
            both["dret"] = both.ret_0 - both.ret_d
            both["dpnl"] = both.pnl_0 - both.pnl_d
            only0 = mg[mg._merge == "left_only"]; only1 = mg[mg._merge == "right_only"]
            o["trades"] = dict(n0=int(len(t0_)), nd=int(len(t1_)), matched=int(len(both)),
                               only_d0=int(len(only0)), only_dd=int(len(only1)),
                               matched_mean_dret=float(both.dret.mean()), matched_median_dret=float(both.dret.median()),
                               matched_n_d0_better=int((both.dret > 0).sum()),
                               only_d0_pnl=float(only0.pnl_0.sum()), only_dd_pnl=float(only1.pnl_d.sum()),
                               only_d0_mean_ret=float(only0.ret_0.mean()) if len(only0) else None,
                               only_dd_mean_ret=float(only1.ret_d.mean()) if len(only1) else None,
                               top_dret=[dict(code=r.code, buy0=r.buy_date_0, ret0=float(r.ret_0), retd=float(r.ret_d),
                                              dret=float(r.dret)) for r in both.sort_values("dret", ascending=False).head(10).itertuples()],
                               worst_dret=[dict(code=r.code, buy0=r.buy_date_0, ret0=float(r.ret_0), retd=float(r.ret_d),
                                                dret=float(r.dret)) for r in both.sort_values("dret").head(5).itertuples()],
                               top_only_d0=[dict(code=r.code, buy=r.buy_date_0, ret=float(r.ret_0), pnl=float(r.pnl_0))
                                            for r in only0.sort_values("pnl_0", ascending=False).head(8).itertuples()])
            # 前 10 筆配對差（以 dret）占配對總 dret 的比例
            sd = both.dret.sort_values(ascending=False)
            o["trades"]["top10_share_of_sum_dret"] = float(sd.head(10).sum() / sd.sum()) if sd.sum() != 0 else None
            o["trades"]["sum_dret"] = float(sd.sum())
            # 個股層：d＝0 實際買進的股票，換股日開盤 → 延遲執行日開盤（含換股日當天與隔天）漲多少
            tr0 = res[cid]["trades"]
            g_ = []
            for i_, tb_ in zip(tr0.i, tr0.tb):
                i_ = int(i_); tb_ = int(tb_)
                if tb_ + d > e:
                    continue
                v = P.Cf[tb_, i_] / P.O[tb_, i_]
                ok = True
                for u in range(tb_ + 1, tb_ + d):
                    r = P.ret[u, i_]
                    v *= (1 + r) if r == r else 1.0
                oo = P.O[tb_ + d, i_]; rf = P.refe[tb_ + d, i_]
                if oo == oo and rf == rf:
                    v *= oo / rf
                else:
                    ok = False
                if ok:
                    g_.append((v - 1.0, P.e50o[tb_ + d] / P.e50o[tb_] - 1.0))
            ga = np.array(g_)
            o["stock_open_to_delayed_open"] = dict(n=int(len(ga)), mean=float(ga[:, 0].mean()),
                                                   median=float(np.median(ga[:, 0])),
                                                   e50_mean=float(ga[:, 1].mean()),
                                                   share_gt5pct=float((ga[:, 0] > 0.05).mean()))
            # 進場與出場拆解（以 d＝0 的交易金額加權，相對 0050，換成占淨值的 log 近似）
            ent = 0.0; exi = 0.0; ent19 = 0.0; exi19 = 0.0
            for i_, tb_, ts_, c_, pr_, dl_ in zip(tr0.i, tr0.tb, tr0.ts, tr0.cost, tr0.proceeds, tr0.delist):
                i_ = int(i_); tb_ = int(tb_); ts_ = int(ts_)
                if tb_ + d <= e:
                    v = P.Cf[tb_, i_] / P.O[tb_, i_]
                    for u in range(tb_ + 1, tb_ + d):
                        r = P.ret[u, i_]; v *= (1 + r) if r == r else 1.0
                    oo = P.O[tb_ + d, i_]; rf = P.refe[tb_ + d, i_]
                    if oo == oo and rf == rf:
                        v *= oo / rf
                        g1 = c_ / n0[tb_ - 1] * ((v - 1.0) - (P.e50o[tb_ + d] / P.e50o[tb_] - 1.0))
                        ent += g1; ent19 += g1 if tb_ > i19 else 0.0
                if dl_ or ts_ + d > e or ts_ <= tb_:
                    continue
                o1 = P.O[ts_, i_]; rf1 = P.refe[ts_, i_]
                if not (o1 == o1):
                    continue
                v = 1.0 * (P.Cf[ts_, i_] / o1)
                for u in range(ts_ + 1, ts_ + d):
                    r = P.ret[u, i_]; v *= (1 + r) if r == r else 1.0
                oo = P.O[ts_ + d, i_]; rf = P.refe[ts_ + d, i_]
                if oo == oo and rf == rf:
                    v *= oo / rf
                    g2 = pr_ / n0[ts_ - 1] * ((P.e50o[ts_ + d] / P.e50o[ts_] - 1.0) - (v - 1.0))
                    exi += g2; exi19 += g2 if ts_ > i19 else 0.0
            o["entry_exit_split"] = dict(entry_full=ent, exit_full=exi, entry_2019=ent19, exit_2019=exi19,
                                         note="正值＝延遲造成的損失（d＝0 比 d 多賺）；以交易金額÷前一日淨值加權，相對 0050")
            rvc["%s_d%d" % (cid, d)] = o
    out["reviewC"] = rvc
    out["runtime_s"] = time.time() - t_start; out["peak_rss_mb"] = rss()

    # ---- 存檔、凍結
    trs = []
    for k in res:
        t = res[k]["trades"].copy(); t["run"] = k; trs.append(t)
    for (cid, d), x in dres.items():
        t = x["trades"].copy(); t["run"] = "%s_d%d" % (cid, d); trs.append(t)
    pd.concat(trs, ignore_index=True).drop(columns=["i"]).to_csv(os.path.join(HERE, "my_trades.csv.gz"), index=False,
                                                                 compression="gzip")
    nv = pd.DataFrame({"date": P.dstr[P.t0:], "e50": e50[P.t0:]})
    for k in res:
        nv[k] = res[k]["nav"][P.t0:]
        nv[k + "_B"] = res[k]["B"][P.t0:]
    for (cid, d), x in dres.items():
        nv["%s_d%d" % (cid, d)] = x["nav"][P.t0:]
    nv.to_csv(os.path.join(HERE, "my_navs.csv.gz"), index=False, compression="gzip", float_format="%.4f")
    js = json.dumps(out, ensure_ascii=False, indent=1, default=float)
    open(os.path.join(HERE, "my_numbers.json"), "w").write(js)
    h = hashlib.sha256(open(os.path.join(HERE, "my_numbers.json"), "rb").read()).hexdigest()
    open(os.path.join(HERE, "my_numbers.sha256"), "w").write(h + "  my_numbers.json\n")
    print("FROZEN", h, "rss", rss(), "s", round(time.time() - t_start))


if __name__ == "__main__":
    main(sys.argv[1])
