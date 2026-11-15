# -*- coding: utf-8 -*-
"""S11 獨立複製：驗收、E0／Q／QF／QR／C／CF／C0 主跑＋八項檢查、20 條路徑、事前判定、相關、回撤逐段、
事件層級、審查 A（Q 為什麼贏不了 0050）、B（法定期限換股 vs 事件型）、C（C 的合併方式），然後凍結。
用法：python3 -I -B rep_s11_run.py <scratch_dir>
需先跑 s1 的 rep_s1_panel.py <scratch_dir> 與 s3 的 rep_s3_names.py <scratch_dir>/innov_names.json。
輸出（本資料夾）：my_numbers.json、my_numbers.sha256、my_trades.csv.gz、my_paths.csv.gz
凍結前不讀研究員的任何輸出（FINDINGS 第 1、2 節以外、results*.json、trades.csv.gz、navs.csv.gz）。
"""
import sys, os, json, hashlib, resource, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import rep_s11_lib as K
C_, S, Q, R, L = K.C, K.S, K.Q, K.R, K.L
ST, X2 = K.ST, K.X2

CAPS = [600_000.0, 800_000.0, 1_000_000.0, 1_500_000.0, 2_000_000.0]
SHIFTS = [0, 1, 2, 3]
KNOWN_E0 = (0.4814, 0.3409)
BEARS = {"2018": ("2018-01-01", "2018-12-31"), "2020-03": ("2020-01-01", "2020-06-30"),
         "2022": ("2022-01-01", "2022-12-31"), "2024-07~2025-04": ("2024-07-01", "2025-04-30")}
SINGLE = ["E0", "Q", "QF", "QR"]
COMBO = {"C": ("E0", "Q"), "CF": ("E0", "QF"), "C0": ("E0", "0050")}


def rss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024


def tst(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return None
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))


def r4(x):
    return None if x is None else round(float(x), 6)


def main(sc, odir=HERE):
    t_start = time.time()
    P = R.build(sc)
    S.attach_e50(P, sc)
    excl = S.innov_mask_fn(P, sc)
    Rv, mi0 = L.load_rev(P)
    S_, _ = L.surge(Rv, mi0, null2013=True)
    F = K.load_fin(P)
    SUE, _ = K.sue(F["EPS"]); RSUE, _ = K.sue(F["REV"])
    sched = K.quarter_sched(P, F)
    ma60 = R.ma(P, 60)
    e = P.T - 1
    i19 = L.didx(P, "2018-12-28"); i24 = L.didx(P, "2023-12-29"); ih = L.didx(P, "2021-02-19")
    da = np.array(P.dstr)

    def win(a, b):
        i0 = int(np.searchsorted(da, a, side="left")); i1 = int(np.searchsorted(da, b, side="right")) - 1
        return i0, i1
    bw = {k: win(a, b) for k, (a, b) in BEARS.items()}
    e50 = P.e50.values.astype(float)
    out = dict(meta=dict(T=P.T, N=P.N, first=P.dstr[0], last=P.dstr[-1], t0=P.dstr[P.t0], i19=P.dstr[i19],
                         i24=P.dstr[i24], ih=P.dstr[ih],
                         bear_windows={k: (P.dstr[a], P.dstr[b]) for k, (a, b) in bw.items()},
                         note="面板為複製員自建（N＝2130，研究員 2142；S6～S10 複製時已知，差的是從未進股池的代號）"))

    def bears(nav):
        return {k: L.maxdd(P, nav, a, b)[0] for k, (a, b) in bw.items()}

    b50 = dict(ann_2019=L.ann(P, e50, i19, e), ann_full=L.ann(P, e50, P.t0, e), ann_2024=L.ann(P, e50, i24, e),
               ann_1923=L.ann(P, e50, i19, i24), mdd_full=L.maxdd(P, e50, P.t0, e)[0],
               ann_h1=L.ann(P, e50, P.t0, ih), ann_h2=L.ann(P, e50, ih, e), bears=bears(e50),
               yearly=L.yearly(P, e50))
    out["bench_0050"] = b50
    print("0050", {k: v for k, v in b50.items() if k != "yearly"}, flush=True)

    # ---------------- 名單（位移 0～3）
    plans = {}
    qinfo = []
    for sh in SHIFTS:
        lists = Q.make_lists(P, S_, mi0, shift=sh, excl=excl)
        plans[("E0", sh)] = Q.cell_plans(lists, "C10", x=0.10)
        for cell in ("Q", "QF", "QR"):
            plans[(cell, sh)] = K.q_plans(P, F, SUE, RSUE, sched, cell, shift=sh, excl=excl, ma60=ma60,
                                          info=(qinfo if (sh == 0 and cell == "Q") else None))
    print("plans", len(plans), "rss", rss(), round(time.time() - t_start), flush=True)

    def run(cell, sh=0, capital=1_000_000.0, cost=None, block_span=None, rec_pos=False):
        b, k = plans[(cell, sh)]
        return K.simulate(P, b, k, sat_w=0.7, sizing="W10", capital=capital, block_span=block_span,
                          rec_pos=rec_pos, **(cost or ST))

    def e50nav(capital):
        return capital * e50 / e50[P.t0]

    def nav_stats(nav, ntr=None):
        md = L.maxdd(P, nav, P.t0, e)
        return dict(ann_2019=L.ann(P, nav, i19, e), ann_full=L.ann(P, nav, P.t0, e), ann_2024=L.ann(P, nav, i24, e),
                    ann_1923=L.ann(P, nav, i19, i24), mdd_full=md[0], mdd_peak=md[1], mdd_trough=md[2],
                    ann_h1=L.ann(P, nav, P.t0, ih), ann_h2=L.ann(P, nav, ih, e), n_trades=ntr)

    # ---------------- 驗收
    acc = {}
    acc["P0"] = dict(T=P.T, N=P.N, last=P.dstr[-1], note="研究員 N＝2142；複製員面板 2130（已知差異，不影響股池）")
    res = {}
    res["E0"] = run("E0")
    y19 = L.ann(P, res["E0"]["nav"], i19, e); yf = L.ann(P, res["E0"]["nav"], P.t0, e)
    acc["A0"] = dict(ann_2019=y19, ann_full=yf, mdd_full=L.maxdd(P, res["E0"]["nav"], P.t0, e)[0],
                     known_rep_s6=KNOWN_E0, ok=bool(abs(y19 - KNOWN_E0[0]) < 0.00006 and abs(yf - KNOWN_E0[1]) < 0.00006))
    print("A0", acc["A0"], flush=True)
    assert acc["A0"]["ok"]
    # A1
    a1 = dict(rows=0, buy_subset_keep=True, Q_buy_eq_keep=True, buys_eligible=True, sig_after_deadline=True)
    for cell in ("Q", "QF", "QR"):
        for sh in SHIFTS:
            b, k = plans[(cell, sh)]
            for s in b:
                a1["rows"] += 1
                if not set(int(i) for i in b[s]) <= k[s]:
                    a1["buy_subset_keep"] = False
                if cell in ("Q", "QR") and set(int(i) for i in b[s]) != k[s]:
                    a1["Q_buy_eq_keep"] = False
                ev = [x for x in sched if x["s0"] + sh == s]
                assert len(ev) == 1
                ev = ev[0]
                el = K.eligible(P, F, SUE, RSUE, ev, s, excl, need_rev=(cell != "QR"))
                if not all(el[int(i)] for i in k[s]):
                    a1["buys_eligible"] = False
                if not P.dstr[s] > ev["deadline"]:
                    a1["sig_after_deadline"] = False
    acc["A1"] = a1
    print("A1", a1, flush=True)

    # ---------------- 主跑
    for cell in ("Q", "QF", "QR"):
        res[cell] = run(cell)
    half = {}
    for cell in ("E0", "Q", "QF"):
        half[cell] = run(cell, capital=500_000.0, rec_pos=(cell in ("E0", "Q")))
    navs = {c: res[c]["nav"] for c in SINGLE}
    navs["0050"] = e50nav(1_000_000.0)
    for cc, (a, b) in COMBO.items():
        navs[cc] = half[a]["nav"] + (half[b]["nav"] if b != "0050" else e50nav(500_000.0))
    acc["A2"] = dict(max_abs_diff_C_vs_halves=float(np.nanmax(np.abs(navs["C"] - (half["E0"]["nav"] + half["Q"]["nav"])))),
                     C_start=float(navs["C"][P.t0]), ok=True)
    out["acceptance"] = acc

    def trades_of(cc):
        if cc in SINGLE:
            return res[cc]["trades"]
        a, b = COMBO[cc]
        t1 = half[a]["trades"].assign(half=a)
        if b == "0050":
            return t1
        return pd.concat([t1, half[b]["trades"].assign(half=b)], ignore_index=True)

    def open_pnl(cc):
        if cc in SINGLE:
            ops = res[cc]["open_pos"]
        else:
            a, b = COMBO[cc]
            ops = half[a]["open_pos"] + (half[b]["open_pos"] if b != "0050" else [])
        return float(sum(v - c for _, _, c, v, _ in ops))

    def run_cell(cc, sh=0, capital=1_000_000.0, cost=None, block_span=None):
        """回傳 (nav, n_trades)。"""
        if cc in SINGLE:
            x = run(cc, sh, capital, cost, block_span)
            return x["nav"], len(x["trades"])
        a, b = COMBO[cc]
        xa = run(a, sh, capital / 2, cost, block_span)
        if b == "0050":
            return xa["nav"] + e50nav(capital / 2), len(xa["trades"])
        xb = run(b, sh, capital / 2, cost, block_span)
        return xa["nav"] + xb["nav"], len(xa["trades"]) + len(xb["trades"])

    cells = {}
    trade_rows = []
    for cc in SINGLE + list(COMBO):
        nav = navs[cc]; tr = trades_of(cc)
        o = nav_stats(nav, int(len(tr)))
        o["bears"] = bears(nav)
        o["yearly"] = L.yearly(P, nav)
        x2nav, _ = run_cell(cc, cost=X2)
        sp, top = R.span_of(tr, 5)
        blnav, _ = run_cell(cc, block_span=sp)
        o["x2"] = dict(ann_2019=L.ann(P, x2nav, i19, e), ann_full=L.ann(P, x2nav, P.t0, e))
        o["block5"] = dict(ann_2019=L.ann(P, blnav, i19, e), ann_full=L.ann(P, blnav, P.t0, e),
                           drop_2019=o["ann_2019"] - L.ann(P, blnav, i19, e),
                           drop_full=o["ann_full"] - L.ann(P, blnav, P.t0, e))
        o["block5_top"] = top
        inc = float(nav[e] - nav[P.t0])
        o["top5_share_of_gain"] = float(sum(t["pnl"] for t in top) / inc)
        o["final_nav"] = float(nav[e]); o["open_pnl"] = open_pnl(cc)
        if len(tr):
            o["trade_stats"] = dict(exp=float(tr.ret.mean()), win=float((tr.ret > 0).mean()),
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
        if cc in SINGLE or cc in ("C", "CF"):
            for r_ in tr.itertuples():
                trade_rows.append((cc, getattr(r_, "half", cc), r_.code, r_.buy_date, r_.sell_date, round(r_.ret, 6),
                                   round(r_.pnl, 1), int(r_.hold), int(r_.delist), int(r_.reduced)))
        print(cc, round(o["ann_2019"], 4), round(o["ann_full"], 4), round(o["ann_2024"], 4), round(o["mdd_full"], 4),
              o["n_trades"], o["n_checks_pass"], o["eight_cls"], "rss", rss(), round(time.time() - t_start), flush=True)
    out["cells"] = cells
    pd.DataFrame(trade_rows, columns=["cell", "half", "code", "buy_date", "sell_date", "ret", "pnl", "hold", "delist",
                                      "reduced"]).to_csv(os.path.join(odir, "my_trades.csv.gz"), index=False,
                                                         compression="gzip")

    # ---------------- 20 條路徑
    pnav = {}   # (cell, sh, cap) -> nav（單格；C 類用兩半相加）
    def single_nav(cell, sh, cap):
        key = (cell, sh, cap)
        if key not in pnav:
            if sh == 0 and cap == 1_000_000.0 and cell in res:
                pnav[key] = (res[cell]["nav"], len(res[cell]["trades"]))
            elif sh == 0 and cap == 500_000.0 and cell in half:
                pnav[key] = (half[cell]["nav"], len(half[cell]["trades"]))
            else:
                x = run(cell, sh, cap)
                pnav[key] = (x["nav"], len(x["trades"]))
        return pnav[key]

    rows = []
    path_nav = {}
    for cc in SINGLE + list(COMBO):
        for sh in SHIFTS:
            for cp in CAPS:
                if cc in SINGLE:
                    nav, n = single_nav(cc, sh, cp)
                else:
                    a, b = COMBO[cc]
                    na, n1 = single_nav(a, sh, cp / 2)
                    if b == "0050":
                        nav, n = na + e50nav(cp / 2), n1
                    else:
                        nb, n2 = single_nav(b, sh, cp / 2)
                        nav, n = na + nb, n1 + n2
                path_nav[(cc, sh, cp)] = nav
                st = nav_stats(nav, n)
                rows.append(dict(cell=cc, shift=sh, capital=cp, **{k: v for k, v in st.items()},
                                 **{"bear_" + k: v for k, v in bears(nav).items()}))
        print("paths", cc, "rss", rss(), round(time.time() - t_start), flush=True)
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

    def paired(a, b, col):
        x = pr[pr.cell == a].set_index(["shift", "capital"])[col]
        y = pr[pr.cell == b].set_index(["shift", "capital"])[col]
        return (x - y.loc[x.index]).values

    pair = {}
    for a, b in (("C", "E0"), ("CF", "E0"), ("C0", "E0"), ("C", "C0"), ("CF", "C0"), ("Q", "E0"), ("QF", "Q"),
                 ("QR", "Q")):
        o = {}
        for col in ["ann_2019", "ann_full", "ann_2024", "ann_1923", "mdd_full"] + ["bear_" + k for k in BEARS]:
            d = paired(a, b, col)
            o[col] = dict(n_pos=int((d > 0).sum()), median=float(np.median(d)), min=float(d.min()), max=float(d.max()))
        pair["%s-%s" % (a, b)] = o
    out["paired"] = pair

    # ---------------- 相關
    def mret(nav):
        return L.monthly(P, nav, P.t0)
    m50 = mret(e50)

    def corr(na, nb, start=None, excess=True):
        a = mret(na); b = mret(nb)
        if excess:
            a = a - m50; b = b - m50
        df = pd.concat([a, b], axis=1).dropna()
        if start:
            df = df[df.index >= pd.Period(start, "M")]
        return float(df.iloc[:, 0].corr(df.iloc[:, 1])), int(len(df)), str(df.index[0]), str(df.index[-1])
    cor = {}
    for a in ("Q", "QF", "QR"):
        c1 = corr(navs[a], navs["E0"]); c2 = corr(navs[a], navs["E0"], "2019-01")
        c3 = corr(navs[a], navs["E0"], excess=False)
        pc = [corr(path_nav[(a, sh, cp)], path_nav[("E0", sh, cp)])[0] for sh in SHIFTS for cp in CAPS]
        cor[a] = dict(excess_full=c1[0], n_months=c1[1], first=c1[2], last=c1[3], excess_2019=c2[0],
                      n_months_2019=c2[1], raw_full=c3[0], paths20_median=float(np.median(pc)),
                      paths20_min=float(np.min(pc)), paths20_max=float(np.max(pc)))
    out["corr"] = cor
    print("corr", cor, flush=True)

    # ---------------- 事前判定
    rq = cor["Q"]["excess_full"]
    corr_cls = "低相關" if rq < 0.3 else ("中度相關" if rq < 0.6 else "高相關")
    pq = p20["Q"]
    if pq["ann_2019_median"] > b50["ann_2019"] and pq["n_beat_0050"] >= 15 and cells["Q"]["ann_full"] > b50["ann_full"]:
        q_cls = "贏 0050"
    elif pq["ann_2019_median"] < b50["ann_2019"] and pq["n_beat_0050"] <= 5:
        q_cls = "輸 0050"
    else:
        q_cls = "不確定"

    def div_rule(cc):
        pc = pair["%s-E0" % cc]
        c_pass = bool(pc["mdd_full"]["n_pos"] >= 15 and pc["ann_2019"]["median"] >= -0.02)
        seg = {k: dict(n_shallower=pc["bear_" + k]["n_pos"], median=pc["bear_" + k]["median"],
                       ok=bool(pc["bear_" + k]["n_pos"] >= 15 and pc["bear_" + k]["median"] >= 0.01)) for k in BEARS}
        d_pass = bool(sum(v["ok"] for v in seg.values()) >= 3)
        attr = bool(pair["%s-C0" % cc]["ann_2019"]["n_pos"] >= 15)
        if c_pass and d_pass:
            ov = "有分散效果" if attr else "回撤變淺，但來自少押 SURGE，不是 Q"
        elif c_pass or d_pass:
            ov = "部分"
        else:
            ov = "沒有分散效果"
        return dict(c_pass=c_pass, d_pass=d_pass, segments=seg, attribution_Q_contrib=attr,
                    attribution_n=pair["%s-C0" % cc]["ann_2019"]["n_pos"], overall=ov)
    dC = div_rule("C"); dCF = div_rule("CF")
    big = dict(C_top5_share=cells["C"]["top5_share_of_gain"], E0_top5_share=cells["E0"]["top5_share_of_gain"],
               C_drop_2019=cells["C"]["block5"]["drop_2019"], E0_drop_2019=cells["E0"]["block5"]["drop_2019"])
    big["cls"] = "較不依賴大贏家" if (big["C_top5_share"] < big["E0_top5_share"] and
                                     big["C_drop_2019"] < big["E0_drop_2019"]) else "沒有降低"
    xq = "可以加" if (dC["overall"] == "有分散效果" or (q_cls == "贏 0050" and corr_cls != "高相關")) else \
        "不建議加進現行策略，只做紙上追蹤／XQ 交叉驗證"
    out["decisions"] = dict(corr=dict(value=rq, cls=corr_cls), Q_vs_0050=q_cls, C=dC, CF_secondary=dCF,
                            big_winner=big, eight_checks={k: cells[k]["eight_cls"] for k in cells}, xq=xq)
    print("decisions", json.dumps(out["decisions"], ensure_ascii=False, default=str), flush=True)

    # ---------------- 事後：持股重疊、大贏家重疊、資金漂移
    sE = half["E0"]["snap"]; sQ = half["Q"]["snap"]
    ov = []; ovd = []
    for t in range(P.t0 + 1, P.T):
        qh = set(sQ.get(t, {})); eh = set(sE.get(t, {}))
        if qh:
            ov.append(len(qh & eh) / len(qh))
            ovd.append(len(qh & eh))
    trE = res["E0"]["trades"].sort_values("pnl", ascending=False).head(10)
    trQ = res["Q"]["trades"].sort_values("pnl", ascending=False).head(10)
    allQ = res["Q"]["trades"]
    caught = []
    for r_ in trE.itertuples():
        m = allQ[(allQ.code == r_.code) & (allQ.tb <= r_.ts) & (allQ.ts >= r_.tb)]
        caught.append(dict(code=r_.code, buy=r_.buy_date, sell=r_.sell_date, pnl=float(r_.pnl), q_overlap=int(len(m) > 0)))
    same_code = sorted(set(trE.code) & set(trQ.code))
    out["posthoc"] = dict(
        overlap=dict(mean_share_of_Q_half=float(np.mean(ov)), mean_names_both=float(np.mean(ovd)), n_days=len(ov)),
        bigwin=dict(E0_top10=caught, n_E0_top10_caught_by_Q=int(sum(c["q_overlap"] for c in caught)),
                    Q_top10=[dict(code=r_.code, buy=r_.buy_date, sell=r_.sell_date, pnl=float(r_.pnl), ret=float(r_.ret))
                             for r_ in trQ.itertuples()], same_codes_top10=same_code),
        C_drift=dict(surge_half_share_end=float(half["E0"]["nav"][e] / navs["C"][e]),
                     surge_half_share_2019=float(half["E0"]["nav"][i19] / navs["C"][i19])),
        yearly={k: cells[k]["yearly"] for k in cells}, yearly_0050=b50["yearly"])

    # ---------------- 事件層級（位移 0，扣成本前）
    AO, Gf = K.adj_open_matrix(P)
    qs = [x for x in sched if x["s0"] in plans[("Q", 0)][0]]
    ev_rows = []
    ew_parts = []  # (x, x_next, pool idx, top10 idx, top5th idx)
    for n_, ev in enumerate(qs):
        s = ev["s0"]; x = s + 1
        x2 = qs[n_ + 1]["s0"] + 1 if n_ + 1 < len(qs) else None
        el = K.eligible(P, F, SUE, RSUE, ev, s, excl)
        order, sc, b = K.scores(P, F, SUE, RSUE, ev, el, "perp")
        n5 = len(order) // 5
        ew_parts.append((x, x2, order, order[:10], order[:n5], ev["label"]))
        if x2 is None:
            continue
        r = AO[x2, order] / AO[x, order] - 1.0
        allm = float(np.nanmean(r))
        ic = float(pd.Series(sc).rank().corr(pd.Series(r).rank()))
        ev_rows.append(dict(label=ev["label"], exec=P.dstr[x], next_exec=P.dstr[x2], n=len(order),
                            top10_ex=float(np.nanmean(r[:10])) - allm, top5th_ex=float(np.nanmean(r[:n5])) - allm,
                            bot5th_ex=float(np.nanmean(r[-n5:])) - allm, all_mean=allm, ic=ic))
    ed = pd.DataFrame(ev_rows)

    def evsum(g):
        return dict(n_q=int(len(g)), top10_mean=float(g.top10_ex.mean()), top10_t=tst(g.top10_ex),
                    top5th_mean=float(g.top5th_ex.mean()), top5th_t=tst(g.top5th_ex),
                    top5th_pos=int((g.top5th_ex > 0).sum()), bot5th_mean=float(g.bot5th_ex.mean()),
                    bot5th_t=tst(g.bot5th_ex), ic_mean=float(g.ic.mean()), ic_t=tst(g.ic))
    out["event_level"] = dict(all=evsum(ed), pre2024=evsum(ed[ed.exec < "2024-01-01"]),
                              from2024=evsum(ed[ed.exec >= "2024-01-01"]),
                              per_quarter=[{k: (r4(v) if isinstance(v, float) else v) for k, v in r_.items()}
                                           for r_ in ev_rows],
                              note="每季從換股執行日開盤到下一個換股執行日開盤（還原開盤，含息），最後一季（沒有下一個執行日）不列")
    print("event", json.dumps({k: out["event_level"][k] for k in ("all", "pre2024", "from2024")}), flush=True)

    # ---------------- 審查 A：同池等權 vs 0050、Q 相對同池等權
    def ew_nav(which):
        """which：2＝全部可選、3＝前 10、4＝前 1/5。日收盤淨值，第一個執行日開盤＝1；之前＝NaN。"""
        nav = np.full(P.T, np.nan)
        v = 1.0
        for x, x2, *grp in ew_parts:
            idx = (x, x2, *grp)[which]
            end = x2 if x2 is not None else P.T
            base = AO[x, idx]
            path = np.nanmean(Gf[x:end][:, idx] / base, axis=1)
            nav[x:end] = v * path
            if x2 is not None:
                v = v * float(np.nanmean(AO[x2, idx] / base))
        return nav
    x0 = ew_parts[0][0]
    nav_pool = ew_nav(2); nav_top10 = ew_nav(3); nav_top5 = ew_nav(4)
    # 把 x0 前一天收盤當 1（讓年報酬從 x0 開盤起算）
    for nv in (nav_pool, nav_top10, nav_top5):
        nv[x0 - 1] = 1.0
    r50 = np.r_[np.nan, e50[1:] / e50[:-1] - 1.0]

    def blend(nav_s):
        rs = np.r_[np.nan, nav_s[1:] / nav_s[:-1] - 1.0]
        rb = 0.3 * r50 + 0.7 * rs
        out_ = np.full(P.T, np.nan); out_[x0 - 1] = 1.0
        for t in range(x0, P.T):
            out_[t] = out_[t - 1] * (1.0 + rb[t])
        return out_
    nav_bpool = blend(nav_pool); nav_btop = blend(nav_top10)
    qn = navs["Q"]

    def yr(nav, y):
        a = int(np.searchsorted(da, "%d-01-01" % y)) - 1
        b = int(np.searchsorted(da, "%d-01-01" % (y + 1))) - 1
        a = max(a, x0 - 1)
        return float(nav[b] / nav[a] - 1.0)
    years = list(range(2016, 2027))
    tab = []
    for y in years:
        row = dict(year=y, e50=yr(e50, y), pool=yr(nav_pool, y), top10=yr(nav_top10, y), top5th=yr(nav_top5, y),
                   Q=yr(qn, y), blend_pool=yr(nav_bpool, y), blend_top10=yr(nav_btop, y))
        row["smallcap_lag"] = row["pool"] - row["e50"]
        row["selection"] = row["top10"] - row["pool"]
        row["selection_top5th"] = row["top5th"] - row["pool"]
        row["Q_minus_0050"] = row["Q"] - row["e50"]
        row["Q_minus_blendpool"] = row["Q"] - row["blend_pool"]
        row["impl_drag"] = row["Q"] - row["blend_top10"]
        tab.append(row)
    ta = pd.DataFrame(tab)

    def annr(nav, i0, i1=e):
        return L.ann(P, nav, i0, i1)
    sumA = {}
    for nm_, i0 in (("from_first_exec", x0 - 1), ("from_2019", i19), ("from_2024", i24), ("2019_2023", i19)):
        i1 = i24 if nm_ == "2019_2023" else e
        sumA[nm_] = dict(e50=annr(e50, i0, i1), pool=annr(nav_pool, i0, i1), top10=annr(nav_top10, i0, i1),
                         top5th=annr(nav_top5, i0, i1), Q=annr(qn, i0, i1), blend_pool=annr(nav_bpool, i0, i1),
                         blend_top10=annr(nav_btop, i0, i1), start=P.dstr[i0], end=P.dstr[i1])
    out["review_A"] = dict(
        yearly=[{k: (r4(v) if isinstance(v, float) else v) for k, v in r_.items()} for r_ in tab],
        summary=sumA,
        n_years_selection_pos=int((ta.selection > 0).sum()), n_years_top5th_pos=int((ta.selection_top5th > 0).sum()),
        n_years_smallcap_lag_neg=int((ta.smallcap_lag < 0).sum()), n_years=len(ta),
        n_years_Q_beats_blendpool=int((ta.Q_minus_blendpool > 0).sum()),
        quarterly_selection=dict(top10_mean=float(ed.top10_ex.mean()), top10_t=tst(ed.top10_ex),
                                 top5th_mean=float(ed.top5th_ex.mean()), top5th_t=tst(ed.top5th_ex)),
        note=("同池＝每季 Q 的可選股票（上市、非金融、非創新板、有 SUE⊥）；等權、季內買進持有、執行日開盤換股、扣成本前、含息。"
              "blend＝每日再平衡 30% 0050＋70% 等權組合（模擬 Q 的底倉結構，扣成本前）。"
              "smallcap_lag＝同池等權−0050；selection＝前 10 名等權−同池等權；impl_drag＝Q 實際−blend_top10（成本、整張、0050 閒錢等）。"
              "2026 只到 10-22。"))
    print("reviewA", json.dumps(sumA), flush=True)

    # ---------------- 審查 B：事件型（市場可得日 A 的次一交易日開盤進場，只用 fsq＝直接觀察的 A）
    rowsB = []
    pit_rows = []
    prev = None   # 上一季的可選股票分布與係數（PIT 事件型門檻）
    for n_, ev in enumerate(qs):
        s = ev["s0"]; x = s + 1
        x2 = qs[n_ + 1]["s0"] + 1 if n_ + 1 < len(qs) else None
        j = ev["qi"] - F["q0"]
        el = K.eligible(P, F, SUE, RSUE, ev, s, excl)
        order, sc, b = K.scores(P, F, SUE, RSUE, ev, el, "perp")
        cur = dict(sue=np.sort(SUE[j][order]), rs=np.sort(RSUE[j][order]), b=b, thr10=float(sc[9]),
                   thr5=float(sc[len(sc) // 5 - 1]))
        if x2 is not None and ev["qi"] >= 2017 * 4:
            meth = F["Ameth"][j]; A = F["A"][j]
            fsq = np.array([meth[i] == "fsq" for i in order])
            a1 = np.array([int(np.searchsorted(da, A[i], side="right")) if A[i] else x for i in order])  # A 之後第一個交易日
            a1 = np.minimum(a1, x)
            pool = order[fsq]; pa1 = a1[fsq]
            # 同窗基準：可選股票（fsq）在 [a, x'] 的等權報酬
            cache = {}

            def pool_ret(a, b_):
                key = (a, b_)
                if key not in cache:
                    cache[key] = float(np.nanmean(AO[b_, pool] / AO[a, pool] - 1.0))
                return cache[key]
            rank_of = {int(i): r_ for r_, i in enumerate(order)}
            n5 = len(order) // 5
            for i, a in zip(pool, pa1):
                rk = rank_of[int(i)]
                pre = AO[x, i] / AO[a, i] - 1.0
                post = AO[x2, i] / AO[x, i] - 1.0
                full = AO[x2, i] / AO[a, i] - 1.0
                rowsB.append(dict(label=ev["label"], code=P.codes[i], rank=rk, top10=rk < 10, top5th=rk < n5,
                                  gap_td=int(x - a), pre=pre, post=post, full=full,
                                  pre_ex=pre - pool_ret(a, x) if a < x else 0.0, post_ex=post - pool_ret(x, x2),
                                  full_ex=full - pool_ret(a, x2)))
            # PIT 事件型：用上一季的分布換算百分位、上一季的係數算殘差、上一季前 10 名／前 1/5 的門檻
            if prev is not None:
                def ecdf(sorted_v, v):
                    lo = np.searchsorted(sorted_v, v, side="left"); hi = np.searchsorted(sorted_v, v, side="right")
                    return (lo + hi) / 2.0 / len(sorted_v) + 0.5 / len(sorted_v)
                for i, a in zip(pool, pa1):
                    ps = ecdf(prev["sue"], SUE[j][i]); pr_ = ecdf(prev["rs"], RSUE[j][i])
                    res_ = ps - (prev["b"][0] + prev["b"][1] * pr_)
                    sel10 = res_ >= prev["thr10"]; sel5 = res_ >= prev["thr5"]
                    if not sel5:
                        continue
                    pre = AO[x, i] / AO[a, i] - 1.0
                    post = AO[x2, i] / AO[x, i] - 1.0
                    full = AO[x2, i] / AO[a, i] - 1.0
                    pit_rows.append(dict(label=ev["label"], code=P.codes[i], sel10=bool(sel10), gap_td=int(x - a),
                                         pre_ex=pre - pool_ret(a, x) if a < x else 0.0, post_ex=post - pool_ret(x, x2),
                                         full_ex=full - pool_ret(a, x2)))
        prev = cur
    db = pd.DataFrame(rowsB); dp = pd.DataFrame(pit_rows)

    def bsum(g, key="label"):
        if len(g) == 0:
            return None
        q = g.groupby(key)[["pre_ex", "post_ex", "full_ex", "gap_td"]].mean()
        return dict(n_events=int(len(g)), n_q=int(len(q)), gap_td_median=float(g.gap_td.median()),
                    pre_ex_mean=float(q.pre_ex.mean()), pre_ex_t=tst(q.pre_ex),
                    post_ex_mean=float(q.post_ex.mean()), post_ex_t=tst(q.post_ex),
                    full_ex_mean=float(q.full_ex.mean()), full_ex_t=tst(q.full_ex))
    yr24 = lambda g: g[g.label >= "2023Q4"]  # 執行日在 2024 起（2023Q4 的執行日是 2024-04）
    pre24 = lambda g: g[g.label < "2023Q4"]
    out["review_B"] = dict(
        same_names=dict(top10=bsum(db[db.top10]), top5th=bsum(db[db.top5th]), all_fsq=bsum(db),
                        top10_pre2024=bsum(pre24(db[db.top10])), top10_from2024=bsum(yr24(db[db.top10])),
                        top5th_pre2024=bsum(pre24(db[db.top5th])), top5th_from2024=bsum(yr24(db[db.top5th]))),
        pit_event=dict(sel10=bsum(dp[dp.sel10]) if len(dp) else None, sel5th=bsum(dp) if len(dp) else None,
                       n_per_q_sel10=float(dp[dp.sel10].groupby("label").size().mean()) if len(dp) else None,
                       sel10_pre2024=bsum(pre24(dp[dp.sel10])), sel10_from2024=bsum(yr24(dp[dp.sel10]))),
        approx_annual_cost_top10=None,
        note=("只用 2017Q1 起、A_method＝fsq（證交所本益比檔直接觀察）的公司。事件型進場＝A 之後第一個交易日開盤；"
              "法定期限型進場＝訊號列次日開盤（Q 的執行日）；出場都是下一季的執行日開盤。pre＝公布後到我們進場之間的報酬，"
              "ex＝減同窗可選股票（fsq）等權報酬；扣成本前；每季先平均再對季做 t。"
              "same_names 用 Q 自己的排名（事件型其實在公布當下不知道完整排名，所以是『等待損失』的上限描述）；"
              "pit_event 用上一季的分布與係數換算 SUE⊥、上一季的前 10 名／前 1/5 門檻，公布當下就能決定。"))
    if out["review_B"]["same_names"]["top10"]:
        pm = out["review_B"]["same_names"]["top10"]["pre_ex_mean"]
        out["review_B"]["approx_annual_cost_top10"] = dict(per_quarter=pm, x4_x07=pm * 4 * 0.7,
                                                           note="每季等待損失 × 4 季 × 衛星 70%：粗估年化影響（扣成本前、不計名額與成本）")
    print("reviewB", json.dumps({k: out["review_B"][k] for k in ("same_names", "pit_event")}, default=str), flush=True)

    # ---------------- 審查 C：每年再平衡一次、名單聯集
    def annual_rebal(na, nb):
        nav = np.full(P.T, np.nan)
        ye = [P.t0] + [int(np.searchsorted(da, "%d-01-01" % y)) - 1 for y in range(2016, 2027)]
        ye = sorted(set(i for i in ye if P.t0 <= i < e)) + [e]
        v = na[P.t0] + nb[P.t0]
        for a, b in zip(ye[:-1], ye[1:]):
            seg = 0.5 * v * (na[a:b + 1] / na[a]) + 0.5 * v * (nb[a:b + 1] / nb[a])
            nav[a:b + 1] = seg
            v = seg[-1]
        return nav
    rowsC = []
    up = {}
    for sh in SHIFTS:
        bE, kE = plans[("E0", sh)]; bQ, kQ = plans[("Q", sh)]
        up[sh] = K.union_plans(bE, kE, bQ, kQ)
    for sh in SHIFTS:
        for cp in CAPS:
            nE, _ = single_nav("E0", sh, cp / 2); nQ, _ = single_nav("Q", sh, cp / 2)
            nav_ar = annual_rebal(nE, nQ)
            bu, ku, both = up[sh]
            xu = K.simulate(P, bu, ku, sat_w=0.7, sizing="W10", capital=cp, **ST)
            for nm_, nav, ntr in (("C_annual", nav_ar, None), ("C_union", xu["nav"], len(xu["trades"])),
                                  ("C", path_nav[("C", sh, cp)], None), ("E0", path_nav[("E0", sh, cp)], None)):
                st = nav_stats(nav, ntr)
                rowsC.append(dict(cell=nm_, shift=sh, capital=cp, **st, **{"bear_" + k: v for k, v in bears(nav).items()},
                                  both_rows=both if nm_ == "C_union" else None))
            if sh == 0 and cp == 1_000_000.0:
                main_union = xu
    dc = pd.DataFrame(rowsC)
    revC = {}
    for nm_, g in dc.groupby("cell"):
        m = g[(g["shift"] == 0) & (g.capital == 1_000_000.0)].iloc[0]
        revC[nm_] = dict(main=dict(ann_2019=float(m.ann_2019), ann_full=float(m.ann_full), ann_2024=float(m.ann_2024),
                                   mdd_full=float(m.mdd_full), n_trades=None if pd.isna(m.n_trades) else int(m.n_trades),
                                   bears={k: float(m["bear_" + k]) for k in BEARS}),
                         p20=dict(ann_2019_median=float(g.ann_2019.median()), n_beat_0050=int((g.ann_2019 > b50["ann_2019"]).sum()),
                                  mdd_median=float(g.mdd_full.median()),
                                  bears_median={k: float(g["bear_" + k].median()) for k in BEARS}))
    for nm_ in ("C_annual", "C_union"):
        x = dc[dc.cell == nm_].set_index(["shift", "capital"]); y = dc[dc.cell == "E0"].set_index(["shift", "capital"])
        z = dc[dc.cell == "C"].set_index(["shift", "capital"])
        o = {}
        for col in ["ann_2019", "mdd_full"] + ["bear_" + k for k in BEARS]:
            d1 = (x[col] - y[col].loc[x.index]).values; d2 = (x[col] - z[col].loc[x.index]).values
            o[col] = dict(vs_E0_n_pos=int((d1 > 0).sum()), vs_E0_median=float(np.median(d1)),
                          vs_C_n_pos=int((d2 > 0).sum()), vs_C_median=float(np.median(d2)))
        revC[nm_]["paired"] = o
        # 依同一套 (c)(d) 規則描述（只描述，不改判定）
        cpass = bool(o["mdd_full"]["vs_E0_n_pos"] >= 15 and o["ann_2019"]["vs_E0_median"] >= -0.02)
        dpass = bool(sum(o["bear_" + k]["vs_E0_n_pos"] >= 15 and o["bear_" + k]["vs_E0_median"] >= 0.01 for k in BEARS) >= 3)
        revC[nm_]["c_rule"] = cpass; revC[nm_]["d_rule"] = dpass
    revC["C_union"]["main_extra"] = dict(avg_npos=float(main_union["npos"][P.t0 + 1:].mean()),
                                         no_slot=int(main_union["cnt"]["no_slot"]), buys=int(main_union["cnt"]["buy"]),
                                         both_rows=int(up[0][2]))
    revC["note"] = ("C_annual：兩半淨值每年最後一個交易日收盤拉回各半（用兩半各自的日報酬相乘，不計移轉成本）。"
                    "C_union：單一帳戶 70% 衛星、W10（每檔＝衛星÷10）、最多 10 檔共用名額；E0 訊號列買 E0 名單、Q 訊號列買 Q 名單；"
                    "續抱＝最近一次 E0 前 20 名 ∪ 最近一次 Q 前 10 名。")
    out["review_C"] = revC
    dc.to_csv(os.path.join(odir, "my_reviewC_paths.csv.gz"), index=False, float_format="%.6f", compression="gzip")
    print("reviewC", json.dumps({k: (v["main"], v["p20"]) for k, v in revC.items() if k != "note"}), flush=True)

    out["q_signal_info"] = [dict(label=d_["label"], date=d_["date"], n_elig=d_["n_elig"], top=d_["top"])
                            for d_ in qinfo]
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
