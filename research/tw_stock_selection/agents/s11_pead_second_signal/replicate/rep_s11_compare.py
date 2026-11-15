# -*- coding: utf-8 -*-
"""S11 獨立複製：凍結後比對。先驗 my_numbers.json 的 sha256，再比對研究員 FINDINGS.md（全文）的數字與 trades.csv.gz。
results*.json、navs.csv.gz 不讀。另做兩個事後診斷（不改 my_numbers.json）：
  D1：金融保險分類改用「同代號最後一季的財報格式」（o_pead 的 ind.get(code) 寫法）重跑 Q（100 萬、50 萬），看交易是否逐筆相同。
  D2：事件層級把最後一季（2026Q2，沒有下一個執行日）算到面板最後一天，看 44 季的 t 是否對上研究員。
用法：python3 -I -B rep_s11_compare.py <scratch_dir>
"""
import sys, os, json, hashlib
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import rep_s11_lib as K

RES_DIR = os.path.dirname(HERE)

# 研究員數字（FINDINGS.md 第 3～8 節與結論；百分比）
RES = {
    "main": {  # 2019 起、全期、2024 起、2019～2023、最大回撤
        "E0": (48.13, 34.08, 70.16, 37.01, -33.40), "Q": (23.35, 18.48, 22.75, 23.69, -35.77),
        "QF": (26.20, 19.84, 36.63, 20.69, -29.39), "QR": (30.80, 22.28, 40.29, 25.74, -42.09),
        "C": (37.24, 26.65, 53.80, 28.73, -32.78), "CF": (38.17, 26.86, 59.62, 27.40, -30.98),
        "C0": (39.19, 27.48, 65.35, 26.34, -30.32)},
    "n_trades": {"E0": 379, "Q": 362, "QF": 131, "QR": 327, "C": 661, "CF": 454, "C0": 333},
    "paths20": {  # 中位數、最少、最多、贏 0050 條數、回撤中位數
        "E0": (46.23, 43.24, 48.51, 20, -34.01), "Q": (23.12, 19.13, 25.85, 0, -35.80),
        "QF": (26.19, 24.59, 28.26, 0, -30.79), "QR": (30.19, 27.67, 31.71, 16, -41.81),
        "C": (35.49, 33.45, 39.15, 20, -34.26), "CF": (36.45, 34.23, 40.08, 20, -32.26),
        "C0": (37.84, 35.92, 41.12, 20, -31.57)},
    "x2": {"Q": (18.65, 14.57), "C": (32.61, 23.13), "E0": (42.28, 29.70)},
    "block5": {"Q": (18.08, 14.96), "C": (34.13, 24.65), "E0": (43.42, 31.12)},
    "paired": {  # 2019 起（條數、中位數）、全期回撤（條數、中位數）、2019～2023 中位數、2024 起中位數
        "C-E0": (0, -10.03, 8, -0.33, -6.85, -16.17), "CF-E0": (0, -9.57, 15, 1.41, -9.25, -10.11),
        "C0-E0": (0, -7.89, 20, 2.56, -9.82, -4.33), "C-C0": (0, -2.07, 0, -2.85, 2.45, -12.08),
        "Q-E0": (0, -23.37, 5, -2.43, -13.35, -43.23), "QF-Q": (20, 2.89, 20, 5.20, -2.72, 14.09),
        "QR-Q": (20, 7.55, 0, -4.83, 3.56, 14.56)},
    "bears_main": {
        "E0": (-16.38, -26.92, -28.79, -33.40), "Q": (-19.46, -32.05, -21.50, -35.77),
        "QF": (-14.33, -28.34, -26.82, -29.39), "C": (-17.37, -29.44, -25.48, -32.78),
        "CF": (-16.89, -27.57, -28.01, -30.98), "C0": (-16.95, -27.80, -30.32, -30.22),
        "0050": (-15.99, -28.22, -33.96, -27.48)},
    "bears_p20": {
        "E0": (-16.79, -26.47, -29.69, -34.01), "Q": (-19.56, -31.82, -22.33, -35.80),
        "QF": (-16.37, -27.73, -29.30, -30.69), "C": (-17.28, -29.17, -25.48, -34.26),
        "CF": (-16.31, -27.56, -28.56, -32.26), "C0": (-16.16, -27.60, -30.53, -31.57)},
    "bears_C_vs_E0": {"2018": (10, 0.00), "2020-03": (0, -2.64), "2022": (20, 3.65), "2024-07~2025-04": (8, -0.33)},
    "corr": {"Q_full": 0.458, "Q_2019": 0.474, "Q_p20": 0.471, "QF": 0.510, "QF_p20": 0.525, "QR": 0.362,
             "Q_raw": 0.776, "overlap": 1.0},
    "event": {"top5th": 2.13, "top5th_t": 4.63, "n_q": 44, "ic": 0.068, "ic_t": 6.47, "top10": 2.07, "top10_t": 1.94,
              "top5th_2024": 1.09, "top5th_2024_t": 0.90, "top10_2024": -0.64},
    "bigwin": {"C_share": 19.1, "E0_share": 24.9, "C_drop": 3.12, "E0_drop": 4.70, "C0_share": 17.6, "C0_drop": 2.92},
    "eight": {"E0": "值得上 XQ 測", "Q": "不值得", "QF": "不值得", "QR": "有條件", "C": "值得上 XQ 測",
              "CF": "值得上 XQ 測", "C0": "值得上 XQ 測"},
    "decisions": {"corr": "中度相關", "Q": "輸 0050", "C": "沒有分散效果", "CF": "沒有分散效果",
                  "c_pass": False, "d_pass": False, "attr": False, "big": "較不依賴大贏家",
                  "xq": "不建議加進現行策略，只做紙上追蹤／XQ 交叉驗證"},
    "drift_end": 77.0,
    "yearly_Q": {2015: -9.6, 2016: 15.5, 2017: 31.1, 2018: -3.6, 2019: 29.7, 2020: 32.0, 2021: 42.5, 2022: -12.3,
                 2023: 35.4, 2024: 15.3, 2025: 26.9, 2026: 21.7},
}


def main(sc):
    js = open(os.path.join(HERE, "my_numbers.json")).read()
    h = hashlib.sha256(js.encode("utf-8")).hexdigest()
    h0 = open(os.path.join(HERE, "my_numbers.sha256")).read().split()[0]
    assert h == h0, "my_numbers.json 已被改動"
    M = json.loads(js)
    cmp = dict(sha256_verified=h, items={})
    worst = dict(ann=0.0, mdd=0.0, trades=0.0)
    rows = {}
    for c, (a19, af, a24, a1923, mdd) in RES["main"].items():
        m = M["cells"][c]
        mine = (m["ann_2019"] * 100, m["ann_full"] * 100, m["ann_2024"] * 100, m["ann_1923"] * 100, m["mdd_full"] * 100)
        d = [round(x - y, 2) for x, y in zip(mine, (a19, af, a24, a1923, mdd))]
        nt = m["n_trades"]; ntr = RES["n_trades"][c]
        rows[c] = dict(mine=[round(x, 2) for x in mine], res=[a19, af, a24, a1923, mdd], diff=d,
                       trades=(nt, ntr, round((nt - ntr) / ntr * 100, 2)))
        worst["ann"] = max(worst["ann"], max(abs(x) for x in d[:4]))
        worst["mdd"] = max(worst["mdd"], abs(d[4]))
        worst["trades"] = max(worst["trades"], abs((nt - ntr) / ntr * 100))
    cmp["items"]["main"] = rows
    p = {}
    for c, (med, mn, mx, nb, mdd) in RES["paths20"].items():
        m = M["paths20"][c]
        mine = (m["ann_2019_median"] * 100, m["ann_2019_min"] * 100, m["ann_2019_max"] * 100, m["n_beat_0050"],
                m["mdd_median"] * 100)
        p[c] = dict(mine=[round(x, 2) for x in mine], res=[med, mn, mx, nb, mdd],
                    diff=[round(x - y, 2) for x, y in zip(mine, (med, mn, mx, nb, mdd))])
    cmp["items"]["paths20"] = p
    pa = {}
    for k, v in RES["paired"].items():
        m = M["paired"][k]
        mine = (m["ann_2019"]["n_pos"], m["ann_2019"]["median"] * 100, m["mdd_full"]["n_pos"], m["mdd_full"]["median"] * 100,
                m["ann_1923"]["median"] * 100, m["ann_2024"]["median"] * 100)
        pa[k] = dict(mine=[round(x, 2) for x in mine], res=list(v), diff=[round(x - y, 2) for x, y in zip(mine, v)])
    cmp["items"]["paired"] = pa
    bk = list(K.__dict__.get("BEARS_KEYS", ["2018", "2020-03", "2022", "2024-07~2025-04"]))
    bm = {}
    for c, v in RES["bears_main"].items():
        mine = [M["bench_0050"]["bears"][k] * 100 for k in bk] if c == "0050" else [M["cells"][c]["bears"][k] * 100 for k in bk]
        bm[c] = dict(mine=[round(x, 2) for x in mine], res=list(v), diff=[round(x - y, 2) for x, y in zip(mine, v)])
    cmp["items"]["bears_main"] = bm
    bp = {}
    for c, v in RES["bears_p20"].items():
        mine = [M["paths20"][c]["bears_median"][k] * 100 for k in bk]
        bp[c] = dict(mine=[round(x, 2) for x in mine], res=list(v), diff=[round(x - y, 2) for x, y in zip(mine, v)])
    cmp["items"]["bears_p20"] = bp
    cmp["items"]["bears_C_vs_E0"] = {k: dict(mine=(M["decisions"]["C"]["segments"][k]["n_shallower"],
                                                    round(M["decisions"]["C"]["segments"][k]["median"] * 100, 2)),
                                              res=v) for k, v in RES["bears_C_vs_E0"].items()}
    co = M["corr"]
    cmp["items"]["corr"] = dict(mine=dict(Q_full=co["Q"]["excess_full"], Q_2019=co["Q"]["excess_2019"],
                                          Q_p20=co["Q"]["paths20_median"], QF=co["QF"]["excess_full"],
                                          QF_p20=co["QF"]["paths20_median"], QR=co["QR"]["excess_full"],
                                          Q_raw=co["Q"]["raw_full"],
                                          overlap=M["posthoc"]["overlap"]["mean_share_of_Q_half"] * 100),
                                res=RES["corr"])
    ev = M["event_level"]
    cmp["items"]["event"] = dict(mine=dict(top5th=ev["all"]["top5th_mean"] * 100, top5th_t=ev["all"]["top5th_t"],
                                           n_q=ev["all"]["n_q"], ic=ev["all"]["ic_mean"], ic_t=ev["all"]["ic_t"],
                                           top10=ev["all"]["top10_mean"] * 100, top10_t=ev["all"]["top10_t"],
                                           top5th_2024=ev["from2024"]["top5th_mean"] * 100,
                                           top5th_2024_t=ev["from2024"]["top5th_t"],
                                           top10_2024=ev["from2024"]["top10_mean"] * 100), res=RES["event"])
    cmp["items"]["x2_block5"] = {c: dict(mine_x2=(round(M["cells"][c]["x2"]["ann_2019"] * 100, 2),
                                                  round(M["cells"][c]["x2"]["ann_full"] * 100, 2)), res_x2=RES["x2"][c],
                                         mine_block5=(round(M["cells"][c]["block5"]["ann_2019"] * 100, 2),
                                                      round(M["cells"][c]["block5"]["ann_full"] * 100, 2)),
                                         res_block5=RES["block5"][c]) for c in RES["x2"]}
    cmp["items"]["bigwin"] = dict(mine=dict(C_share=M["cells"]["C"]["top5_share_of_gain"] * 100,
                                            E0_share=M["cells"]["E0"]["top5_share_of_gain"] * 100,
                                            C_drop=M["cells"]["C"]["block5"]["drop_2019"] * 100,
                                            E0_drop=M["cells"]["E0"]["block5"]["drop_2019"] * 100,
                                            C0_share=M["cells"]["C0"]["top5_share_of_gain"] * 100,
                                            C0_drop=M["cells"]["C0"]["block5"]["drop_2019"] * 100), res=RES["bigwin"])
    d = M["decisions"]
    mine_dec = dict(corr=d["corr"]["cls"], Q=d["Q_vs_0050"], C=d["C"]["overall"], CF=d["CF_secondary"]["overall"],
                    c_pass=d["C"]["c_pass"], d_pass=d["C"]["d_pass"], attr=d["C"]["attribution_Q_contrib"],
                    big=d["big_winner"]["cls"], xq=d["xq"])
    cmp["items"]["decisions"] = dict(mine=mine_dec, res=RES["decisions"],
                                     all_same=bool(all(mine_dec[k] == RES["decisions"][k] for k in mine_dec)))
    cmp["items"]["eight"] = dict(mine=d["eight_checks"], res=RES["eight"],
                                 all_same=bool(all(d["eight_checks"][k] == RES["eight"][k] for k in RES["eight"])))
    cmp["items"]["yearly_Q"] = {y: dict(mine=round(M["cells"]["Q"]["yearly"][str(y)] * 100, 1), res=v)
                                for y, v in RES["yearly_Q"].items()}
    cmp["items"]["drift_end"] = dict(mine=M["posthoc"]["C_drift"]["surge_half_share_end"] * 100, res=RES["drift_end"])
    cmp["worst_main"] = worst
    cmp["thresholds"] = dict(ann_pp=0.5, mdd_pp=0.5, trades_pct=2.0)
    cmp["main_within_threshold"] = {c: bool(max(abs(x) for x in r["diff"][:4]) <= 0.5 and abs(r["diff"][4]) <= 0.5 and
                                            abs(r["trades"][2]) <= 2.0) for c, r in rows.items()}

    # ---------------- 逐筆交易（研究員 trades.csv.gz：why＝rule 是已平倉，open_end 是期末未平倉）
    t = pd.read_csv(os.path.join(RES_DIR, "trades.csv.gz"), dtype={"code": str})
    m = pd.read_csv(os.path.join(HERE, "my_trades.csv.gz"), dtype={"code": str})
    tc = {}
    for run in ("E0", "Q", "QF", "QR", "C", "CF"):
        a = t[(t.run == run) & (t.why == "rule")]; b = m[m.cell == run]
        A = set(zip(a.sleeve, a.code, a.ent_date, a.exit_date)); B = set(zip(b.half, b.code, b.buy_date, b.sell_date))
        tc[run] = dict(res=len(a), mine=len(b), common=len(A & B), only_res=sorted(A - B), only_me=sorted(B - A))
    tc["C0"] = dict(res=int(((t.run == "C0") & (t.why == "rule")).sum()), mine=int(M["cells"]["C0"]["n_trades"]))
    cmp["trades"] = tc

    # ---------------- 事後診斷 D1、D2
    P = K.R.build(sc); K.S.attach_e50(P, sc); excl = K.S.innov_mask_fn(P, sc)
    F = K.load_fin(P); SUE, _ = K.sue(F["EPS"]); RS, _ = K.sue(F["REV"])
    sched = K.quarter_sched(P, F); ma60 = K.R.ma(P, 60)
    e = P.T - 1; i19 = K.L.didx(P, "2018-12-28"); i24 = K.L.didx(P, "2023-12-29")
    fl = pd.read_parquet(K.PEAD + "/fin_long.parquet").sort_values(["code", "y", "q"])
    last = fl.groupby("code").industry.last()
    fin_last = np.zeros(P.N, bool)
    for c, ind in last.items():
        if c in P.cidx:
            fin_last[P.cidx[c]] = ind != "ci"
    F2 = dict(F); F2["FIN"] = np.repeat(fin_last[None, :], F["nq"], axis=0)
    diff_cls = []
    for ev in sched:
        j = ev["qi"] - F["q0"]
        u = K.R.universe(P, ev["s0"]) & ~excl(ev["s0"])
        for i in np.where(u & (F["FIN"][j] != F2["FIN"][j]))[0]:
            diff_cls.append((ev["label"], P.codes[i], "mine_fin" if F["FIN"][j, i] else "mine_nonfin"))
    d1 = dict(classification_differences_in_universe=diff_cls)
    for cap, run in ((1e6, "Q"), (5e5, "C")):
        b, k = K.q_plans(P, F2, SUE, RS, sched, "Q", excl=excl, ma60=ma60)
        x = K.simulate(P, b, k, sat_w=0.7, sizing="W10", capital=cap, **K.ST)
        nav = x["nav"]; tr = x["trades"]
        a = t[(t.run == run) & (t.sleeve == "Q") & (t.why == "rule")]
        A = set(zip(a.code, a.ent_date, a.exit_date)); B = set(zip(tr.code, tr.buy_date, tr.sell_date))
        d1["Q_%d" % cap] = dict(ann_2019=K.L.ann(P, nav, i19, e), ann_full=K.L.ann(P, nav, P.t0, e),
                                ann_2024=K.L.ann(P, nav, i24, e), mdd=K.L.maxdd(P, nav, P.t0, e)[0], n=len(tr),
                                only_res=sorted(A - B), only_me=sorted(B - A))
    # D1b：同一套設定下的 20 條路徑中位數（Q）
    p_ = []
    for sh in range(4):
        b, k = K.q_plans(P, F2, SUE, RS, sched, "Q", shift=sh, excl=excl, ma60=ma60)
        for cp in (6e5, 8e5, 1e6, 1.5e6, 2e6):
            x = K.simulate(P, b, k, sat_w=0.7, sizing="W10", capital=cp, **K.ST)
            p_.append((K.L.ann(P, x["nav"], i19, e), K.L.maxdd(P, x["nav"], P.t0, e)[0]))
    p_ = np.array(p_)
    d1["Q_paths20"] = dict(ann_2019_median=float(np.median(p_[:, 0])), ann_2019_min=float(p_[:, 0].min()),
                           ann_2019_max=float(p_[:, 0].max()), mdd_median=float(np.median(p_[:, 1])))
    cmp["D1_per_code_last_format"] = d1
    # D2：事件層級含最後一季（到最後一天）
    AO, Gf = K.adj_open_matrix(P)
    bq, _ = K.q_plans(P, F, SUE, RS, sched, "Q", excl=excl, ma60=ma60)
    qs = [x for x in sched if x["s0"] in bq]
    out = {}
    for endmode in ("close_last_day", "adjopen_last_day"):
        rws = []
        for n_, ev in enumerate(qs):
            s = ev["s0"]; x0 = s + 1
            el = K.eligible(P, F, SUE, RS, ev, s, excl)
            order, sc_, _ = K.scores(P, F, SUE, RS, ev, el, "perp")
            if n_ + 1 < len(qs):
                r = AO[qs[n_ + 1]["s0"] + 1, order] / AO[x0, order] - 1.0
            else:
                endv = Gf[e, order] if endmode == "close_last_day" else AO[e, order]
                r = endv / AO[x0, order] - 1.0
            n5 = len(order) // 5; al = np.nanmean(r)
            rws.append(dict(exec=P.dstr[x0], t10=np.nanmean(r[:10]) - al, t5=np.nanmean(r[:n5]) - al,
                            ic=pd.Series(sc_).rank().corr(pd.Series(r).rank())))
        g = pd.DataFrame(rws)
        tt = lambda v: float(v.mean() / (v.std(ddof=1) / np.sqrt(len(v))))
        h24 = g[g.exec >= "2024-01-01"]
        out[endmode] = dict(n_q=len(g), top5th=float(g.t5.mean()), top5th_t=tt(g.t5), top10=float(g.t10.mean()),
                            top10_t=tt(g.t10), ic=float(g.ic.mean()), ic_t=tt(g.ic), n_q_2024=len(h24),
                            top5th_2024=float(h24.t5.mean()), top5th_2024_t=tt(h24.t5), top10_2024=float(h24.t10.mean()))
    cmp["D2_event_with_last_quarter"] = out
    js2 = json.dumps(cmp, ensure_ascii=False, indent=1, default=lambda o: (bool(o) if isinstance(o, np.bool_) else
                                                                          (int(o) if isinstance(o, np.integer) else float(o))))
    open(os.path.join(HERE, "compare_vs_researcher.json"), "w").write(js2)
    print(json.dumps(dict(worst=worst, within=cmp["main_within_threshold"],
                          trades={k: (v.get("common"), v["res"], v["mine"]) for k, v in tc.items()},
                          dec=cmp["items"]["decisions"]["all_same"], eight=cmp["items"]["eight"]["all_same"],
                          D1={k: (v if k == "classification_differences_in_universe" else
                                  {kk: vv for kk, vv in v.items()}) for k, v in d1.items()},
                          D2=out), ensure_ascii=False, default=str, indent=0))


if __name__ == "__main__":
    main(sys.argv[1])
