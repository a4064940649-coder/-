# -*- coding: utf-8 -*-
"""S20 獨立複製：驗收（P0、A0、A1、A2、A3）→ 特徵分位數與門檻 → 訓練窗 7 規則＋LOW（20 條路徑）→ 機械凍結
→ 驗證（重新起跑，20 條路徑，T0、V1～V4）→ 全期連續主跑（CUR／FZ／LOW）→ 審查 A（隨機篩掉同比例，20 種子）、
B（F1 內 μ ≤ 0 組對 0050 超額）、C（往下遞補 L_MA 第 11～20 名，事後）。
用法：python3 -I -B rep_s20_run.py <scratch_dir>
需先跑 s1 的 rep_s1_panel.py <scratch_dir> 與 s3 的 rep_s3_names.py <scratch_dir>/innov_names.json。
輸出（本資料夾）：my_numbers.json、my_numbers.sha256、my_trades.csv.gz、my_paths.csv.gz
凍結前只讀 PREREG.json、FROZEN.json、data_check.json、FINDINGS.md 第 1、6 節。
"""
import sys, os, json, hashlib, resource, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import rep_s20_lib as K
S, Q, R, L = K.S, K.Q, K.R, K.L
ST, X2 = K.ST, K.X2
SPEC = os.path.dirname(HERE)

CAPS = [600_000.0, 800_000.0, 1_000_000.0, 1_500_000.0, 2_000_000.0]
SHIFTS = [0, 1, 2, 3]
BEARS = {"2018": ("2018-01-01", "2018-12-31"), "2020-03": ("2020-01-01", "2020-06-30"),
         "2022": ("2022-01-01", "2022-12-31"), "2024-07~2025-04": ("2024-07-01", "2025-04-30")}
VBEARS = ["2022", "2024-07~2025-04"]
BIG = ["2408", "3653", "1325", "2603", "6213", "2059"]


def rss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024


def jdef(o):
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


def qstats(v):
    v = np.asarray(v, float)
    return dict(n=int(len(v)), median=float(np.median(v)), min=float(v.min()), max=float(v.max()), n_pos=int((v > 0).sum()))


def main(sc, odir=HERE):
    t_start = time.time()
    prereg = json.load(open(os.path.join(SPEC, "PREREG.json")))
    frozen = json.load(open(os.path.join(SPEC, "FROZEN.json")))
    P = R.build(sc)
    excl = S.innov_mask_fn(P, sc)
    T = P.T; e = T - 1; t0 = P.t0
    da = np.array(P.dstr)
    codes = np.array(P.codes)
    i19 = L.didx(P, "2018-12-28"); ih = L.didx(P, "2021-02-19")
    itr1 = L.didx(P, "2021-12-30")
    assert P.dstr[i19] == "2018-12-28" and P.dstr[itr1] == "2021-12-30" and P.dstr[ih] == "2021-02-19"
    # 訓練窗起點：任務說明與 FINDINGS 標題都寫 2016-01-30（週六補班交易日，面板有這天）；PREREG 文字寫 2016-01-29。
    # 主用 2016-01-30，2016-01-29 列為敏感度（鍵名 sens_start_20160129）。
    itr0 = L.didx(P, "2016-01-30")
    assert P.dstr[itr0] == "2016-01-30"
    itr0b = L.didx(P, "2016-01-29")
    assert P.dstr[itr0b] == "2016-01-29"

    def win(a, b):
        return int(np.searchsorted(da, a, side="left")), int(np.searchsorted(da, b, side="right")) - 1
    bw = {k: win(a, b) for k, (a, b) in BEARS.items()}

    out = dict(meta=dict(T=T, N=P.N, first=P.dstr[0], last=P.dstr[-1], t0=P.dstr[t0], train_window=(P.dstr[itr0], P.dstr[itr1]),
                         valid_window=(P.dstr[itr1], P.dstr[e]), i19=P.dstr[i19], ih=P.dstr[ih],
                         bear_windows={k: (P.dstr[a], P.dstr[b]) for k, (a, b) in bw.items()},
                         note="面板為複製員自建（N＝2130；研究員 2142，差 12 檔為 91xx，股池本來就排除）。"
                              "0050 停牌日用 S14 複製員修正。"))
    print("meta", out["meta"], flush=True)

    # ================================================================ 特徵、名單
    Rv, mi0 = L.load_rev(P)
    Sstd, F = K.features(Rv, mi0)
    S_l, _ = L.surge(Rv, mi0, null2013=True)
    lists = {}; plans = {}
    for sh in SHIFTS:
        lists[sh] = Q.make_lists(P, Sstd, mi0, shift=sh, excl=excl)
        plans[sh] = Q.cell_plans(lists[sh], "C10", x=0.10)
    ll0 = Q.make_lists(P, S_l, mi0, shift=0, excl=excl)
    pl0 = Q.cell_plans(ll0, "C10", x=0.10)
    a1 = bool(sorted(pl0[0]) == sorted(plans[0][0]) and all(np.array_equal(pl0[0][s], plans[0][0][s]) for s in pl0[0]))
    print("lists", round(time.time() - t_start), "rss", rss(), "A1", a1, flush=True)

    # F1 名單（位移 0）
    rowsF = []
    for s in sorted(plans[0][0]):
        m = lists[0][s]["m"]
        for i in plans[0][0][s]:
            i = int(i)
            rowsF.append(dict(s=s, sig=P.dstr[s], m=m, rev_month="%04d-%02d" % ((mi0 + m) // 12, (mi0 + m) % 12 + 1),
                              i=i, code=codes[i], MS=F["MS"][m, i], MU=F["MU"][m, i], NPOS12=F["NPOS12"][m, i],
                              SURGE=Sstd[m, i]))
    f1 = pd.DataFrame(rowsF)
    f1["period"] = np.where(f1.m + mi0 >= K.VALID_MI, "valid", "train")
    tr_ = f1[f1.period == "train"]; va_ = f1[f1.period == "valid"]
    fc = dict(n_total=int(len(f1)), n_months_total=int(f1.s.nunique()),
              train=dict(n=int(len(tr_)), n_months=int(tr_.s.nunique()), first_sig=tr_.sig.min(), last_sig=tr_.sig.max()),
              valid=dict(n=int(len(va_)), n_months=int(va_.s.nunique()), first_sig=va_.sig.min(), last_sig=va_.sig.max()))

    def fstats(d):
        o = {}
        for f in ["MS", "MU", "NPOS12"]:
            x = d[f].values.astype(float)
            o[f] = dict(coverage=float(np.isfinite(x).mean()), mean=float(np.nanmean(x)),
                        q=[float(np.percentile(x, 100 / 3)), float(np.percentile(x, 50)), float(np.percentile(x, 200 / 3))],
                        min=float(np.nanmin(x)), max=float(np.nanmax(x)))
        o["spearman"] = dict(MS_NPOS12=K17_spear(d.MS, d.NPOS12), MS_MU=K17_spear(d.MS, d.MU),
                             NPOS12_MU=K17_spear(d.NPOS12, d.MU), MS_SURGE=K17_spear(d.MS, d.SURGE))
        o["NPOS12_hist"] = {int(k): int(v) for k, v in d.NPOS12.value_counts().sort_index().items()}
        return o
    feat = dict(f1=fc, train=fstats(tr_), valid=fstats(va_), full_MS_q=fstats(f1)["MS"]["q"])
    qms = feat["train"]["MS"]["q"]; qnp = feat["train"]["NPOS12"]["q"]
    RULES = [dict(id="MS33", feat="MS", op=">=", thr=qms[0]), dict(id="MS50", feat="MS", op=">=", thr=qms[1]),
             dict(id="MS67", feat="MS", op=">=", thr=qms[2]), dict(id="MU", feat="MU", op=">", thr=0.5),
             dict(id="NP6", feat="NPOS12", op=">=", thr=float(round(qnp[0]))),
             dict(id="NP8", feat="NPOS12", op=">=", thr=float(round(qnp[1]))),
             dict(id="NP9", feat="NPOS12", op=">=", thr=float(round(qnp[2])))]
    LOW = dict(id="LOW", feat="MS", op="<", thr=qms[1])
    pr_c = {c["id"]: c["thr"] for c in prereg["rule_form"]["candidates（7 條，全部計入 N；門檻＝訓練期位移 0 F1 名單的分位數，numpy linear，凍結成數字）"]}
    feat["thr_mine"] = {r["id"]: r["thr"] for r in RULES + [LOW]}
    feat["thr_prereg"] = dict(pr_c, LOW=prereg["rule_form"]["placebo（只描述，不參與選擇與判定）"]["thr"])
    feat["thr_max_abs_diff"] = float(max(abs(feat["thr_mine"][k] - feat["thr_prereg"][k]) for k in feat["thr_mine"]))
    # 門檻是事前凍結的「數字」：規則一律用 PREREG 的數字（自算分位數只用來核對）。
    # 注意 MS 的 1/3 分位數剛好等於某一筆資料的值；自算值與 PREREG 差 1e-16，用自算值會讓那一筆落到門檻外（另列敏感度）。
    own = {r["id"]: r["thr"] for r in RULES + [LOW]}
    for r in RULES + [LOW]:
        r["thr"] = float(feat["thr_prereg"][r["id"]])
    ALLR = {r["id"]: r for r in RULES + [LOW]}
    ALLR["MS33own"] = dict(id="MS33own", feat="MS", op=">=", thr=own["MS33"])
    feat["MS33_boundary_entries"] = [dict(code=r_.code, sig=r_.sig, MS=float(r_.MS), MU=float(r_.MU))
                                     for r_ in tr_.itertuples() if abs(r_.MS - own["MS33"]) < 1e-12]

    def kf(d, r):
        return float(np.mean([K.rule_pass(F, r, int(m), int(i)) for m, i in zip(d.m, d.i)]))
    feat["keep_frac"] = {k: dict(train=kf(tr_, r), valid=kf(va_, r), full=kf(f1, r)) for k, r in ALLR.items()}
    out["features"] = feat
    print("features", json.dumps(feat, default=jdef, ensure_ascii=False)[:3000], flush=True)

    # ================================================================ 引擎
    K.attach_e50_fixed(P, sc)
    e50 = P.e50.values.astype(float)

    def sim(pl, cap=1_000_000.0, cost=None, block_span=None):
        b, k = pl
        return K.simulate(P, b, k, sat_w=0.7, sizing="W10", capital=cap, block_span=block_span, **(cost or ST))

    def bears(nav):
        return {k: L.maxdd(P, nav, a, b)[0] for k, (a, b) in bw.items()}

    def nav_stats(nav):
        md = L.maxdd(P, nav, t0, e)
        return dict(ann_2019=L.ann(P, nav, i19, e), ann_full=L.ann(P, nav, t0, e), mdd_full=md[0], mdd_peak=md[1],
                    mdd_trough=md[2], ann_h1=L.ann(P, nav, t0, ih), ann_h2=L.ann(P, nav, ih, e),
                    ann_train=L.ann(P, nav, itr0, itr1), ann_valid=L.ann(P, nav, itr1, e),
                    mdd_train=L.maxdd(P, nav, itr0, itr1)[0], mdd_valid=L.maxdd(P, nav, itr1, e)[0],
                    ann_1618=L.ann(P, nav, itr0, i19), ann_1921=L.ann(P, nav, i19, itr1),
                    ann_train_b=L.ann(P, nav, itr0b, itr1), ann_1618b=L.ann(P, nav, itr0b, i19))

    # A0
    cur0 = sim(plans[0])
    a0 = dict(ann_2019=L.ann(P, cur0["nav"], i19, e), ann_full=L.ann(P, cur0["nav"], t0, e),
              mdd_full=L.maxdd(P, cur0["nav"], t0, e)[0], n_trades=int(len(cur0["trades"])))
    a0["ok"] = bool(abs(a0["ann_2019"] - 0.4813) < 0.00025 and abs(a0["ann_full"] - 0.3408) < 0.00025 and
                    abs(a0["mdd_full"] + 0.3340) < 0.00025)
    print("A0", a0, flush=True)
    assert a0["ok"]
    # A2：全部通過
    allpass = dict(id="ALL", feat="MS", op=">=", thr=-np.inf)
    x = sim(K.filt_plans(lists[0], plans[0], F, allpass))
    a2 = bool(np.array_equal(x["nav"][t0:], cur0["nav"][t0:]) and len(x["trades"]) == len(cur0["trades"]) and
              (x["trades"][["code", "tb", "ts", "sh"]].values == cur0["trades"][["code", "tb", "ts", "sh"]].values).all())
    out["acceptance"] = dict(P0=dict(T=T, N=P.N, last=P.dstr[-1], ok=bool(T == 2876 and P.dstr[-1] == "2026-10-22")),
                             A0=a0, A1_same_as_rep_s1_surge=a1, A2_allpass_identical=a2)
    print("A2", a2, flush=True)

    # ================================================================ 連續模擬：20 條路徑 × (CUR＋7＋LOW)
    pnav = {}; prow = []; mains = {}
    ids = ["CUR"] + [r["id"] for r in RULES] + ["LOW", "MS33own"]
    for rid in ids:
        rule = None if rid == "CUR" else ALLR[rid]
        for sh in SHIFTS:
            pl = K.filt_plans(lists[sh], plans[sh], F, rule)
            for cp in CAPS:
                x = cur0 if (rid == "CUR" and sh == 0 and cp == 1e6) else sim(pl, cp)
                nav = x["nav"]
                pnav[("cont", rid, sh, cp)] = nav
                if sh == 0 and cp == 1e6:
                    mains[("cont", rid)] = x
                st = nav_stats(nav)
                prow.append(dict(run="cont", rule=rid, shift=sh, capital=cp, n_trades=int(len(x["trades"])),
                                 **{k: v for k, v in st.items() if k not in ("mdd_peak", "mdd_trough")},
                                 **{"bear_" + k: v for k, v in bears(nav).items()}))
        print("cont", rid, round(time.time() - t_start), "rss", rss(), flush=True)
    pr = pd.DataFrame(prow)

    def paired(run_a, a, run_b, b, col):
        xa = pr[(pr.run == run_a) & (pr.rule == a)].set_index(["shift", "capital"])
        xb = pr[(pr.run == run_b) & (pr.rule == b)].set_index(["shift", "capital"]).loc[xa.index]
        return (xa[col] - xb[col]).values

    # ---------------------------------------------------------------- 訓練窗與機械選法
    trn = {}
    for rid in [r["id"] for r in RULES] + ["LOW", "MS33own"]:
        d = paired("cont", rid, "cont", "CUR", "ann_train")
        g = pr[(pr.run == "cont") & (pr.rule == rid)]
        trn[rid] = dict(train_median=float(g.ann_train.median()), train_min=float(g.ann_train.min()),
                        train_max=float(g.ann_train.max()), train_dd_median=float(g.mdd_train.median()),
                        score_median=float(np.median(d)), npos=int((d > 0).sum()), diff_min=float(d.min()),
                        diff_max=float(d.max()),
                        dd_diff_median=float(np.median(paired("cont", rid, "cont", "CUR", "mdd_train"))),
                        y1618_median=float(np.median(paired("cont", rid, "cont", "CUR", "ann_1618"))),
                        y1618_npos=int((paired("cont", rid, "cont", "CUR", "ann_1618") > 0).sum()),
                        y1921_median=float(np.median(paired("cont", rid, "cont", "CUR", "ann_1921"))),
                        y1921_npos=int((paired("cont", rid, "cont", "CUR", "ann_1921") > 0).sum()),
                        keep_frac_train=feat["keep_frac"][rid]["train"],
                        sens_start_20160129=dict(train_median=float(g.ann_train_b.median()), train_min=float(g.ann_train_b.min()),
                                                 train_max=float(g.ann_train_b.max()),
                                                 score_median=float(np.median(paired("cont", rid, "cont", "CUR", "ann_train_b"))),
                                                 y1618_median=float(np.median(paired("cont", rid, "cont", "CUR", "ann_1618b")))))
    g = pr[(pr.run == "cont") & (pr.rule == "CUR")]
    trn["CUR"] = dict(train_median=float(g.ann_train.median()), train_min=float(g.ann_train.min()),
                      train_max=float(g.ann_train.max()), train_dd_median=float(g.mdd_train.median()),
                      sens_start_20160129=dict(train_median=float(g.ann_train_b.median())))
    out["meta"]["train_start_sens"] = P.dstr[itr0b]
    cand = sorted([r["id"] for r in RULES], key=lambda k: (-trn[k]["score_median"], -trn[k]["npos"], -trn[k]["keep_frac_train"]))
    sel = cand[0]
    cand_b = sorted([r["id"] for r in RULES], key=lambda k: (-trn[k]["sens_start_20160129"]["score_median"], -trn[k]["npos"],
                                                             -trn[k]["keep_frac_train"]))
    out["meta"]["train_ranking_sens_20160129"] = cand_b
    T0 = bool(trn[sel]["score_median"] >= 0.01 and trn[sel]["npos"] >= 14)
    out["train"] = dict(rules=trn, ranking=cand, selected=sel, selected_rule=ALLR[sel], T0_pass=T0,
                        same_as_FROZEN=bool(sel == frozen["frozen_rule"]["id"] and T0 == frozen["T0_pass"]),
                        frozen_file_rule=frozen["frozen_rule"]["id"])
    print("train", json.dumps(out["train"], default=jdef, ensure_ascii=False), flush=True)
    FZ = ALLR[sel]

    # ---------------------------------------------------------------- 驗證：重新起跑
    fr_cur = sim(K.fresh_plans(lists[0], plans[0], mi0))
    ratio = float(fr_cur["nav"][itr1] / 1e6)
    a3 = []
    vids = {"CUR": None, "FZ": FZ, "LOW": LOW}
    bf_info = {}
    fpl = {}
    for sh in SHIFTS:
        for rid, rule in vids.items():
            fpl[(rid, sh)] = K.fresh_plans(lists[sh], K.filt_plans(lists[sh], plans[sh], F, rule), mi0)
        (bfp, info) = K.backfill_plans(lists[sh], plans[sh], F, FZ)
        bf_info[sh] = info
        fpl[("BF", sh)] = K.fresh_plans(lists[sh], bfp, mi0)
        fpl[("BFc", sh)] = bfp
    navat = []
    for rid in ["CUR", "FZ", "LOW", "BF"]:
        for sh in SHIFTS:
            for cp in CAPS:
                x = sim(fpl[(rid, sh)], cp / ratio)
                nav = x["nav"]
                if rid == "CUR":
                    a3.append(int((x["trades"].tb <= itr1).sum()) + int(sum(1 for o in x["open_pos"] if o[1] <= P.dstr[itr1])))
                    navat.append(float(nav[itr1] / cp))
                if sh == 0 and cp == 1e6:
                    mains[("fresh", rid)] = x
                st = nav_stats(nav)
                prow.append(dict(run="fresh", rule=rid, shift=sh, capital=cp, n_trades=int(len(x["trades"])),
                                 **{k: v for k, v in st.items() if k not in ("mdd_peak", "mdd_trough")},
                                 **{"bear_" + k: v for k, v in bears(nav).items()}))
                pnav[("fresh", rid, sh, cp)] = nav
        print("fresh", rid, round(time.time() - t_start), "rss", rss(), flush=True)
    # 審查 C 連續
    for sh in SHIFTS:
        for cp in CAPS:
            x = sim(fpl[("BFc", sh)], cp)
            nav = x["nav"]
            if sh == 0 and cp == 1e6:
                mains[("cont", "BF")] = x
            st = nav_stats(nav)
            prow.append(dict(run="cont", rule="BF", shift=sh, capital=cp, n_trades=int(len(x["trades"])),
                             **{k: v for k, v in st.items() if k not in ("mdd_peak", "mdd_trough")},
                             **{"bear_" + k: v for k, v in bears(nav).items()}))
    pr = pd.DataFrame(prow)
    out["acceptance"]["A3_fresh_no_stock_trade_before_2021-12-30"] = bool(max(a3) == 0)
    fr_tr = mains[("fresh", "CUR")]["trades"]
    out["acceptance"]["A3_first_fresh_buy"] = str(fr_tr.buy_date.min())
    out["acceptance"]["fresh_ratio"] = ratio
    out["acceptance"]["fresh_nav_at_2021-12-30_over_cap"] = dict(min=min(navat), max=max(navat))

    # V1～V4
    dv = paired("fresh", "FZ", "fresh", "CUR", "ann_valid")
    V1 = int((dv > 0).sum()) >= 15
    V2 = float(np.median(dv)) >= 0.02
    spF, topF = R.span_of(mains[("fresh", "FZ")]["trades"], 5)
    spC, topC = R.span_of(mains[("fresh", "CUR")]["trades"], 5)
    spU = {}
    for sp in (spF, spC):
        for c, v in sp.items():
            for ab in v:
                if ab not in spU.setdefault(c, []):
                    spU[c].append(ab)
    v3d = []; v3rows = []
    for sh in SHIFTS:
        for cp in CAPS:
            xa = sim(fpl[("FZ", sh)], cp / ratio, block_span=spU)
            xb = sim(fpl[("CUR", sh)], cp / ratio, block_span=spU)
            a_ = L.ann(P, xa["nav"], itr1, e); b_ = L.ann(P, xb["nav"], itr1, e)
            v3d.append(a_ - b_); v3rows.append((sh, cp, a_, b_))
    V3 = float(np.median(v3d)) > 0
    v4 = {}
    for k in VBEARS:
        d = paired("fresh", "FZ", "fresh", "CUR", "bear_" + k)
        v4[k] = dict(median_diff=float(np.median(d)), n_shallower=int((d > 0).sum()),
                     fz_median=float(pr[(pr.run == "fresh") & (pr.rule == "FZ")]["bear_" + k].median()),
                     cur_median=float(pr[(pr.run == "fresh") & (pr.rule == "CUR")]["bear_" + k].median()))
    V4 = all(v["median_diff"] >= -0.02 for v in v4.values())
    gF = pr[(pr.run == "fresh")]
    val = dict(T0=T0, V1=dict(ok=bool(V1), **qstats(dv)), V2=dict(ok=bool(V2), median=float(np.median(dv))),
               V3=dict(ok=bool(V3), block_union={c: v for c, v in spU.items()}, top5_FZ=topF, top5_CUR=topC,
                       **qstats(np.array(v3d)), fz_median=float(np.median([r_[2] for r_ in v3rows])),
                       cur_median=float(np.median([r_[3] for r_ in v3rows]))),
               V4=dict(ok=bool(V4), **v4),
               passed=bool(T0 and V1 and V2 and V3 and V4),
               path_medians={rid: dict(ann_valid=float(gF[gF.rule == rid].ann_valid.median()),
                                       ann_valid_min=float(gF[gF.rule == rid].ann_valid.min()),
                                       ann_valid_max=float(gF[gF.rule == rid].ann_valid.max()),
                                       mdd_valid=float(gF[gF.rule == rid].mdd_valid.median()),
                                       n_trades=float(gF[gF.rule == rid].n_trades.median()))
                             for rid in ["CUR", "FZ", "LOW", "BF"]},
               LOW_minus_CUR=qstats(paired("fresh", "LOW", "fresh", "CUR", "ann_valid")),
               BF_minus_CUR=qstats(paired("fresh", "BF", "fresh", "CUR", "ann_valid")),
               BF_minus_FZ=qstats(paired("fresh", "BF", "fresh", "FZ", "ann_valid")),
               cont_valid_FZ_minus_CUR=qstats(paired("cont", "MS33" if sel == "MS33" else sel, "cont", "CUR", "ann_valid")),
               cont_valid_LOW_minus_CUR=qstats(paired("cont", "LOW", "cont", "CUR", "ann_valid")))
    out["validation"] = val
    print("validation", json.dumps(val, default=jdef, ensure_ascii=False)[:4000], flush=True)

    # ================================================================ 主跑描述
    b50 = dict(ann_2019=L.ann(P, e50, i19, e), ann_full=L.ann(P, e50, t0, e), mdd_full=L.maxdd(P, e50, t0, e)[0],
               ann_h1=L.ann(P, e50, t0, ih), ann_h2=L.ann(P, e50, ih, e), ann_valid=L.ann(P, e50, itr1, e),
               ann_train=L.ann(P, e50, itr0, itr1), mdd_valid=L.maxdd(P, e50, itr1, e)[0],
               bears=bears(e50), yearly=L.yearly(P, e50))
    out["bench_0050"] = b50
    mres = {}; trows = []
    for run in ["cont", "fresh"]:
        for rid in ["CUR", "FZ", "LOW", "BF"]:
            key = (run, sel if (run == "cont" and rid == "FZ") else rid)
            x = mains[key]
            nav = x["nav"]; tr = x["trades"]
            o = nav_stats(nav)
            o["bears"] = bears(nav); o["yearly"] = L.yearly(P, nav)
            o["n_trades"] = int(len(tr)); o["n_open"] = len(x["open_pos"])
            o["hold_mean"] = float(tr.hold.mean()); o["win"] = float((tr.ret > 0).mean())
            o["npos_mean_valid"] = float(np.mean(x["npos"][itr1 + 1:])); o["npos_mean_full"] = float(np.mean(x["npos"][t0 + 1:]))
            cpm = 1e6 if run == "cont" else 1e6 / ratio
            plm = (fpl[("BFc", 0)] if rid == "BF" else K.filt_plans(lists[0], plans[0], F, None if rid == "CUR" else (FZ if rid == "FZ" else LOW))) \
                if run == "cont" else fpl[(rid, 0)]
            sp, top = R.span_of(tr, 5)
            x2 = sim(plm, cpm, cost=X2); bl = sim(plm, cpm, block_span=sp)
            if run == "cont":
                o["x2"] = dict(ann_2019=L.ann(P, x2["nav"], i19, e), ann_full=L.ann(P, x2["nav"], t0, e))
                o["block5"] = dict(ann_2019=L.ann(P, bl["nav"], i19, e), ann_full=L.ann(P, bl["nav"], t0, e))
                ck = dict(a2019=o["ann_2019"] > b50["ann_2019"], afull=o["ann_full"] > b50["ann_full"],
                          x2_2019=o["x2"]["ann_2019"] > b50["ann_2019"], x2_full=o["x2"]["ann_full"] > b50["ann_full"],
                          bl_2019=o["block5"]["ann_2019"] > b50["ann_2019"], bl_full=o["block5"]["ann_full"] > b50["ann_full"],
                          h1=o["ann_h1"] > b50["ann_h1"], h2=o["ann_h2"] > b50["ann_h2"])
                o["checks"] = ck; o["n_checks_pass"] = int(sum(ck.values()))
            else:
                o["x2"] = dict(ann_valid=L.ann(P, x2["nav"], itr1, e))
                o["block5"] = dict(ann_valid=L.ann(P, bl["nav"], itr1, e))
                o["checks_valid"] = dict(ann=o["ann_valid"] > b50["ann_valid"], x2=o["x2"]["ann_valid"] > b50["ann_valid"],
                                         bl=o["block5"]["ann_valid"] > b50["ann_valid"])
            o["top5"] = top
            bwr = {}
            for c in BIG:
                g_ = tr[tr.code == c]
                bwr[c] = [dict(buy=r_.buy_date, sell=r_.sell_date, ret=float(r_.ret), pnl=float(r_.pnl)) for r_ in g_.itertuples()]
                bwr[c] += [dict(buy=bd, sell=None, open=True, cost=cst, val=val_) for cc_, bd, cst, val_, i_ in x["open_pos"] if cc_ == c]
            o["big_winners"] = bwr
            mres["%s|%s" % (run, rid)] = o
            for r_ in tr.itertuples():
                trows.append((run, rid, r_.code, r_.buy_date, r_.sell_date, int(r_.sh), round(r_.ret, 6), round(r_.pnl, 1), int(r_.hold)))
            for cc_, bd, cst, val_, i_ in x["open_pos"]:
                trows.append((run, rid, cc_, bd, "OPEN", -1, round(val_ / cst - 1, 6), round(val_ - cst, 1), -1))
            print("main", run, rid, round(o["ann_2019"], 4), round(o["ann_full"], 4), round(o["ann_valid"], 4),
                  round(o["mdd_full"], 4), o["n_trades"], flush=True)
    out["main"] = mres
    pd.DataFrame(trows, columns=["run", "rule", "code", "buy_date", "sell_date", "sh", "ret", "pnl", "hold"]).to_csv(
        os.path.join(odir, "my_trades.csv.gz"), index=False, compression="gzip")
    # 全期連續 20 條路徑中位數
    pc = {}
    for rid in ["CUR", sel, "LOW", "BF"]:
        g_ = pr[(pr.run == "cont") & (pr.rule == rid)]
        pc[rid] = dict(ann_2019=float(g_.ann_2019.median()), ann_full=float(g_.ann_full.median()),
                       mdd_full=float(g_.mdd_full.median()), n_trades=float(g_.n_trades.median()))
        if rid != "CUR":
            pc[rid]["paired_2019"] = qstats(paired("cont", rid, "cont", "CUR", "ann_2019"))
            pc[rid]["paired_full"] = qstats(paired("cont", rid, "cont", "CUR", "ann_full"))
            pc[rid]["paired_mdd"] = qstats(paired("cont", rid, "cont", "CUR", "mdd_full"))
            pc[rid]["paired_train"] = qstats(paired("cont", rid, "cont", "CUR", "ann_train"))
    out["cont_paths"] = pc
    pr.to_csv(os.path.join(odir, "my_paths.csv.gz"), index=False, compression="gzip", float_format="%.6f")

    # ================================================================ 審查 A：隨機篩掉同樣比例（全期連續主跑）
    ent = K.f1_entries(lists[0], plans[0])
    nE = len(ent)
    rA = dict(n_entries=nE)
    stat_of = {}
    for rid in ["CUR", sel, "LOW"]:
        n_ = mains[("cont", rid)]["nav"]
        stat_of[rid] = dict(ann_2019=L.ann(P, n_, i19, e), ann_full=L.ann(P, n_, t0, e), mdd_full=L.maxdd(P, n_, t0, e)[0],
                            ann_train=L.ann(P, n_, itr0, itr1), ann_valid=L.ann(P, n_, itr1, e))
    for lab, rid in [("keep_FZ", sel), ("keep_LOW", "LOW")]:
        p = feat["keep_frac"][rid]["full"]
        nk = int(round(p * nE))
        res = []
        for k in range(20):
            rng = np.random.default_rng(20_000 + k)
            pick = rng.choice(nE, nk, replace=False)
            kp = set(ent[j] for j in pick)
            x = sim(K.random_keep_plans(lists[0], plans[0], kp))
            n_ = x["nav"]
            res.append(dict(seed=20_000 + k, ann_2019=L.ann(P, n_, i19, e), ann_full=L.ann(P, n_, t0, e),
                            mdd_full=L.maxdd(P, n_, t0, e)[0], ann_train=L.ann(P, n_, itr0, itr1),
                            ann_valid=L.ann(P, n_, itr1, e), n_trades=int(len(x["trades"]))))
        df = pd.DataFrame(res)
        tgt = stat_of[rid]
        o = dict(keep_frac=p, n_keep=nk, seeds=res, rule_stats=tgt, cur_stats=stat_of["CUR"])
        for col in ["ann_2019", "ann_full", "mdd_full", "ann_train", "ann_valid"]:
            v = df[col].values
            o[col] = dict(min=float(v.min()), p10=float(np.percentile(v, 10)), median=float(np.median(v)),
                          p90=float(np.percentile(v, 90)), max=float(v.max()), rule=tgt[col],
                          rank_rule_from_top=int(1 + (v > tgt[col]).sum()),
                          inside_range=bool(v.min() <= tgt[col] <= v.max()), cur=stat_of["CUR"][col])
        rA[lab] = o
        print("reviewA", lab, json.dumps({c: o[c] for c in ["ann_2019", "ann_full", "mdd_full"]}, default=jdef), flush=True)
    out["reviewA"] = rA

    # ================================================================ 審查 B：F1 內 μ ≤ 0 組
    top20 = {s: set(int(i) for i in lists[0][s]["lma"][:20]) for s in lists[0]}
    rows_sorted = sorted(lists[0])
    exc = []
    for r_ in f1.itertuples():
        o_ = K.outcome_e0(P, rows_sorted, top20, int(r_.s), int(r_.i), e)
        o2 = K.outcome_e0(P, rows_sorted, top20, int(r_.s), int(r_.i), itr1) if r_.period == "train" else None
        exc.append((np.nan if o_ is None else o_["exc"], np.nan if o2 is None else o2["exc"],
                    np.nan if o_ is None else o_["hold"], None if o_ is None else o_["kind"]))
    f1["exc"] = [a[0] for a in exc]; f1["exc_tc_train"] = [a[1] for a in exc]; f1["hold_e0"] = [a[2] for a in exc]
    f1["kind"] = [a[3] for a in exc]

    def grp(d, col="exc"):
        d = d[np.isfinite(d[col])]
        if len(d) == 0:
            return None
        mm = d.groupby("s")[col].mean()
        sd = float(mm.std(ddof=1)) if len(mm) > 1 else np.nan
        x = np.sort(d[col].values)[::-1]
        return dict(n=int(len(d)), n_months=int(len(mm)), mean=float(d[col].mean()), median=float(d[col].median()),
                    month_mean=float(mm.mean()), month_t=float(mm.mean() / (sd / np.sqrt(len(mm)))) if len(mm) > 1 else None,
                    mean_drop_top5=float(x[5:].mean()) if len(x) > 5 else None,
                    top5=[(c, sg, float(v)) for c, sg, v in d.sort_values(col, ascending=False).head(5)[["code", "sig", col]].values],
                    win=float((d[col] > 0).mean()))

    def dmonth(d, grpmask, col="exc"):
        d = d[np.isfinite(d[col])]
        allm = d.groupby("s")[col].mean()
        gm = d[grpmask.reindex(d.index)].groupby("s")[col].mean()
        dd = gm - allm.reindex(gm.index)
        return dict(n_months=int(len(dd)), mean=float(dd.mean()), t=float(dd.mean() / (dd.std(ddof=1) / np.sqrt(len(dd)))))
    rB = {}
    for per, d in [("train", f1[f1.period == "train"]), ("valid", f1[f1.period == "valid"]), ("full", f1)]:
        mneg = d.MU == 0
        rB[per] = dict(mu_le0=grp(d[mneg]), mu_gt0=grp(d[~mneg]), all=grp(d),
                       ms_lt_thr=grp(d[~d.apply(lambda r: K.rule_pass(F, FZ, int(r.m), int(r.i)), axis=1)]),
                       mu_le0_minus_month_all=dmonth(d, mneg))
        if per == "train":
            rB[per]["mu_le0_tc_2021-12-30"] = grp(d[mneg], "exc_tc_train")
            rB[per]["mu_gt0_tc_2021-12-30"] = grp(d[~mneg], "exc_tc_train")
    rB["note"] = "E0 持有期（rep_s13 outcome_e0：訊號列次日開盤進場、掉出 L_MA 前 20 名後次一可賣開盤出場、下市以最後收盤），對 0050 同期（開盤到開盤）超額，不扣成本；估值截止＝2026-10-22（訓練另列截止 2021-12-30 版）。月份群聚 t＝組內每月平均超額的平均÷(標準差÷√月數)。"
    out["reviewB"] = rB
    print("reviewB", json.dumps({k: v for k, v in rB.items() if k != "note"}, default=jdef, ensure_ascii=False)[:3000], flush=True)

    # 大贏家在 F1 名單中的 μ÷σ
    out["big_winner_features"] = {c: [dict(sig=r_.sig, rev_month=r_.rev_month, MS=float(r_.MS), MU=float(r_.MU),
                                           NPOS12=float(r_.NPOS12), kept_by_FZ=K.rule_pass(F, FZ, int(r_.m), int(r_.i)))
                                      for r_ in f1[f1.code == c].itertuples()] for c in BIG}

    # ================================================================ 審查 C
    rC = dict(fill_info={str(k): v for k, v in bf_info.items()},
              cont_main=dict(BF=stat_of_from(P, mains[("cont", "BF")]["nav"], L, i19, t0, e, itr0, itr1),
                             CUR=stat_of["CUR"], FZ=stat_of[sel]),
              cont_paths=dict(BF_minus_CUR_2019=qstats(paired("cont", "BF", "cont", "CUR", "ann_2019")),
                              BF_minus_CUR_full=qstats(paired("cont", "BF", "cont", "CUR", "ann_full")),
                              BF_minus_CUR_train=qstats(paired("cont", "BF", "cont", "CUR", "ann_train")),
                              BF_minus_CUR_mdd=qstats(paired("cont", "BF", "cont", "CUR", "mdd_full")),
                              BF_minus_FZ_2019=qstats(paired("cont", "BF", "cont", sel, "ann_2019")),
                              BF_minus_FZ_train=qstats(paired("cont", "BF", "cont", sel, "ann_train"))),
              fresh_paths=dict(BF_minus_CUR_valid=val["BF_minus_CUR"], BF_minus_FZ_valid=val["BF_minus_FZ"],
                               BF_bears={k: qstats(paired("fresh", "BF", "fresh", "CUR", "bear_" + k)) for k in VBEARS}))
    out["reviewC"] = rC
    print("reviewC", json.dumps(rC, default=jdef, ensure_ascii=False)[:3000], flush=True)

    out["resources"] = dict(runtime_s=round(time.time() - t_start, 1), maxrss_mb=rss())
    js = json.dumps(out, ensure_ascii=False, indent=1, default=jdef)
    fp = os.path.join(odir, "my_numbers.json")
    open(fp, "w").write(js)
    h = hashlib.sha256(js.encode("utf-8")).hexdigest()
    open(os.path.join(odir, "my_numbers.sha256"), "w").write(h + "  my_numbers.json\n")
    print("done", h, out["resources"], flush=True)


def stat_of_from(P, n_, L, i19, t0, e, itr0, itr1):
    return dict(ann_2019=L.ann(P, n_, i19, e), ann_full=L.ann(P, n_, t0, e), mdd_full=L.maxdd(P, n_, t0, e)[0],
                ann_train=L.ann(P, n_, itr0, itr1), ann_valid=L.ann(P, n_, itr1, e))


def K17_spear(a, b):
    return K.K17.spearman(np.asarray(a, float), np.asarray(b, float))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else HERE)
