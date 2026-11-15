# -*- coding: utf-8 -*-
"""S13 獨立複製：候選、特徵、E0 結果變數、訓練期單變量與 18 條規則、機械選法凍結、驗證期 V1～V4、F1 子集合、
審查 A（大輸家 AUC、RET60 前 1/5 的超額離散）、B（V3 的 Newey-West／區塊自助法 t）、C（固定持有一個月的結果變數）。
用法：python3 -I -B rep_s13_run.py <scratch_dir>
需先跑 s1 的 rep_s1_panel.py <scratch_dir> 與 s3 的 rep_s3_names.py <scratch_dir>/innov_names.json。
輸出（本資料夾）：my_numbers.json、my_numbers.sha256、my_cands.csv.gz
凍結前不讀研究員的 FINDINGS 第 1 節以外、results*.json、cand_*.csv.gz。
"""
import sys, os, json, hashlib, resource, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import rep_s13_lib as K
S, Q, R, L = K.S, K.Q, K.R, K.L
FEATS = K.FEATS
QL = [0.1, 1 / 3, 0.5, 2 / 3, 0.9]   # data_check 的分位數水準（由其數值反推：10%、1/3、1/2、2/3、90%）


def rss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024


def fl(x):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else float(x)


def univariate(d, bw_col="bw", exc_col="exc"):
    out = {}
    for f in FEATS:
        dd = d[np.isfinite(d[exc_col])]
        a = K.auc(dd[bw_col].values, dd[f].values)
        out[f] = dict(auc=a, dir=1 if a >= 0.5 else -1, auc_dir=max(a, 1 - a), n=int(np.isfinite(dd[f]).sum()),
                      quint=K.quintiles(dd, f, bw_col, exc_col))
    return out


def rules(d, uni, exc_col="exc"):
    dd = d[np.isfinite(d[exc_col])]
    out = []
    for f in FEATS:
        dr = uni[f]["dir"]
        v = dd[f].values; v = v[np.isfinite(v)]
        qs = [1 / 3, 1 / 2, 2 / 3] if dr > 0 else [2 / 3, 1 / 2, 1 / 3]
        lab = ["2/3", "1/2", "1/3"]
        for q, lb in zip(qs, lab):
            thr = float(np.quantile(v, q))
            k = K.rule_keep(dd, f, dr, thr)
            dm = K.month_d(dd, k, exc_col)
            out.append(dict(feature=f, dir=dr, q=q, keep_approx=lb, thr=thr, keep_frac=float(k.mean()),
                            n_keep=int(k.sum()), n_months=int(len(dm)), d_mean=float(dm.mean()), t=K.tstat(dm.values),
                            auc_dir=uni[f]["auc_dir"], bw_share_keep=float(dd.loc[k, "bw"].mean()) if "bw" in dd else None,
                            mean_keep=float(dd.loc[k, exc_col].mean()), mean_all=float(dd[exc_col].mean())))
    return out


def select(rl, uni):
    elig = [r for r in rl if uni[r["feature"]]["auc_dir"] >= 0.55]
    weak = len(elig) == 0
    if weak:
        elig = rl
    best = sorted(elig, key=lambda r: (-r["t"], -r["auc_dir"], r["keep_frac"]))[0]
    return best, weak, sorted(set(r["feature"] for r in elig))


def main(sc, odir=HERE):
    t_start = time.time()
    P = R.build(sc)
    S.attach_e50(P, sc)
    excl = S.innov_mask_fn(P, sc)
    Rv, mi0 = L.load_rev(P)
    S_, _ = L.surge(Rv, mi0, null2013=True)
    out = dict(meta=dict(T=P.T, N=P.N, first=P.dstr[0], last=P.dstr[-1],
                         note="面板為複製員自建（N＝2130，研究員 2142；S6～S12 複製時已知，差的是從未進股池的代號）"))

    lists = Q.make_lists(P, S_, mi0, shift=0, excl=excl)
    f1buy, keep20 = Q.cell_plans(lists, "C10", x=0.10)
    rows = sorted(lists)
    top20 = {s: set(int(i) for i in lists[s]["lma"][:20]) for s in rows}
    cand = K.candidates(P, lists, f1buy)
    tc_train = L.didx(P, "2021-12-31") - 1
    assert P.dstr[tc_train] == "2021-12-30", P.dstr[tc_train]
    tc_valid = P.T - 1
    print("cands", len(cand), "rss", rss(), flush=True)

    # ---------------- 特徵
    F = K.load_gm(P)
    feat = K.features(P, cand, S_, Rv, F, variants=True)
    cand = pd.concat([cand, feat], axis=1)

    # ---------------- 結果變數（E0）
    res = []
    for r in cand.itertuples():
        tc = tc_train if r.period == "train" else tc_valid
        o = K.outcome_e0(P, rows, top20, r.s, r.i, tc)
        res.append(o if o is not None else dict(e=r.s + 1, x=-1, kind="no_open", ret=np.nan, r50=np.nan, exc=np.nan,
                                                hold=-1, lim_up=False, mark=-1))
    oc = pd.DataFrame(res, index=cand.index)
    cand = pd.concat([cand, oc], axis=1)
    cand["entry"] = [P.dstr[e] for e in cand.e]
    cand["exit"] = [P.dstr[x] if x >= 0 else "" for x in cand.x]
    print("outcomes", round(time.time() - t_start), "rss", rss(), flush=True)

    # ---------------- 審查 C 結果變數（固定持有到下一個換股日次日開盤）
    AO, Gf = K11adj(P)
    nxt = {s: (rows[n + 1] if n + 1 < len(rows) else None) for n, s in enumerate(rows)}
    rc = []
    for r in cand.itertuples():
        tc = tc_train if r.period == "train" else tc_valid
        o = K.outcome_fixed(P, AO, Gf, r.s, nxt[r.s], r.i, tc)
        rc.append(dict(excC=np.nan, kindC="no_open") if o is None else dict(excC=o["exc"], kindC=o["kind"]))
    cand = pd.concat([cand, pd.DataFrame(rc, index=cand.index)], axis=1)
    del AO, Gf

    tr = cand[cand.period == "train"].copy(); va = cand[cand.period == "valid"].copy()

    # ---------------- 資料檢查重現
    dc = {}
    for nm, d in (("train", tr), ("valid", va)):
        dc[nm] = dict(n_months=int(d.s.nunique()), n_cand=int(len(d)), n_per_month_min=int(d.groupby("s").size().min()),
                      n_f1=int(d.f1.sum()), sig_from=d.sig.min(), sig_to=d.sig.max(),
                      coverage={f: float(np.isfinite(d[f]).mean()) for f in FEATS},
                      quantiles={f: [float(x) for x in np.nanquantile(d[f].values, QL)] for f in FEATS},
                      variants=dict(
                          coverage={v: float(np.isfinite(d[v]).mean()) for v in
                                    ["NEAR52_G", "NEAR52_Hbars", "GMYOY_fb", "ADVCHG_inbar"]},
                          quantiles={v: [float(x) for x in np.nanquantile(d[v].values, QL)] for v in
                                     ["NEAR52_G", "NEAR52_Hbars", "GMYOY_fb", "ADVCHG_inbar"]}))
    sp = cand[FEATS].rank().corr()
    dc["spearman_all"] = {a: {b: round(float(sp.loc[a, b]), 3) for b in FEATS} for a in FEATS}
    dc["corr_with_surge_rank"] = {f: round(float(-cand[[f, "rank"]].rank().corr().iloc[0, 1]), 3) for f in FEATS}
    cand["surge"] = [S_[m, i] for m, i in zip(cand.m, cand.i)]
    dc["corr_with_surge_value"] = {f: round(float(cand[[f, "surge"]].rank().corr().iloc[0, 1]), 3) for f in FEATS}
    dc["gm_quarter_examples"] = [[r.sig, r.code, r.gm_q] for r in cand.groupby("s").head(1).head(8).itertuples()]
    dc["n_months"] = int(cand.s.nunique()); dc["first_sig"] = cand.sig.min(); dc["last_sig"] = cand.sig.max()
    out["data_check_repro"] = dc
    print("datacheck", json.dumps({k: dc[k]["coverage"] for k in ("train", "valid")}), flush=True)

    # ---------------- 驗收 A1（F1 ⊂ 候選，且等於 S10 E0 的 C10 買進名單）、A2（出場日對實際交易）
    a1_sub = bool(all(set(int(i) for i in f1buy[s]) <= top20[s] for s in rows))
    eod = K.K10.make_eod(P, keep20, f1buy, R.ma(P, 60))
    sim = K.K10.simulate_s10(P, f1buy, keep20, sat_w=0.7, sizing="W10", eod=eod, **K.ST)
    nav = sim["nav"]; e = P.T - 1; i19 = L.didx(P, "2018-12-28")
    trd = sim["trades"]
    a0 = dict(ann_2019=L.ann(P, nav, i19, e), ann_full=L.ann(P, nav, P.t0, e), mdd=L.maxdd(P, nav, P.t0, e)[0],
              n_trades=int(len(trd)), known_rep=(0.4814, 0.3409), researcher=(0.4813, 0.3408, -0.3340))
    key = {(int(r.s) + 1, int(r.i)): (int(r.x), r.kind) for r in cand.itertuples()}
    tt = trd[(trd.buy_date <= "2021-12-30") & (trd.sell_date <= "2021-12-30") & (trd.delist == 0) & (trd.why == "sig")]
    same = 0; diffs = []
    for r in tt.itertuples():
        x = key.get((int(r.tb), int(r.i)))
        if x is not None and x[0] == int(r.ts):
            same += 1
        else:
            diffs.append([r.code, r.buy_date, r.sell_date, None if x is None else P.dstr[x[0]]])
    tv = trd[(trd.buy_date > "2021-12-30") & (trd.delist == 0) & (trd.why == "sig")]
    same_v = sum(1 for r in tv.itertuples() if key.get((int(r.tb), int(r.i)), (-9,))[0] == int(r.ts))
    diffs_v = [[r.code, r.buy_date, r.sell_date, P.dstr[key[(int(r.tb), int(r.i))][0]], int(r.reduced)]
               for r in tv.itertuples() if key.get((int(r.tb), int(r.i)), (-9,))[0] != int(r.ts)]
    out["acceptance"] = dict(P0=dict(T=P.T, N=P.N, last=P.dstr[-1]), A0_W10_70=a0, A1_f1_subset_of_top20=a1_sub,
                             A1_f1_counts=dict(train=int(tr.f1.sum()), valid=int(va.f1.sum())),
                             A2=dict(n=int(len(tt)), same=int(same), diffs=diffs[:20],
                                     valid_n=int(len(tv)), valid_same=int(same_v), valid_diffs=diffs_v,
                                     sell_split_note="引擎單筆賣出 ≤ 當日成交值 5%，被拆單的交易最後賣出日會晚於結果變數的出場日"))
    print("A0/A2", a0, len(tt), same, len(tv), same_v, flush=True)
    del sim

    # ---------------- 樣本概況
    def prof(d):
        return dict(n=int(len(d)), n_outcome=int(np.isfinite(d.exc).sum()), n_no_open=int((d.kind == "no_open").sum()),
                    n_trunc=int((d.kind == "trunc").sum()), n_delist=int((d.kind == "delist").sum()),
                    n_limup_entry=int(d.lim_up.sum()), mean_exc=float(d.exc.mean()), median_exc=float(d.exc.median()),
                    mean_ret=float(d.ret.mean()), hold_mean=float(d.loc[d.hold >= 0, "hold"].mean()),
                    hold_median=float(d.loc[d.hold >= 0, "hold"].median()),
                    f1_mean_exc=float(d.loc[d.f1 == 1, "exc"].mean()), nonf1_mean_exc=float(d.loc[d.f1 == 0, "exc"].mean()))
    thr_bw = float(np.quantile(tr.exc.dropna().values, 0.9))
    for d in (tr, va, cand):
        d["bw"] = d.exc >= thr_bw
    out["profile"] = dict(train=prof(tr), valid=prof(va), big_winner_threshold=thr_bw,
                          n_bw_train=int(tr.bw.sum()), n_bw_valid=int(va.bw.sum()),
                          bw_share_train=float(tr.bw[np.isfinite(tr.exc)].mean()),
                          bw_share_valid=float(va.bw[np.isfinite(va.exc)].mean()))
    print("profile", out["profile"]["big_winner_threshold"], out["profile"]["n_bw_train"], out["profile"]["n_bw_valid"],
          flush=True)

    # ---------------- 訓練期
    uni = univariate(tr)
    rl = rules(tr, uni)
    best, weak, elig = select(rl, uni)
    frozen = dict(feature=best["feature"], direction=best["dir"], threshold=best["thr"], keep_approx=best["keep_approx"],
                  big_winner_threshold_excess=thr_bw, training_weak=weak, training_t=best["t"],
                  training_auc_dir=best["auc_dir"], eligible_features=elig)
    out["train"] = dict(univariate=uni, rules=rl, frozen=frozen)
    print("FROZEN(mine)", frozen, flush=True)
    for f in FEATS:
        print("  ", f, round(uni[f]["auc"], 4), [(round(r["t"], 3), r["keep_approx"]) for r in rl if r["feature"] == f])

    # ---------------- 驗證期（用自己凍結的規則；另算 FROZEN.json 的規則以防不同）
    fz_file = json.load(open(os.path.join(os.path.dirname(HERE), "FROZEN.json")))

    def validate(feature, direction, thr):
        d = va[np.isfinite(va.exc)].copy()
        a = K.auc(d.bw.values, d[feature].values)
        a_dir = a if direction > 0 else 1 - a
        k = K.rule_keep(d, feature, direction, thr)
        dm = K.month_d(d, k)
        bw_keep = float(d.loc[k, "bw"].mean()); bw_all = float(d.bw.mean())
        kk = d[k].sort_values("exc", ascending=False)
        top5 = kk.head(5)
        d2 = d.drop(top5.index); k2 = k.drop(top5.index)
        v4 = float(d2.loc[k2, "exc"].mean() - d2.exc.mean())
        yearly = {}
        for y in range(2022, 2027):
            dmy = dm[[P.dstr[s][:4] == str(y) for s in dm.index]]
            yearly[str(y)] = dict(n=int(len(dmy)), mean=fl(dmy.mean()), t=fl(K.tstat(dmy.values)))
        o = dict(auc=a, auc_dir=a_dir, V1=bool(a_dir >= 0.55),
                 bw_share_keep=bw_keep, bw_share_all=bw_all, V2=bool(bw_keep > bw_all),
                 d_mean=float(dm.mean()), t=K.tstat(dm.values), n_months=int(len(dm)),
                 V3=bool(dm.mean() > 0 and K.tstat(dm.values) >= 2.0),
                 mean_keep=float(d.loc[k, "exc"].mean()), mean_all=float(d.exc.mean()), n_keep=int(k.sum()),
                 keep_frac=float(k.mean()), top5_removed=[[r.code, r.sig, float(r.exc)] for r in top5.itertuples()],
                 V4_diff=v4, V4=bool(v4 > 0), yearly_dm=yearly)
        o["pass"] = bool(o["V1"] and o["V2"] and o["V3"] and o["V4"])
        o["dm_series"] = {P.dstr[s]: float(v) for s, v in dm.items()}
        return o, k

    vmine, kmine = validate(frozen["feature"], frozen["direction"], frozen["threshold"])
    vfile, kfile = validate(fz_file["feature"], fz_file["direction"], fz_file["threshold"])
    out["valid"] = dict(mine=vmine, frozen_file=vfile,
                        univariate={f: dict(auc=K.auc(va.bw[np.isfinite(va.exc)].values, va.loc[np.isfinite(va.exc), f].values),
                                            quint=K.quintiles(va[np.isfinite(va.exc)], f)) for f in FEATS})
    # 驗證期大贏家名單
    dv = va[np.isfinite(va.exc)]
    kf = K.rule_keep(dv, fz_file["feature"], fz_file["direction"], fz_file["threshold"])
    out["valid"]["big_winners"] = [[r.code, r.sig, round(float(r.exc), 4), int(r.f1), bool(kf.loc[r.Index])]
                                   for r in dv[dv.bw].sort_values("exc", ascending=False).itertuples()]
    print("VALID(file rule)", {k: vfile[k] for k in ("auc_dir", "V1", "bw_share_keep", "bw_share_all", "V2", "d_mean", "t",
                                                     "V3", "V4_diff", "V4", "pass")}, flush=True)

    # ---------------- F1 子集合（訓練、驗證）
    def f1_eff(d, feature, direction, thr):
        d = d[np.isfinite(d.exc) & (d.f1 == 1)]
        k = K.rule_keep(d, feature, direction, thr)
        dm = K.month_d(d, k)
        # 每月 留下 − 剔除（兩邊都有的月份）
        a = d[k].groupby("s").exc.mean(); b = d[~k].groupby("s").exc.mean()
        both = a.index.intersection(b.index)
        pf = (a[both] - b[both])
        return dict(n=int(len(d)), n_keep=int(k.sum()), n_drop=int((~k).sum()), mean_all=float(d.exc.mean()),
                    mean_keep=float(d.loc[k, "exc"].mean()), mean_drop=float(d.loc[~k, "exc"].mean()),
                    median_keep=float(d.loc[k, "exc"].median()), median_drop=float(d.loc[~k, "exc"].median()),
                    bw_keep=float(d.loc[k, "bw"].mean()), bw_drop=float(d.loc[~k, "bw"].mean()),
                    dm_keep_vs_all=dict(mean=float(dm.mean()), t=K.tstat(dm.values), n=int(len(dm))),
                    dm_keep_vs_drop=dict(mean=float(pf.mean()), t=K.tstat(pf.values), n=int(len(pf))))
    out["f1_subset"] = {nm: f1_eff(d, fz_file["feature"], fz_file["direction"], fz_file["threshold"])
                        for nm, d in (("train", tr), ("valid", va))}
    print("F1", json.dumps(out["f1_subset"], ensure_ascii=False)[:600], flush=True)

    # ---------------- 審查 A：大輸家
    thr_bl = float(np.quantile(tr.exc.dropna().values, 0.1))
    ra = dict(big_loser_threshold=thr_bl, auc_loser={}, auc_extreme={})
    for nm, d in (("train", tr), ("valid", va)):
        dd = d[np.isfinite(d.exc)]
        bl = dd.exc <= thr_bl
        ext = bl | dd.bw
        ra["n_bl_" + nm] = int(bl.sum())
        ra["auc_loser"][nm] = {f: K.auc(bl.values, dd[f].values) for f in FEATS}
        ra["auc_extreme"][nm] = {f: K.auc(ext.values, dd[f].values) for f in FEATS}
        # RET60 五組的超額標準差
        q = K.quintiles(dd, "RET60")
        ra["ret60_quint_" + nm] = q
        g = pd.qcut(dd.RET60, 5, labels=False)
        sd_top = float(dd.exc[g == 4].std(ddof=1)); sd_rest = float(dd.exc[g < 4].std(ddof=1))
        sd_mid = float(np.median([dd.exc[g == j].std(ddof=1) for j in range(4)]))
        ra["ret60_sd_" + nm] = dict(sd_top=sd_top, sd_rest=sd_rest, ratio_top_rest=sd_top / sd_rest,
                                    ratio_top_median_other=sd_top / sd_mid,
                                    ratio_top_bottom=sd_top / float(dd.exc[g == 0].std(ddof=1)))
        # 凍結規則（ADVCHG）
        g2 = pd.qcut(dd.ADVCHG, 5, labels=False)
        ra["advchg_sd_" + nm] = dict(sd_top=float(dd.exc[g2 == 4].std(ddof=1)), sd_rest=float(dd.exc[g2 < 4].std(ddof=1)))
        kk = K.rule_keep(dd, fz_file["feature"], fz_file["direction"], fz_file["threshold"])
        ra["frozen_rule_" + nm] = dict(bl_share_keep=float(bl[kk].mean()), bl_share_drop=float(bl[~kk].mean()),
                                       bw_share_keep=float(dd.bw[kk].mean()), bw_share_drop=float(dd.bw[~kk].mean()),
                                       sd_keep=float(dd.exc[kk].std(ddof=1)), sd_drop=float(dd.exc[~kk].std(ddof=1)))
    out["review_A"] = ra
    print("reviewA", json.dumps({k: ra[k] for k in ("auc_loser",)}, ensure_ascii=False), ra["ret60_sd_train"],
          ra["ret60_sd_valid"], flush=True)

    # ---------------- 審查 B：V3 的 t（持有期重疊）
    dm = pd.Series(vfile["dm_series"])
    dv = va[np.isfinite(va.exc)]
    kk = K.rule_keep(dv, fz_file["feature"], fz_file["direction"], fz_file["threshold"])
    hold_rows = dv.loc[kk, "hold"]
    # 持有期跨幾個訊號月：出場列之前的訊號列數
    rows_a = np.array(rows)
    span = [(int(np.searchsorted(rows_a, x, side="left")) - int(np.searchsorted(rows_a, s, side="right")))
            for s, x in zip(dv.loc[kk, "s"], dv.loc[kk, "x"])]
    rb = dict(plain_t=K.tstat(dm.values), n=int(len(dm)), mean=float(dm.mean()),
              hold_days_mean=float(hold_rows.mean()), hold_days_median=float(hold_rows.median()),
              months_spanned_mean=float(np.mean(span)), months_spanned_p90=float(np.percentile(span, 90)),
              acf={str(l): float(pd.Series(dm.values).autocorr(l)) for l in range(1, 7)},
              nw={}, boot={})
    nauto = int(np.floor(4 * (len(dm) / 100.0) ** (2.0 / 9.0)))
    for lag in sorted(set([1, 2, 3, 4, 6, nauto, int(np.ceil(np.mean(span)))])):
        t, se = K.nw_t(dm.values, lag)
        rb["nw"][str(lag)] = dict(t=t, se=se)
    rb["nw_auto_lag"] = nauto
    for b in (2, 3, 6):
        t, se, p = K.block_boot_t(dm.values, b)
        rb["boot"][str(b)] = dict(t=t, se=se, p_one_sided=p)
    # 訓練期同一條規則的 t 也算
    dmt = K.month_d(tr[np.isfinite(tr.exc)], K.rule_keep(tr[np.isfinite(tr.exc)], fz_file["feature"],
                                                       fz_file["direction"], fz_file["threshold"]))
    rb["train_plain_t"] = K.tstat(dmt.values)
    rb["train_nw"] = {str(l): K.nw_t(dmt.values, l)[0] for l in (1, 3, 6)}
    rb["train_boot3"] = K.block_boot_t(dmt.values, 3)[0]
    out["review_B"] = rb
    print("reviewB", rb["plain_t"], rb["nw"], rb["boot"], flush=True)

    # ---------------- 審查 C：固定持有一個月
    rcC = {}
    trc = tr.copy(); vac = va.copy()
    thrC = float(np.quantile(trc.excC.dropna().values, 0.9))
    trc["bw"] = trc.excC >= thrC; vac["bw"] = vac.excC >= thrC
    uniC = univariate(trc, "bw", "excC")
    rlC = rules(trc, uniC, "excC")
    bestC, weakC, eligC = select(rlC, uniC)
    rcC = dict(big_winner_threshold=thrC, n_bw=int(trc.bw.sum()),
               profile=dict(train_mean=float(trc.excC.mean()), valid_mean=float(vac.excC.mean()),
                            n_trunc_train=int((trc.kindC == "trunc").sum()), n_delist=int((cand.kindC == "delist").sum())),
               univariate={f: dict(auc=uniC[f]["auc"], auc_dir=uniC[f]["auc_dir"], dir=uniC[f]["dir"],
                                   quint=uniC[f]["quint"]) for f in FEATS},
               rules=rlC,
               frozen=dict(feature=bestC["feature"], direction=bestC["dir"], threshold=bestC["thr"],
                           keep_approx=bestC["keep_approx"], t=bestC["t"], auc_dir=bestC["auc_dir"], weak=weakC,
                           eligible=eligC),
               corr_E0_vs_fixed_train=float(trc[["exc", "excC"]].corr(method="spearman").iloc[0, 1]),
               bw_overlap_train=dict(both=int((tr.bw & trc.bw).sum()), e0_only=int((tr.bw & ~trc.bw).sum()),
                                     fixed_only=int((~tr.bw & trc.bw).sum())))
    # 原凍結規則（ADVCHG ≥ 0.296）在固定持有結果下的訓練 t（描述）
    kC = K.rule_keep(trc[np.isfinite(trc.excC)], fz_file["feature"], fz_file["direction"], fz_file["threshold"])
    dmc = K.month_d(trc[np.isfinite(trc.excC)], kC, "excC")
    rcC["frozen_file_rule_train_t_fixed"] = K.tstat(dmc.values)
    # 描述：固定持有凍結規則若拿到驗證期（固定持有結果）的四條（不改判定）
    dvC = vac[np.isfinite(vac.excC)]
    kv = K.rule_keep(dvC, bestC["feature"], bestC["dir"], bestC["thr"])
    dmv = K.month_d(dvC, kv, "excC")
    av = K.auc(dvC.bw.values, dvC[bestC["feature"]].values)
    rcC["alt_rule_valid_desc"] = dict(auc_dir=av if bestC["dir"] > 0 else 1 - av, bw_keep=float(dvC.bw[kv].mean()),
                                      bw_all=float(dvC.bw.mean()), d_mean=float(dmv.mean()), t=K.tstat(dmv.values))
    out["review_C"] = rcC
    print("reviewC", rcC["frozen"], {f: round(uniC[f]["auc"], 4) for f in FEATS}, flush=True)

    out["resources"] = dict(peak_rss_mb=int(rss()), runtime_s=round(time.time() - t_start, 1))
    keepcols = ["sig", "code", "rank", "f1", "period", "entry", "exit", "kind", "hold", "lim_up", "ret", "r50", "exc",
                "bw", "excC", "gm_q"] + FEATS
    cand[keepcols].to_csv(os.path.join(odir, "my_cands.csv.gz"), index=False, float_format="%.8g")
    js = json.dumps(out, ensure_ascii=False, indent=1, default=lambda o: o if not isinstance(o, (np.generic,))
                    else o.item())
    open(os.path.join(odir, "my_numbers.json"), "w").write(js)
    h = hashlib.sha256(js.encode("utf-8")).hexdigest()
    open(os.path.join(odir, "my_numbers.sha256"), "w").write(h + "  my_numbers.json\n")
    print("done", out["resources"], h, flush=True)


def K11adj(P):
    return K.K11.adj_open_matrix(P)


if __name__ == "__main__":
    main(sys.argv[1])
