# -*- coding: utf-8 -*-
"""S16 獨立複製：資料檢查（上櫃日 K、下市、轉上市、2018-10 營收缺洞與補洞）、補洞檔逐筆比對、驗收、
CUR／ALL／OTC／ALL_O20／ALL_RAW 主跑＋八項、20 條路徑、逐條配對、四段空頭、事前判定、分市場每筆期望值與報酬貢獻、
審查 A（上櫃期望值拆解）、B（上市優先名額，事後）、C（下市股營收缺失的偏誤範圍），然後凍結。
用法：python3 -I -B rep_s16_run.py <scratch_dir>
需先跑 s1 的 rep_s1_panel.py <scratch_dir> 與 s3 的 rep_s3_names.py <scratch_dir>/innov_names.json。
輸出（本資料夾）：my_numbers.json、my_numbers.sha256、my_trades.csv.gz、my_paths.csv.gz、my_rev_patch_otc_201810.csv
凍結前不讀研究員的 FINDINGS 第 1 節以外、results*.json、navs.csv.gz、trades.csv.gz。
"""
import sys, os, json, hashlib, resource, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import rep_s16_lib as K
S, Q, R, L = K.S, K.Q, K.R, K.L
ST, X2 = K.ST, K.X2

CAPS = [600_000.0, 800_000.0, 1_000_000.0, 1_500_000.0, 2_000_000.0]
SHIFTS = [0, 1, 2, 3]
KNOWN = (0.4813, 0.3408, -0.3340)
BEARS = {"2018": ("2018-01-01", "2018-12-31"), "2020-03": ("2020-01-01", "2020-06-30"),
         "2022": ("2022-01-01", "2022-12-31"), "2024-07~2025-04": ("2024-07-01", "2025-04-30")}
CELLS = ["CUR", "ALL", "OTC", "ALL_O20", "ALL_RAW"]
MKT = {1: "twse", 2: "tpex"}


def rss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024


def jdef(o):
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, set):
        return sorted(o)
    return str(o)


def main(sc, odir=HERE):
    t_start = time.time()
    P = R.build(sc)
    fixed = K.attach_e50_fixed(P, sc)
    excl = S.innov_mask_fn(P, sc)
    excl_M = P.innov_M
    AO, Gf = K.adj_open_matrix(P)
    T = P.T; e = T - 1; t0 = P.t0
    i19 = L.didx(P, "2018-12-28"); i24 = L.didx(P, "2023-12-29"); ih = L.didx(P, "2021-02-19")
    assert P.dstr[i19] == "2018-12-28" and P.dstr[i24] == "2023-12-29"
    da = np.array(P.dstr)
    yrs_arr = np.array([int(d[:4]) for d in P.dstr])
    codes = np.array(P.codes)
    Mk = P.M

    def win(a, b):
        return int(np.searchsorted(da, a, side="left")), int(np.searchsorted(da, b, side="right")) - 1
    bw = {k: win(a, b) for k, (a, b) in BEARS.items()}
    e50 = P.e50.values.astype(float)

    def bears(nav):
        return {k: L.maxdd(P, nav, a, b)[0] for k, (a, b) in bw.items()}

    def nav_stats(nav):
        md = L.maxdd(P, nav, t0, e)
        return dict(ann_2019=L.ann(P, nav, i19, e), ann_full=L.ann(P, nav, t0, e), ann_2024=L.ann(P, nav, i24, e),
                    ann_1923=L.ann(P, nav, i19, i24), mdd_full=md[0], mdd_peak=md[1], mdd_trough=md[2],
                    mdd_2019=L.maxdd(P, nav, i19, e)[0], ann_h1=L.ann(P, nav, t0, ih), ann_h2=L.ann(P, nav, ih, e))

    out = dict(meta=dict(T=T, N=P.N, first=P.dstr[0], last=P.dstr[-1], t0=P.dstr[t0], i19=P.dstr[i19],
                         i24=P.dstr[i24], ih=P.dstr[ih],
                         bear_windows={k: (P.dstr[a], P.dstr[b]) for k, (a, b) in bw.items()},
                         note="面板為複製員自建（N＝2130；研究員 2142，差 12 檔推測為 91xx（研究員面板含、股池排除；"
                              "s1 複製員面板直接不收）。0050 停牌日用 S14 複製員的修正。",
                         e50_fixed_days=[x["date"] for x in fixed]))
    b50 = dict(nav_stats(e50), bears=bears(e50), yearly=L.yearly(P, e50))
    out["bench_0050"] = b50

    # ================================================================ 1. 資料檢查（價格面）
    dc = {}
    has_tp = (Mk == 2).any(axis=1)
    dc["tpex_first_date"] = P.dstr[int(np.argmax(has_tp))]
    dc["tpex_every_day"] = bool(has_tp.all())
    dc["codes_by_year"] = {}
    for y in range(2015, 2027):
        rr = yrs_arr == y
        dc["codes_by_year"][y] = dict(twse=int((Mk[rr] == 1).any(axis=0).sum()), tpex=int((Mk[rr] == 2).any(axis=0).sum()))
    last = P.last
    dl = np.where(last < T - 1 - 5)[0]
    mk_last = np.array([Mk[last[i], i] for i in dl])
    dc["delisted_total"] = dict(twse=int((mk_last == 1).sum()), tpex=int((mk_last == 2).sum()))
    dby = {}
    for i, m_ in zip(dl, mk_last):
        y = P.dstr[last[i]][:4]
        dby.setdefault(y, dict(twse=0, tpex=0))[MKT[int(m_)]] += 1
    dc["delisted_by_year"] = dict(sorted(dby.items()))
    # 轉上市（同一代號 M 在相鄰兩個有資料的交易日改變）
    sw = []
    for i in range(P.N):
        tt = np.where(Mk[:, i] > 0)[0]
        if len(tt) < 2:
            continue
        mm = Mk[tt, i]
        ch = np.where(mm[1:] != mm[:-1])[0]
        for c in ch:
            a, b = tt[c], tt[c + 1]
            sw.append(dict(code=codes[i], frm=int(mm[c]), to=int(mm[c + 1]), last_old=P.dstr[a], first_new=P.dstr[b],
                           gap_days=int(b - a - 1)))
    swd = pd.DataFrame(sw)
    dc["market_switch"] = dict(n=int(len(swd)), tpex_to_twse=int(((swd.frm == 2) & (swd.to == 1)).sum()),
                               twse_to_tpex=int(((swd.frm == 1) & (swd.to == 2)).sum()),
                               max_gap=int(swd.gap_days.max()), n_codes=int(swd.code.nunique()),
                               list=swd.to_dict("records"))
    print("datacheck price", dc["tpex_first_date"], dc["delisted_total"], dc["market_switch"]["n"],
          dc["market_switch"]["tpex_to_twse"], dc["market_switch"]["twse_to_tpex"], dc["market_switch"]["max_gap"],
          flush=True)

    # ================================================================ 2. 月營收、缺洞、補洞
    raw = K.read_rev_csv()
    cnt_m = raw.groupby(["month", "market"]).size().unstack(fill_value=0)
    holes = {}
    for mk_ in cnt_m.columns:
        v = cnt_m[mk_].values
        for k_ in range(len(v)):
            nb = np.r_[v[max(0, k_ - 5):k_], v[k_ + 1:k_ + 6]]
            med = float(np.median(nb))
            if v[k_] < 0.8 * med:
                holes["%s_%s" % (cnt_m.index[k_][:7], mk_)] = dict(n=int(v[k_]), median_nearby=med)
    dc["revenue_holes"] = holes
    dc["revenue_rows_by_market"] = raw.market.value_counts().to_dict()
    mops, mops_sha, n_tot = K.parse_mops()
    mops = mops.set_index("code")
    o10 = raw[(raw.month == "2018-10-01") & (raw.market == "otc")].set_index("code")
    o09 = raw[(raw.month == "2018-09-01") & (raw.market == "otc")].set_index("code")
    o17 = raw[(raw.month == "2017-10-01") & (raw.market == "otc")].set_index("code")
    any10 = raw[raw.month == "2018-10-01"]

    def vchk(csvd, col):
        com = csvd.index.intersection(mops.index)
        ex = int((np.abs(csvd.loc[com, "revenue_k_twd"].values - mops.loc[com, col].values) < 0.5).sum())
        return dict(csv_rows=int(len(csvd)), matched=int(len(com)), exact=ex)
    val = {"2018-10_rev": vchk(o10, "cur"), "2018-09_prev_month": vchk(o09, "prev"), "2017-10_last_year": vchk(o17, "ly")}
    newc = [c for c in mops.index if c not in set(any10.code)]
    my_patch = pd.DataFrame(dict(month="2018-10-01", market="otc", code=newc, name=mops.loc[newc, "name"].values,
                                 revenue_k_twd=mops.loc[newc, "cur"].values))
    my_patch = my_patch.sort_values("code").reset_index(drop=True)
    my_patch.to_csv(os.path.join(odir, "my_rev_patch_otc_201810.csv"), index=False)
    # 研究員補洞檔逐筆比對
    rp_sha = hashlib.sha256(open(K.RPATCH, "rb").read()).hexdigest()
    rp = pd.read_csv(K.RPATCH, dtype={"code": str})
    mm_ = my_patch.merge(rp, on="code", how="outer", suffixes=("_me", "_rs"), indicator=True)
    both = mm_[mm_._merge == "both"]
    pcmp = dict(researcher_sha256=rp_sha, sha_ok=bool(rp_sha == K.RPATCH_SHA), researcher_rows=int(len(rp)),
                my_rows=int(len(my_patch)), only_me=mm_[mm_._merge == "left_only"].code.tolist(),
                only_researcher=mm_[mm_._merge == "right_only"].code.tolist(), both=int(len(both)),
                value_exact=int((np.abs(both.revenue_k_twd_me - both.revenue_k_twd_rs) < 0.5).sum()),
                name_exact=int((both.name_me.str.strip() == both.name_rs.str.strip()).sum()),
                month_market_ok=bool((rp.month == "2018-10-01").all() and (rp.market == "otc").all()),
                mops_rows=int(len(mops)), mops_sha256=mops_sha, validation=val,
                patch_codes_with_sii_row_same_month=int(any10[(any10.market == "sii") & any10.code.isin(newc)].shape[0]))
    pcmp["identical"] = bool(pcmp["sha_ok"] and not pcmp["only_me"] and not pcmp["only_researcher"] and
                             pcmp["value_exact"] == pcmp["both"] == len(rp) and pcmp["name_exact"] == pcmp["both"])
    out["patch_compare"] = pcmp
    print("patch", json.dumps({k: v for k, v in pcmp.items() if k not in ("only_me", "only_researcher")},
                              ensure_ascii=False, default=jdef), flush=True)
    assert pcmp["identical"]
    Rv_raw, mi0 = K.load_rev_df(P, raw)
    Rv_l, mi0_l = L.load_rev(P)
    a3_raw_same = bool(mi0 == mi0_l and Rv_raw.shape == Rv_l.shape and
                       np.array_equal(np.nan_to_num(Rv_raw, nan=-1), np.nan_to_num(Rv_l, nan=-1)))
    del Rv_l
    Rv_p, mi0p = K.load_rev_df(P, raw, my_patch)
    assert mi0p == mi0
    S_raw, _ = L.surge(Rv_raw, mi0, null2013=True)
    S_p, _ = L.surge(Rv_p, mi0, null2013=True)
    ch_ = np.isfinite(S_raw) != np.isfinite(S_p)
    ch_ |= np.isfinite(S_raw) & np.isfinite(S_p) & (np.abs(np.nan_to_num(S_raw) - np.nan_to_num(S_p)) > 1e-12)
    myear = np.array([(mi0 + m) // 12 for m in range(S_p.shape[0])])
    dc["patch_effect_surge_cells_changed_by_year"] = {int(y): int(ch_[myear == y].sum()) for y in np.unique(myear)
                                                      if ch_[myear == y].sum() > 0}
    del raw

    # ================================================================ 3. 股池與名單
    U = K.univ_mats(P, excl_M)
    a2 = {}
    a2["ALL_eq_CUR_or_OTC"] = bool(np.array_equal(U["ALL"], U["CUR"] | U["OTC"]))
    a2["CUR_and_OTC_empty"] = bool(not (U["CUR"] & U["OTC"]).any())
    a2["O20_subset_ALL"] = bool(not (U["ALL_O20"] & ~U["ALL"]).any())
    a2["O20_and_twse_eq_CUR"] = bool(np.array_equal(U["ALL_O20"] & (Mk == 1), U["CUR"]))
    cur_eq = True
    for t in range(60, T):
        if not np.array_equal(U["CUR"][t], R.universe(P, t) & ~excl(t)):
            cur_eq = False; break
    a2["CUR_eq_rep_s2_universe"] = cur_eq
    print("A2 sets", a2, flush=True)

    lists = {}; plans = {}
    for sh in SHIFTS:
        for cc in ("CUR", "ALL", "OTC", "ALL_O20"):
            lists[(cc, sh)] = K.make_lists_u(P, S_p, mi0, U[cc], shift=sh)
            plans[(cc, sh)] = Q.cell_plans(lists[(cc, sh)], "C10", x=0.10)
        lists[("ALL_RAW", sh)] = K.make_lists_u(P, S_raw, mi0, U["ALL"], shift=sh)
        plans[("ALL_RAW", sh)] = Q.cell_plans(lists[("ALL_RAW", sh)], "C10", x=0.10)
        lists[("CUR_RAW", sh)] = K.make_lists_u(P, S_raw, mi0, U["CUR"], shift=sh)
    print("plans", "rss", rss(), round(time.time() - t_start), flush=True)
    acc = dict(P0=dict(T=T, N=P.N, last=P.dstr[-1], note="研究員 N＝2142；複製員面板 2130（已知差異，91xx）"))
    # A1：CUR 名單＝rep_s3 make_lists（rep_s2 universe＋創新板排除）＋C10，位移 0～3
    a1 = True
    for sh in SHIFTS:
        lq = Q.make_lists(P, S_p, mi0, shift=sh, excl=excl)
        bq, kq = Q.cell_plans(lq, "C10", x=0.10)
        b_, k_ = plans[("CUR", sh)]
        a1 &= sorted(bq) == sorted(b_) and all(np.array_equal(bq[s], b_[s]) and kq[s] == k_[s] for s in bq)
        a1 &= all(np.array_equal(lq[s]["lma"], lists[("CUR", sh)][s]["lma"]) for s in lq)
    acc["A1"] = dict(identical_to_rep_s3_make_lists=bool(a1))
    # A3：補洞不改變 CUR 訊號
    d3 = dict(buy_cells_diff=0, exit_cells_diff=0, months_diff=[])
    for sh in SHIFTS:
        lr = lists[("CUR_RAW", sh)]
        br, kr = Q.cell_plans(lr, "C10", x=0.10)
        b_, k_ = plans[("CUR", sh)]
        assert sorted(br) == sorted(b_)
        for s in b_:
            nb_ = len(set(map(int, br[s])) ^ set(map(int, b_[s]))); nk_ = len(kr[s] ^ k_[s])
            d3["buy_cells_diff"] += nb_; d3["exit_cells_diff"] += nk_
            if nb_ or nk_:
                d3["months_diff"].append((sh, P.dstr[s]))
    acc["A3"] = dict(unpatched_rev_equals_rep_s1_load_rev=a3_raw_same, patch_effect_on_CUR=d3)
    print("A1", acc["A1"], "A3", acc["A3"], flush=True)

    # 資料檢查重現：換股日（位移 0）各格股池大小、SURGE 有值比例、站上季線
    ma60 = R.ma(P, 60)
    rowsdc = []
    sig0 = sorted(lists[("CUR", 0)])
    for s in sig0:
        m = lists[("CUR", 0)][s]["m"]
        sp_ = np.isfinite(S_p[m]); sr_ = np.isfinite(S_raw[m])
        with np.errstate(invalid="ignore"):
            abv = P.G[s] > ma60[s]
        for cc in ("CUR", "ALL", "OTC", "ALL_O20"):
            u = U[cc][s]
            rowsdc.append(dict(s=s, year=int(P.dstr[s][:4]), cell=cc, pool=int(u.sum()), surge_ok=int((u & sp_).sum()),
                               surge_ok_raw=int((u & sr_).sum()), above_ma60=int((u & sp_ & abv).sum())))
    ddc = pd.DataFrame(rowsdc)
    pby = {}
    for (y, cc), g in ddc.groupby(["year", "cell"]):
        pby.setdefault(int(y), {})[cc] = dict(pool=float(g.pool.mean()), surge_ok=float(g.surge_ok.mean()),
                                              surge_share=float(g.surge_ok.sum() / g.pool.sum()),
                                              surge_share_raw=float(g.surge_ok_raw.sum() / g.pool.sum()),
                                              above_ma60=float(g.above_ma60.mean()))
    dc["pool_by_year_on_signal_days"] = pby
    g16 = ddc[ddc.year >= 2016]
    dc["pool_avg_since2016"] = {cc: dict(pool=float(g.pool.mean()), surge_ok=float(g.surge_ok.mean()))
                                for cc, g in g16.groupby("cell")}
    # 不足 50 檔的月份（全部 A10 訊號日，未位移）
    lt50 = {}
    for cc in ("CUR", "ALL", "OTC", "ALL_O20"):
        for nmS, SS in (("patched", S_p), ("raw", S_raw)):
            bad = []
            for m in range(SS.shape[0]):
                d10, j, s0 = L.rev_avail(P, mi0 + m)
                if s0 < t0 or s0 >= T - 1:
                    continue
                if int((U[cc][s0] & np.isfinite(SS[m])).sum()) < 50:
                    bad.append("%04d-%02d" % ((mi0 + m) // 12, (mi0 + m) % 12 + 1))
            lt50["%s_%s" % (cc, nmS)] = dict(n=len(bad), months=bad if len(bad) <= 40 else bad[:40])
    dc["months_lt50_surge"] = lt50
    out["data_check"] = dc
    print("pool avg", dc["pool_avg_since2016"], {k: v["n"] for k, v in lt50.items()}, flush=True)

    # ================================================================ 4. 引擎
    def sim(cc, sh=0, cap=1_000_000.0, cost=None, block_span=None, rec_pos=False, pl=None):
        b, k = pl if pl is not None else plans[(cc, sh)]
        return K.simulate(P, b, k, sat_w=0.7, sizing="W10", capital=cap, block_span=block_span, rec_pos=rec_pos,
                          **(cost or ST))

    def top5(tr, nav):
        inc = float(nav[e] - nav[t0])
        return float(tr.sort_values("pnl", ascending=False).head(5).pnl.sum() / inc)

    main = {}
    main["CUR"] = sim("CUR", rec_pos=True)
    nav0 = main["CUR"]["nav"]
    x9 = K.K9.simulate_s9(P, *plans[("CUR", 0)], sat_w=0.7, sizing="W10", **ST)
    same9 = bool(np.allclose(x9["nav"][t0:], nav0[t0:], rtol=0, atol=1e-6) and len(x9["trades"]) == len(main["CUR"]["trades"])
                 and (x9["trades"][["code", "tb", "ts", "sh"]].values == main["CUR"]["trades"][["code", "tb", "ts", "sh"]].values).all())
    del x9
    acc["A0"] = dict(ann_2019=L.ann(P, nav0, i19, e), ann_full=L.ann(P, nav0, t0, e), mdd_full=L.maxdd(P, nav0, t0, e)[0],
                     n_trades_closed=int(len(main["CUR"]["trades"])), n_open=len(main["CUR"]["open_pos"]), known=KNOWN,
                     same_as_simulate_s9=same9)
    acc["A0"]["ok"] = bool(abs(acc["A0"]["ann_2019"] - KNOWN[0]) < 0.00025 and abs(acc["A0"]["ann_full"] - KNOWN[1]) < 0.00025
                           and abs(acc["A0"]["mdd_full"] - KNOWN[2]) < 0.00025 and same9)
    print("A0", acc["A0"], flush=True)
    assert acc["A0"]["ok"]
    for cc in CELLS[1:]:
        main[cc] = sim(cc, rec_pos=cc in ("ALL", "OTC"))

    # A2（續）：每個買進訊號都在該格股池內；CUR 每筆進場是上市、OTC 每筆是上櫃
    a2b = {}
    for cc in CELLS:
        ucc = U["ALL" if cc == "ALL_RAW" else cc]
        nb = 0; nbad = 0
        for sh in SHIFTS:
            for s, b in plans[(cc, sh)][0].items():
                nb += len(b); nbad += int((~ucc[s, b]).sum()) if len(b) else 0
        a2b[cc] = dict(n_buy_signals=nb, not_in_pool=nbad)
    trc = main["CUR"]["trades"]; tro = main["OTC"]["trades"]
    a2b["CUR_trades_twse_at_entry"] = [int((Mk[trc.tb.values, trc.i.values] == 1).sum()), int(len(trc)),
                                       int((Mk[trc.tb.values - 1, trc.i.values] == 1).sum())]
    a2b["OTC_trades_tpex_at_entry"] = [int((Mk[tro.tb.values, tro.i.values] == 2).sum()), int(len(tro)),
                                       int((Mk[tro.tb.values - 1, tro.i.values] == 2).sum())]
    acc["A2"] = dict(sets=a2, signals=a2b)
    out["acceptance"] = acc
    print("A2", a2b, flush=True)

    # ================================================================ 5. 主跑統計、八項
    def cand_stats(cc, sh=0):
        b, k = plans[(cc, sh)]; ls = lists[(cc, sh)]
        nb = np.array([len(b[s]) for s in sorted(b)])
        nl = np.array([len(ls[s]["lma"]) for s in sorted(ls)])
        o = dict(n_months=int(len(nb)), buy_mean=float(nb.mean()), buy_empty_months=int((nb == 0).sum()),
                 lma_mean=float(nl.mean()), first=P.dstr[min(b)], last=P.dstr[max(b)])
        if cc in ("ALL", "ALL_O20", "ALL_RAW"):
            tot = sum(len(b[s]) for s in b)
            ot = sum(int((Mk[s, b[s]] == 2).sum()) for s in b if len(b[s]))
            o["buy_list_tpex_share"] = float(ot / tot) if tot else None
            lt = sum(min(len(ls[s]["lma"]), 10) for s in ls)
            lo = sum(int((Mk[s, ls[s]["lma"][:10]] == 2).sum()) for s in ls)
            o["lma_top10_tpex_share"] = float(lo / lt)
            o["pool_tpex_share"] = float(np.mean([(Mk[s][U["ALL"][s]] == 2).mean() for s in ls]))
        return o

    def tstats(tr):
        if len(tr) == 0:
            return dict(n=0)
        w = tr.ret > 0
        return dict(n=int(len(tr)), win=float(w.mean()), avg_win=float(tr.ret[w].mean()),
                    avg_loss=float(-tr.ret[~w].mean()) if (~w).any() else 0.0,
                    exp=float(tr.ret.mean()), hold_mean=float(tr.hold.mean()), n_delist=int(tr.delist.sum()))

    cells = {}
    trade_rows = []
    years_full = (P.dates[e] - P.dates[t0]).days / 365.25
    for cc in CELLS:
        x = main[cc]; nav = x["nav"]; tr = x["trades"]
        o = nav_stats(nav)
        o["n_trades"] = int(len(tr)); o["n_open"] = len(x["open_pos"])
        o["n_trades_2019"] = int((tr.tb > i19).sum())
        o["bears"] = bears(nav); o["yearly"] = L.yearly(P, nav)
        x2 = sim(cc, cost=X2)
        sp, top = R.span_of(tr, 5)
        bl = sim(cc, block_span=sp)
        o["x2"] = dict(ann_2019=L.ann(P, x2["nav"], i19, e), ann_full=L.ann(P, x2["nav"], t0, e))
        o["block5"] = dict(ann_2019=L.ann(P, bl["nav"], i19, e), ann_full=L.ann(P, bl["nav"], t0, e))
        o["top5"] = top
        o["top5_share_of_gain"] = top5(tr, nav)
        o["final_nav"] = float(nav[e])
        o["trade_stats_full"] = tstats(tr); o["trade_stats_2019"] = tstats(tr[tr.tb > i19])
        cn = x["cnt"]
        o["blocked"] = dict(blk_cap=int(cn["liq_capped"]), blk_lot=int(cn["lot_skip"]), blk_limup=int(cn["limit_up"]),
                            blk_min=int(cn["blk_min"]), no_open=int(cn["no_open"]), no_cash=int(cn["no_cash"]),
                            sell_partial_cap=int(cn["sell_split"]), sell_defer_limdn=int(cn["sell_defer_limdn"]),
                            sell_defer=int(cn["sell_defer"]), cand=int(cn["cand"]), buy=int(cn["buy"]),
                            no_slot=int(cn["no_slot"]))
        mnav = float(np.mean(nav[t0:e + 1]))
        o["turnover_oneway_ann"] = float((cn["buy_amt"] + cn["sell_amt"]) / 2 / mnav / years_full)
        o["turnover_buy_ann"] = float(cn["buy_amt"] / mnav / years_full)
        o["cost_over_nav_ann"] = float((cn["cost_total"] + cn["e50_cost"]) / mnav / years_full)
        o["cost_stock_over_nav_ann"] = float(cn["cost_total"] / mnav / years_full)
        o["avg_npos"] = float(x["npos"][t0 + 1:].mean())
        satv = nav[t0 + 1:] - x["B"][t0 + 1:]
        o["sat_idle_share"] = float(np.mean(1.0 - x["stkw"][t0 + 1:] * nav[t0 + 1:] / np.where(satv > 0, satv, np.nan)))
        o["candidates"] = cand_stats(cc)
        ck = dict(a2019=o["ann_2019"] > b50["ann_2019"], afull=o["ann_full"] > b50["ann_full"],
                  x2_2019=o["x2"]["ann_2019"] > b50["ann_2019"], x2_full=o["x2"]["ann_full"] > b50["ann_full"],
                  bl_2019=o["block5"]["ann_2019"] > b50["ann_2019"], bl_full=o["block5"]["ann_full"] > b50["ann_full"],
                  h1=o["ann_h1"] > b50["ann_h1"], h2=o["ann_h2"] > b50["ann_h2"])
        o["checks"] = ck; o["n_checks_pass"] = int(sum(ck.values()))
        o["eight_cls"] = "值得上 XQ 測" if all(ck.values()) else (
            "不值得" if (not ck["a2019"] and not ck["afull"]) else "有條件")
        cells[cc] = o
        for r_ in tr.itertuples():
            trade_rows.append((cc, r_.code, r_.buy_date, r_.sell_date, round(r_.ret, 6), round(r_.pnl, 1), int(r_.hold),
                               int(r_.delist), int(r_.reduced), int(Mk[r_.tb, r_.i]), r_.why))
        print(cc, round(o["ann_2019"], 4), round(o["ann_full"], 4), round(o["ann_1923"], 4), round(o["ann_2024"], 4),
              round(o["mdd_full"], 4), o["n_trades"], round(o["top5_share_of_gain"], 3), o["n_checks_pass"], o["eight_cls"],
              "rss", rss(), round(time.time() - t_start), flush=True)
    out["cells"] = cells
    tdf = pd.DataFrame(trade_rows, columns=["cell", "code", "buy_date", "sell_date", "ret", "pnl", "hold", "delist",
                                            "reduced", "mkt_entry", "why"])
    tdf.to_csv(os.path.join(odir, "my_trades.csv.gz"), index=False, compression="gzip")

    # ================================================================ 6. 20 條路徑
    rows = []
    for cc in CELLS:
        for sh in SHIFTS:
            for cp in CAPS:
                x = main[cc] if (sh == 0 and cp == 1_000_000.0) else sim(cc, sh, cp)
                nav = x["nav"]
                rows.append(dict(cell=cc, shift=sh, capital=cp, **nav_stats(nav), n_trades=int(len(x["trades"])),
                                 top5_share=top5(x["trades"], nav), blk_cap=int(x["cnt"]["liq_capped"]),
                                 **{"bear_" + k: v for k, v in bears(nav).items()}))
        print("paths", cc, "rss", rss(), round(time.time() - t_start), flush=True)
    pr = pd.DataFrame(rows)

    def p20_of(g):
        return dict(ann_2019_median=float(g.ann_2019.median()), ann_2019_min=float(g.ann_2019.min()),
                    ann_2019_max=float(g.ann_2019.max()), n_beat_0050=int((g.ann_2019 > b50["ann_2019"]).sum()),
                    mdd_median=float(g.mdd_full.median()), ann_full_median=float(g.ann_full.median()),
                    ann_2024_median=float(g.ann_2024.median()), ann_1923_median=float(g.ann_1923.median()),
                    top5_share_median=float(g.top5_share.median()), n_trades_median=float(g.n_trades.median()),
                    blk_cap_by_capital={int(k): float(v) for k, v in g.groupby("capital").blk_cap.mean().items()},
                    bears_median={k: float(g["bear_" + k].median()) for k in BEARS})
    p20 = {cc: p20_of(g) for cc, g in pr.groupby("cell")}

    PCOLS = ["ann_2019", "ann_full", "ann_1923", "ann_2024", "mdd_full", "top5_share"] + ["bear_" + k for k in BEARS]

    def paired(prx, a, b):
        o = {}
        xa = prx[prx.cell == a].set_index(["shift", "capital"])
        xb = prx[prx.cell == b].set_index(["shift", "capital"]).loc[xa.index]
        for col in PCOLS:
            d = (xa[col] - xb[col]).values
            o[col] = dict(n_pos=int((d > 0).sum()), median=float(np.median(d)), min=float(d.min()), max=float(d.max()))
        return o
    PAIRS = [("ALL", "CUR"), ("OTC", "CUR"), ("ALL_O20", "CUR"), ("ALL_O20", "ALL"), ("ALL_RAW", "ALL"), ("ALL_RAW", "CUR")]
    pair = {"%s-%s" % ab: paired(pr, *ab) for ab in PAIRS}

    # ================================================================ 7. 事前判定
    def main_rule(pc):
        npos = pc["ann_2019"]["n_pos"]; m1 = pc["ann_1923"]["median"]; m2 = pc["ann_2024"]["median"]
        md = pc["mdd_full"]["median"]
        nseg = int(m1 >= 0) + int(m2 >= 0)
        if npos >= 15:
            if md < -0.02:
                cls = "報酬較高，但回撤代價超過 2 個百分點"
            elif nseg == 2:
                cls = "加上櫃較好"
            elif nseg == 1:
                cls = "報酬較高，但只靠一段"
            else:
                cls = "差不多／不確定（≥15 條為正但兩段中位數都 < 0）"
        elif npos <= 5:
            cls = "加上櫃較差"
        else:
            cls = "差不多／不確定"
        return dict(n_pos_2019=npos, median_2019=pc["ann_2019"]["median"], seg_1923_median=m1, seg_2024_median=m2,
                    mdd_diff_median=md, cls=cls)

    dec = {}
    dec["main_ALL_vs_CUR"] = main_rule(pair["ALL-CUR"])
    dec["sens_ALL_O20_vs_CUR"] = main_rule(pair["ALL_O20-CUR"])
    dec["sens_ALL_O20_vs_ALL"] = main_rule(pair["ALL_O20-ALL"])
    dec["data_ALL_RAW_vs_ALL"] = main_rule(pair["ALL_RAW-ALL"])
    pt = pair["ALL-CUR"]["top5_share"]
    xa = pr[pr.cell == "ALL"].set_index(["shift", "capital"]); xb = pr[pr.cell == "CUR"].set_index(["shift", "capital"]).loc[xa.index]
    nlow = int((xa.top5_share < xb.top5_share).sum())
    dec["concentration"] = dict(n_lower=nlow, paired_median=pt["median"], all_median=p20["ALL"]["top5_share_median"],
                                cur_median=p20["CUR"]["top5_share_median"],
                                cls=("較分散" if (nlow >= 15 and pt["median"] < 0) else ("較集中" if nlow <= 5 else "差不多")))
    bc = {}
    for k in BEARS:
        v = pair["ALL-CUR"]["bear_" + k]
        bc[k] = dict(n_shallower=v["n_pos"], median=v["median"],
                     cls=("較淺" if (v["n_pos"] >= 15 and v["median"] >= 0.01) else
                          ("較深" if (v["n_pos"] <= 5 and v["median"] <= -0.01) else "差不多")))
    dec["bears_ALL_vs_CUR"] = bc
    po = p20["OTC"]
    dec["OTC_alone"] = dict(median=po["ann_2019_median"], n_beat_0050=po["n_beat_0050"], e50=b50["ann_2019"],
                            cls=("SURGE 在上櫃有效" if (po["ann_2019_median"] > b50["ann_2019"] and po["n_beat_0050"] >= 15) else
                                 ("在上櫃無效" if (po["ann_2019_median"] < b50["ann_2019"] and po["n_beat_0050"] <= 5) else "不確定")),
                            paired_vs_CUR=main_rule(pair["OTC-CUR"]))
    dec["eight_checks"] = {cc: (cells[cc]["n_checks_pass"], cells[cc]["eight_cls"]) for cc in CELLS}
    mc = dec["main_ALL_vs_CUR"]["cls"]
    if mc == "加上櫃較好" and cells["ALL"]["eight_cls"] == "值得上 XQ 測":
        xq = "給 XQ 的做法（20／21 號腳本股池改成上市＋上櫃）"
    elif mc.startswith("報酬較高"):
        xq = "現行不改（股池維持只上市）；上市＋上櫃可選、紙上追蹤"
    else:
        xq = "現行不改（股池維持只上市）"
    dec["xq"] = xq
    out["paths20"] = p20
    out["paired"] = pair
    out["decisions"] = dec
    print("decisions", json.dumps(dec, ensure_ascii=False, default=jdef), flush=True)

    # ================================================================ 8. ALL 分市場：每筆期望值、報酬貢獻
    def mkt_split(cc):
        x = main[cc]; tr = x["trades"].copy(); nav = x["nav"]
        tr["mk"] = Mk[tr.tb.values, tr.i.values]
        o = {}
        for nm_, g in (("full", tr), ("from2019", tr[tr.tb > i19]), ("2019_2023", tr[(tr.tb > i19) & (tr.tb <= i24)]),
                       ("from2024", tr[tr.tb > i24])):
            o[nm_] = {MKT[int(k)]: dict(tstats(h), pnl=float(h.pnl.sum())) for k, h in g.groupby("mk")}
        spans = {}
        for r_ in tr.itertuples():
            spans.setdefault(r_.i, []).append((r_.tb, r_.ts, int(r_.mk)))
        for code, bd, cost, val, i in x["open_pos"]:
            tb = int(np.searchsorted(da, bd))
            spans.setdefault(i, []).append((tb, e + 1, int(Mk[tb, i])))

        def mk_at(i, t):
            for a, b_, mv in spans.get(i, ()):
                if a <= t <= b_:
                    return mv
            return -1
        snap, flows = x["snap"], x["flows"]
        per = {"full": (t0 + 1, e), "from2019": (i19 + 1, e), "2019_2023": (i19 + 1, i24), "from2024": (i24 + 1, e)}
        res = {}
        for pn, (a, b_) in per.items():
            mny = {1: 0.0, 2: 0.0, -1: 0.0}; con = {1: 0.0, 2: 0.0, -1: 0.0}
            for t in range(a, b_ + 1):
                prev = snap.get(t - 1, {}); cur = snap.get(t, {})
                for i in set(prev) | set(cur):
                    pnl = cur.get(i, 0.0) - prev.get(i, 0.0) + flows.get((t, i), 0.0)
                    mv = mk_at(i, t)
                    mny[mv] += pnl; con[mv] += pnl / nav[t - 1]
            tm = sum(mny.values()); tc = sum(con.values())
            res[pn] = dict(money={MKT.get(k, "?"): v for k, v in mny.items()},
                           money_share={MKT.get(k, "?"): (v / tm if tm else None) for k, v in mny.items()},
                           contrib={MKT.get(k, "?"): v for k, v in con.items()},
                           contrib_share={MKT.get(k, "?"): (v / tc if tc else None) for k, v in con.items()},
                           contrib_total=tc, nav_ann=L.ann(P, nav, a - 1, b_))
        o["contribution"] = res
        # 檢查：每日個股損益加總＝個股淨值變動（現金流已含）——用 flows 中不在 snap 的鍵確認
        miss = [k for k in flows if k[1] not in snap.get(k[0], {}) and k[1] not in snap.get(k[0] - 1, {})]
        o["flows_outside_snap"] = len(miss)
        return o
    out["market_split"] = {cc: mkt_split(cc) for cc in ("ALL",)}
    # 持有期間轉換市場的交易（ALL、ALL_O20 主跑；含期末未平倉）
    swi = {}
    for cc in ("ALL", "ALL_O20", "CUR", "OTC"):
        x = main[cc]; tr = x["trades"]
        segs = [(int(r_.i), int(r_.tb), min(int(r_.ts), e)) for r_ in tr.itertuples()]
        segs += [(int(i), int(np.searchsorted(da, bd)), e) for code, bd, cost, val, i in x["open_pos"]]
        hits = []
        for i, a, b in segs:
            mv = Mk[a:b + 1, i]; mv = mv[mv > 0]
            if len(mv) and (mv != mv[0]).any():
                hits.append((codes[i], P.dstr[a], P.dstr[b]))
        swi[cc] = dict(n=len(hits), list=hits)
    out["market_switch_during_holding"] = swi
    out["data_check"]["91xx_note"] = ("另以原始日 K 掃描：91xx 共 7 檔（皆上市），其中 3 檔在資料結束前下市（9106 2017-06-26、"
                                      "9157 2019-11-11、9188 2022-03-03）；複製員面板不收 91xx，所以上市下市 70 檔＋3＝研究員 73。")
    ms = out["market_split"]["ALL"]
    print("ALL by market", json.dumps({k: {m: (v["n"], round(v["exp"], 4)) for m, v in ms[k].items()}
                                       for k in ("full", "from2019")}),
          json.dumps({k: v["contrib_share"] for k, v in ms["contribution"].items()}), flush=True)

    # ================================================================ 9. 審查 A：上櫃期望值拆解（扣成本前）
    Utw, Utp = U["CUR"], U["OTC"]
    op_ok = np.isfinite(P.O)

    def pool_ew(Um, s, a, b):
        idx = np.where(Um[s] & op_ok[a])[0]
        bb = AO[b, idx] if b <= e else Gf[e, idx]
        return float(np.nanmean(bb / AO[a, idx] - 1.0)), len(idx)

    xA = main["ALL"]; trA = xA["trades"].copy()
    trA["mk"] = Mk[trA.tb.values, trA.i.values]
    gr = []; pw = []; pwo = []; r5 = []
    for r_ in trA.itertuples():
        a, b = int(r_.tb), int(r_.ts)
        gr.append(float(AO[b, r_.i] / AO[a, r_.i] - 1.0))
        own = Utw if r_.mk == 1 else Utp
        oth = Utp if r_.mk == 1 else Utw
        pw.append(pool_ew(own, a - 1, a, b)[0]); pwo.append(pool_ew(oth, a - 1, a, b)[0])
        r5.append(float(P.e50o[b] / P.e50o[a] - 1.0))
    trA["gross"] = gr; trA["pool_own"] = pw; trA["pool_other"] = pwo; trA["r50"] = r5
    revA = {"trade_level": {}}
    for nm_, g in (("full", trA), ("from2019", trA[trA.tb > i19]), ("2019_2023", trA[(trA.tb > i19) & (trA.tb <= i24)]),
                   ("from2024", trA[trA.tb > i24])):
        o = {}
        for k, h in g.groupby("mk"):
            o[MKT[int(k)]] = dict(n=int(len(h)), exp_net=float(h.ret.mean()), gross=float(h.gross.mean()),
                                  cost_drag=float((h.gross - h.ret).mean()), pool_own_ew=float(h.pool_own.mean()),
                                  pool_other_ew=float(h.pool_other.mean()), r50=float(h.r50.mean()),
                                  selection=float((h.gross - h.pool_own).mean()),
                                  pool_vs_0050=float((h.pool_own - h.r50).mean()),
                                  own_vs_other_pool=float((h.pool_own - h.pool_other).mean()),
                                  hold_mean=float(h.hold.mean()))
        revA["trade_level"][nm_] = o
    # 同一套交易視窗以外：每月時窗（訊號列 s → 執行 s＋1 到下一個執行日），扣成本前
    ws = sorted(lists[("ALL", 0)])
    wrow = []
    bA = plans[("ALL", 0)][0]; bO = plans[("OTC", 0)][0]; bC = plans[("CUR", 0)][0]
    for n_, s in enumerate(ws):
        ee = s + 1; e2 = ws[n_ + 1] + 1 if n_ + 1 < len(ws) else None
        op = op_ok[ee]
        r50 = K.e50_win(P, ee, e2)
        ew_tw = K.win_ret(P, AO, Gf, np.where(Utw[s] & op)[0], ee, e2)
        ew_tp = K.win_ret(P, AO, Gf, np.where(Utp[s] & op)[0], ee, e2)
        la = np.array([i for i in bA.get(s, []) if op[i]], dtype=int)
        la_tp = la[Mk[s, la] == 2] if len(la) else la
        la_tw = la[Mk[s, la] == 1] if len(la) else la
        lo = np.array([i for i in bO.get(s, []) if op[i]], dtype=int)
        lc = np.array([i for i in bC.get(s, []) if op[i]], dtype=int)
        wrow.append(dict(exec=P.dstr[ee], end=P.dstr[e2] if e2 is not None else P.dstr[e], year=int(P.dstr[ee][:4]),
                         r50=r50, ew_tw=ew_tw, ew_tp=ew_tp,
                         l_all_tp=K.win_ret(P, AO, Gf, la_tp, ee, e2) if len(la_tp) else ew_tp,
                         l_all_tw=K.win_ret(P, AO, Gf, la_tw, ee, e2) if len(la_tw) else ew_tw,
                         l_otc=K.win_ret(P, AO, Gf, lo, ee, e2) if len(lo) else ew_tp,
                         l_cur=K.win_ret(P, AO, Gf, lc, ee, e2) if len(lc) else ew_tw,
                         n_all_tp=len(la_tp), n_all_tw=len(la_tw), n_otc=len(lo), n_cur=len(lc)))
    wdf = pd.DataFrame(wrow)

    def chain(g, col):
        v = float(np.prod(1.0 + g[col].values))
        yrs = (pd.Timestamp(g.end.iloc[-1]) - pd.Timestamp(g.exec.iloc[0])).days / 365.25
        return v ** (1.0 / yrs) - 1.0
    wo = {}
    for nm_, g in (("full", wdf), ("2019_2023", wdf[(wdf.year >= 2019) & (wdf.year <= 2023)]), ("from2024", wdf[wdf.year >= 2024]),
                   ("from2019", wdf[wdf.year >= 2019])):
        a = {c: chain(g, c) for c in ("r50", "ew_tw", "ew_tp", "l_all_tp", "l_all_tw", "l_otc", "l_cur")}
        wo[nm_] = dict(ann=a, otc_pool_vs_twse_pool=a["ew_tp"] - a["ew_tw"], otc_pool_vs_0050=a["ew_tp"] - a["r50"],
                       twse_pool_vs_0050=a["ew_tw"] - a["r50"],
                       sel_otc_in_ALL=a["l_all_tp"] - a["ew_tp"], sel_twse_in_ALL=a["l_all_tw"] - a["ew_tw"],
                       sel_OTC_cell=a["l_otc"] - a["ew_tp"], sel_CUR=a["l_cur"] - a["ew_tw"],
                       months=int(len(g)), months_no_otc_in_ALL=int((g.n_all_tp == 0).sum()),
                       start=g.exec.iloc[0], end=g.end.iloc[-1],
                       monthly_mean_sel_otc_in_ALL=float((g.l_all_tp - g.ew_tp)[g.n_all_tp > 0].mean()),
                       monthly_mean_sel_twse_in_ALL=float((g.l_all_tw - g.ew_tw)[g.n_all_tw > 0].mean()))
    revA["monthly_windows"] = wo
    revA["note"] = ("trade_level：ALL 主跑每筆已平倉交易（市場別看進場日），gross＝還原開盤（含息）進場到出場的報酬；"
                    "pool_own_ew＝同市場股池（訊號日、進場日有開盤）同一持有期等權；selection＝gross − pool_own_ew；"
                    "cost_drag＝gross − 淨報酬（含成本、部分賣出差異）。monthly_windows：每個換股時窗（執行日到下一個執行日）"
                    "名單等權與同池等權的年化串接（名單空月用同池等權代替），扣成本前。")
    out["reviewA"] = revA
    print("reviewA", json.dumps({k: {m: (v["n"], round(v["exp_net"], 4), round(v["pool_own_ew"], 4), round(v["selection"], 4),
                                         round(v["cost_drag"], 4)) for m, v in d.items()}
                                 for k, d in revA["trade_level"].items()}),
          json.dumps({k: {kk: round(vv, 4) for kk, vv in v.items() if isinstance(vv, float)} for k, v in wo.items()}),
          flush=True)

    # ================================================================ 10. 審查 B：上市優先名額（事後）
    pb = {}
    for sh in SHIFTS:
        pb[("B1", sh)] = K.prio_plans(lists[("CUR", sh)], lists[("OTC", sh)])
        pb[("B3", sh)] = K.prio_plans(lists[("CUR", sh)], lists[("ALL", sh)],
                                      xfilter=(lambda s, i: Mk[s, i] == 2))
    rowsB = []
    revB = {}
    for vb in ("B1", "B3"):
        for sh in SHIFTS:
            b, k, nf = pb[(vb, sh)]
            for cp in CAPS:
                x = sim(vb, sh, cp, pl=(b, k))
                nav = x["nav"]
                rowsB.append(dict(cell=vb, shift=sh, capital=cp, **nav_stats(nav), n_trades=int(len(x["trades"])),
                                  top5_share=top5(x["trades"], nav), blk_cap=int(x["cnt"]["liq_capped"]),
                                  **{"bear_" + kk: v for kk, v in bears(nav).items()}))
                if sh == 0 and cp == 1_000_000.0:
                    tr = x["trades"]
                    mkv = Mk[tr.tb.values, tr.i.values]
                    nfa = np.array([nf[s] for s in sorted(nf)])
                    revB[vb] = dict(main=dict(nav_stats(nav), n_trades=int(len(tr)), n_trades_tpex=int((mkv == 2).sum()),
                                              exp=float(tr.ret.mean()),
                                              exp_twse=float(tr.ret[mkv == 1].mean()), exp_tpex=float(tr.ret[mkv == 2].mean()),
                                              avg_npos=float(x["npos"][t0 + 1:].mean()), top5_share=top5(tr, nav),
                                              bears=bears(nav)),
                                    fill_mean=float(nfa.mean()), fill_months=int((nfa > 0).sum()), months=int(len(nfa)))
    prB = pd.concat([pr, pd.DataFrame(rowsB)], ignore_index=True)
    for vb in ("B1", "B3"):
        revB[vb]["p20"] = p20_of(prB[prB.cell == vb])
        revB[vb]["paired_vs_CUR"] = paired(prB, vb, "CUR")
        revB[vb]["paired_vs_ALL"] = paired(prB, vb, "ALL")
        revB[vb]["rule_vs_CUR_descriptive"] = main_rule(revB[vb]["paired_vs_CUR"])
    revB["note"] = ("事後、只描述。B1：買進名單＝上市 F1（上市 L_MA 前 10 名中 RET20 > 10%），不足 10 檔時接上上櫃 F1"
                    "（只上櫃股池 L_MA 前 10 名中 RET20 > 10%，依 SURGE），合計最多 10 檔；續抱＝上市 L_MA 前 20 ∪ 上櫃 L_MA 前 20。"
                    "B3：遞補來源改為 ALL 合併排名 F1 名單中的上櫃股（續抱＝上市 L_MA 前 20 ∪ ALL L_MA 前 20 中的上櫃股）。"
                    "引擎、成本、W10_70 同主跑。")
    out["reviewB"] = revB
    print("reviewB", json.dumps({vb: (revB[vb]["main"]["ann_2019"], revB[vb]["main"]["ann_full"], revB[vb]["p20"]["ann_2019_median"],
                                      revB[vb]["paired_vs_CUR"]["ann_2019"]["n_pos"], revB[vb]["paired_vs_CUR"]["ann_2019"]["median"],
                                      revB[vb]["fill_mean"]) for vb in ("B1", "B3")}), flush=True)
    prB.to_csv(os.path.join(odir, "my_paths.csv.gz"), index=False, float_format="%.6f", compression="gzip")

    # ================================================================ 11. 審查 C：下市股營收缺失
    bA0, kA0 = plans[("ALL", 0)]
    lA0 = lists[("ALL", 0)]
    sigs = sorted(lA0)
    ever = U["ALL"][t0:].any(axis=0)
    dli = [int(i) for i in dl if ever[i]]
    has_rev = np.isfinite(Rv_p).any(axis=0)
    last_rev_m = np.array([np.where(np.isfinite(Rv_p[:, i]))[0].max() if has_rev[i] else -1 for i in range(P.N)])
    # 基準率：存活股（非下市）在訊號日「價格條件成立」（股池內、G > 季線、RET20 > 10%）且 SURGE 有值者，被選入 F1 名單的比例
    n_po_s = 0; n_sel_s = 0; fw_sel = []; fw_po = []
    rec = []
    dset = set(dli)
    isdl = np.isin(np.arange(P.N), dli)
    for n_, s in enumerate(sigs):
        ee = s + 1; e2 = sigs[n_ + 1] + 1 if n_ + 1 < len(sigs) else None
        m = lA0[s]["m"]; sv = np.isfinite(S_p[m])
        with np.errstate(invalid="ignore"):
            po = U["ALL"][s] & (P.G[s] > ma60[s]) & (lA0[s]["r20"] > 0.10) & op_ok[ee]
        sel = np.zeros(P.N, bool); sel[bA0[s]] = True
        surv = po & sv & ~isdl
        n_po_s += int(surv.sum()); n_sel_s += int((surv & sel).sum())
        bfw = AO[e2] if e2 is not None else Gf[e]
        fr = bfw / AO[ee] - 1.0
        if (sel & op_ok[ee]).any():
            fw_sel.append(float(np.nanmean(fr[sel & op_ok[ee]])))
        fw_po.append(float(np.nanmean(fr[surv])) if surv.any() else np.nan)
        for i in np.where(po)[0]:
            if int(i) in dset and s > last[i] - 250:
                rec.append(dict(code=codes[i], mkt=MKT[int(Mk[s, i])], sig=P.dstr[s], fwd=float(fr[i]),
                                surge_ok=bool(sv[i]), selected=bool(sel[i]), to_delist=float(Gf[e, i] / AO[ee, i] - 1.0),
                                last=P.dstr[last[i]]))
    base_rate = n_sel_s / n_po_s
    rc = pd.DataFrame(rec)
    dlist = []
    for i in dli:
        mk_ = int(Mk[last[i], i])
        a = max(int(last[i]) - 250, 0)
        r12 = float(P.G[last[i], i] / P.G[a, i] - 1.0) if np.isfinite(P.G[a, i]) else None
        a3 = max(int(last[i]) - 60, 0)
        r3 = float(P.G[last[i], i] / P.G[a3, i] - 1.0) if np.isfinite(P.G[a3, i]) else None
        inpool12 = int(U["ALL"][a:last[i] + 1, i].sum())
        h = rc[rc.code == codes[i]] if len(rc) else rc
        lrm = int(last_rev_m[i])
        dlist.append(dict(code=codes[i], mkt=MKT[mk_], name=P.names.get(codes[i], ""), last=P.dstr[last[i]],
                          has_rev=bool(has_rev[i]),
                          last_rev=("%04d-%02d" % ((mi0 + lrm) // 12, (mi0 + lrm) % 12 + 1)) if lrm >= 0 else None,
                          inpool_days_12m=inpool12, ret_12m_to_last=r12, ret_3m_to_last=r3,
                          n_sig_price_ok_12m=int(len(h)), n_sig_price_ok_surge_ok=int(h.surge_ok.sum()) if len(h) else 0,
                          n_selected=int(h.selected.sum()) if len(h) else 0,
                          fwd_mean=float(h.fwd.mean()) if len(h) else None))
    dld = pd.DataFrame(dlist)
    years_sig = (P.dates[e] - P.dates[sigs[0]]).days / 365.25
    fw_sel_m = float(np.mean(fw_sel)); fw_po_m = float(np.nanmean(fw_po))
    revC = dict(n_delisted_ever_in_ALL_pool=len(dli),
                by_market={m: int((dld.mkt == m).sum()) for m in ("twse", "tpex")},
                with_any_revenue={m: int(((dld.mkt == m) & dld.has_rev).sum()) for m in ("twse", "tpex")},
                inpool_within_12m={m: int(((dld.mkt == m) & (dld.inpool_days_12m > 0)).sum()) for m in ("twse", "tpex")},
                price_ok_stocks={m: int(((dld.mkt == m) & (dld.n_sig_price_ok_12m > 0)).sum()) for m in ("twse", "tpex")},
                price_ok_stock_months={m: int(dld[dld.mkt == m].n_sig_price_ok_12m.sum()) for m in ("twse", "tpex")},
                selected_actual=int(dld.n_selected.sum()),
                base_rate_selected_given_price_ok_surviving=base_rate, n_price_ok_surviving=n_po_s, n_selected_surviving=n_sel_s,
                fwd_1m_mean_selected=fw_sel_m, fwd_1m_mean_price_ok_surviving=fw_po_m,
                fwd_1m_mean_delisted_price_ok=float(rc.fwd.mean()) if len(rc) else None,
                fwd_1m_median_delisted_price_ok=float(rc.fwd.median()) if len(rc) else None,
                fwd_1m_delisted_price_ok_by_mkt={m: dict(n=int(len(g)), mean=float(g.fwd.mean()), median=float(g.fwd.median()),
                                                         min=float(g.fwd.min()), share_neg=float((g.fwd < 0).mean()))
                                                 for m, g in rc.groupby("mkt")} if len(rc) else {},
                ret_12m_to_last_median={m: float(dld[dld.mkt == m].ret_12m_to_last.median()) for m in ("twse", "tpex")},
                ret_3m_to_last_median={m: float(dld[dld.mkt == m].ret_3m_to_last.median()) for m in ("twse", "tpex")},
                years=years_sig)
    # 影響估計：預期多選到的筆數 × (該股 1 個月報酬 − 名單平均 1 個月報酬) × 每檔權重 7%，攤到年
    if len(rc):
        exp_picks = len(rc) * base_rate
        dlt = float(rc.fwd.mean()) - fw_sel_m
        imp_expected = exp_picks * 0.07 * dlt / years_sig
        worst = rc.nsmallest(max(1, int(round(exp_picks))), "fwd")
        imp_worst = float(((worst.fwd - fw_sel_m) * 0.07).sum()) / years_sig
        imp_all = float(((rc.fwd - fw_sel_m) * 0.07).sum()) / years_sig
        revC["impact"] = dict(expected_picks=exp_picks, delta_vs_selected=dlt,
                              ann_impact_expected=imp_expected, ann_impact_worst_k=imp_worst, ann_impact_if_all_picked=imp_all,
                              by_mkt={m: dict(expected_picks=len(g) * base_rate,
                                              ann_impact_expected=len(g) * base_rate * 0.07 * (float(g.fwd.mean()) - fw_sel_m) / years_sig)
                                      for m, g in rc.groupby("mkt")})
    revC["list"] = dld.to_dict("records")
    revC["note"] = ("下市＝最後一根日 K 早於資料結束前 5 個交易日；只列 2015-07 起曾入 ALL 股池者（價格面板 2015 年起，2013～2014 下市的不在面板）。"
                    "價格條件＝ALL 股池內、G > 60 日均線、RET20 > 10%、隔日有開盤（F1 除了 SURGE 以外的條件），在下市前 250 個交易日內的 A10 訊號日。"
                    "基準率＝存活股中價格條件成立且 SURGE 有值者被選入 ALL 的 F1 名單的比例。fwd＝執行日到下一個執行日的還原開盤報酬（下市則到最後市值）。"
                    "影響＝預期被選筆數 ×（fwd − 名單平均 fwd）× 7%（W10_70 每檔目標）÷ 年數；只是量級估計。")
    out["reviewC"] = revC
    print("reviewC", json.dumps({k: v for k, v in revC.items() if k not in ("list",)}, ensure_ascii=False, default=jdef)[:2500],
          flush=True)

    # ================================================================ 12. 事後小項
    out["posthoc"] = dict(yearly={cc: cells[cc]["yearly"] for cc in CELLS},
                          yearly_ALL_minus_CUR={y: cells["ALL"]["yearly"][y] - cells["CUR"]["yearly"][y]
                                                for y in cells["CUR"]["yearly"]})
    out["resources"] = dict(peak_rss_mb=int(rss()), seconds=round(time.time() - t_start, 1))
    js = json.dumps(out, ensure_ascii=False, indent=1, default=jdef)
    open(os.path.join(odir, "my_numbers.json"), "w").write(js)
    h = hashlib.sha256(js.encode("utf-8")).hexdigest()
    open(os.path.join(odir, "my_numbers.sha256"), "w").write(h + "  my_numbers.json\n")
    print("frozen", h, "rss", rss(), round(time.time() - t_start), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else HERE)
