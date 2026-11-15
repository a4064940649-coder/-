# -*- coding: utf-8 -*-
"""S12 獨立複製：凍結後比對（my_numbers.json 已寫好 sha256 之後才執行）。
讀研究員 FINDINGS.md 全文的數字（手抄在 RS 字典）與 trades.csv.gz；不讀 results*.json、navs.csv.gz。
另做凍結後才加的對照（標「凍結後」）：研究員「實際權重」拆法的重算、β 的時窗口徑敏感度、Q55 大帳戶占比的月底取樣。
用法：python3 -I -B rep_s12_compare.py <scratch_dir>
輸出：compare_vs_researcher.json
"""
import sys, os, json, hashlib
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import rep_s12_lib as K
S, Q, R, L = K.S, K.Q, K.R, K.L
SRC = os.path.dirname(HERE)

# 研究員 FINDINGS 第 3～8 節（%／個百分點）
RS = dict(
    main={"CUR": (48.13, 34.08, 37.01, 70.16, -33.40, 379), "TL": (44.48, 30.48, 27.11, 81.42, -42.58, 184),
          "TM": (25.22, 20.28, 21.02, 33.03, -38.97, 315), "TS": (35.98, 26.62, 31.94, 43.47, -38.98, 435),
          "Q55": (45.34, 32.12, 36.05, 63.46, -34.89, 464), "TL_ADV": (38.82, 26.57, 31.35, 53.16, -49.79, 272)},
    p20={"CUR": (46.23, 43.24, 48.51, 20, -34.01, 24.4), "TL": (42.48, 40.82, 44.66, 20, -41.03, 27.7),
         "TM": (26.90, 24.53, 28.17, 0, -38.45, 31.3), "TS": (36.70, 31.59, 38.58, 20, -39.38, 23.0),
         "Q55": (41.83, 34.61, 46.58, 20, -39.50, 29.0), "TL_ADV": (42.15, 38.82, 44.86, 20, -49.80, 29.6)},
    top5_block={"CUR": (24.9, 4.70), "TL": (29.4, 4.35), "TM": (25.4, 2.83), "TS": (22.4, 4.96), "Q55": (29.0, 4.20),
                "TL_ADV": (28.0, 4.45)},
    paired={"TL": ((0, -3.34), (0, -8.44), (18, 9.14), (0, -7.05), 0), "TM": ((0, -19.23), (0, -14.57), (0, -28.25), (2, -4.65), 0),
            "TS": ((0, -10.07), (0, -3.96), (0, -23.59), (0, -5.98), 10), "Q55": ((3, -5.44), (12, 1.00), (0, -17.73), (0, -4.70), 2),
            "TL_ADV": ((0, -3.53), (3, -3.06), (5, -6.84), (0, -14.41), 0)},
    sel={"ALL": (66.8, 90.1, 79.8, 110.0, 4.36, 10, 4), "L": (27.2, 22.1, 3.0, 69.6, 2.61, 6, 16),
         "M": (24.7, 22.7, 24.8, 19.1, 2.50, 9, 3), "S": (39.5, 52.6, 67.3, 29.7, 3.19, 9, 3)},
    sel_shift_full={"L": (26.7, 15.1, 26.7), "M": (13.6, 27.1, 24.8), "S": (38.2, 25.6, 15.0), "ALL": (55.1, 60.5, 41.0)},
    L_yearly={2016: -18.5, 2017: 46.7, 2018: 108.0, 2019: -14.5, 2020: 36.9, 2021: 95.4, 2022: -21.7, 2023: -30.7,
              2024: -2.9, 2025: 110.0, 2026: 126.4},
    decomp2019={"CUR": (48.75, 29.12, 19.63, -7.69, 61.76, -34.44), "TL": (44.74, 29.12, 15.62, -3.07, 16.44, 2.25),
                "TM": (25.46, 29.12, -3.65, -7.53, 17.41, -13.53), "TS": (36.27, 29.12, 7.16, -8.96, 38.02, -21.91)},
    decomp_actual={"from2019": {"CUR": (23.76, 5.67, -2.12), "TL": (12.33, 2.64, 3.71), "TM": (4.96, 5.36, -6.44),
                                "TS": (15.06, 6.75, -5.69)},
                   "2019_2023": {"CUR": (1.68, 20.76, 0.77, -2.38, 20.83)}, "from2024": {"CUR": (-28.15, 29.71, 16.18, -1.47, 16.28)}},
    w10_dil_CUR=-32.32,
    beta={"CUR": 0.23, "TL": 0.18, "TM": 0.35, "TS": 0.45, "Q55": 0.34, "TL_ADV": 0.36},
    bears_med={"CUR": (-16.79, -26.47, -29.69, -34.01), "TL": (-17.05, -28.12, -31.12, -28.73),
               "TM": (-16.69, -28.93, -33.29, -34.96), "TS": (-25.52, -28.77, -23.86, -39.38),
               "Q55": (-18.13, -26.32, -27.57, -39.50), "TL_ADV": (-21.05, -29.56, -36.16, -32.97)},
    bears_cls={"TL": ("差不多", "較深", "差不多", "較淺"), "TM": ("差不多", "較深", "較深", "差不多"),
               "TS": ("較深", "較深", "較淺", "較深"), "Q55": ("差不多", "差不多", "差不多", "較深"),
               "TL_ADV": ("較深", "較深", "較深", "較淺")},
    eight={"CUR": "值得上 XQ 測", "TL": "有條件", "TM": "不值得", "TS": "值得上 XQ 測", "Q55": "值得上 XQ 測", "TL_ADV": "值得上 XQ 測"},
    decisions=dict(TL="SURGE 在大型股有效", sel=dict(L="弱", M="選股超額存在", S="選股超額存在", ALL="選股超額存在"),
                   Q55="沒有降低依賴", TL_dep="沒有降低依賴", beta="沒有明顯降低", xq="不建議改現行"),
    cur_tier_share={"full": (64, 19, 17), "2019_2023": (19, 39, 42), "from2024": (76, 13, 11)},
    q55_drift=(27, 51),
)


def main(sc):
    my = json.load(open(os.path.join(HERE, "my_numbers.json")))
    h = hashlib.sha256(open(os.path.join(HERE, "my_numbers.json"), "rb").read()).hexdigest()
    frozen = open(os.path.join(HERE, "my_numbers.sha256")).read().split()[0]
    assert h == frozen
    out = dict(my_numbers_sha256=h)
    pc = lambda x: None if x is None else round(100 * x, 2)

    # ---------------- 主跑
    tab = {}
    worst = dict(ann=0.0, mdd=0.0, ntr=0.0)
    for cc, r in RS["main"].items():
        c = my["cells"][cc]
        m = (pc(c["ann_2019"]), pc(c["ann_full"]), pc(c["ann_1923"]), pc(c["ann_2024"]), pc(c["mdd_full"]), c["n_trades"])
        d = tuple(round(a - b, 2) for a, b in zip(m, r))
        tab[cc] = dict(mine=m, researcher=r, diff=d)
        worst["ann"] = max(worst["ann"], max(abs(x) for x in d[:4])); worst["mdd"] = max(worst["mdd"], abs(d[4]))
        worst["ntr"] = max(worst["ntr"], abs(d[5]) / r[5])
    out["main"] = tab; out["main_worst"] = worst
    p20 = {}
    for cc, r in RS["p20"].items():
        c = my["paths20"][cc]
        m = (pc(c["ann_2019_median"]), pc(c["ann_2019_min"]), pc(c["ann_2019_max"]), c["n_beat_0050"], pc(c["mdd_median"]),
             round(100 * c["top5_share_median"], 1))
        p20[cc] = dict(mine=m, researcher=r, diff=tuple(round(a - b, 2) for a, b in zip(m, r)))
    out["paths20"] = p20
    tb = {}
    for cc, r in RS["top5_block"].items():
        c = my["cells"][cc]
        tb[cc] = dict(mine=(round(100 * c["top5_share_of_gain"], 1), pc(c["block5"]["drop_2019"])), researcher=r)
    out["top5_block"] = tb
    pr = {}
    for cc, r in RS["paired"].items():
        c = my["paired_vs_CUR"]["%s-CUR" % cc]
        m = ((c["ann_2019"]["n_pos"], pc(c["ann_2019"]["median"])), (c["ann_1923"]["n_pos"], pc(c["ann_1923"]["median"])),
             (c["ann_2024"]["n_pos"], pc(c["ann_2024"]["median"])), (c["mdd_full"]["n_pos"], pc(c["mdd_full"]["median"])),
             20 - c["top5_share"]["n_pos"])
        pr[cc] = dict(mine=m, researcher=r)
    out["paired"] = pr
    sel = {}
    for tn, r in RS["sel"].items():
        c = my["S1_selection"]["shift0"][tn]
        m = (round(100 * c["full"]["diff"], 1), round(100 * c["from2019"]["diff"], 1), round(100 * c["2019_2023"]["diff"], 1),
             round(100 * c["from2024"]["diff"], 1), round(c["t_month"], 2), c["n_years_pos"], c["n_months_list_empty"])
        sel[tn] = dict(mine=m, researcher=r)
    out["selection"] = sel
    out["selection_shift_full"] = {tn: dict(mine=tuple(round(100 * my["S1_selection"]["shift%d" % s][tn]["full"]["diff"], 1)
                                                       for s in (1, 2, 3)), researcher=r)
                                   for tn, r in RS["sel_shift_full"].items()}
    ly = my["S1_selection"]["shift0"]["L"]["yearly_diff"]
    out["L_yearly"] = {y: dict(mine=round(100 * ly[str(y)], 1), researcher=v) for y, v in RS["L_yearly"].items()}
    dc = {}
    for cc, r in RS["decomp2019"].items():
        x = my["S7_decomposition"][cc]["from2019"]
        m = (pc(x["ann"]["act"]), pc(x["ann"]["r50"]), pc(x["total_vs_0050"]), pc(x["tier_lag"]), pc(x["selection"]),
             pc(x["exec_loss"]))
        dc[cc] = dict(mine=m, researcher=r)
    out["decomp_2019_prereg"] = dc
    out["w10_dilution_CUR_2019"] = dict(mine=pc(my["S7_decomposition"]["CUR"]["from2019"]["w10_dilution"]),
                                        researcher=RS["w10_dil_CUR"])
    out["beta"] = {cc: dict(mine=round(my["S6_beta"][cc]["beta"], 3), t=round(my["S6_beta"][cc]["t"], 2), researcher=v)
                   for cc, v in RS["beta"].items()}
    bm = {}
    ks = ["2018", "2020-03", "2022", "2024-07~2025-04"]
    for cc, r in RS["bears_med"].items():
        m = tuple(pc(my["paths20"][cc]["bears_median"][k]) for k in ks)
        bm[cc] = dict(mine=m, researcher=r, cls_mine=None if cc == "CUR" else tuple(my["decisions"]["bears"][cc][k]["cls"] for k in ks),
                      cls_researcher=RS["bears_cls"].get(cc))
    out["bears"] = bm
    dm = my["decisions"]
    out["decisions"] = dict(
        TL=dict(mine=dm["TL_surge_large"], researcher=RS["decisions"]["TL"]),
        selection=dict(mine=dm["selection_by_tier"], researcher=RS["decisions"]["sel"]),
        Q55=dict(mine=dm["Q55_dependency"]["cls"], researcher=RS["decisions"]["Q55"]),
        TL_dep=dict(mine=dm["TL_dependency_secondary"]["cls"], researcher=RS["decisions"]["TL_dep"]),
        beta=dict(mine={k: v["cls"] for k, v in dm["beta"].items()}, researcher=RS["decisions"]["beta"]),
        eight=dict(mine={k: v[1] for k, v in dm["eight_checks"].items()}, researcher=RS["eight"]),
        xq=dict(mine=dm["xq"], researcher=RS["decisions"]["xq"]))
    out["decisions"]["all_same"] = bool(
        dm["TL_surge_large"] == RS["decisions"]["TL"] and dm["selection_by_tier"] == RS["decisions"]["sel"] and
        dm["Q55_dependency"]["cls"] == RS["decisions"]["Q55"] and dm["TL_dependency_secondary"]["cls"] == RS["decisions"]["TL_dep"]
        and all(v["cls"] == RS["decisions"]["beta"] for v in dm["beta"].values()) and
        {k: v[1] for k, v in dm["eight_checks"].items()} == RS["eight"] and dm["xq"] == RS["decisions"]["xq"] and
        all(bm[cc]["cls_mine"] == bm[cc]["cls_researcher"] for cc in RS["bears_cls"]))
    ra = my["review_A"]
    out["cur_tier_share"] = {p: dict(mine_money_daily=tuple(round(100 * ra[p]["money_share"][t]) for t in "LMS"),
                                     mine_contrib_daily=tuple(round(100 * ra[p]["contrib_share"][t]) for t in "LMS"),
                                     researcher=r) for p, r in RS["cur_tier_share"].items()}

    # ---------------- 逐筆
    mt = pd.read_csv(os.path.join(HERE, "my_trades.csv.gz"), dtype={"code": str})
    rt = pd.read_csv(os.path.join(SRC, "trades.csv.gz"), dtype={"code": str})
    rt = rt[rt.why != "open_end"].copy()
    rt["sub"] = rt.sleeve.map(lambda s: {"Q55_L": "L", "Q55_MS": "MS"}.get(s, s))
    mt["sub"] = mt["sub"].astype(str)
    tr = {}
    for cc in RS["main"]:
        a = mt[mt.cell == cc].copy(); b = rt[rt.run == cc].copy()
        key = ["sub", "code", "buy_date"]
        b = b.rename(columns={"ent_date": "buy_date"})
        j = a.merge(b, on=key, how="outer", suffixes=("_m", "_r"), indicator=True)
        both = j[j._merge == "both"]
        tr[cc] = dict(n_mine=int(len(a)), n_researcher=int(len(b)), matched_entry=int(len(both)),
                      only_mine=int((j._merge == "left_only").sum()), only_researcher=int((j._merge == "right_only").sum()),
                      same_exit=int((both.sell_date == both.exit_date).sum()),
                      max_abs_ret_diff=float((both.ret_m - both.ret_r).abs().max()) if len(both) else None,
                      max_abs_pnl_diff=float((both.pnl_m - both.pnl_r).abs().max()) if len(both) else None,
                      rel_pnl_diff_p99=float(((both.pnl_m - both.pnl_r).abs() / both.cost.abs()).quantile(0.99)) if len(both) else None,
                      same_tier=int((both.entry_tier_mc == both.tier_mc + 1).sum()),
                      tier_note="研究員 tier_mc 0／1／2＝大／中／小，複製員 1／2／3",
                      tier_diff_rows=[(r_.code, r_.buy_date, int(r_.entry_tier_mc), int(r_.tier_mc) + 1)
                                      for r_ in both[both.entry_tier_mc != both.tier_mc + 1].itertuples()][:10])
    out["trades"] = tr
    print(json.dumps(tr, ensure_ascii=False), flush=True)
    # CUR 最賺 5 筆的層
    t5 = mt[mt.cell == "CUR"].sort_values("pnl", ascending=False).head(5)
    out["cur_top5_tiers"] = [(r_.code, r_.buy_date, int(r_.entry_tier_mc), round(r_.ret, 4)) for r_ in t5.itertuples()]

    # ---------------- 凍結後：研究員實際權重拆法、β 口徑、Q55 漂移月底取樣
    P = R.build(sc); S.attach_e50(P, sc); excl = S.innov_mask_fn(P, sc)
    Rv, mi0 = L.load_rev(P); S_, _ = L.surge(Rv, mi0, null2013=True)
    K.load_shares(P); ma60 = R.ma(P, 60); AO, Gf = K.adj_open_matrix(P)
    e = P.T - 1; e50 = P.e50.values.astype(float)
    lists = Q.make_lists(P, S_, mi0, shift=0, excl=excl)
    tof = {s: K.tiers(P, s, excl, "mc") for s in lists}
    spec = {"CUR": None, "TL": {1}, "TM": {2}, "TS": {3}, "Q55L": {1}, "Q55S": {2, 3}}
    pl = {nm: K.tier_plans(P, S_, lists, tof, ts, excl, ma60=ma60)[:2] for nm, ts in spec.items()}
    navs = {nm: K.simulate_s12(P, *pl[nm], sat_w=0.7, sizing="W10", **K.ST)["nav"] for nm in ("CUR", "TL", "TM", "TS")}
    qa = K.simulate_s12(P, *pl["Q55L"], sat_w=0.7, sizing="W10", nmax=5, ntgt=5, capital=500_000.0, **K.ST)["nav"]
    qb = K.simulate_s12(P, *pl["Q55S"], sat_w=0.7, sizing="W10", nmax=5, ntgt=5, capital=500_000.0, **K.ST)["nav"]
    navs["Q55"] = qa + qb
    assert abs(L.ann(P, navs["CUR"], L.didx(P, "2018-12-28"), e) - my["cells"]["CUR"]["ann_2019"]) < 1e-12
    ws = K.windows(P, lists)
    rows = []
    for s, ee, e2 in ws:
        pool = R.universe(P, s) & ~excl(s); op = np.isfinite(P.O[ee]); tt = tof[s]
        r50 = K.e50_win(P, ee, e2)
        for tn, nm, tv in (("ALL", "CUR", None), ("L", "TL", 1), ("M", "TM", 2), ("S", "TS", 3)):
            idx = np.where(pool & op if tv is None else pool & op & (tt == tv))[0]
            rew = K.win_ret(P, AO, Gf, idx, ee, e2)
            lst = np.array([i for i in pl[nm][0][s] if op[i]], dtype=int)
            rl = K.win_ret(P, AO, Gf, lst, ee, e2) if len(lst) else rew
            a = navs[nm][ee - 1]; b = navs[nm][e2 - 1] if e2 is not None else navs[nm][e]
            ao = None
            rows.append(dict(tier=tn, e=ee, exec=P.dstr[ee], end=P.dstr[e2] if e2 is not None else P.dstr[e],
                             year=int(P.dstr[ee][:4]), n=len(lst), r_ew=rew, r_list=rl, r50=r50, act=b / a - 1.0,
                             r50c=(e50[e2 - 1] if e2 is not None else e50[e]) / e50[ee - 1] - 1.0))
    g = pd.DataFrame(rows)

    def cann(x, sub):
        v = float(np.prod(1.0 + x)); yrs = (pd.Timestamp(sub.end.iloc[-1]) - pd.Timestamp(sub.exec.iloc[0])).days / 365.25
        return v ** (1 / yrs) - 1
    da_ = {}
    for tn, nm in (("ALL", "CUR"), ("L", "TL"), ("M", "TM"), ("S", "TS")):
        h = g[g.tier == tn].sort_values("e")
        nl = np.minimum(h.n.values, 10) / 10.0
        sat_ew = np.where(h.n.values > 0, nl * h.r_list + (1 - nl) * h.r_ew, h.r_ew)   # 名單每檔 1/10、空位放同層等權
        sat_50 = np.where(h.n.values > 0, nl * h.r_list + (1 - nl) * h.r50, h.r50)     # 名單每檔 1/10、空位放 0050
        h = h.assign(b_ew=0.3 * h.r50 + 0.7 * h.r_ew, b_selw=0.3 * h.r50 + 0.7 * sat_ew, b_dil=0.3 * h.r50 + 0.7 * sat_50)
        o = {}
        for pn, sub in (("from2019", h[h.year >= 2019]), ("2019_2023", h[(h.year >= 2019) & (h.year <= 2023)]),
                        ("from2024", h[h.year >= 2024]), ("full", h)):
            A = {c: cann(sub[c].values, sub) for c in ("r50", "b_ew", "b_selw", "b_dil", "act")}
            o[pn] = dict(tier_lag=pc(A["b_ew"] - A["r50"]), selection_actualw=pc(A["b_selw"] - A["b_ew"]),
                         empty_to_0050=pc(A["b_dil"] - A["b_selw"]), exec_other=pc(A["act"] - A["b_dil"]),
                         total=pc(A["act"] - A["r50"]))
        da_[nm] = o
    out["postfreeze_decomp_researcher_def"] = dict(mine=da_, researcher=RS["decomp_actual"],
                                                  note="凍結後：照研究員 FINDINGS 第 4 節的事後分組定義（選股＝名單每檔 1/10、空位放同層等權；"
                                                       "空位放 0050；其餘執行）用複製員的時窗表重算。")
    # β 口徑：(1) 複製員主口徑：格的收盤時窗報酬 − 0050 收盤時窗；(2) 格收盤時窗 − 0050 開盤時窗；(3) 用日曆月報酬
    h = g[g.tier == "ALL"].sort_values("e")
    xf = (h.r_ew - h.r50).values

    def ols(y, x):
        X = np.column_stack([np.ones(len(x)), x]); b, *_ = np.linalg.lstsq(X, y, rcond=None)
        res = y - X @ b; s2 = res @ res / (len(y) - 2); cv = s2 * np.linalg.inv(X.T @ X)
        return round(float(b[1]), 3), round(float(b[1] / np.sqrt(cv[1, 1])), 2)
    bv = {}
    for nm in ("CUR", "TL", "TM", "TS", "Q55"):
        a = navs[nm][h.e.values - 1]
        e2s = list(h.e.values[1:] - 1) + [e]
        b = navs[nm][np.array(e2s)]
        y = b / a - 1.0
        bv[nm] = dict(close_vs_close=ols(y - h.r50c.values, xf), close_vs_open50=ols(y - h.r50.values, xf))
    out["postfreeze_beta_sensitivity"] = dict(values=bv, researcher=RS["beta"],
                                              note="凍結後：研究員的 β 系統性低 0.02～0.03；這裡看時窗口徑造成多少差。")
    # Q55 大帳戶占比：日資料 vs 月底取樣
    sh = pd.Series(qa / navs["Q55"], index=P.dates)[P.t0:]
    me = sh.groupby(sh.index.to_period("M")).last()
    out["postfreeze_q55_drift"] = dict(daily_min=round(float(sh.min()), 4), daily_max=round(float(sh.max()), 4),
                                       monthend_min=round(float(me.min()), 4), monthend_min_at=str(me.idxmin()),
                                       monthend_max=round(float(me.max()), 4), monthend_max_at=str(me.idxmax()),
                                       at_2020_08=round(float(me[pd.Period("2020-08", "M")]), 4),
                                       at_2024_09=round(float(me[pd.Period("2024-09", "M")]), 4),
                                       end=round(float(sh.iloc[-1]), 4), researcher=RS["q55_drift"])
    # 凍結後：最賺 5 筆占比改成「含期末未平倉」（研究員事後列的 CUR 最賺 5 筆含未平倉的 6213，但占比用已平倉）
    t5o = {}
    for cc in ("CUR", "TL", "Q55"):
        vals_c = []; vals_o = []
        for sh_ in (0, 1, 2, 3):
            ls_ = Q.make_lists(P, S_, mi0, shift=sh_, excl=excl)
            tf_ = {s: K.tiers(P, s, excl, "mc") for s in ls_}
            for cp in (600_000.0, 800_000.0, 1_000_000.0, 1_500_000.0, 2_000_000.0):
                subs = [("TL", {1}, 10, None, 1.0)] if cc == "TL" else (
                    [("CUR", None, 10, None, 1.0)] if cc == "CUR" else [("L", {1}, 5, 5, 0.5), ("MS", {2, 3}, 5, 5, 0.5)])
                pn = []; po = []; nv = 0.0
                for _, ts, nmx, ntg, f in subs:
                    b_, k_, _ = K.tier_plans(P, S_, ls_, tf_, ts, excl, ma60=ma60)
                    x = K.simulate_s12(P, b_, k_, sat_w=0.7, sizing="W10", nmax=nmx, ntgt=ntg, capital=cp * f, **K.ST)
                    pn += list(x["trades"].pnl.values); po += [v - c for _, _, c, v, _ in x["open_pos"]]
                    nv = nv + x["nav"]
                inc = nv[e] - nv[P.t0]
                vals_c.append(float(np.sort(pn)[::-1][:5].sum() / inc))
                vals_o.append(float(np.sort(pn + po)[::-1][:5].sum() / inc))
        t5o[cc] = dict(closed_median=round(float(np.median(vals_c)), 4), with_open_median=round(float(np.median(vals_o)), 4),
                       main_closed=round(vals_c[2], 4), main_with_open=round(vals_o[2], 4))
    t5o["Q55_rule_a_with_open"] = bool(t5o["Q55"]["with_open_median"] < t5o["CUR"]["with_open_median"])
    out["postfreeze_top5_with_open"] = t5o
    js = json.dumps(out, ensure_ascii=False, indent=1, default=lambda o: (bool(o) if isinstance(o, np.bool_) else
                                                                          (int(o) if isinstance(o, np.integer) else float(o))))
    open(os.path.join(HERE, "compare_vs_researcher.json"), "w").write(js)
    print(js[:200])


if __name__ == "__main__":
    main(sys.argv[1])
