# -*- coding: utf-8 -*-
"""凍結後的審查用加跑（不改 my_numbers.json）。輸出 review_numbers.json。
用法：python3 -I -B rep_review.py <scratch_dir>
"""
import sys, os, json, math, hashlib, subprocess, re, io, csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import rep_lib as L

HERE = os.path.dirname(os.path.abspath(__file__))
S = sys.argv[1]
FROZEN = json.load(open(os.path.join(HERE, "my_numbers.json")))
assert hashlib.sha256(open(os.path.join(HERE, "my_numbers.json"), "rb").read()).hexdigest() == open(os.path.join(HERE, "my_numbers.sha256")).read().split()[0]
P = L.load_panel(S); T = P.T; t0 = P.t0
i19 = max(i for i, d in enumerate(P.dstr) if d < "2019-01-01")
b50 = P.e50.values
FP = L.fundamentals(P, True); K = L.tech(P)
R = {}

# ---------------- 0. 凍結數字可由現行（加了選用參數的）函式庫重現
sig_tw = L.signals(P, FP, K, "twse"); sig_all = L.signals(P, FP, K, "both")
r0 = L.run_exec(P, sig_tw, K)
a_full = L.ann(r0["nav"][t0], r0["nav"][-1], P.dstr[t0], P.dstr[-1])[0]
assert abs(a_full - FROZEN["exec"]["main_twse_pit_normal"]["full"]["ann"]) < 1e-12, "函式庫變動影響了凍結數字"
R["frozen_reproduced"] = True

# ---------------- 1. 0050 開盤／收盤（供「空位放 0050」精確版）
def load_0050():
    cache = os.path.join(S, "e0050_ohlc.csv")
    if os.path.exists(cache):
        return pd.read_csv(cache, index_col=0)
    env = dict(os.environ, GIT_NO_LAZY_FETCH="1")
    ls = subprocess.run(["git", "-C", "/home/user/-/research/tw_stock_selection/raw/jb7k6295r7-bot_tw-stock-data", "ls-tree", "HEAD", "data/universe/daily/"],
                        capture_output=True, text=True, env=env, check=True).stdout
    files = []
    for line in ls.splitlines():
        meta, path = line.split("\t"); m = re.search(r"(\d{4}-\d{2}-\d{2})\.csv$", path)
        if m and m.group(1) <= "2026-10-22": files.append((m.group(1), meta.split()[2]))
    files.sort()
    hl = os.path.join(S, "_hashes.txt")
    open(hl, "w").write("\n".join(h for _, h in files) + "\n")
    p = subprocess.Popen(["git", "-C", "/home/user/-/research/tw_stock_selection/raw/jb7k6295r7-bot_tw-stock-data", "cat-file", "--batch"], stdin=open(hl, "rb"), stdout=subprocess.PIPE, env=env)
    rows = []
    for d, h in files:
        hdr = p.stdout.readline().split(); data = p.stdout.read(int(hdr[2])); p.stdout.read(1)
        txt = data.decode("utf-8"); k = txt.find("\n%s_0050," % d)
        if k < 0: rows.append((d, np.nan, np.nan, np.nan)); continue
        line = txt[k + 1: txt.find("\n", k + 1)]
        head = txt[:txt.find("\n")].split(",")
        rec = dict(zip(head, next(csv.reader([line]))))
        f = lambda x: float(x) if x not in ("", "--") else np.nan
        rows.append((d, f(rec["open"]), f(rec["close"]), f(rec["change"])))
    p.wait()
    df = pd.DataFrame(rows, columns=["date", "open", "close", "change"]).set_index("date")
    df.to_csv(cache)
    return df
e = load_0050().reindex(P.dstr)
r_tot = (P.e50 / P.e50.shift(1) - 1).fillna(0).values
intr = (e.close / e.open - 1).fillna(0).values
ov = (1 + r_tot) / (1 + intr) - 1
R["e0050_ohlc"] = {"days_missing": int(e.open.isna().sum()), "mean_overnight_bp": float(ov[t0 + 1:].mean() * 1e4), "mean_intraday_bp": float(intr[t0 + 1:].mean() * 1e4),
                   "check_total": float(np.prod(1 + ov[t0 + 1:]) * np.prod(1 + intr[t0 + 1:]) / (b50[-1] / b50[t0]))}

def seg_ann(nav, d0, d1):
    i0 = max(i for i, d in enumerate(P.dstr) if d <= d0); i1 = max(i for i, d in enumerate(P.dstr) if d <= d1)
    return L.ann(nav[i0], nav[i1], P.dstr[i0], P.dstr[i1])[0], L.ann(b50[i0], b50[i1], P.dstr[i0], P.dstr[i1])[0], i0, i1

def pack(r, start_i=None):
    nav = r["nav"]; s0 = t0 if start_i is None else start_i
    out = {}
    for nm, i0 in (("full", s0), ("from2019", i19)):
        a_s = L.ann(nav[i0], nav[-1], P.dstr[i0], P.dstr[-1])[0]; a_b = L.ann(b50[i0], b50[-1], P.dstr[i0], P.dstr[-1])[0]
        x, _, _ = L.month_excess(P, nav, b50, i0, T - 1); tt, m, sd, n = L.tstat(x)
        out[nm] = {"ann": a_s, "ann_0050": a_b, "excess_pp": (a_s - a_b) * 100, "t": tt, "n": n, "nw_t6": L.nw_t(x, 6), "maxdd": L.maxdd(nav[i0:]),
                   "pos_month_share": float((x > 0).mean())}
    segs = {}
    for nm, d0, d1 in (("2017-03-31~2018", "2017-03-31", "2018-12-31"), ("2019~2023", "2018-12-31", "2023-12-31"), ("2024~2026-10", "2023-12-31", "2026-10-22"),
                       ("pre_odd(2017-03-31~2020-10-23)", "2017-03-31", "2020-10-25"), ("post_odd(2020-10-26~)", "2020-10-25", "2026-10-22")):
        a_s, a_b, i0, i1 = seg_ann(nav, d0, d1); x, _, _ = L.month_excess(P, nav, b50, i0, i1)
        segs[nm] = {"ann": a_s, "ann_0050": a_b, "excess_pp": (a_s - a_b) * 100, "t": L.tstat(x)[0]}
    out["segments"] = segs
    yr = {}
    for y in range(2017, 2027):
        a_s, a_b, i0, i1 = seg_ann(nav, "%d-12-31" % (y - 1) if y > 2017 else "2017-03-31", "%d-12-31" % y)
        yr[str(y)] = {"strat": nav[i1] / nav[i0] - 1, "e0050": b50[i1] / b50[i0] - 1}
    out["yearly"] = yr
    yrs = (P.dates[-1] - P.dates[s0]).days / 365.25
    out["turnover"] = r["cnt"]["buy_amt"] / float(np.nanmean(nav[s0:])) / yrs
    out["avg_npos"] = float(r["npos"][s0 + 1:].mean()); out["avg_cash_w"] = float(r["cashw"][s0 + 1:].mean())
    out["cnt"] = {k: float(v) for k, v in r["cnt"].items()}
    return out

EXK = dict(cash_0050="exact", e50_ov=ov, e50_in=intr)
V = {}
V["main"] = pack(r0)
V["main_e50_exact"] = pack(L.run_exec(P, sig_tw, K, **EXK))
V["main_e50_approx_mine"] = pack(L.run_exec(P, sig_tw, K, cash_0050=True))
Ks = {tr: L.tech(P, trail=float(tr)) for tr in (6, 8, 10, 12, 15, 20)}
for tr in (6, 8, 10, 12, 15, 20):
    V["both_e50_trail%d" % tr] = pack(L.run_exec(P, sig_all, Ks[tr], **EXK))
    V["twse_e50_trail%d" % tr] = pack(L.run_exec(P, sig_tw, Ks[tr], **EXK))
    V["twse_cash_trail%d" % tr] = pack(L.run_exec(P, sig_tw, Ks[tr]))
    V["both_cash_trail%d" % tr] = pack(L.run_exec(P, sig_all, Ks[tr]))
V["best_stress"] = pack(L.run_exec(P, sig_all, Ks[10], cost="stress", **EXK))
V["best_zero"] = pack(L.run_exec(P, sig_all, Ks[10], cost="zero", **EXK))
V["best_nolot"] = pack(L.run_exec(P, sig_all, Ks[10], board_lot_rule=False, **EXK))
V["best_no_same_day_slot"] = pack(L.run_exec(P, sig_all, Ks[10], same_day_slot=False, **EXK))
V["best_3m"] = pack(L.run_exec(P, sig_all, Ks[10], capital=3e6, **EXK))
V["best_20slots"] = pack(L.run_exec(P, sig_all, Ks[10], nmax=20, **EXK))
V["main_nolot"] = pack(L.run_exec(P, sig_tw, K, board_lot_rule=False))
V["main_no_same_day_slot"] = pack(L.run_exec(P, sig_tw, K, same_day_slot=False))
# 進出場打架的修正診斷：訊號日出場條件已成立者不進場
ex = K["ex1"] | K["ex2"]
V["main_noconflict"] = pack(L.run_exec(P, sig_tw & ~ex, K))
V["main_noconflict_zero"] = pack(L.run_exec(P, sig_tw & ~ex, K, cost="zero"))
trc = L.post_trades(P, sig_tw & ~ex, K); cl = trc[trc.closed == 1]
R["post_noconflict_twse"] = {"n": int(len(trc)), "win_raw": float((trc.ret_raw > 0).mean()), "mean_raw": float(trc.ret_raw.mean()), "hold1_share": float((cl.hold == 1).mean())}
# 起始偏移 13 條（確認是否近乎退化）
offs = []
for j in range(13):
    ts = t0 + 5 * j
    rr = L.run_exec(P, L.signals(P, FP, K, "twse", t_start=ts), K, t_start=ts)
    nav = rr["nav"]; a_s = L.ann(nav[i19], nav[-1], P.dstr[i19], P.dstr[-1])[0]; a_b = L.ann(b50[i19], b50[-1], P.dstr[i19], P.dstr[-1])[0]
    offs.append((a_s - a_b) * 100)
R["j_offsets_2019_excess_pp"] = {"min": min(offs), "max": max(offs), "median": float(np.median(offs)), "all": offs}
R["variants"] = V

# ---------------- 2. PIT 品質：進場訊號所用季報／年報的公告日來源
a = pd.read_parquet(os.path.join(L.PEAD, "avail_dates.parquet"))
f = pd.read_parquet(os.path.join(L.PEAD, "fin_long.parquet"))[["code", "y", "q"]]
f = f.merge(a[["code", "y", "q", "A", "A_method", "deadline", "late"]], on=["code", "y", "q"], how="left")
dl = {1: (0, 5, 15), 2: (0, 8, 14), 3: (0, 11, 14), 4: (1, 3, 31)}
f["Ause"] = f.A.where(f.A.notna(), pd.to_datetime([pd.Timestamp(y + dl[q][0], dl[q][1], dl[q][2]) for y, q in zip(f.y, f.q)]))
f["A_method"] = f.A_method.fillna("no_row→deadline")
f["ai"] = np.searchsorted(P.dates.values, f.Ause.values.astype("datetime64[ns]"), side="left"); f["qi"] = f.y * 4 + f.q - 1
tr_all = L.post_trades(P, sig_all, K)
byc = {c: g.sort_values("ai") for c, g in f.groupby("code")}
mq = []; my = []; age = []
for c, i, a_ in zip(tr_all.code, tr_all.i, tr_all.a):
    g = byc[c]; t = a_ - 1
    gg = g[g.ai <= t]
    q0 = gg.loc[gg.qi.idxmax()]; mq.append(q0.A_method); age.append(t - q0.ai)
    g4 = gg[gg.q == 4]; y0 = g4.loc[g4.qi.idxmax()]; my.append(y0.A_method)
tr_all["mq"] = mq; tr_all["my"] = my; tr_all["age"] = age
R["pit_quality_entries_both"] = {"n": int(len(tr_all)), "q0_method_share": tr_all.mq.value_counts(normalize=True).to_dict(), "y0_method_share": tr_all.my.value_counts(normalize=True).to_dict(),
                                 "q0_method_share_twse": tr_all[tr_all.mkt == 1].mq.value_counts(normalize=True).to_dict(),
                                 "q0_method_share_tpex": tr_all[tr_all.mkt == 2].mq.value_counts(normalize=True).to_dict(),
                                 "entries_within_1d_of_A": float((tr_all.age <= 1).mean()), "entries_on_A_day": float((tr_all.age == 0).mean()),
                                 "mean_raw_ret_by_q0_method": tr_all.groupby("mq").ret_raw.agg(["mean", "count"]).to_dict("index"),
                                 "mean_ret_age0": float(tr_all[tr_all.age == 0].ret_raw.mean()), "mean_ret_age_gt5": float(tr_all[tr_all.age > 5].ret_raw.mean())}

# ---------------- 3. 貼文對帳推論強度
def lst(df, col, asc, n=6):
    d = df.sort_values(col, ascending=asc).head(n)
    return [dict(code=r.code, mkt=int(r.mkt), buy=r.buy_date, sell=r.sell_date, bpx=float(r.buy_px), spx=float(r.sell_px), raw=float(r.ret_raw), div=float(r.ret_div)) for r in d.itertuples()]
tr_tw = L.post_trades(P, sig_tw, K)
R["recon"] = {"top_gain_both": lst(tr_all, "ret_raw", False), "top_loss_both_div": lst(tr_all, "ret_div", True), "top_gain_twse": lst(tr_tw, "ret_raw", False, 3)}
g5508 = 218.5 / 105.5; l3665 = 2250 / 3000
R["recon"]["implied_cost_proportional"] = {
    "5508_point": 1 - 2.0629 / g5508, "5508_range": [1 - 2.06295 / g5508, 1 - 2.06285 / g5508],
    "3665_point": 1 - 0.7469 / l3665, "3665_range": [1 - 0.74695 / l3665, 1 - 0.74685 / l3665]}
sep = {}
for u, tr in (("both", tr_all), ("twse", tr_tw)):
    m = (tr.closed == 1) & tr.sell_date.str.startswith("2026-09")
    for c in (0.0, 0.002, 0.004, 0.00585):
        nd = (1 + tr.ret_div[m]) * (1 - c) - 1; nr = (1 + tr.ret_raw[m]) * (1 - c) - 1
        sep["%s_cost%.3f%%" % (u, c * 100)] = {"n": int(m.sum()), "wins_div": int((nd > 0).sum()), "wins_raw": int((nr > 0).sum())}
R["recon"]["sep2026"] = sep
for c in (0.0, 0.004, 0.00585):
    nr = (1 + tr_all.ret_raw) * (1 - c) - 1
    R["recon"]["both_win_raw_cost%.3f%%" % (c * 100)] = float((nr > 0).mean())
R["recon"]["trades_per_year_both"] = len(tr_all) / ((P.dates[-1] - P.dates[t0]).days / 365.25)
R["recon"]["n_gain_ge_90pct_both"] = int((tr_all.ret_raw >= 0.9).sum()); R["recon"]["n_loss_le_-20pct_both_div"] = int((tr_all.ret_div <= -0.20).sum())

# ---------------- 4. 起點：2017-03-31 可算 CAGR 的檔數；2017-01～03 的 PIT 訊號
fl = pd.read_parquet(os.path.join(L.PEAD, "fin_long.parquet"))
pv = fl[fl.q == 4].pivot(index="code", columns="y", values="eps_ytd")
ci = {c: i for i, c in enumerate(P.codes)}
both_pos = pv.index[(pv[2016] > 0) & (pv[2013] > 0)]
tw_now = [c for c in both_pos if c in ci and P.M[t0, ci[c]] == 1]; all_now = [c for c in both_pos if c in ci and P.M[t0, ci[c]] >= 1]
R["start"] = {"twse_FY2016&FY2013_pos_regardless_of_A": len(tw_now), "both_mkts_same": len(all_now),
              "twse_pit_available_on_2017-03-31": int((np.isfinite(FP[4][t0]) & (P.M[t0] == 1)).sum()),
              "both_pit_available_on_2017-03-31": int((np.isfinite(FP[4][t0]) & (P.M[t0] >= 1)).sum()),
              "twse_pit_available_on_2017-04-28": int((np.isfinite(FP[4][t0 + 18]) & (P.M[t0 + 18] == 1)).sum())}
first_valid = int(np.where(np.isfinite(FP[4]).any(axis=1))[0][0])
R["start"]["first_day_any_CAGR_pit"] = P.dstr[first_valid]
sig_early = L.signals(P, FP, K, "both", t_start=first_valid)
R["start"]["signal_rows_before_2017-03-31_both"] = int(sig_early[:t0].sum()); R["start"]["signal_rows_before_2017-03-31_twse"] = int((sig_early[:t0] & (P.M[:t0] == 1)).sum())

# ---------------- 5. 延伸診斷：用「收盤÷本益比」推回 FY2011、FY2012 年度 EPS（只上市），把起點提前
try:
    per = pd.read_parquet(os.path.join(L.PEAD, "per_long.parquet"), columns=["date", "code", "mkt", "per"])
    cr = pd.read_parquet("/home/user/-/research/tw_stock_selection/data_panel/panel/close_raw.parquet")
    cr.index = pd.to_datetime(cr.index); cr.columns = cr.columns.astype(str)
    def implied(year):
        d = per[(per.date >= "%d-04-10" % (year + 1)) & (per.date <= "%d-04-30" % (year + 1)) & (per.per > 0)]
        cl = cr.loc["%d-04-10" % (year + 1): "%d-04-30" % (year + 1)].stack().rename("cl").reset_index(); cl.columns = ["date", "code", "cl"]
        d = d.merge(cl, on=["date", "code"]); d["e"] = d.cl / d.per
        return d.groupby("code").e.median()
    val = {}
    for y in (2013, 2014):
        im = implied(y); act = fl[(fl.y == y) & (fl.q == 4)].set_index("code").eps_ytd
        j = pd.concat([im, act], axis=1, keys=["im", "act"]).dropna(); j = j[j.act > 0]; rr = j.im / j.act - 1
        val[str(y)] = {"n": int(len(j)), "within2pct": float((rr.abs() < 0.02).mean()), "within5pct": float((rr.abs() < 0.05).mean())}
    extra = {}
    for y in (2011, 2012):
        for c, v in implied(y).items(): extra[(c, y)] = float(v)
    FX = L.fundamentals(P, True, extra_epsy=extra)
    ts = int(np.where(P.barno.max(axis=1) >= 240)[0][0])
    sx = L.signals(P, FX, K, "twse", t_start=ts)
    R["extension"] = {"validation_implied_vs_actual": val, "n_implied_FY2011": int(sum(1 for k in extra if k[1] == 2011)), "n_implied_FY2012": int(sum(1 for k in extra if k[1] == 2012)),
                      "first_signal_day_possible": P.dstr[ts], "signal_rows_before_2017-03-31": int(sx[:t0].sum()),
                      "signal_rows_after_identical": bool((sx[t0:] == sig_tw[t0:]).all())}
    trx = L.post_trades(P, sx, K, t_start=ts); pre = trx[trx.buy_date < "2017-04-05"]
    R["extension"]["post_trades_pre_2017-04-05"] = {"n": int(len(pre)), "win_raw": float((pre.ret_raw > 0).mean()), "mean_raw": float(pre.ret_raw.mean()),
                                                   "per_year": len(pre) / ((P.dates[t0] - P.dates[ts]).days / 365.25)}
    for nm, kw in (("normal", {}), ("zero", {"cost": "zero"}), ("nolot", {"board_lot_rule": False})):
        rx = L.run_exec(P, sx, K, t_start=ts, **kw); nav = rx["nav"]
        a1 = L.ann(nav[ts], nav[t0], P.dstr[ts], P.dstr[t0])[0]; b1 = L.ann(b50[ts], b50[t0], P.dstr[ts], P.dstr[t0])[0]
        a2 = L.ann(nav[ts], nav[-1], P.dstr[ts], P.dstr[-1])[0]; b2 = L.ann(b50[ts], b50[-1], P.dstr[ts], P.dstr[-1])[0]
        x, _, _ = L.month_excess(P, nav, b50, ts, T - 1)
        R["extension"]["exec_" + nm] = {"window_ann": a1, "window_0050": b1, "ext_full_ann": a2, "ext_full_0050": b2, "ext_full_t": L.tstat(x)[0], "n_m": len(x),
                                        "lot_skip": int(rx["cnt"]["lot_skip"]), "buys": int(rx["cnt"]["buy"])}
except Exception as ex_:
    R["extension"] = {"error": repr(ex_)}

# ---------------- 6. t 值補充
x, rs, rb = L.month_excess(P, r0["nav"], b50, t0, T - 1)
xv = x.values
R["tstat"] = {"n": len(xv), "t": L.tstat(xv)[0], "nw3": L.nw_t(xv, 3), "nw6": L.nw_t(xv, 6), "nw12": L.nw_t(xv, 12),
              "ac1": float(pd.Series(xv).autocorr(1)), "ac2": float(pd.Series(xv).autocorr(2)), "ac3": float(pd.Series(xv).autocorr(3)),
              "pos_months": int((xv > 0).sum()), "corr_strat_0050_m": float(np.corrcoef(rs.values, rb.values)[0, 1]),
              "beta_to_0050": float(np.cov(rs.values, rb.values)[0, 1] / rb.values.var(ddof=1)),
              "sd_strat_m": float(rs.std()), "sd_0050_m": float(rb.std())}
bta = R["tstat"]["beta_to_0050"]; al = rs.values - bta * rb.values
R["tstat"]["alpha_m"] = float(al.mean()); R["tstat"]["alpha_t"] = L.tstat(al)[0]
# 加一個 0 報酬的 2017-03 月（研究員 n=116 的作法）
x116 = np.concatenate([[0.0], xv]); R["tstat"]["t_n116_with_zero_month"] = L.tstat(x116)[0]
rng = np.random.default_rng(20261114); bs = []
for _ in range(5000):
    idx = []
    while len(idx) < len(xv):
        s = rng.integers(0, len(xv)); idx.extend(range(s, min(s + 6, len(xv))))
    bs.append(xv[idx[:len(xv)]].mean())
R["tstat"]["block_boot_p_mean_ge_0"] = float((np.array(bs) >= 0).mean())

def clean(o):
    if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.bool_,)): return bool(o)
    return o
json.dump(clean(R), open(os.path.join(HERE, "review_numbers.json"), "w"), ensure_ascii=False, indent=1)
print("ok")
