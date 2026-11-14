# -*- coding: utf-8 -*-
"""獨立複製主程式：產生 my_numbers.json 並記 sha256（凍結）。
用法：python3 -I -B rep_run.py <scratch_dir>
此檔不讀研究員的 results.json／trades／signals／navs，也不 import cslib／run。
"""
import sys, os, json, math, hashlib, datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import rep_lib as L

HERE = os.path.dirname(os.path.abspath(__file__))
S = sys.argv[1]
P = L.load_panel(S)
T = P.T; t0 = P.t0
i19 = max(i for i, d in enumerate(P.dstr) if d < "2019-01-01")   # 2018 最後交易日＝2019 起的基期
OUT = {"meta": {"utc": datetime.datetime.utcnow().isoformat() + "Z", "days": T, "codes": P.N, "first": P.dstr[0], "last": P.dstr[-1],
                "start_signal": P.dstr[t0], "first_buy": P.dstr[t0 + 1], "base_2019": P.dstr[i19],
                "idx_missing": P.idx_missing, "e50_missing_filled": P.e50_missing, "bad_ret_cells": P.n_bad}}

# ---- 0050 含息
e = P.e50_full
c0, c1 = e.loc["2014-11-03"], e.loc["2026-10-22"]
a, yrs = L.ann(c0, c1, "2014-11-03", "2026-10-22")
OUT["e0050"] = {"cum_2014-11-03_2026-10-22": c1 / c0 - 1, "ann": a, "years": yrs,
                "post_ann_12y": (1 + 6.4844) ** (1 / 12) - 1}
b50 = P.e50.values

FP = L.fundamentals(P, True)
FN = L.fundamentals(P, False)
K = L.tech(P)


def bench_stats(nav, tag_from):
    i0 = t0 if tag_from == "full" else i19
    a_s, y = L.ann(nav[i0], nav[-1], P.dstr[i0], P.dstr[-1])
    a_b, _ = L.ann(b50[i0], b50[-1], P.dstr[i0], P.dstr[-1])
    x, rs, rb = L.month_excess(P, nav, b50, i0, T - 1)
    tt, m, sd, n = L.tstat(x)
    return {"ann": a_s, "ann_0050": a_b, "excess_pp": (a_s - a_b) * 100, "t": tt, "mean_m": m, "sd_m": sd, "n_m": n,
            "years": y, "maxdd": L.maxdd(nav[i0:]), "maxdd_0050": L.maxdd(b50[i0:]), "nw_t6": L.nw_t(x, 6),
            "ac1": float(pd.Series(x.values).autocorr(1))}


def exec_pack(sig, Kx=K, **kw):
    r = L.run_exec(P, sig, Kx, **kw)
    nav = r["nav"]
    d = {"full": bench_stats(nav, "full"), "from2019": bench_stats(nav, "2019"), "end_nav": float(nav[-1])}
    yrs = (P.dates[-1] - P.dates[t0]).days / 365.25
    d["turnover_1side"] = r["cnt"]["buy_amt"] / float(np.nanmean(nav[t0:])) / yrs
    d["avg_npos"] = float(r["npos"][t0 + 1:].mean()); d["avg_cash_w"] = float(r["cashw"][t0 + 1:].mean())
    d["cnt"] = {k: (float(v) if isinstance(v, float) else int(v)) for k, v in r["cnt"].items()}
    avgnav = float(np.nanmean(nav[t0:]))
    d["cost_acct_pct_per_yr"] = {k: r["cnt"][k] / avgnav / yrs * 100 for k in ("fee", "tax", "slip")}
    tr = r["trades"]
    if len(tr):
        pr = tr.proceeds / tr.cost - 1
        d["trades"] = {"n_closed": int(len(tr)), "open": int(r["open_pos"]), "win": float((pr > 0).mean()), "mean": float(pr.mean()),
                       "avg_win": float(pr[pr > 0].mean()), "avg_loss": float(pr[pr <= 0].mean()), "hold1": float((tr.hold == 1).mean())}
    return d, r


def post_pack(sig, name):
    tr = L.post_trades(P, sig, K)
    cl = tr[tr.closed == 1]
    d = {"n_all": int(len(tr)), "n_closed": int(len(cl)), "n_open": int((tr.closed == 0).sum()),
         "win_raw_all": float((tr.ret_raw > 0).mean()), "win_raw_closed": float((cl.ret_raw > 0).mean()),
         "mean_raw_all": float(tr.ret_raw.mean()), "mean_div_all": float(tr.ret_div.mean()),
         "win_div_all": float((tr.ret_div > 0).mean())}
    w = tr[tr.ret_raw > 0].ret_raw; l = tr[tr.ret_raw <= 0].ret_raw
    d["avg_win"] = float(w.mean()); d["avg_loss"] = float(l.mean()); d["payoff"] = float(w.mean() / -l.mean())
    for key, col, fn in [("max_gain_raw", "ret_raw", "idxmax"), ("max_loss_raw", "ret_raw", "idxmin"),
                         ("max_gain_div", "ret_div", "idxmax"), ("max_loss_div", "ret_div", "idxmin")]:
        j = getattr(tr[col], fn)(); r = tr.loc[j]
        d[key] = {"code": r.code, "mkt": int(r.mkt), "buy_date": r.buy_date, "sell_date": r.sell_date, "buy_px": float(r.buy_px),
                  "sell_px": float(r.sell_px), "ret_raw": float(r.ret_raw), "ret_div": float(r.ret_div), "closed": int(r.closed)}
    d["hold1_share_closed"] = float((cl.hold == 1).mean()); d["hold1_n"] = int((cl.hold == 1).sum())
    d["hold_mean_closed"] = float(cl.hold.mean())
    d["hold1_mean_div"] = float(cl[cl.hold == 1].ret_div.mean()); d["hold1_win_div"] = float((cl[cl.hold == 1].ret_div > 0).mean())
    d["exit_only_trail"] = int((cl.ex1 & ~cl.ex2).sum()); d["exit_only_cross"] = int((~cl.ex1 & cl.ex2).sum()); d["exit_both"] = int((cl.ex1 & cl.ex2).sum())
    for cst, nm in [(0.0, "tw_nocost"), (0.004, "tw_cost040")]:
        nav, n = L.post_tw_curve(P, tr, cst)
        aa, _ = L.ann(1.0, nav[-1], P.dstr[t0], P.dstr[-1])
        pk = int(np.argmax(nav))
        d[nm] = {"cum": float(nav[-1] - 1), "ann": aa, "maxdd": L.maxdd(nav), "peak_date": P.dstr[t0 + pk],
                 "peak_to_end": float(nav[-1] / nav[pk] - 1), "max_concurrent": int(n.max()), "avg_concurrent": float(n[1:].mean()),
                 "days_empty": float((n[1:] == 0).mean())}
    # 扣 0.40% 來回成本後的統計（含息口徑）與 2026 年依出場月份
    net = (1 + tr.ret_div) * (1 - 0.004) - 1
    netraw = (1 + tr.ret_raw) * (1 - 0.004) - 1
    d["cost040"] = {"win_raw_all": float((netraw > 0).mean()), "mean_raw_all": float(netraw.mean()), "max_gain_raw": float(netraw.max()),
                    "min_div": float(net.min()), "win_div_all": float((net > 0).mean())}
    bym = {}
    for mth in ["2026-05", "2026-06", "2026-07", "2026-08", "2026-09", "2026-10"]:
        mk = (tr.closed == 1) & tr.sell_date.str.startswith(mth)
        mk_all = tr.sell_date.str.startswith(mth)
        bym[mth] = {"n_closed": int(mk.sum()), "wins_div": int((net[mk] > 0).sum()), "win_div": float((net[mk] > 0).mean()) if mk.sum() else None,
                    "mean_div": float(net[mk].mean()) if mk.sum() else None, "wins_raw": int((netraw[mk] > 0).sum()),
                    "n_incl_open": int(mk_all.sum()), "wins_div_incl_open": int((net[mk_all] > 0).sum())}
    d["by_exit_month_2026_cost040"] = bym
    return d, tr


sigs = {}
for u in ("twse", "both"):
    for pit, F in (("pit", FP), ("nonpit", FN)):
        sigs[(u, pit)] = L.signals(P, F, K, u)
OUT["signals"] = {f"{u}_{p}": {"rows": int(s.sum()), "stocks": int(s.any(axis=0).sum())} for (u, p), s in sigs.items()}

# ---- 貼文口徑
OUT["post"] = {}; TR = {}
for (u, p), s in sigs.items():
    OUT["post"][f"{u}_{p}"], TR[(u, p)] = post_pack(s, f"{u}_{p}")

# ---- 訊號日當天出場條件已成立的比例（審查用）
ex = K["ex1"] | K["ex2"]
for u in ("twse", "both"):
    s = sigs[(u, "pit")]
    OUT["post"][f"{u}_pit"]["sigday_exit_true_share_all_signal_rows"] = float((s & ex).sum() / s.sum())
    OUT["post"][f"{u}_pit"]["sigday_ex1_true_share"] = float((s & K["ex1"]).sum() / s.sum())
    tr = TR[(u, "pit")]
    ii = tr.i.values; aa = tr.a.values - 1
    OUT["post"][f"{u}_pit"]["entry_sigday_exit_true_share_trades"] = float(ex[aa, ii].mean())
    cl = tr[tr.closed == 1]
    h1 = cl[cl.hold == 1]
    OUT["post"][f"{u}_pit"]["hold1_with_sigday_exit_true"] = float(ex[h1.a.values - 1, h1.i.values].mean())
    # 1 天交易中，訊號日「距 20 日最高收盤」的回落幅度分布
    hc20 = pd.DataFrame(P.Cf).rolling(20, min_periods=20).max().values
    dd_sig = P.Cf[h1.a.values - 1, h1.i.values] / hc20[h1.a.values - 1, h1.i.values] - 1
    OUT["post"][f"{u}_pit"]["hold1_sigday_dd20_median"] = float(np.median(dd_sig))
    OUT["post"][f"{u}_pit"]["hold1_sigday_dd20_le_-3pct_share"] = float((dd_sig <= -0.03).mean())
    # 1 天交易的成因：進場日單日（訊號日收盤→進場日收盤）下跌幅度
    r1 = P.Cf[h1.a.values, h1.i.values] / P.Cf[h1.a.values - 1, h1.i.values] - 1
    OUT["post"][f"{u}_pit"]["hold1_entryday_c2c_median"] = float(np.median(r1))

# ---- 可執行
EX = {}
def add(name, sig, **kw):
    d, r = exec_pack(sig, **kw)
    OUT.setdefault("exec", {})[name] = d
    EX[name] = r
    return d

add("main_twse_pit_normal", sigs[("twse", "pit")])
add("main_zero", sigs[("twse", "pit")], cost="zero")
add("main_stress", sigs[("twse", "pit")], cost="stress")
add("nonpit_normal", sigs[("twse", "nonpit")])
add("nonpit_zero", sigs[("twse", "nonpit")], cost="zero")
add("both_pit_normal", sigs[("both", "pit")])
add("both_pit_zero", sigs[("both", "pit")], cost="zero")
add("main_cash0050", sigs[("twse", "pit")], cash_0050=True)
add("main_no_lot_rule", sigs[("twse", "pit")], board_lot_rule=False)
add("main_no_same_day_slot", sigs[("twse", "pit")], same_day_slot=False)
add("cap_3m", sigs[("twse", "pit")], capital=3_000_000.0)
add("cap_10m", sigs[("twse", "pit")], capital=10_000_000.0)
add("cap_300k", sigs[("twse", "pit")], capital=300_000.0)
# 最有利的合理口徑：上市＋上櫃、空位放 0050、移動停損 10%
K10 = L.tech(P, trail=10.0)
add("trail10_twse", sigs[("twse", "pit")], Kx=K10)
add("best_both_cash0050_trail10", sigs[("both", "pit")], Kx=K10, cash_0050=True)
add("best_both_cash0050_trail10_nolot", sigs[("both", "pit")], Kx=K10, cash_0050=True, board_lot_rule=False)
add("best_both_cash0050_trail10_zero", sigs[("both", "pit")], Kx=K10, cash_0050=True, cost="zero")
add("both_cash0050_trail6", sigs[("both", "pit")], cash_0050=True)
add("both_trail10", sigs[("both", "pit")], Kx=K10)

m = OUT["exec"]
OUT["derived"] = {
    "cost_drag_full_pp": (m["main_zero"]["full"]["ann"] - m["main_twse_pit_normal"]["full"]["ann"]) * 100,
    "cost_drag_2019_pp": (m["main_zero"]["from2019"]["ann"] - m["main_twse_pit_normal"]["from2019"]["ann"]) * 100,
    "nonpit_minus_pit_exec_full_pp": (m["nonpit_normal"]["full"]["ann"] - m["main_twse_pit_normal"]["full"]["ann"]) * 100,
    "nonpit_minus_pit_exec_2019_pp": (m["nonpit_normal"]["from2019"]["ann"] - m["main_twse_pit_normal"]["from2019"]["ann"]) * 100,
    "nonpit_minus_pit_zero_full_pp": (m["nonpit_zero"]["full"]["ann"] - m["main_zero"]["full"]["ann"]) * 100,
    "nonpit_minus_pit_tw_twse_pp": (OUT["post"]["twse_nonpit"]["tw_nocost"]["ann"] - OUT["post"]["twse_pit"]["tw_nocost"]["ann"]) * 100,
}
# 2020-10-26 前後分段（主欄；審查整股限制用）
navm = EX["main_twse_pit_normal"]["nav"]; navl = EX["main_no_lot_rule"]["nav"]
io = P.t_odd - 1
def seg(nav, i0, i1):
    return L.ann(nav[i0], nav[i1], P.dstr[i0], P.dstr[i1])[0]
OUT["lot_review"] = {
    "pre_odd_main_ann": seg(navm, t0, io), "pre_odd_nolot_ann": seg(navl, t0, io), "pre_odd_0050_ann": seg(b50, t0, io),
    "post_odd_main_ann": seg(navm, io, T - 1), "post_odd_nolot_ann": seg(navl, io, T - 1), "post_odd_0050_ann": seg(b50, io, T - 1),
    "pre_odd_avg_npos_main": float(EX["main_twse_pit_normal"]["npos"][t0 + 1:io + 1].mean()),
    "pre_odd_avg_npos_nolot": float(EX["main_no_lot_rule"]["npos"][t0 + 1:io + 1].mean()),
    "post_odd_avg_npos_main": float(EX["main_twse_pit_normal"]["npos"][io + 1:].mean()),
    "pre_odd_cash_w_main": float(EX["main_twse_pit_normal"]["cashw"][t0 + 1:io + 1].mean()),
    "pre_odd_cash_w_nolot": float(EX["main_no_lot_rule"]["cashw"][t0 + 1:io + 1].mean()),
}

txt = json.dumps(OUT, ensure_ascii=False, indent=1, sort_keys=True, default=float)
with open(os.path.join(HERE, "my_numbers.json"), "w") as f:
    f.write(txt)
h = hashlib.sha256(txt.encode()).hexdigest()
with open(os.path.join(HERE, "my_numbers.sha256"), "w") as f:
    f.write(h + "  my_numbers.json\n" + datetime.datetime.utcnow().isoformat() + "Z\n")
# 我方逐筆（凍結後比對用）
TR[("twse", "pit")].drop(columns=["i", "a", "b"]).to_csv(os.path.join(HERE, "my_post_trades_twse_pit.csv.gz"), index=False)
TR[("both", "pit")].drop(columns=["i", "a", "b"]).to_csv(os.path.join(HERE, "my_post_trades_both_pit.csv.gz"), index=False)
EX["main_twse_pit_normal"]["trades"].to_csv(os.path.join(HERE, "my_exec_trades_main.csv.gz"), index=False)
pd.DataFrame({"date": P.dstr[t0:], "main": EX["main_twse_pit_normal"]["nav"][t0:], "zero": EX["main_zero"]["nav"][t0:],
              "e0050": b50[t0:] / b50[t0] * 1e6}).to_csv(os.path.join(HERE, "my_navs.csv.gz"), index=False)
print("sha256", h)
mm = m["main_twse_pit_normal"]
print("MAIN 2019+: ann %.4f 0050 %.4f t %.3f | full ann %.4f 0050 %.4f t %.3f n %d" % (mm["from2019"]["ann"], mm["from2019"]["ann_0050"], mm["from2019"]["t"], mm["full"]["ann"], mm["full"]["ann_0050"], mm["full"]["t"], mm["full"]["n_m"]))
print("ZERO full %.4f drag %.2f turnover %.2f" % (m["main_zero"]["full"]["ann"], OUT["derived"]["cost_drag_full_pp"], mm["turnover_1side"]))
for k in ("twse_pit", "both_pit"):
    q = OUT["post"][k]; print(k, q["n_all"], q["n_closed"], "%.4f" % q["win_raw_all"], q["max_gain_raw"], q["max_loss_div"], q["by_exit_month_2026_cost040"]["2026-09"])
print("nonpit-pit", OUT["derived"])
