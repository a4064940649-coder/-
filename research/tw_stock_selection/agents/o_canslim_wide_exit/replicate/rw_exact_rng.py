# -*- coding: utf-8 -*-
"""審查用：用研究員「精簡面板」的亂數矩陣形狀 (2,876 日 × 1,723 欄)、(142 月 × 1,723 欄) 逐條重現路徑擾動與安慰劑。
兩種均線實作（我凍結時的逐項相加、pandas rolling）都跑。用法：python3 -I -B rw_exact_rng.py <scratch_dir>
輸出 exact_rng_numbers.json
"""
import sys, os, json, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import rw_lib as W
S = sys.argv[1]
d = W.load(S); T, N, t0 = d.T, d.N, d.t0; b50 = d.e50
i19 = W.last_before(d, "2019-01-01")
cols = np.load(os.path.join(S, "cols.npz"))["any_mkt_full"]
kept = np.flatnonzero(cols); NK = len(kept); assert NK == 1723
pos_in_kept = np.full(N, -1); pos_in_kept[kept] = np.arange(NK)
PD = pd.DataFrame(d.Cf)
p8 = PD.rolling(8, min_periods=8).mean().values; p20 = PD.rolling(20, min_periods=20).mean().values
with np.errstate(invalid="ignore"):
    cross_pd = np.zeros((T, N), bool); cross_pd[1:] = (p8[1:] < p20[1:]) & (p8[:-1] >= p20[:-1])
del p8, p20, PD
IMPL = {"mine": d.cross, "pandas": cross_pd}
C_TW = W.cands_from(d.sig_tw, d.excess)
sig_days = [t for t in range(t0, T) if C_TW[t]]
sig_ix = {t: np.flatnonzero(d.sig_tw[t]) for t in sig_days}
pool_ix = {t: np.flatnonzero(d.pool[t]) for t in sig_days}
assert all((pos_in_kept[pool_ix[t]] >= 0).all() for t in sig_days)
Z = {}
for t in sig_days:
    x = d.excess[t, sig_ix[t]]; sd = x.std()
    Z[t] = (x - x.mean()) / sd if (len(x) > 1 and sd > 0) else np.zeros(len(x))
ym = [s[:7] for s in d.dates]
midx = np.array([(int(s[:4]) - 2015) * 12 + int(s[5:7]) - 1 for s in ym]); NM = int(midx.max()) + 1
a50_19 = W.ann(d, b50, i19, T - 1); a50_f = W.ann(d, b50, t0, T - 1)


def q(v):
    v = np.asarray(v)
    return dict(median=float(np.median(v)), mean=float(v.mean()), p5=float(np.percentile(v, 5)), p10=float(np.percentile(v, 10)),
                p90=float(np.percentile(v, 90)), p95=float(np.percentile(v, 95)), min=float(v.min()), max=float(v.max()), share_gt0=float((v > 0).mean()))


OUT = {"NM": NM, "NK": NK}
tic = time.time()
acc = {k: {im: dict(e19=[], ef=[], a19=[], af=[], buys=[]) for im in IMPL} for k in ("P1", "P2", "PL", "PLZ", "P1E50")}
for i in range(200):
    eps = np.random.default_rng(1000 + i).standard_normal((T, NK))
    c1 = [None] * T
    for t in sig_days:
        ix = sig_ix[t]; key = Z[t] + eps[t, pos_in_kept[ix]]
        c1[t] = ix[np.argsort(-key, kind="stable")].tolist()
    u = np.random.default_rng(5000 + i).random((T, NK))
    c2 = [None] * T
    for t in sig_days:
        ix = sig_ix[t]
        c2[t] = ix[np.argsort(-u[t, pos_in_kept[ix]], kind="stable")].tolist()
    U = np.random.default_rng(9000 + i).random((NM, NK))
    c3 = [None] * T
    for t in sig_days:
        ix = pool_ix[t]; k = len(sig_ix[t])
        ch = np.sort(ix[np.argsort(U[midx[t], pos_in_kept[ix]], kind="stable")[:k]]) if k < len(ix) else ix
        c3[t] = ch[np.argsort(-d.excess[t, ch], kind="stable")].tolist()
    for im, cx in IMPL.items():
        kw = dict(exit_static=d.tr15 | cx, why_parts=None, record=False)
        for nm, cc, extra in (("P1", c1, {}), ("P2", c2, {}), ("PL", c3, {}), ("PLZ", c3, dict(cost="zero")), ("P1E50", c1, dict(idle="e50"))):
            r = W.simulate(d, cc, **kw, **extra); nav = r["nav"]
            a = acc[nm][im]
            a["a19"].append(W.ann(d, nav, i19, T - 1)); a["af"].append(W.ann(d, nav, t0, T - 1)); a["buys"].append(r["cnt"]["buy"])
print("sec", time.time() - tic)
main = {}
for im, cx in IMPL.items():
    kw = dict(exit_static=d.tr15 | cx, why_parts=None, record=False)
    main[im] = {}
    for nm, extra in (("MAIN", {}), ("ZERO", dict(cost="zero")), ("E50", dict(idle="e50"))):
        nav = W.simulate(d, C_TW, **kw, **extra)["nav"]
        main[im][nm] = (float(W.ann(d, nav, i19, T - 1)), float(W.ann(d, nav, t0, T - 1)))
for nm in acc:
    OUT[nm] = {}
    for im in IMPL:
        a = acc[nm][im]; a19 = np.array(a["a19"]); af = np.array(a["af"])
        base = main[im]["ZERO" if nm == "PLZ" else ("E50" if nm == "P1E50" else "MAIN")]
        OUT[nm][im] = dict(ex2019_pp=q((a19 - a50_19) * 100), exfull_pp=q((af - a50_f) * 100), ann2019=q(a19), annfull=q(af),
                           buys_mean=float(np.mean(a["buys"])), buys_min=int(np.min(a["buys"])), buys_max=int(np.max(a["buys"])),
                           main_ann2019=base[0], main_pctile_2019=float((a19 < base[0]).mean()), main_pctile_full=float((af < base[1]).mean()),
                           beat_0050_2019=float((a19 > a50_19).mean()))
json.dump(OUT, open(os.path.join(HERE, "exact_rng_numbers.json"), "w"), indent=1)
R = json.load(open(os.path.join(os.path.dirname(HERE), "results.json")))
print("THEIRS P1", R["PATH"]["P1_z_plus_noise"]["from2019_excess_pp"], R["PATH"]["P1_z_plus_noise"]["main_pctile_in_paths"])
for im in IMPL: print("MINE", im, {k: round(v, 4) for k, v in OUT["P1"][im]["ex2019_pp"].items()}, OUT["P1"][im]["main_pctile_2019"])
print("THEIRS P2", R["PATH"]["P2_pure_random"]["from2019_excess_pp"])
for im in IMPL: print("MINE", im, {k: round(v, 4) for k, v in OUT["P2"][im]["ex2019_pp"].items()}, OUT["P2"][im]["main_pctile_2019"])
print("THEIRS PLACEBO", R["PLACEBO"]["from2019_cagr"], R["PLACEBO"]["full_cagr"], R["PLACEBO"]["n_buy"], R["PLACEBO"]["placebo_share_beating_0050_from2019"])
for im in IMPL: print("MINE", im, {k: round(v, 5) for k, v in OUT["PL"][im]["ann2019"].items()}, OUT["PL"][im]["main_pctile_2019"], "full med", round(OUT["PL"][im]["annfull"]["median"], 5), OUT["PL"][im]["main_pctile_full"], OUT["PL"][im]["buys_mean"], OUT["PL"][im]["buys_min"], OUT["PL"][im]["buys_max"], OUT["PL"][im]["beat_0050_2019"])
print("THEIRS PLACEBO ZERO", R["POSTHOC"]["placebo_zero_cost"])
for im in IMPL: print("MINE", im, {k: round(v, 5) for k, v in OUT["PLZ"][im]["ann2019"].items()}, OUT["PLZ"][im]["main_pctile_2019"], "full", round(OUT["PLZ"][im]["annfull"]["median"], 5), OUT["PLZ"][im]["main_pctile_full"])
print("THEIRS P1 E50", R["POSTHOC"]["e50_version"]["path_P1_from2019_excess_pp"])
for im in IMPL: print("MINE", im, {k: round(v, 4) for k, v in OUT["P1E50"][im]["ex2019_pp"].items()})
