# -*- coding: utf-8 -*-
"""審查用：安慰劑的抽樣誤差。用研究員的亂數形狀 (142 月 × 1,723 欄)、pandas 均線，種子 9000..10999 共 2,000 次，
每 200 次一批，看「種子 9000..9199 那一批」在批次分布中的位置。用法：python3 -I -B rw_placebo_more.py <scratch_dir>
輸出 placebo_more.json
"""
import sys, os, json, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import rw_lib as W
S = sys.argv[1]
d = W.load(S); T, N, t0 = d.T, d.N, d.t0; b50 = d.e50
i19 = W.last_before(d, "2019-01-01")
cols = np.load(os.path.join(S, "cols.npz"))["any_mkt_full"]; kept = np.flatnonzero(cols); NK = len(kept)
pik = np.full(N, -1); pik[kept] = np.arange(NK)
PD = pd.DataFrame(d.Cf)
p8 = PD.rolling(8, min_periods=8).mean().values; p20 = PD.rolling(20, min_periods=20).mean().values
with np.errstate(invalid="ignore"):
    cx = np.zeros((T, N), bool); cx[1:] = (p8[1:] < p20[1:]) & (p8[:-1] >= p20[:-1])
del p8, p20, PD
EX = d.tr15 | cx
C_TW = W.cands_from(d.sig_tw, d.excess)
sig_days = [t for t in range(t0, T) if C_TW[t]]
kt = {t: int(d.sig_tw[t].sum()) for t in sig_days}
pool_ix = {t: np.flatnonzero(d.pool[t]) for t in sig_days}
ym = [s[:7] for s in d.dates]; midx = np.array([(int(s[:4]) - 2015) * 12 + int(s[5:7]) - 1 for s in ym]); NM = int(midx.max()) + 1
main = {}
for nm, kw in (("normal", {}), ("zero", dict(cost="zero")), ("e50", dict(idle="e50"))):
    nav = W.simulate(d, C_TW, exit_static=EX, record=False, **kw)["nav"]
    main[nm] = (float(W.ann(d, nav, i19, T - 1)), float(W.ann(d, nav, t0, T - 1)))
NB = 2000
A = {k: dict(a19=[], af=[]) for k in main}
tic = time.time()
for i in range(NB):
    U = np.random.default_rng(9000 + i).random((NM, NK))
    c3 = [None] * T
    for t in sig_days:
        ix = pool_ix[t]; k = kt[t]
        ch = np.sort(ix[np.argsort(U[midx[t], pik[ix]], kind="stable")[:k]]) if k < len(ix) else ix
        c3[t] = ch[np.argsort(-d.excess[t, ch], kind="stable")].tolist()
    for nm, kw in (("normal", {}), ("zero", dict(cost="zero")), ("e50", dict(idle="e50"))):
        nav = W.simulate(d, c3, exit_static=EX, record=False, **kw)["nav"]
        A[nm]["a19"].append(float(W.ann(d, nav, i19, T - 1))); A[nm]["af"].append(float(W.ann(d, nav, t0, T - 1)))
OUT = {"n": NB, "main": main, "sec": time.time() - tic}
for nm in A:
    a19 = np.array(A[nm]["a19"]); af = np.array(A[nm]["af"])
    OUT[nm] = dict(median2019=float(np.median(a19)), mean2019=float(a19.mean()), sd2019=float(a19.std(ddof=1)), p5=float(np.percentile(a19, 5)), p95=float(np.percentile(a19, 95)),
                   pctile2019=float((a19 < main[nm][0]).mean() * 100), medianfull=float(np.median(af)), pctilefull=float((af < main[nm][1]).mean() * 100),
                   pctile2019_vs_researcher_main=float((a19 < {"normal": 0.321721, "zero": 0.376983, "e50": 0.404327}[nm]).mean() * 100),
                   batches=[dict(seeds="%d-%d" % (9000 + k, 9000 + k + 199), median2019=float(np.median(a19[k:k + 200])), mean2019=float(a19[k:k + 200].mean()),
                                 pctile2019=float((a19[k:k + 200] < main[nm][0]).mean() * 100), pctilefull=float((af[k:k + 200] < main[nm][1]).mean() * 100)) for k in range(0, NB, 200)])
json.dump(OUT, open(os.path.join(HERE, "placebo_more.json"), "w"), indent=1)
for nm in A:
    q = OUT[nm]; print(nm, "ALL median %.2f%% mean %.2f%% sd %.2f pct2019 %.1f (vs researcher main %.1f) | full median %.2f%% pct %.1f" % (q["median2019"] * 100, q["mean2019"] * 100, q["sd2019"] * 100, q["pctile2019"], q["pctile2019_vs_researcher_main"], q["medianfull"] * 100, q["pctilefull"]))
    print("   batches:", [(b["seeds"][:4], round(b["median2019"] * 100, 2), b["pctile2019"]) for b in q["batches"]])
