# -*- coding: utf-8 -*-
"""審查加跑第 2 部分（凍結後）：(1) SPA 修正重算（讀 my_navs.csv.gz，不重跑引擎）；(2) 零碎買進（現金只剩一點時買 1 股）的筆數與敏感度。
用法：python3 -I -B rw_review2.py <scratch_dir>     輸出 review2_numbers.json，並把修正後的 SPA 寫回 review_numbers.json
"""
import sys, os, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import rw_lib as W
S = sys.argv[1]; RD = os.path.dirname(HERE)
d = W.load(S); T, t0 = d.T, d.t0; b50 = d.e50
i19 = W.last_before(d, "2019-01-01"); i23 = W.last_before(d, "2024-01-01")
FROZEN = json.load(open(os.path.join(HERE, "my_numbers.json")))
OUT = {}
CANDS = ["T15_MA", "T12_MA", "T20_MA", "T12", "T15", "T20", "FS8_T15", "FS8_T15_MA", "MA50_FS8", "HOLD60"]
# ---------------- (1) SPA
nv = pd.read_csv(os.path.join(HERE, "my_navs.csv.gz"), index_col=0)
XS = []
for c in CANDS:
    nav = nv[c + "|MAIN"].reindex(d.dates).values
    XS.append(W.mret(d, nav, t0, T - 1)[0] - W.mret(d, b50, t0, T - 1)[0])
XS = np.array(XS).T


def spa(X, B=10000, q=1 / 6, seed=20261114):
    n, k = X.shape; rng = np.random.default_rng(seed); dbar = X.mean(axis=0)
    idx = np.zeros((B, n), int); idx[:, 0] = rng.integers(0, n, B)
    jump = rng.random((B, n)) < q; newi = rng.integers(0, n, (B, n))
    for t in range(1, n):
        idx[:, t] = np.where(jump[:, t], newi[:, t], (idx[:, t - 1] + 1) % n)
    bm = X[idx].mean(axis=1); om = np.sqrt(n) * bm.std(axis=0, ddof=1)
    Tn = max(0.0, float((np.sqrt(n) * dbar / om).max()))
    mu_c = dbar * ((np.sqrt(n) * dbar / om) >= -np.sqrt(2 * np.log(np.log(n))))
    mu_l = np.minimum(dbar, 0.0)
    res = dict(T=Tn, k=k, n=n)
    for nm, mu in (("p_consistent", mu_c), ("p_lower", mu_l), ("p_upper", np.zeros(k))):
        Tb = np.maximum(0.0, (np.sqrt(n) * (bm - mu) / om).max(axis=1))
        res[nm] = float((Tb >= Tn).mean())
    return res
OUT["spa_full_10"] = spa(XS); OUT["spa_2019_10"] = spa(XS[-94:])
rv = json.load(open(os.path.join(HERE, "review_numbers.json")))
rv["spa_full_10_SUPERSEDED_buggy"] = rv.get("spa_full_10"); rv["spa_2019_10_SUPERSEDED_buggy"] = rv.get("spa_2019_10")
rv["spa_full_10"] = OUT["spa_full_10"]; rv["spa_2019_10"] = OUT["spa_2019_10"]
json.dump(rv, open(os.path.join(HERE, "review_numbers.json"), "w"), ensure_ascii=False, indent=1, default=float)

# ---------------- (2) 零碎買進
mt = pd.read_csv(os.path.join(HERE, "my_trades_main.csv.gz"), dtype={"code": str})
navM = nv["T15_MA|MAIN"]
prev = {d.dates[t]: d.dates[t - 1] for t in range(1, T)}
mt["target"] = [navM[prev[x]] / 10 for x in mt.buy_date]
mt["frac"] = mt.cost / mt.target
rt = pd.read_csv(os.path.join(RD, "trades.csv.gz"), dtype={"code": str}); rr = rt[rt.variant == "T15_MA|MAIN"].copy()
rn = pd.read_csv(os.path.join(RD, "navs.csv.gz"), index_col=0); rn.index = [str(x)[:10] for x in rn.index]
rr["target"] = [rn["T15_MA|MAIN"][prev[x]] / 10 for x in rr.ent_date]; rr["frac"] = rr.cost / rr.target
def dust(df, shares_col):
    return dict(n=int(len(df)), lt50=int((df.frac < 0.5).sum()), lt25=int((df.frac < 0.25).sum()), lt10=int((df.frac < 0.10).sum()), lt1=int((df.frac < 0.01).sum()),
                le10_shares=int((df[shares_col] <= 10).sum()), one_share=int((df[shares_col] == 1).sum()),
                lt10_after_odd=int(((df.frac < 0.10) & (df.iloc[:, 1 if shares_col == "shares" and "buy_date" in df else 3] >= "2020-10-26")).sum()) if False else None,
                median_frac=float(df.frac.median()))
OUT["dust_mine_main"] = dust(mt, "shares"); OUT["dust_researcher_main"] = dust(rr, "shares")
x = rr[rr.frac < 0.01].sort_values("ent_date")
OUT["dust_researcher_examples"] = [dict(code=a.code, ent=a.ent_date, exit=a.exit_date, shares=int(a.shares), cost=float(a.cost), ret=float(a.ret)) for a in x.head(12).itertuples()]
for v in sorted(rt.variant.unique()):
    s = rt[rt.variant == v]
    OUT.setdefault("dust_researcher_all_variants_le10shares", {})[v] = int((s.shares <= 10).sum())
# 敏感度：預算不到目標 50%／25% 就不買（空位留到下次）
TRL = {12: d.tr12, 15: d.tr15, 20: d.tr20}
def spec(name):
    parts = []; fs = None; hd = None
    if name == "HOLD60": hd = 60
    elif name == "MA50_FS8": parts = [("ma50", d.ma50)]; fs = 8.0
    else:
        for tk in name.split("_"):
            if tk.startswith("T") and tk[1:].isdigit(): parts.append(("trail", TRL[int(tk[1:])]))
            elif tk == "MA": parts.append(("cross", d.cross))
            elif tk == "FS8": fs = 8.0
    st = None
    for _, a in parts: st = a if st is None else (st | a)
    return dict(exit_static=st, why_parts=parts, fstop=fs, hold=hd)
C_TW = W.cands_from(d.sig_tw, d.excess)
r0 = W.simulate(d, C_TW, **spec("T15_MA"))
assert abs(r0["nav"][-1] - FROZEN["main"]["MAIN"]["end_nav"]) < 1e-6, "加了 min_frac 選項後主欄仍須等於凍結值"
a19 = W.ann(d, b50, i19, T - 1); af = W.ann(d, b50, t0, T - 1); a24 = W.ann(d, b50, i23, T - 1)
OUT["min_frac"] = {}
for mf in (None, 0.25, 0.5):
    q = {}
    for c in CANDS:
        for col, kw in (("MAIN", {}), ("STRESS", dict(cost="stress"))):
            if col == "STRESS" and c != "T15_MA": continue
            r = W.simulate(d, C_TW, min_frac=mf, **spec(c), **kw); nav = r["nav"]
            q[c + "|" + col] = dict(ex2019=float((W.ann(d, nav, i19, T - 1) - a19) * 100), exfull=float((W.ann(d, nav, t0, T - 1) - af) * 100),
                                    ex2024=float((W.ann(d, nav, i23, T - 1) - a24) * 100), t2019=W.window(d, nav, b50, i19, T - 1)["t"],
                                    buys=int(r["cnt"]["buy"]), dust_skip=int(r["cnt"].get("dust_skip", 0)))
    OUT["min_frac"][str(mf)] = q
json.dump(OUT, open(os.path.join(HERE, "review2_numbers.json"), "w"), ensure_ascii=False, indent=1, default=float)
print("SPA", OUT["spa_full_10"], OUT["spa_2019_10"])
print("dust mine", OUT["dust_mine_main"]); print("dust theirs", OUT["dust_researcher_main"]); print(OUT["dust_researcher_examples"])
print({k: v for k, v in OUT["dust_researcher_all_variants_le10shares"].items()})
for mf, q in OUT["min_frac"].items():
    print(mf, {k: (round(v["ex2019"], 2), round(v["ex2024"], 2), v["buys"], v["dust_skip"]) for k, v in q.items()})
