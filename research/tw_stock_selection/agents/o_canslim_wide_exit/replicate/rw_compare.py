# -*- coding: utf-8 -*-
"""凍結後比對：我的引擎（重跑、先斷言等於凍結值）對研究員的 trades.csv.gz / navs.csv.gz / results.json。
用法：python3 -I -B rw_compare.py <scratch_dir>     輸出 compare_vs_researcher.json
"""
import sys, os, json, gzip, csv
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import rw_lib as W

S = sys.argv[1]
RD = os.path.dirname(HERE)
d = W.load(S)
T, t0 = d.T, d.t0
FROZEN = json.load(open(os.path.join(HERE, "my_numbers.json")))
TR = {12: d.tr12, 15: d.tr15, 20: d.tr20}


def spec(name):
    parts = []; fs = None; hd = None
    if name == "HOLD60": hd = 60
    elif name == "MA50_FS8": parts = [("ma50", d.ma50)]; fs = 8.0
    else:
        for tk in name.split("_"):
            if tk.startswith("T") and tk[1:].isdigit(): parts.append(("trail", TR[int(tk[1:])]))
            elif tk == "MA": parts.append(("cross", d.cross))
            elif tk == "FS8": fs = 8.0
    st = None
    for _, a in parts: st = a if st is None else (st | a)
    return dict(exit_static=st, why_parts=parts, fstop=fs, hold=hd)


C = {"TW": W.cands_from(d.sig_tw, d.excess), "ALL": W.cands_from(d.sig_all, d.excess), "TP": W.cands_from(d.sig_tp, d.excess),
     "EXT": W.cands_from(d.sig_ext, d.excess, t_from=d.t_ext), "POOL": W.cands_from(d.pool, d.excess)}
COLS = {"MAIN": ("TW", {}), "STRESS": ("TW", dict(cost="stress")), "ZERO": ("TW", dict(cost="zero")), "E50": ("TW", dict(idle="e50")),
        "E50_STRESS": ("TW", dict(idle="e50", cost="stress")), "ALL": ("ALL", {}), "ALL_E50": ("ALL", dict(idle="e50")),
        "NEXTDAY": ("TW", dict(same_day=False)), "NOLOT": ("TW", dict(lot_rule=False)), "TPEX": ("TP", {}),
        "S20": ("TW", dict(nmax=20)), "C30": ("TW", dict(capital=300_000.0)), "C300": ("TW", dict(capital=3_000_000.0)),
        "C1000": ("TW", dict(capital=10_000_000.0)),
        "EXT": ("EXT", dict(t_start=d.t_ext)), "EXT_NOLOT": ("EXT", dict(t_start=d.t_ext, lot_rule=False))}


def my_run(variant):
    if variant == "POOL_RS":
        return W.simulate(d, C["POOL"], **spec("T15_MA"))
    if variant == "POOL_RS_ZERO":
        return W.simulate(d, C["POOL"], cost="zero", **spec("T15_MA"))
    cand, col = variant.split("|")
    ck, kw = COLS[col]
    return W.simulate(d, C[ck], **spec(cand), **kw)


r0 = my_run("T15_MA|MAIN")
assert abs(r0["nav"][-1] - FROZEN["main"]["MAIN"]["end_nav"]) < 1e-6, "主欄不等於凍結值"

rt = pd.read_csv(os.path.join(RD, "trades.csv.gz"), dtype={"code": str})
rn = pd.read_csv(os.path.join(RD, "navs.csv.gz"), index_col=0)
rn.index = [str(x)[:10] for x in rn.index]
OUT = {"trades": {}, "navs": {}}
i19 = W.last_before(d, "2019-01-01")
for variant in list(rn.columns):
    if variant in ("0050", "V7X_jmean_norm"): continue
    try:
        r = my_run(variant)
    except KeyError:
        continue
    nav = pd.Series(r["nav"], index=d.dates).dropna()
    theirs = rn[variant].dropna()
    ix = nav.index.intersection(theirs.index)
    rel = (nav[ix] / theirs[ix] - 1)
    first_1bp = next((k for k in ix if abs(rel[k]) > 1e-4), None)
    first_10bp = next((k for k in ix if abs(rel[k]) > 1e-3), None)
    q = dict(n_days=int(len(ix)), end_mine=float(nav.iloc[-1]), end_theirs=float(theirs.iloc[-1]), end_rel=float(rel.iloc[-1]),
             max_abs_rel=float(rel.abs().max()), first_gt_1bp=first_1bp, first_gt_10bp=first_10bp)
    if variant.split("|")[-1] not in ("EXT", "EXT_NOLOT"):
        a_m = W.ann(d, r["nav"], i19, T - 1)
        th = theirs.reindex(d.dates).values
        a_t = W.ann(d, th, i19, T - 1)
        q["ann2019_mine"] = float(a_m); q["ann2019_theirs"] = float(a_t); q["ann2019_diff_pp"] = float((a_m - a_t) * 100)
        a_m = W.ann(d, r["nav"], t0, T - 1); a_t = W.ann(d, th, t0, T - 1)
        q["annfull_diff_pp"] = float((a_m - a_t) * 100)
    OUT["navs"][variant] = q
    sub = rt[rt.variant == variant]
    if len(sub):
        mine = pd.DataFrame([(d.codes[i], d.dates[tb], d.dates[te], c, p, w, sh) for (i, tb, te, c, p, w, sh, bo, tg) in r["trades"]],
                            columns=["code", "ent_date", "exit_date", "cost", "proceeds", "why", "shares"])
        m = mine.merge(sub, on=["code", "ent_date"], how="outer", suffixes=("_m", "_r"), indicator=True).sort_values(["ent_date", "code"])
        only_m = m[m._merge == "left_only"]; only_r = m[m._merge == "right_only"]; both = m[m._merge == "both"]
        dexit = both[both.exit_date_m != both.exit_date_r]
        dsh = both[both.shares_m != both.shares_r]
        wm = both.why_m.str.replace("open.*", "open_end", regex=True)
        dwhy = both[(wm != both.why_r)]
        first_diff = None
        cand_rows = pd.concat([only_m, only_r, dexit]).sort_values(["ent_date", "code"])
        if len(cand_rows):
            x = cand_rows.iloc[0]
            first_diff = {k: (None if (isinstance(x[k], float) and x[k] != x[k]) else (x[k] if not hasattr(x[k], "item") else x[k].item()))
                          for k in ("code", "ent_date", "exit_date_m", "exit_date_r", "shares_m", "shares_r", "cost_m", "cost_r", "why_m", "why_r")}
            first_diff["kind"] = str(x["_merge"])
        first_sh = None
        if len(dsh):
            x = dsh.iloc[0]
            first_sh = dict(code=x.code, ent_date=x.ent_date, shares_m=int(x.shares_m), shares_r=int(x.shares_r), cost_m=float(x.cost_m), cost_r=float(x.cost_r))
        OUT["trades"][variant] = dict(n_mine=int(len(mine)), n_theirs=int(len(sub)), n_both=int(len(both)), only_mine=int(len(only_m)),
                                      only_theirs=int(len(only_r)), exit_date_diff=int(len(dexit)), shares_diff=int(len(dsh)),
                                      why_diff=int(len(dwhy)), first_diff=first_diff, first_shares_diff=first_sh,
                                      max_abs_ret_diff=float(((both.proceeds_m / both.cost_m) - (both.proceeds_r / both.cost_r)).abs().max()))
json.dump(OUT, open(os.path.join(HERE, "compare_vs_researcher.json"), "w"), ensure_ascii=False, indent=1, default=str)
print("%-22s %6s %6s %5s %5s %5s %5s %8s %9s %9s  first-diff" % ("variant", "mine", "theirs", "onlyM", "onlyR", "exitD", "shD", "endRel%", "d2019pp", "1st>10bp"))
for v, q in OUT["navs"].items():
    tr = OUT["trades"].get(v)
    fd = ""
    if tr and tr["first_diff"]:
        f = tr["first_diff"]; fd = "%s %s ent %s exit m=%s r=%s" % (f["kind"], f["code"], f["ent_date"], f["exit_date_m"], f["exit_date_r"])
    print("%-22s %6s %6s %5s %5s %5s %5s %8.3f %9s %9s  %s" % (
        v, tr["n_mine"] if tr else "", tr["n_theirs"] if tr else "", tr["only_mine"] if tr else "", tr["only_theirs"] if tr else "",
        tr["exit_date_diff"] if tr else "", tr["shares_diff"] if tr else "", q["end_rel"] * 100,
        ("%+.3f" % q["ann2019_diff_pp"]) if "ann2019_diff_pp" in q else "", q["first_gt_10bp"], fd))
