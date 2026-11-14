# -*- coding: utf-8 -*-
"""獨立複製主程式（o_canslim_wide_exit）：產生 my_numbers.json 並記 sha256（凍結）。
用法：python3 -I -B rw_run.py <scratch_dir>
此檔不讀研究員的 results.json／trades.csv.gz／navs.csv.gz，也不 import cwlib／run／cslib。
"""
import sys, os, json, math, hashlib, datetime, time, resource, gzip
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import rw_lib as W

S = sys.argv[1]
NPATH = int(sys.argv[2]) if len(sys.argv) > 2 else 200
OUTDIR = HERE if NPATH == 200 else S      # 乾跑（路徑數不是 200）只寫暫存夾，不產生凍結檔
d = W.load(S)
T, N, t0 = d.T, d.N, d.t0
b50 = d.e50
i19 = W.last_before(d, "2019-01-01")
i23 = W.last_before(d, "2024-01-01")
OUT = {"meta": {"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "T": T, "N": N, "start": d.dates[t0],
                "base_2019": d.dates[i19], "base_2024": d.dates[i23], "end": d.dates[-1], "ext_start": d.dates[d.t_ext],
                "signals_twse": int(d.sig_tw.sum()), "signals_all": int(d.sig_all.sum()), "signals_tpex": int(d.sig_tp.sum()),
                "signals_ext_segment": int(d.sig_ext[d.t_ext:t0].sum()), "npath": NPATH}}

# ---------------------------------------------------------------- 候選的出場規則
TR = {12: d.tr12, 15: d.tr15, 20: d.tr20}
def spec(name):
    """回傳 simulate 的出場參數。"""
    parts = []; fs = None; hd = None
    if name == "HOLD60":
        hd = 60
    elif name == "MA50_FS8":
        parts = [("ma50", d.ma50)]; fs = 8.0
    else:
        toks = name.split("_")
        for tk in toks:
            if tk.startswith("T") and tk[1:].isdigit():
                parts.append(("trail", TR[int(tk[1:])]))
            elif tk == "MA":
                parts.append(("cross", d.cross))
            elif tk == "FS8":
                fs = 8.0
    st = None
    for _, a in parts:
        st = a if st is None else (st | a)
    return dict(exit_static=st, why_parts=parts, fstop=fs, hold=hd)

CANDS = ["T15_MA", "T12_MA", "T20_MA", "T12", "T15", "T20", "FS8_T15", "FS8_T15_MA", "MA50_FS8", "HOLD60"]
SPEC = {c: spec(c) for c in CANDS}
C_TW = W.cands_from(d.sig_tw, d.excess)
C_ALL = W.cands_from(d.sig_all, d.excess)
C_TP = W.cands_from(d.sig_tp, d.excess)
C_EXT = W.cands_from(d.sig_ext, d.excess, t_from=d.t_ext)
C_POOL = W.cands_from(d.pool, d.excess)
YEARS = (d.dnum[-1] - d.dnum[t0]).astype(int) / 365.25


def pack(r, full=True, i0=None, i1=None):
    nav = r["nav"]; q = {}
    if i0 is not None:
        q["seg"] = W.window(d, nav, b50, i0, i1)
        return q
    q["full"] = W.window(d, nav, b50, t0, T - 1)
    q["from2019"] = W.window(d, nav, b50, i19, T - 1)
    q["seg_2017_2018"] = W.window(d, nav, b50, t0, i19)
    q["seg_2019_2023"] = W.window(d, nav, b50, i19, i23)
    q["seg_2024_2026"] = W.window(d, nav, b50, i23, T - 1)
    q["drop_best3"] = W.drop_best3(d, nav, b50, t0, T - 1)
    q["end_nav"] = float(nav[-1])
    avgnav = float(np.mean(nav[t0:]))
    c = r["cnt"]
    q["turnover_1side"] = float(c["buy_amt"] / avgnav / YEARS)
    q["cost_acct_pct_yr"] = {k: float(c[k] / avgnav / YEARS * 100) for k in ("fee", "tax", "slip", "e50_cost")}
    q["avg_npos"] = float(r["npos"][t0 + 1:].mean()); q["avg_idle_w"] = float(r["idlew"][t0 + 1:].mean())
    q["cnt"] = {k: (float(v) if isinstance(v, float) else int(v)) for k, v in c.items()}
    tr = r["trades"]
    if tr:
        ret = np.array([x[4] / x[3] - 1 for x in tr]); hold = np.array([x[2] - x[1] for x in tr])
        opn = np.array([x[5].startswith("open") for x in tr])
        w = ret > 0
        q["trades"] = dict(n=int(len(tr)), n_open=int(opn.sum()), win=float(w.mean()), avg_win=float(ret[w].mean()),
                           avg_loss=float(-ret[~w].mean()), expectancy=float(w.mean() * ret[w].mean() + (1 - w.mean()) * ret[~w].mean()),
                           mean=float(ret.mean()), hold_mean_all=float(hold.mean()), hold_median_all=float(np.median(hold)),
                           hold_mean_closed=float(hold[~opn].mean()))
        by = {}
        for x, rr, hh in zip(tr, ret, hold):
            k = "open" if x[5].startswith("open") else x[5]
            by.setdefault(k, []).append((rr, hh))
        q["exit_reason"] = {k: dict(n=len(v), mean=float(np.mean([a for a, _ in v])), win=float(np.mean([a > 0 for a, _ in v])),
                                    hold=float(np.mean([h for _, h in v]))) for k, v in by.items()}
        pnl = np.array([x[4] - x[3] for x in tr])
        q["pnl_sum"] = float(pnl.sum())
    return q


def run(cands, name, **kw):
    a = dict(SPEC[name]); a.update(kw)
    return W.simulate(d, cands, **a)


tic = time.time()
RES = {}; NAVS = {}
# ---------------------------------------------------------------- 主規格各欄
M = "T15_MA"
cols = {
    "MAIN": (C_TW, {}), "STRESS": (C_TW, dict(cost="stress")), "ZERO": (C_TW, dict(cost="zero")),
    "E50": (C_TW, dict(idle="e50")), "E50_STRESS": (C_TW, dict(idle="e50", cost="stress")), "E50_ZERO": (C_TW, dict(idle="e50", cost="zero")),
    "ALL": (C_ALL, {}), "ALL_E50": (C_ALL, dict(idle="e50")), "S20": (C_TW, dict(nmax=20)),
    "C30": (C_TW, dict(capital=300_000.0)), "C300": (C_TW, dict(capital=3_000_000.0)), "C1000": (C_TW, dict(capital=10_000_000.0)),
    "NOLOT": (C_TW, dict(lot_rule=False)), "NEXTDAY": (C_TW, dict(same_day=False)),
    "TPEX": (C_TP, {}), "TPEX_ZERO": (C_TP, dict(cost="zero")),
}
OUT["main"] = {}
for k, (cc, kw) in cols.items():
    r = run(cc, M, **kw)
    OUT["main"][k] = pack(r); NAVS[M + "|" + k] = r["nav"]; RES[k] = r
# 延伸段
for k, kw in (("EXT", {}), ("EXT_NOLOT", dict(lot_rule=False))):
    r = run(C_EXT, M, t_start=d.t_ext, **kw)
    q = W.window(d, r["nav"], b50, d.t_ext, t0)
    tr = [x for x in r["trades"] if x[1] <= t0]
    ret = np.array([x[4] / x[3] - 1 for x in tr])
    q.update(n_buys_in_segment=len(tr), win=float((ret > 0).mean()), mean=float(ret.mean()),
             full_from_ext=W.window(d, r["nav"], b50, d.t_ext, T - 1))
    OUT["main"][k] = q; NAVS[M + "|" + k] = r["nav"]
mm = OUT["main"]
OUT["derived"] = {
    "cost_drag_full_pp": (mm["ZERO"]["full"]["ann"] - mm["MAIN"]["full"]["ann"]) * 100,
    "cost_drag_2019_pp": (mm["ZERO"]["from2019"]["ann"] - mm["MAIN"]["from2019"]["ann"]) * 100,
    "stress_drag_full_pp": (mm["ZERO"]["full"]["ann"] - mm["STRESS"]["full"]["ann"]) * 100,
    "identity_gap": mm["MAIN"]["end_nav"] - 1_000_000.0 - mm["MAIN"]["pnl_sum"],
}
print("main cols done", time.time() - tic)

# ---------------------------------------------------------------- 10 個候選 × 7 欄
OUT["cands"] = {}
grid = {"MAIN": (C_TW, {}), "STRESS": (C_TW, dict(cost="stress")), "ZERO": (C_TW, dict(cost="zero")), "E50": (C_TW, dict(idle="e50")),
        "ALL": (C_ALL, {}), "NEXTDAY": (C_TW, dict(same_day=False)), "NOLOT": (C_TW, dict(lot_rule=False))}
for c in CANDS:
    OUT["cands"][c] = {}
    for k, (cc, kw) in grid.items():
        r = run(cc, c, **kw)
        q = pack(r)
        if k != "MAIN":
            q = {kk: q[kk] for kk in ("full", "from2019", "seg_2024_2026", "turnover_1side")}
        OUT["cands"][c][k] = q
        NAVS[c + "|" + k] = r["nav"]
print("cands done", time.time() - tic)

# ---------------------------------------------------------------- 路徑擾動
sig_days = [t for t in range(t0, T) if C_TW[t]]
sig_ix = {t: np.flatnonzero(d.sig_tw[t]) for t in sig_days}
Z = {}
for t in sig_days:
    x = d.excess[t, sig_ix[t]]
    sd = x.std()
    Z[t] = (x - x.mean()) / sd if (len(x) > 1 and sd > 0) else np.zeros(len(x))


def perturb(kind, i):
    out = [None] * T
    if kind == "P1":
        eps = np.random.default_rng(1000 + i).standard_normal((T, N))
        for t in sig_days:
            ix = sig_ix[t]; key = Z[t] + eps[t, ix]
            out[t] = ix[np.argsort(-key, kind="stable")].tolist()
    else:
        u = np.random.default_rng(5000 + i).random((T, N))
        for t in sig_days:
            ix = sig_ix[t]
            out[t] = ix[np.argsort(-u[t, ix], kind="stable")].tolist()
    return out


def dist(v):
    v = np.asarray(v)
    return dict(median=float(np.median(v)), p10=float(np.percentile(v, 10)), p90=float(np.percentile(v, 90)), p5=float(np.percentile(v, 5)),
                p95=float(np.percentile(v, 95)), mean=float(v.mean()), pos_share=float((v > 0).mean()), n=int(len(v)), min=float(v.min()), max=float(v.max()))


a50_19 = W.ann(d, b50, i19, T - 1); a50_f = W.ann(d, b50, t0, T - 1)
OUT["perturb"] = {}
for kind, variants in (("P1", ("MAIN", "E50")), ("P2", ("MAIN",))):
    acc = {v: {"ex19": [], "exf": [], "buys": []} for v in variants}
    for i in range(NPATH):
        cc = perturb(kind, i)
        for v in variants:
            r = run(cc, M, record=False, **({"idle": "e50"} if v == "E50" else {}))
            nav = r["nav"]
            acc[v]["ex19"].append((W.ann(d, nav, i19, T - 1) - a50_19) * 100)
            acc[v]["exf"].append((W.ann(d, nav, t0, T - 1) - a50_f) * 100)
            acc[v]["buys"].append(r["cnt"]["buy"])
    for v in variants:
        base = OUT["main"][v]
        OUT["perturb"][kind + "_" + v] = dict(
            ex2019=dist(acc[v]["ex19"]), exfull=dist(acc[v]["exf"]), buys_mean=float(np.mean(acc[v]["buys"])),
            main_pctile_2019=float((np.array(acc[v]["ex19"]) < base["from2019"]["excess_pp"]).mean() * 100),
            raw_ex2019=[float(x) for x in acc[v]["ex19"]])
    print(kind, "done", time.time() - tic)

# ---------------------------------------------------------------- 安慰劑與池子基準
ym = [s[:7] for s in d.dates]
y0, m0 = int(ym[0][:4]), int(ym[0][5:7])
midx = np.array([(int(s[:4]) - y0) * 12 + int(s[5:7]) - m0 for s in ym])
NM = int(midx.max()) + 1
pool_ix = {t: np.flatnonzero(d.pool[t]) for t in sig_days}
kt = {t: len(sig_ix[t]) for t in sig_days}
OUT["meta"]["placebo_k_gt_pool_days"] = int(sum(kt[t] > len(pool_ix[t]) for t in sig_days))
OUT["meta"]["signal_not_in_pool"] = int(sum(len(np.setdiff1d(sig_ix[t], pool_ix[t])) for t in sig_days))


def placebo(i, kscale=1.0):
    U = np.random.default_rng(9000 + i).random((NM, N))
    out = [None] * T
    for t in sig_days:
        ix = pool_ix[t]; k = kt[t]
        if kscale != 1.0:
            k = int(round(k * kscale))
            if k < 1: k = 1
        if k < len(ix):
            u = U[midx[t], ix]
            ch = np.sort(ix[np.argsort(u, kind="stable")[:k]])
        else:
            ch = ix
        out[t] = ch[np.argsort(-d.excess[t, ch], kind="stable")].tolist()
    return out


pl = {"normal": {"a19": [], "af": [], "buys": [], "dd": []}, "zero": {"a19": [], "af": [], "buys": [], "dd": []},
      "e50": {"a19": [], "af": [], "buys": [], "dd": []}}
for i in range(NPATH):
    cc = placebo(i)
    for v, kw in (("normal", {}), ("zero", dict(cost="zero")), ("e50", dict(idle="e50"))):
        r = run(cc, M, record=False, **kw); nav = r["nav"]
        pl[v]["a19"].append(W.ann(d, nav, i19, T - 1)); pl[v]["af"].append(W.ann(d, nav, t0, T - 1))
        pl[v]["buys"].append(r["cnt"]["buy"]); pl[v]["dd"].append(W.maxdd(nav[t0:]))
OUT["placebo"] = {}
for v, col in (("normal", "MAIN"), ("zero", "ZERO"), ("e50", "E50")):
    a19 = np.array(pl[v]["a19"]); af = np.array(pl[v]["af"]); base = OUT["main"][col]
    OUT["placebo"][v] = dict(
        ann2019=dist(a19), annfull=dist(af), buys_mean=float(np.mean(pl[v]["buys"])), buys_min=int(np.min(pl[v]["buys"])),
        buys_max=int(np.max(pl[v]["buys"])), maxdd_median=float(np.median(pl[v]["dd"])),
        main_pctile_2019=float((a19 < base["from2019"]["ann"]).mean() * 100), main_pctile_full=float((af < base["full"]["ann"]).mean() * 100),
        beat_0050_2019=float((a19 > a50_19).mean()), raw_ann2019=[float(x) for x in a19], raw_annfull=[float(x) for x in af],
        raw_buys=[int(x) for x in pl[v]["buys"]])
print("placebo done", time.time() - tic)
OUT["pool"] = {}
for k, kw in (("POOL_RS", {}), ("POOL_RS_ZERO", dict(cost="zero")), ("POOL_RS_E50", dict(idle="e50"))):
    r = run(C_POOL, M, **kw)
    OUT["pool"][k] = pack(r); NAVS["POOL|" + k] = r["nav"]

# ---------------------------------------------------------------- β 調整
OUT["beta"] = {}
for k in ("MAIN", "E50", "ALL"):
    nav = NAVS[M + "|" + k]
    for wn, i0 in (("full", t0), ("from2019", i19)):
        rs, _ = W.mret(d, nav, i0, T - 1); rb, _ = W.mret(d, b50, i0, T - 1)
        q = W.ols_beta(rs, rb)
        mix = np.prod(1 + q["beta"] * rb)
        q["beta_matched_ann_m"] = float(mix ** (12 / len(rb)) - 1)
        q["strat_ann_m"] = float(np.prod(1 + rs) ** (12 / len(rs)) - 1); q["e50_ann_m"] = float(np.prod(1 + rb) ** (12 / len(rb)) - 1)
        OUT["beta"][k + "_" + wn] = q

# ---------------------------------------------------------------- 檢定力
def norm_sf(x):
    return 0.5 * math.erfc(x / math.sqrt(2))
f = mm["MAIN"]["full"]; g = mm["MAIN"]["from2019"]
se = f["sd_m"] / math.sqrt(f["n"])
OUT["power"] = {
    "sd_full": f["sd_m"], "sd_2019": g["sd_m"], "need_m_full_t3": 3 * se, "need_ann_full_t3_pp": 3 * se * 12 * 100,
    "need_m_2019_t2": 2 * g["sd_m"] / math.sqrt(g["n"]), "need_ann_2019_t2_pp": 2 * g["sd_m"] / math.sqrt(g["n"]) * 12 * 100,
    "mde80_ann_pp": (3 + 0.84) * se * 12 * 100,
    "p_pass_true3": norm_sf(3 - 0.03 / 12 / se), "p_pass_true5": norm_sf(3 - 0.05 / 12 / se), "p_pass_true10": norm_sf(3 - 0.10 / 12 / se),
    "ci95_ann_pp": [(f["mean_m"] - 1.96 * se) * 1200, (f["mean_m"] + 1.96 * se) * 1200],
}

OUT["meta"]["runtime_sec"] = time.time() - tic
OUT["meta"]["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
txt = json.dumps(OUT, ensure_ascii=False, indent=1, sort_keys=True, default=float)
with open(os.path.join(OUTDIR, "my_numbers.json"), "w") as fh:
    fh.write(txt)
h = hashlib.sha256(txt.encode()).hexdigest()
with open(os.path.join(OUTDIR, "my_numbers.sha256"), "w") as fh:
    fh.write(h + "  my_numbers.json\n" + datetime.datetime.now(datetime.timezone.utc).isoformat() + "\n")
    for fn in ("rw_lib.py", "rw_run.py", "rw_prep.py", "rw_e50.py"):
        fh.write(hashlib.sha256(open(os.path.join(HERE, fn), "rb").read()).hexdigest() + "  " + fn + "(at freeze)\n")
# 我方逐筆與淨值（凍結後比對用）
with gzip.open(os.path.join(OUTDIR, "my_trades_main.csv.gz"), "wt") as fh:
    fh.write("code,buy_date,sell_date,cost,proceeds,why,shares,buy_open,trig_date\n")
    for (i, tb, te, c, p, w, sh, bo, tg) in RES["MAIN"]["trades"]:
        fh.write("%s,%s,%s,%.4f,%.4f,%s,%d,%.4f,%s\n" % (d.codes[i], d.dates[tb], d.dates[te], c, p, w, sh, bo, d.dates[tg] if tg >= 0 else ""))
keys = sorted(NAVS)
with gzip.open(os.path.join(OUTDIR, "my_navs.csv.gz"), "wt") as fh:
    fh.write("date,e0050," + ",".join(keys) + "\n")
    for t in range(d.t_ext, T):
        fh.write(d.dates[t] + ",%.6f," % b50[t] + ",".join("" if not (NAVS[k][t] == NAVS[k][t]) else "%.2f" % NAVS[k][t] for k in keys) + "\n")
print("sha256", h)
x = mm["MAIN"]
print("MAIN 2019+: %.4f vs %.4f ex %.2f t %.3f | full %.4f vs %.4f ex %.2f t %.3f | dd %.4f" % (
    x["from2019"]["ann"], x["from2019"]["ann_0050"], x["from2019"]["excess_pp"], x["from2019"]["t"],
    x["full"]["ann"], x["full"]["ann_0050"], x["full"]["excess_pp"], x["full"]["t"], x["full"]["maxdd"]))
print("peak MB", OUT["meta"]["peak_rss_mb"], "sec", OUT["meta"]["runtime_sec"])
