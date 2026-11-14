# -*- coding: utf-8 -*-
"""凍結後的審查加跑（不影響凍結數字；開頭斷言主欄等於凍結值）。
用法：python3 -I -B rw_review.py <scratch_dir>     輸出 review_numbers.json
"""
import sys, os, json, math, time, resource
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import rw_lib as W

S = sys.argv[1]
RD = os.path.dirname(HERE)
d = W.load(S)
T, N, t0 = d.T, d.N, d.t0
b50 = d.e50
i19 = W.last_before(d, "2019-01-01"); i23 = W.last_before(d, "2024-01-01")
FROZEN = json.load(open(os.path.join(HERE, "my_numbers.json")))
OUT = {}
tic = time.time()


def mk_spec(cross, ma50):
    TRL = {12: d.tr12, 15: d.tr15, 20: d.tr20}
    def spec(name):
        parts = []; fs = None; hd = None
        if name == "HOLD60": hd = 60
        elif name == "MA50_FS8": parts = [("ma50", ma50)]; fs = 8.0
        else:
            for tk in name.split("_"):
                if tk.startswith("T") and tk[1:].isdigit(): parts.append(("trail", TRL[int(tk[1:])]))
                elif tk == "MA": parts.append(("cross", cross))
                elif tk == "FS8": fs = 8.0
        st = None
        for _, a in parts: st = a if st is None else (st | a)
        return dict(exit_static=st, why_parts=parts, fstop=fs, hold=hd)
    return spec


spec = mk_spec(d.cross, d.ma50)
CANDS = ["T15_MA", "T12_MA", "T20_MA", "T12", "T15", "T20", "FS8_T15", "FS8_T15_MA", "MA50_FS8", "HOLD60"]
C_TW = W.cands_from(d.sig_tw, d.excess); C_ALL = W.cands_from(d.sig_all, d.excess); C_TP = W.cands_from(d.sig_tp, d.excess)
C_EXT = W.cands_from(d.sig_ext, d.excess, t_from=d.t_ext); C_POOL = W.cands_from(d.pool, d.excess)
M = spec("T15_MA")
rM = W.simulate(d, C_TW, **M)
assert abs(rM["nav"][-1] - FROZEN["main"]["MAIN"]["end_nav"]) < 1e-6
rE = W.simulate(d, C_TW, idle="e50", **M)
assert abs(rE["nav"][-1] - FROZEN["main"]["E50"]["end_nav"]) < 1e-6
a50_19 = W.ann(d, b50, i19, T - 1); a50_f = W.ann(d, b50, t0, T - 1)


def ex(nav):
    return dict(ex2019=float((W.ann(d, nav, i19, T - 1) - a50_19) * 100), exfull=float((W.ann(d, nav, t0, T - 1) - a50_f) * 100),
                ex2024=float((W.ann(d, nav, i23, T - 1) - W.ann(d, b50, i23, T - 1)) * 100),
                t2019=W.window(d, nav, b50, i19, T - 1)["t"], tfull=W.window(d, nav, b50, t0, T - 1)["t"],
                ann2019=float(W.ann(d, nav, i19, T - 1)), annfull=float(W.ann(d, nav, t0, T - 1)), maxdd=W.maxdd(nav[t0:]))


# ================================================================ R0 均線平手（SMA8＝SMA20、收盤＝SMA50）的實作敏感度
C100 = np.round(d.Cf * 100); okc = np.isfinite(C100); Ci = np.where(okc, C100, 0).astype(np.int64)
def rsum(n):
    cs = np.cumsum(Ci, axis=0); o = cs.copy(); o[n:] = cs[n:] - cs[:-n]
    vv = np.cumsum(okc, axis=0); v = vv.copy(); v[n:] = vv[n:] - vv[:-n]
    return o, (v == n)
s8, _ = rsum(8); s20, v20 = rsum(20); s50, v50 = rsum(50)
lt = (5 * s8 < 2 * s20) & v20; ge = (5 * s8 >= 2 * s20) & v20
cross_ex = np.zeros_like(lt); cross_ex[1:] = lt[1:] & ge[:-1]
ma50_ex = (50 * Ci < s50) & v50
n_tie = int(((5 * s8 == 2 * s20) & v20).sum()); n_tie50 = int(((50 * Ci == s50) & v50).sum())
del s8, s20, s50, lt, ge
PD = pd.DataFrame(d.Cf)
p8 = PD.rolling(8, min_periods=8).mean().values; p20 = PD.rolling(20, min_periods=20).mean().values
with np.errstate(invalid="ignore"):
    cross_pd = np.zeros((T, N), bool); cross_pd[1:] = (p8[1:] < p20[1:]) & (p8[:-1] >= p20[:-1])
    del p8, p20
    p50 = PD.rolling(50, min_periods=50).mean().values; ma50_pd = d.Cf < p50
    del p50, PD
sigcols = d.sig_all.any(axis=0)
OUT["ties"] = dict(cells_sma8_eq_sma20=n_tie, cells_close_eq_sma50=n_tie50,
                   cross_cells=dict(mine=int(d.cross.sum()), pandas=int(cross_pd.sum()), exact=int(cross_ex.sum())),
                   cross_diff_mine_vs_exact=int((d.cross != cross_ex).sum()), cross_diff_pandas_vs_exact=int((cross_pd != cross_ex).sum()),
                   cross_diff_mine_vs_pandas=int((d.cross != cross_pd).sum()),
                   ma50_diff_mine_vs_exact=int((d.ma50 != ma50_ex).sum()), ma50_diff_pandas_vs_exact=int((ma50_pd != ma50_ex).sum()),
                   cross_diff_mine_vs_exact_signal_stocks_from_t0=int((d.cross != cross_ex)[t0:, sigcols].sum()), impl={})
rt = pd.read_csv(os.path.join(RD, "trades.csv.gz"), dtype={"code": str})
for nm, (cx, mx) in (("mine_shifted_sum", (d.cross, d.ma50)), ("pandas_rolling", (cross_pd, ma50_pd)), ("exact_integer", (cross_ex, ma50_ex))):
    sp = mk_spec(cx, mx); q = {}
    for c in CANDS:
        r = W.simulate(d, C_TW, **sp(c))
        q[c] = ex(r["nav"]); q[c]["buys"] = r["cnt"]["buy"]
        sub = rt[rt.variant == c + "|MAIN"]
        mine = pd.DataFrame([(d.codes[i], d.dates[tb], d.dates[te]) for (i, tb, te, *_r) in r["trades"]], columns=["code", "ent_date", "exit_m"])
        mg = mine.merge(sub, on=["code", "ent_date"], how="outer", indicator=True)
        q[c]["trade_mismatch_vs_researcher"] = int((mg._merge != "both").sum() + ((mg._merge == "both") & (mg.exit_m != mg.exit_date)).sum())
    for k, kw in (("POOL_RS", {}), ("POOL_RS_ZERO", dict(cost="zero"))):
        r = W.simulate(d, C_POOL, **sp("T15_MA"), **kw); q[k] = ex(r["nav"])
    for k, cc, kw in (("T15_MA|STRESS", C_TW, dict(cost="stress")), ("T15_MA|E50", C_TW, dict(idle="e50")), ("T15_MA|TPEX", C_TP, {}),
                      ("T15_MA|ALL", C_ALL, {})):
        r = W.simulate(d, cc, **sp("T15_MA"), **kw); q[k] = ex(r["nav"])
    OUT["ties"]["impl"][nm] = q
print("R0 ties done", time.time() - tic)

# ================================================================ R1 安慰劑：1,000 次、筆數對齊
sig_days = [t for t in range(t0, T) if C_TW[t]]
sig_ix = {t: np.flatnonzero(d.sig_tw[t]) for t in sig_days}
pool_ix = {t: np.flatnonzero(d.pool[t]) for t in sig_days}
kt = {t: len(sig_ix[t]) for t in sig_days}
ym = [s[:7] for s in d.dates]
midx = np.array([(int(s[:4]) - int(ym[0][:4])) * 12 + int(s[5:7]) - int(ym[0][5:7]) for s in ym]); NM = int(midx.max()) + 1


def placebo(seed, kscale=1.0):
    U = np.random.default_rng(seed).random((NM, N))
    out = [None] * T
    for t in sig_days:
        ix = pool_ix[t]; k = kt[t]
        if kscale != 1.0: k = max(1, int(round(k * kscale)))
        if k < len(ix):
            ch = np.sort(ix[np.argsort(U[midx[t], ix], kind="stable")[:k]])
        else:
            ch = ix
        out[t] = ch[np.argsort(-d.excess[t, ch], kind="stable")].tolist()
    return out


def pstats(a, base):
    a = np.asarray(a)
    return dict(median=float(np.median(a)), mean=float(a.mean()), p5=float(np.percentile(a, 5)), p95=float(np.percentile(a, 95)),
                sd=float(a.std(ddof=1)), main=float(base), main_pctile=float((a < base).mean() * 100), n=int(len(a)))


NBIG = 1000
acc = {v: dict(a19=[], af=[], buys=[], idle=[], npos=[]) for v in ("normal", "zero", "e50")}
for i in range(NBIG):
    cc = placebo(9000 + i)
    for v, kw in (("normal", {}), ("zero", dict(cost="zero")), ("e50", dict(idle="e50"))):
        r = W.simulate(d, cc, record=False, **M, **kw); nav = r["nav"]
        acc[v]["a19"].append(W.ann(d, nav, i19, T - 1)); acc[v]["af"].append(W.ann(d, nav, t0, T - 1))
        acc[v]["buys"].append(r["cnt"]["buy"]); acc[v]["idle"].append(float(r["idlew"][t0 + 1:].mean())); acc[v]["npos"].append(float(r["npos"][t0 + 1:].mean()))
OUT["placebo_1000"] = {}
fm = FROZEN["main"]
for v, col in (("normal", "MAIN"), ("zero", "ZERO"), ("e50", "E50")):
    a = acc[v]
    q = dict(ann2019=pstats(a["a19"], fm[col]["from2019"]["ann"]), annfull=pstats(a["af"], fm[col]["full"]["ann"]),
             buys_mean=float(np.mean(a["buys"])), idle_w_mean=float(np.mean(a["idle"])), npos_mean=float(np.mean(a["npos"])),
             main_buys=fm[col]["cnt"]["buy"], main_idle_w=fm[col]["avg_idle_w"], main_npos=fm[col]["avg_npos"],
             beat_0050_2019=float((np.array(a["a19"]) > a50_19).mean()))
    # 前 200 次（種子 9000..9199＝規格的 200 次）與其後每 200 次一批
    q["batches_of_200"] = [dict(median=float(np.median(a["a19"][k:k + 200])), main_pctile=float((np.array(a["a19"][k:k + 200]) < fm[col]["from2019"]["ann"]).mean() * 100))
                           for k in range(0, NBIG, 200)]
    X = np.column_stack([np.ones(NBIG), np.array(a["buys"]), np.array(a["idle"])])
    co, *_ = np.linalg.lstsq(X, np.array(a["a19"]), rcond=None)
    q["xs_reg_ann2019_on_buys_idle"] = dict(const=float(co[0]), per_buy=float(co[1]), per_idle_w=float(co[2]))
    OUT["placebo_1000"][v] = q
print("R1 placebo 1000 done", time.time() - tic)
OUT["placebo_kscale"] = {}
for ks in (0.3, 0.4, 0.5, 0.6, 0.8):
    a = dict(n19=[], z19=[], buys=[], idle=[])
    for i in range(200):
        cc = placebo(9000 + i, ks)
        r = W.simulate(d, cc, record=False, **M); a["n19"].append(W.ann(d, r["nav"], i19, T - 1)); a["buys"].append(r["cnt"]["buy"]); a["idle"].append(float(r["idlew"][t0 + 1:].mean()))
        r = W.simulate(d, cc, record=False, cost="zero", **M); a["z19"].append(W.ann(d, r["nav"], i19, T - 1))
    OUT["placebo_kscale"][str(ks)] = dict(buys_mean=float(np.mean(a["buys"])), idle_w_mean=float(np.mean(a["idle"])),
                                          normal=pstats(a["n19"], fm["MAIN"]["from2019"]["ann"]), zero=pstats(a["z19"], fm["ZERO"]["from2019"]["ann"]))
print("R1 kscale done", time.time() - tic)

# ================================================================ R2 空位放 0050：β 對照、整股限制
def monthly(nav, i0, i1=T - 1):
    return W.mret(d, nav, i0, i1)[0]


def timing_only(r):
    """個股部位換成 0050、閒置資金照原樣（現金）：w[t-1]×隔夜 + w[t]×盤中。"""
    w = 1 - r["idlew"]; w[:t0 + 1] = 0.0
    nav = np.full(T, np.nan); nav[t0] = 1.0
    for t in range(t0 + 1, T):
        nav[t] = nav[t - 1] * (1 + w[t - 1] * d.ov[t]) * (1 + w[t] * d.intr[t])
    return nav


OUT["e50"] = {}
navM, navE = rM["nav"], rE["nav"]
navTM = timing_only(rM)
for wn, i0, i1 in (("full", t0, T - 1), ("from2019", i19, T - 1), ("seg_2017_2018", t0, i19), ("seg_2019_2023", i19, i23), ("seg_2024_2026", i23, T - 1)):
    rb = monthly(b50, i0, i1); q = {}
    for nm, nav in (("MAIN", navM), ("E50", navE)):
        rs = monthly(nav, i0, i1)
        o = W.ols_beta(rs, rb)
        o["excess_t"] = W.tstat(rs - rb)[0]; o["excess_mean_m"] = float((rs - rb).mean()); o["excess_sd_m"] = float((rs - rb).std(ddof=1))
        o["ann"] = float(W.ann(d, nav, i0, i1)); o["ann_0050"] = float(W.ann(d, b50, i0, i1))
        o["beta_matched_ann_m"] = float(np.prod(1 + o["beta"] * rb) ** (12 / len(rb)) - 1)
        o["strat_ann_m"] = float(np.prod(1 + rs) ** (12 / len(rs)) - 1)
        o["alpha_ann_arith_pp"] = o["alpha_m"] * 1200
        q[nm] = o
    # 主欄：個股換成 0050 的對照（同一條曝險路徑）
    rt_ = monthly(navTM, i0, i1); rs = monthly(navM, i0, i1)
    q["MAIN_stock_to_0050"] = dict(ann=float(W.ann(d, navTM, i0, i1)), main_minus_ctrl_pp=float((W.ann(d, navM, i0, i1) - W.ann(d, navTM, i0, i1)) * 100),
                                   t=W.tstat(rs - rt_)[0], nw6=W.nw_t(rs - rt_), mean_m=float((rs - rt_).mean()), avg_stock_w=float((1 - rM["idlew"][i0 + 1:i1 + 1]).mean()))
    # 放 0050 版：個股換成 0050 ＝ 100% 0050（忽略換手成本）→ 對照就是 0050 本身
    rs = monthly(navE, i0, i1)
    q["E50_stock_to_0050"] = dict(ann=float(W.ann(d, b50, i0, i1)), e50_minus_ctrl_pp=float((W.ann(d, navE, i0, i1) - W.ann(d, b50, i0, i1)) * 100),
                                  t=W.tstat(rs - rb)[0], nw6=W.nw_t(rs - rb), avg_stock_w=float((1 - rE["idlew"][i0 + 1:i1 + 1]).mean()))
    q["E50_minus_MAIN_pp"] = float((W.ann(d, navE, i0, i1) - W.ann(d, navM, i0, i1)) * 100)
    OUT["e50"][wn] = q
# 0050 整股限制（2020-10-26 前 0050 只能整張）
rL = W.simulate(d, C_TW, idle="e50", e50_lot=True, **M)
OUT["e50"]["lot_constrained_0050"] = dict(ex(rL["nav"]), seg_2017_2018_pp=float((W.ann(d, rL["nav"], t0, i19) - W.ann(d, b50, t0, i19)) * 100),
                                          spec_seg_2017_2018_pp=float((W.ann(d, navE, t0, i19) - W.ann(d, b50, t0, i19)) * 100),
                                          spec=ex(navE), idle_w_pre_odd=float(rL["idlew"][t0 + 1:d.t_odd].mean()),
                                          pre_odd_ann_lot=float(W.ann(d, rL["nav"], t0, d.t_odd - 1)), pre_odd_ann_spec=float(W.ann(d, navE, t0, d.t_odd - 1)))
# 池子基準與上櫃的放 0050 版
rPE = W.simulate(d, C_POOL, idle="e50", **M); rP = W.simulate(d, C_POOL, **M)
rTE = W.simulate(d, C_TP, idle="e50", **M); rT = W.simulate(d, C_TP, **M)
for nm, r in (("POOL_RS_E50", rPE), ("POOL_RS", rP), ("TPEX_E50", rTE), ("TPEX", rT)):
    q = ex(r["nav"])
    for wn, i0 in (("full", t0), ("from2019", i19)):
        o = W.ols_beta(monthly(r["nav"], i0), monthly(b50, i0)); q["beta_" + wn] = o
    q["avg_idle_w"] = float(r["idlew"][t0 + 1:].mean()); q["avg_npos"] = float(r["npos"][t0 + 1:].mean())
    q["timing_only_ex2019"] = ex(timing_only(r))["ex2019"] if "E50" not in nm else None
    OUT["e50"][nm] = q
# 主欄 vs 放 0050 版的月相關
OUT["e50"]["corr_monthly_excess_MAIN_E50"] = float(np.corrcoef(monthly(navM, t0) - monthly(b50, t0), monthly(navE, t0) - monthly(b50, t0))[0, 1])
print("R2 e50 done", time.time() - tic)

# ================================================================ R3 集中度
tr = rM["trades"]
pnl = np.array([x[4] - x[3] for x in tr]); order = np.argsort(-pnl)
tot = float(pnl.sum())
OUT["concentration"] = dict(total_pnl=tot, n=len(tr), top5_share=float(pnl[order[:5]].sum() / tot), top10_share=float(pnl[order[:10]].sum() / tot),
                            top20_share=float(pnl[order[:20]].sum() / tot),
                            top10=[dict(code=d.codes[tr[k][0]], ent=d.dates[tr[k][1]], exit=d.dates[tr[k][2]], ret=float(tr[k][4] / tr[k][3] - 1),
                                        pnl=float(pnl[k]), why=tr[k][5]) for k in order[:10]])
by_year = {}
for x, p in zip(tr, pnl):
    by_year[d.dates[x[2]][:4]] = by_year.get(d.dates[x[2]][:4], 0.0) + float(p)
OUT["concentration"]["pnl_by_exit_year"] = by_year


def ban_set(trs, ks):
    b = set()
    for k in ks:
        i, tb, te = trs[k][0], trs[k][1], trs[k][2]
        for t in range(tb, te + 1): b.add((i, t))
    return b


OUT["concentration"]["drop_top"] = {}
for n in (1, 3, 5, 10):
    r = W.simulate(d, C_TW, ban=ban_set(tr, order[:n]), **M)
    q = ex(r["nav"]); q["buys"] = r["cnt"]["buy"]
    # 算術口徑：最賺 n 筆的損益直接歸零（不補別的股票、不調整後續複利）
    q["naive_end_nav"] = float(navM[-1] - pnl[order[:n]].sum())
    q["naive_annfull"] = float((q["naive_end_nav"] / 1e6) ** (365.25 / int((d.dnum[-1] - d.dnum[t0]).astype(int))) - 1)
    OUT["concentration"]["drop_top"][str(n)] = q
trE = rE["trades"]; pnlE = np.array([x[4] - x[3] for x in trE]); oE = np.argsort(-pnlE)
OUT["concentration"]["e50_top5_share_of_stock_pnl"] = float(pnlE[oE[:5]].sum() / pnlE.sum())
OUT["concentration"]["e50_drop_top"] = {}
for n in (1, 5, 10):
    r = W.simulate(d, C_TW, idle="e50", ban=ban_set(trE, oE[:n]), **M)
    OUT["concentration"]["e50_drop_top"][str(n)] = ex(r["nav"])
# 依報酬率（不是金額）拿掉最高 5 筆
ret = np.array([x[4] / x[3] - 1 for x in tr]); o2 = np.argsort(-ret)
r = W.simulate(d, C_TW, ban=ban_set(tr, o2[:5]), **M)
OUT["concentration"]["drop_top5_by_ret"] = dict(ex(r["nav"]), trades=[dict(code=d.codes[tr[k][0]], ent=d.dates[tr[k][1]], ret=float(ret[k])) for k in o2[:5]])
print("R3 concentration done", time.time() - tic)

# ================================================================ R4 2024 起、延伸段、只上櫃
OUT["recent"] = {"cands": {}}
XS = {}
for c in CANDS:
    r = W.simulate(d, C_TW, **spec(c)); nav = r["nav"]
    w3 = W.window(d, nav, b50, i23, T - 1)
    o = W.ols_beta(monthly(nav, i23), monthly(b50, i23))
    XS[c] = monthly(nav, t0) - monthly(b50, t0)
    tm = timing_only(r)
    OUT["recent"]["cands"][c] = dict(ex2024_pp=w3["excess_pp"], t=w3["t"], nw6=w3["nw6"], mean_m=w3["mean_m"], sd_m=w3["sd_m"], n=w3["n"],
                                     beta=o["beta"], alpha_m=o["alpha_m"], t_alpha=o["t_alpha"], avg_stock_w=float((1 - r["idlew"][i23 + 1:]).mean()),
                                     minus_stock_to_0050_pp=float((W.ann(d, nav, i23, T - 1) - W.ann(d, tm, i23, T - 1)) * 100),
                                     months_positive=int((monthly(nav, i23) - monthly(b50, i23) > 0).sum()))
Cm = np.corrcoef(np.array([XS[c] for c in CANDS]))
OUT["recent"]["avg_pairwise_corr_monthly_excess_full"] = float((Cm.sum() - len(CANDS)) / (len(CANDS) * (len(CANDS) - 1)))
C24 = np.corrcoef(np.array([XS[c][-34:] for c in CANDS]))
OUT["recent"]["avg_pairwise_corr_monthly_excess_2024"] = float((C24.sum() - len(CANDS)) / (len(CANDS) * (len(CANDS) - 1)))
ev = np.linalg.eigvalsh(Cm)[::-1]
OUT["recent"]["corr_eigen_share_first"] = float(ev[0] / ev.sum())
# 延伸段
rX = W.simulate(d, C_EXT, t_start=d.t_ext, **M)
rs = monthly(rX["nav"], d.t_ext, t0); rb = monthly(b50, d.t_ext, t0)
o = W.ols_beta(rs, rb)
tmx = np.full(T, np.nan); tmx[d.t_ext] = 1.0
wx = 1 - rX["idlew"]; wx[:d.t_ext + 1] = 0.0
for t in range(d.t_ext + 1, t0 + 1):
    tmx[t] = tmx[t - 1] * (1 + wx[t - 1] * d.ov[t]) * (1 + wx[t] * d.intr[t])
OUT["recent"]["EXT"] = dict(W.window(d, rX["nav"], b50, d.t_ext, t0), beta=o["beta"], alpha_m=o["alpha_m"], t_alpha=o["t_alpha"],
                            avg_stock_w=float(wx[d.t_ext + 1:t0 + 1].mean()), avg_npos=float(rX["npos"][d.t_ext + 1:t0 + 1].mean()),
                            stock_to_0050_ann=float(W.ann(d, tmx, d.t_ext, t0)),
                            minus_stock_to_0050_pp=float((W.ann(d, rX["nav"], d.t_ext, t0) - W.ann(d, tmx, d.t_ext, t0)) * 100),
                            cnt={k: (float(v) if isinstance(v, float) else int(v)) for k, v in rX["cnt"].items()})
rXE = W.simulate(d, C_EXT, t_start=d.t_ext, idle="e50", **M)
OUT["recent"]["EXT_E50"] = W.window(d, rXE["nav"], b50, d.t_ext, t0)
# 只上櫃
tmT = timing_only(rT)
OUT["recent"]["TPEX"] = dict(ex(rT["nav"]), avg_stock_w=float((1 - rT["idlew"][t0 + 1:]).mean()), avg_npos=float(rT["npos"][t0 + 1:].mean()),
                             minus_stock_to_0050_2019_pp=float((W.ann(d, rT["nav"], i19, T - 1) - W.ann(d, tmT, i19, T - 1)) * 100),
                             minus_stock_to_0050_full_pp=float((W.ann(d, rT["nav"], t0, T - 1) - W.ann(d, tmT, t0, T - 1)) * 100),
                             t_vs_ctrl_full=W.tstat(monthly(rT["nav"], t0) - monthly(tmT, t0))[0],
                             beta_full=W.ols_beta(monthly(rT["nav"], t0), monthly(b50, t0)))
OUT["recent"]["TPEX_E50"] = ex(rTE["nav"])
print("R4 done", time.time() - tic)

# ================================================================ R5 檢定力（事實陳述用）
def norm_ppf(p):
    lo, hi = -10.0, 10.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if 0.5 * math.erfc(-mid / math.sqrt(2)) < p: lo = mid
        else: hi = mid
    return (lo + hi) / 2
f = fm["MAIN"]["full"]; sd = f["sd_m"]; n = f["n"]
zb = norm_ppf(1 - 0.05 / (2 * 2700))
e5 = OUT["e50"]["full"]["E50"]
OUT["power"] = dict(sd_m=sd, n=n, need_ann_t3_pp=3 * sd / math.sqrt(n) * 1200, bonferroni_z_two_sided_N2700=zb,
                    need_ann_bonferroni_pp=zb * sd / math.sqrt(n) * 1200,
                    months_needed_for_Et3_true5pp=(3 * sd / (0.05 / 12)) ** 2, months_needed_for_Et3_true10pp=(3 * sd / (0.10 / 12)) ** 2,
                    months_needed_for_Et3_at_observed_mean=(3 * sd / f["mean_m"]) ** 2, observed_mean_m=f["mean_m"],
                    g8_12m_need_m_for_t2=2 * sd / math.sqrt(12), g8_12m_need_ann_pp=2 * sd / math.sqrt(12) * 1200,
                    e50_sd_m=e5["excess_sd_m"], e50_need_ann_t3_pp=3 * e5["excess_sd_m"] / math.sqrt(n) * 1200,
                    e50_p_pass_at_true_equal_observed=0.5 * math.erfc((3 - e5["excess_t"]) / math.sqrt(2)),
                    alpha_resid_sd_m=FROZEN["beta"]["MAIN_full"]["resid_sd"],
                    need_alpha_ann_t3_pp=3 * FROZEN["beta"]["MAIN_full"]["resid_sd"] / math.sqrt(n) * 1200)

# ================================================================ R6 SPA（Hansen 2005，consistent；10 個候選、全期月超額）
def spa(X, B=5000, q=1 / 6, seed=20261114):
    X = np.asarray(X); n, k = X.shape
    rng = np.random.default_rng(seed)
    dbar = X.mean(axis=0)
    # 以拔靴估 ω
    idx = np.zeros((B, n), int)
    idx[:, 0] = rng.integers(0, n, B)
    jump = rng.random((B, n)) < q
    newi = rng.integers(0, n, (B, n))
    for t in range(1, n):
        idx[:, t] = np.where(jump[:, t], newi[:, t], (idx[:, t - 1] + 1) % n)
    bm = X[idx].mean(axis=1)                         # B×k
    om = np.sqrt(n) * bm.std(axis=0, ddof=1)
    Tn = max(0.0, float((np.sqrt(n) * dbar / om).max()))
    thr = -np.sqrt(2 * np.log(np.log(n)))
    mu_c = dbar * ((np.sqrt(n) * dbar / om) >= thr)
    Tb = np.maximum(0.0, (np.sqrt(n) * (bm - mu_c) / om).max(axis=1))   # 2026-11-14 修正：原本寫成 bm - dbar + mu_c（置中方向錯）
    return dict(T=Tn, p_consistent=float((Tb >= Tn).mean()), k=k, n=n)
OUT["spa_full_10"] = spa(np.array([XS[c] for c in CANDS]).T)
OUT["spa_2019_10"] = spa(np.array([XS[c][-94:] for c in CANDS]).T)

OUT["meta"] = dict(runtime_sec=time.time() - tic, peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                   frozen_sha256=open(os.path.join(HERE, "my_numbers.sha256")).read().split()[0])
json.dump(OUT, open(os.path.join(HERE, "review_numbers.json"), "w"), ensure_ascii=False, indent=1, default=float)
print("done", OUT["meta"])
