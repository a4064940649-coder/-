# -*- coding: utf-8 -*-
"""S1 C1 獨立審查：第二部分三個問題（凍結後執行；全部標為事後）。
用法：python3 -I -B rep_s1_review.py <scratch_dir>     輸出：review_numbers.json
"""
import sys, os, json, resource
HERE = os.path.dirname(os.path.abspath(__file__)); UP = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import rep_s1_lib as L

scratch = sys.argv[1]
P = L.build(scratch)
R, mi0 = L.load_rev(P)
S, D = L.surge(R, mi0, null2013=True)
plans, info = L.c1_plans(P, S, mi0)
T, N = P.T, P.N
i19 = L.didx(P, "2018-12-28"); E = T - 1
e50 = P.e50.values
B = L.bench(P)
out = dict(bench=dict(ann_2019=B["ann_2019"], ann_full=B["ann_full"]))
pc = lambda v: round(float(v) * 100, 2)


def S2(res):
    s = L.summary(P, res)
    return dict(ann_2019=pc(s["ann_2019"]), ann_full=pc(s["ann_full"]), mdd_2019=pc(s["mdd_2019"][0]), n=s["trades"].get("n", 0),
                win=pc(s["trades"].get("win", 0)), exp=pc(s["trades"].get("exp", 0)), hold=round(s["trades"].get("hold_mean", 0), 1),
                n_2019=s["trades_2019"].get("n", 0), exp_2019=pc(s["trades_2019"].get("exp", 0)), win_2019=pc(s["trades_2019"].get("win", 0)),
                avg_npos=round(s["avg_npos"], 2), yearly={k: pc(v) for k, v in s["yearly"].items()}, cnt=s["cnt"])


base = L.simulate(P, plans)
out["c1_base"] = S2(base)
frozen = json.load(open(os.path.join(HERE, "my_numbers.json")))
assert abs(frozen["c1_base"]["ann_2019"] - L.summary(P, base)["ann_2019"]) < 1e-12, "函式庫變更影響了凍結數字"

# ================================================================= 0. 差異追查：停牌缺列口徑
z = np.load(os.path.join(scratch, "panel.npz")); pres = z["P"].astype(bool); del z
full60 = pd.DataFrame(pres.astype(np.float32)).rolling(60, min_periods=60).sum().values >= 60
adv_keep = P.adv60
P.adv60 = np.where(full60, adv_keep, np.nan)                        # 近 60 根有缺列（停牌）者不進股池＝研究員口徑（推定）
plans_s, info_s = L.c1_plans(P, S, mi0)
bs = L.simulate(P, plans_s)
x2s = L.simulate(P, plans_s, cmult=2.0)
t5 = bs["trades"].sort_values("pnl", ascending=False).head(5)
d5s = L.simulate(P, plans_s, block=set(zip(t5.code, t5.buy_date)))
out["diag_strict60"] = dict(base=S2(bs), cost_x2=S2(x2s), drop5=S2(d5s),
                            full=L.summary(P, bs)["trades"], mdd=L.summary(P, bs)["mdd_2019"],
                            h1=pc(L.summary(P, bs)["ann_h1"]), h2=pc(L.summary(P, bs)["ann_h2"]))
rt = pd.read_csv(os.path.join(UP, "trades.csv.gz"), dtype={"code": str}); rt = rt[rt.run == "C1_REVSURGE|twse|cash"]
kr = set(zip(rt.code, rt.ent_date)); km = set(zip(bs["buys"].code, bs["buys"].buy_date))
out["diag_strict60"]["buys_only_r"] = sorted(kr - km, key=lambda x: x[1]); out["diag_strict60"]["buys_only_m"] = sorted(km - kr, key=lambda x: x[1])
out["diag_strict60"]["n_buys"] = (len(kr), len(km))
P.adv60 = adv_keep
print("strict60", out["diag_strict60"]["base"]["ann_2019"], out["diag_strict60"]["cost_x2"]["ann_2019"], out["diag_strict60"]["drop5"]["ann_2019"],
      out["diag_strict60"]["buys_only_r"], out["diag_strict60"]["buys_only_m"])

# ================================================================= 問題一
q1 = {}
# ---- (a) 主策略校準
G = P.G
hi252 = pd.DataFrame(G).rolling(252, min_periods=252).max().values
A_all = G / hi252
nm = R.shape[0]
Y = np.full_like(R, np.nan); Y[12:] = R[12:] / R[:-12] - 1.0
avail = np.array([L.rev_avail(P, mi0 + m)[2] for m in range(nm)])
dl = np.array([L.rev_avail(P, mi0 + m)[1] for m in range(nm)])
fin = np.array([(c[:2] == "28") or c in ("5876", "5880", "6005", "6024") for c in P.codes])
ky = np.array([("KY" in P.names.get(c, "")) or ("DR" in P.names.get(c, "")) for c in P.codes])
codes_arr = np.array(P.codes)


def main_plans(mode):
    """mode: 'pit'＝規則書訊號日（每月第一個 ≥ 11 日的交易日）＋嚴格 PIT 營收；
    'deadline'＝同訊號日，營收以期限日當天即視為已知；'c1day'＝改用 C1 的訊號日（期限日的下一個交易日）。"""
    pl = {}; meta = []
    ym = sorted(set(d[:7] for d in P.dstr[P.t0:]))
    for mth in ym:
        if mode == "c1day":
            y, mo = int(mth[:4]), int(mth[5:7])
            pm = y * 12 + mo - 2                                        # 上個月的營收
            s = L.rev_avail(P, pm)[2]
        else:
            cand = [k for k in range(T) if P.dstr[k][:7] == mth and int(P.dstr[k][8:10]) >= 11]
            if not cand: continue
            s = cand[0]
        if s >= T - 1 or s < P.t0: continue
        lim = avail if mode != "deadline" else dl
        ms = np.where(lim <= s)[0]
        if len(ms) == 0: continue
        m = int(ms.max())
        b = (Y[m] + Y[m - 1] + Y[m - 2]) / 3.0
        tw = P.inbar[s] & P.is_twse[s] & np.isfinite(P.adv60[s])
        idx = np.where(tw)[0]
        top = idx[np.argsort(-P.adv60[s, idx], kind="stable")[:300]]
        a = A_all[s]
        with np.errstate(invalid="ignore"):
            ok = (P.adv60[s, top] >= 1e7) & (P.C[s, top] >= 10) & P.traded[s, top] & ~fin[top] & ~ky[top] & np.isfinite(b[top]) & np.isfinite(a[top])
        pool = top[ok]
        if len(pool) < 20: continue
        ra = pd.Series(a[pool]).rank(pct=True).values; rb = pd.Series(b[pool]).rank(pct=True).values
        sc = 0.5 * ra + 0.5 * rb
        order = pool[np.lexsort((codes_arr[pool], -sc))]
        pl[s] = order
        meta.append((mth, P.dstr[s], "%04d-%02d" % ((mi0 + m) // 12, (mi0 + m) % 12 + 1), len(pool)))
    return pl, meta


mains = {}
for mode in ("pit", "deadline", "c1day"):
    pl, meta = main_plans(mode)
    r = L.simulate(P, pl, fallback=True)
    mains[mode] = (pl, meta, r)
    q1["main_" + mode] = S2(r)
    q1["main_" + mode]["first_signal"] = meta[0][1]; q1["main_" + mode]["n_signals"] = len(meta)
    q1["main_" + mode]["n_stale_rev_signals"] = int(sum(1 for (mth, sd, rm, _) in meta
                                                        if (int(mth[:4]) * 12 + int(mth[5:7]) - 1) - (int(rm[:4]) * 12 + int(rm[5:7]) - 1) > 1))
    print("main", mode, q1["main_" + mode]["ann_2019"], q1["main_" + mode]["ann_full"], q1["main_" + mode]["n"], q1["main_" + mode]["n_stale_rev_signals"])
pl_main, meta_main, r_main = mains["pit"]
for nmv, kw in (("main_pit_extra0.3", dict(extra=0.003)), ("main_pit_cost_x2", dict(cmult=2.0))):
    q1[nmv] = S2(L.simulate(P, pl_main, fallback=True, **kw))
c = r_main["cnt"]; q1["main_pit_rt_cost"] = (c["fee"] + c["tax"] + c["slip"]) / c["turnover_buy"]
c = base["cnt"]; q1["c1_rt_cost"] = (c["fee"] + c["tax"] + c["slip"]) / c["turnover_buy"]
yrs19 = (P.dates[E] - P.dates[i19]).days / 365.25
q1["c1_buys_per_year_2019"] = float((base["buys"].tb > i19).sum() / yrs19)
q1["main_buys_per_year_2019"] = float((r_main["buys"].tb > i19).sum() / yrs19)

# ---- (b) 加嚴成本
grid = {}
for ex in (0.0, 0.002, 0.003, 0.005, 0.0075, 0.010, 0.0125, 0.015):
    r = L.simulate(P, plans, extra=ex); s = L.summary(P, r)
    grid[ex] = (s["ann_2019"], s["ann_full"])
q1["c1_extra_slip"] = {("%.2f%%" % (k * 100)): (pc(v[0]), pc(v[1])) for k, v in grid.items()}


def breakeven(col, target):
    lo, hi = 0.0, 0.03
    f = lambda ex: L.summary(P, L.simulate(P, plans, extra=ex))[col] - target
    if f(hi) > 0: return None
    for _ in range(12):
        mid = (lo + hi) / 2
        if f(mid) > 0: lo = mid
        else: hi = mid
    return (lo + hi) / 2


q1["breakeven_extra_2019"] = breakeven("ann_2019", B["ann_2019"])
q1["breakeven_extra_full"] = breakeven("ann_full", B["ann_full"])
gm = {}
for ex in (0.002, 0.005, 0.010):
    s = L.summary(P, L.simulate(P, pl_main, fallback=True, extra=ex)); gm["%.2f%%" % (ex * 100)] = (pc(s["ann_2019"]), pc(s["ann_full"]))
q1["main_pit_extra_slip"] = gm
print("q1b", q1["c1_extra_slip"], q1["breakeven_extra_2019"], q1["breakeven_extra_full"])

# ---- (c) 逐筆衝擊
b = base["buys"].copy()
b["half_tick_pct"] = [L.tick(o) / 2 / o for o in b.open]
b["part_adv"] = b.amt / b.adv60


def dist(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return dict(mean=float(x.mean()), p50=float(np.percentile(x, 50)), p75=float(np.percentile(x, 75)), p90=float(np.percentile(x, 90)),
                p99=float(np.percentile(x, 99)), max=float(x.max()), n=int(len(x)))


imp = dict(all=dict(part_day=dist(b.part), part_adv60=dist(b.part_adv), gap=dist(b.gap), half_tick=dist(b.half_tick_pct), amt=dist(b.amt)))
for lab, mk in (("2016-2018", b.tb <= i19), ("2019-2022", (b.tb > i19) & (b.buy_date < "2023-01-01")), ("2023-2026", b.buy_date >= "2023-01-01")):
    g = b[mk]; imp[lab] = dict(part_day=dist(g.part), gap=dist(g.gap), amt_mean=float(g.amt.mean()), n=int(len(g)))
imp["gap_pos_share"] = float((b.gap > 0).mean()); imp["gap_gt2pct_share"] = float((b.gap > 0.02).mean())
imp["gap_2019"] = dist(b[b.tb > i19].gap)
# 對照：同一批買進日，股池全體的平均開盤跳空
ug = []
for x in info:
    s = x["s"]; u = L.universe(P, s)
    g = P.O[s + 1, u] / P.refe[s + 1, u] - 1
    ug.append(np.nanmean(g))
imp["gap_universe_mean_same_days"] = float(np.mean(ug)); imp["gap_universe_median_of_means"] = float(np.median(ug))
bk = pd.cut(b.open, [0, 20, 50, 100, 1e9], labels=["<20", "20-50", "50-100", ">100"], right=False)
imp["price_bucket"] = {str(k): dict(share=float((bk == k).mean()), n=int((bk == k).sum()), half_tick_mean=float(b.half_tick_pct[bk == k].mean()),
                                    gap_mean=float(b.gap[bk == k].mean())) for k in bk.cat.categories}
b19 = b[b.tb > i19]; bk19 = pd.cut(b19.open, [0, 20, 50, 100, 1e9], labels=["<20", "20-50", "50-100", ">100"], right=False)
imp["price_bucket_2019"] = {str(k): float((bk19 == k).mean()) for k in bk19.cat.categories}
c = base["cnt"]
imp["lot_block"] = dict(cand_pre_odd=c["cand_pre_odd"], lot_skip=c["lot_skip"], share=c["lot_skip"] / c["cand_pre_odd"])
imp["liq_capped"] = c["liq_capped"]; imp["sell_split"] = c["sell_split"]; imp["no_slot"] = c["no_slot"]
# 被整張擋掉的名單股：若買得到，紙上報酬如何（方向）
q1["impact"] = imp
print("q1c", imp["all"]["part_day"], imp["all"]["gap"], imp["price_bucket"], imp["lot_block"])


# ---- 紙上期間報酬（開盤→下一個換股日開盤；含息）
def entry_level(i, a):
    o = P.O[a, i]
    return G[a, i] * (o / P.Cf[a, i]) if o == o else G[a, i]


def exit_level(i, bday):
    la = P.last[i]
    if bday > la: return G[la, i]
    o = P.O[bday, i]
    return G[bday, i] * (o / P.Cf[bday, i]) if o == o else G[bday, i]


def per_ret(i, a, bday):
    e = entry_level(i, a)
    return exit_level(i, bday) / e - 1.0 if (e == e and e > 0) else np.nan


sig_list = sorted(plans)
nxt = {s: (sig_list[k + 1] if k + 1 < len(sig_list) else None) for k, s in enumerate(sig_list)}

# ---- (d) 與主策略的重疊
by_month_main = {sd[:7]: s for (mth, sd, rm, _), s in zip(meta_main, sorted(pl_main))}
ov = []
for x in info:
    mth = x["signal"][:7]
    if mth in by_month_main:
        a = set(int(i) for i in plans[x["s"]][:10]); bb = set(int(i) for i in pl_main[by_month_main[mth]][:10])
        ov.append((mth, len(a & bb)))
ovs = pd.DataFrame(ov, columns=["m", "k"])
q1["overlap"] = dict(n_months=int(len(ovs)), mean=float(ovs.k.mean() / 10), median=float(ovs.k.median() / 10), max=int(ovs.k.max()),
                     zero_months=int((ovs.k == 0).sum()), mean_2019=float(ovs[ovs.m >= "2019-01"].k.mean() / 10),
                     dist={int(k): int(v) for k, v in ovs.k.value_counts().sort_index().items()})
i16 = L.didx(P, "2016-02-17")
ma = L.monthly(P, base["nav"], i19); mb = L.monthly(P, r_main["nav"], i19); me = L.monthly(P, e50, i19)
q1["corr_monthly_2019"] = dict(c1_main=float(np.corrcoef(ma, mb)[0, 1]), c1_0050=float(np.corrcoef(ma, me)[0, 1]), main_0050=float(np.corrcoef(mb, me)[0, 1]),
                               excess_c1_main=float(np.corrcoef(ma - me, mb - me)[0, 1]), n=int(len(ma)))
ma = L.monthly(P, base["nav"], i16); mb = L.monthly(P, r_main["nav"], i16); me = L.monthly(P, e50, i16)
q1["corr_monthly_2016"] = dict(c1_main=float(np.corrcoef(ma, mb)[0, 1]), excess_c1_main=float(np.corrcoef(ma - me, mb - me)[0, 1]), n=int(len(ma)))
# C1 名單股在訊號日的「距 52 週高」百分位（是不是也在買接近新高的股票）
pa = []
for x in info:
    s = x["s"]; u = L.universe(P, s); a = A_all[s]
    uu = np.where(u & np.isfinite(a))[0]
    rk = pd.Series(a[uu], index=uu).rank(pct=True)
    pa += [rk.get(int(i), np.nan) for i in plans[s][:10]]
q1["c1_list_52wk_high_pctile_mean"] = float(np.nanmean(pa))
q1["c1_list_within10pct_of_high_share"] = float(np.nanmean([A_all[x["s"], int(i)] >= 0.9 for x in info for i in plans[x["s"]][:10]]))
out["q1"] = q1
print("q1d", q1["overlap"], q1["corr_monthly_2019"], q1["c1_list_52wk_high_pctile_mean"])

# ================================================================= 問題二
q2 = {}
has_rev_any = np.array([c in P.rev_codes_all for c in P.codes])
has_rev_2014 = np.isfinite(R[(2014 * 12 - mi0):]).any(axis=0)       # 2014 年以後有任何營收列
delisted = P.last < E
rows = []; ever = np.zeros(N, bool); stock_months = 0
for x in info:
    s = x["s"]; m = x["m"]; u = L.universe(P, s); ever |= u; stock_months += int(u.sum())
    ns = u & ~np.isfinite(S[m])
    rows.append(dict(y=int(x["signal"][:4]), n_univ=int(u.sum()), n_nosurge=int(ns.sum()),
                     n_norev_delisted=int((ns & ~has_rev_2014 & delisted).sum()),
                     n_norev_alive=int((ns & ~has_rev_2014 & ~delisted).sum()),
                     n_hasrev_short=int((ns & has_rev_2014).sum()),
                     n_hasrev_delisted=int((u & has_rev_2014 & delisted).sum())))
df = pd.DataFrame(rows)
g = df.groupby("y").mean()
q2["a_by_year"] = {int(y): dict(n_univ=round(r.n_univ, 1), no_surge_share=round(r.n_nosurge / r.n_univ * 100, 2), n_nosurge=round(r.n_nosurge, 1),
                                delisted_norev=round(r.n_norev_delisted, 1), alive_norev=round(r.n_norev_alive, 1), short_history=round(r.n_hasrev_short, 1))
                   for y, r in g.iterrows()}
q2["a_total"] = dict(ever_in_universe=int(ever.sum()), ever_delisted=int((ever & delisted).sum()),
                     ever_delisted_no_rev2014=int((ever & delisted & ~has_rev_2014).sum()),
                     ever_delisted_with_rev=int((ever & delisted & has_rev_2014).sum()),
                     ever_alive_no_rev=int((ever & ~delisted & ~has_rev_2014).sum()),
                     alive_no_rev_codes=[P.codes[i] for i in np.where(ever & ~delisted & ~has_rev_2014)[0]][:40],
                     stock_months=stock_months,
                     delisted_stock_months=int(df.n_norev_delisted.sum()), delisted_stock_month_share=float(df.n_norev_delisted.sum() / stock_months))
# (b) 下市股下市前表現
dl_idx = np.where(ever & delisted)[0]
rec = []
for i in dl_idx:
    la = int(P.last[i]); a = max(int(P.first[i]), la - 252)
    r12 = G[la, i] / G[a, i] - 1.0
    a3 = max(int(P.first[i]), la - 60); r3 = G[la, i] / G[a3, i] - 1.0
    # 同期股池中位數（以下市日前 252 根的全體上市股）
    oth = np.where(P.inbar[la] & P.inbar[a] & P.is_twse[la])[0]
    med = float(np.nanmedian(G[la, oth] / G[a, oth] - 1.0))
    vol20 = float(np.nanstd(P.ret[la - 19:la + 1, i]))
    rev_res = bool(has_rev_any[i]); last_rev = None
    if rev_res:
        k = np.where(np.isfinite(R[:, i]))[0]
        last_rev = "%04d-%02d" % ((mi0 + k.max()) // 12, (mi0 + k.max()) % 12 + 1) if len(k) else None
    rec.append(dict(code=P.codes[i], name=P.names.get(P.codes[i], ""), last=P.dstr[la], r12=float(r12), r3=float(r3), mkt_med12=med,
                    last_close=float(P.Cf[la, i]), vol20=vol20, last_rev=last_rev))
dd = pd.DataFrame(rec)
q2["b_delisted"] = dict(n=int(len(dd)), r12_mean=float(dd.r12.mean()), r12_median=float(dd.r12.median()), mkt_med12_mean=float(dd.mkt_med12.mean()),
                        r3_mean=float(dd.r3.mean()), r3_median=float(dd.r3.median()),
                        n_r12_lt_m30=int((dd.r12 < -0.3).sum()), n_r12_m30_0=int(((dd.r12 >= -0.3) & (dd.r12 < 0)).sum()),
                        n_r12_0_30=int(((dd.r12 >= 0) & (dd.r12 < 0.3)).sum()), n_r12_gt30=int((dd.r12 >= 0.3).sum()),
                        n_low_vol_end=int((dd.vol20 < 0.005).sum()), n_last_close_lt10=int((dd.last_close < 10).sum()),
                        n_with_residual_rev=int(dd.last_rev.notna().sum()), residual_rev_last_months=dd.last_rev.dropna().value_counts().to_dict(),
                        by_year=dd.groupby(dd["last"].str[:4]).agg(n=("r12", "size"), r12=("r12", "mean")).round(3).to_dict("index"),
                        rows=dd.sort_values("last").to_dict("records"))
# 在股池期間：下市股 vs 股池 vs 名單 的下一期紙上報酬；動能百分位
mom = G / np.vstack([np.full((252, N), np.nan), G[:-252]]) - 1.0
perL = []; perX = []; perU = []; pct_top = []; pct_del = []; npool = []; adj = []
for x in info:
    s = x["s"]; m = x["m"]; s2 = nxt[s]
    a = s + 1; bday = (s2 + 1) if s2 is not None else E
    u = L.universe(P, s)
    top = [int(i) for i in plans[s][:10]]
    dpool = np.where(u & delisted & ~np.isfinite(S[m]))[0]
    Lk = np.nanmean([per_ret(i, a, bday) for i in top])
    Uk = np.nanmean([per_ret(int(i), a, bday) for i in np.where(u)[0]])
    Xk = np.nanmean([per_ret(int(i), a, bday) for i in dpool]) if len(dpool) else np.nan
    perL.append(Lk); perU.append(Uk); perX.append(Xk); npool.append(len(dpool))
    uu = np.where(u & np.isfinite(mom[s]))[0]
    rk = pd.Series(mom[s, uu], index=uu).rank(pct=True)
    pct_top += [rk.get(i, np.nan) for i in top]; pct_del += [rk.get(int(i), np.nan) for i in dpool]
perL = np.array(perL); perX = np.array(perX); perU = np.array(perU); npool = np.array(npool)
ok = np.isfinite(perX)
q2["b_in_pool"] = dict(months=int(len(perL)), months_with_delisted_pool=int(ok.sum()), avg_pool_size=float(npool.mean()),
                       list_mean=float(np.nanmean(perL)), univ_mean=float(np.nanmean(perU)), delisted_mean=float(np.nanmean(perX)),
                       list_mean_same_months=float(perL[ok].mean()), univ_mean_same_months=float(perU[ok].mean()),
                       mom_pctile_list=float(np.nanmean(pct_top)), mom_pctile_delisted=float(np.nanmean(pct_del)),
                       mom_pctile_delisted_n=int(np.isfinite(pct_del).sum()))
# (c) 偏誤上界：每個換股月 1 檔名單股換成當月下市股池平均報酬
sig_dates = [x["signal"] for x in info]


def adj_ann(adjv):
    f_full = np.prod(1 + adjv); m19 = np.array([d > "2018-12-28" for d in sig_dates])
    f19 = np.prod(1 + adjv[m19])
    nav = base["nav"]
    a19 = (nav[E] * f19 / nav[i19]) ** (1 / yrs19) - 1
    yf = (P.dates[E] - P.dates[P.t0]).days / 365.25
    af = (nav[E] * f_full / nav[P.t0]) ** (1 / yf) - 1
    return a19, af


b19_, bf_ = adj_ann(np.zeros(len(perL)))
adjv = np.where(ok, 0.1 * (perX - perL), 0.0)
a1 = adj_ann(adjv)
adjv2 = 0.1 * (np.where(ok, perX, perU) - perL)                      # 沒有下市股的月份改用股池平均
a2 = adj_ann(adjv2)
adjv3 = np.where(ok, 0.1 * (-1.0 - perL), 0.0)                       # 極端：該檔當月歸零（不合理的上界，只當參考）
adjv4 = 0.1 * (perU - perL)                                           # 每月 1 檔換成股池平均
a4 = adj_ann(adjv4)
q2["c_bound"] = dict(base=(pc(b19_), pc(bf_)), replace1_delisted_avg=(pc(a1[0]), pc(a1[1])), drop_pp=(pc(b19_ - a1[0]), pc(bf_ - a1[1])),
                     replace1_delisted_or_univ=(pc(a2[0]), pc(a2[1])), drop_pp2=(pc(b19_ - a2[0]), pc(bf_ - a2[1])),
                     replace1_univ_avg_every_month=(pc(a4[0]), pc(a4[1])), drop_pp4=(pc(b19_ - a4[0]), pc(bf_ - a4[1])),
                     months_adjusted_2019=int(sum(1 for d, o in zip(sig_dates, ok) if o and d > "2018-12-28")),
                     months_2019=int(sum(1 for d in sig_dates if d > "2018-12-28")),
                     mean_gap_list_minus_delisted=float((perL[ok] - perX[ok]).mean()), mean_gap_list_minus_univ=float(np.nanmean(perL - perU)))
out["q2"] = q2
print("q2", q2["a_total"], {k: v for k, v in q2["b_delisted"].items() if k not in ("rows",)}, q2["b_in_pool"], q2["c_bound"])
print("q2a", q2["a_by_year"])

# ================================================================= 問題三
q3 = {}
sg = frozen["signals"]
q3["a_signal_check"] = dict(n=len(sg), all_signal_after_deadline=all(x["signal"] > x["deadline"] for x in sg),
                            all_deadline_ge_10th=all(x["deadline"] >= x["d10"] for x in sg),
                            n_deadline_postponed=sum(1 for x in sg if x["deadline"] != x["d10"]),
                            min_signal_dom=min(int(x["signal"][8:]) for x in sg), max_signal_dom=max(int(x["signal"][8:]) for x in sg),
                            dom_dist=pd.Series([int(x["signal"][8:]) for x in sg]).value_counts().sort_index().to_dict(),
                            buy_is_next_trading_day=all(P.dstr[L.didx(P, x["signal"]) + 1] == x["buy"] for x in sg))
# (b) 2013 不設缺值
Sn, Dn = L.surge(R, mi0, null2013=False)
pn, infn = L.c1_plans(P, Sn, mi0)
rn_ = L.simulate(P, pn)
q3["b_no2013_null"] = S2(rn_); q3["b_no2013_null"]["first_signal"] = infn[0]["signal"]; q3["b_no2013_null"]["n_signals"] = len(infn)
same = sum(1 for x in info if list(pn[x["s"]][:10]) == list(plans[x["s"]][:10]))
q3["b_no2013_null"]["lists_identical_from_2016_02"] = (same, len(info))
# (c) 訊號日後移
q3["c_shift"] = {}
for k in (1, 3, 5, 10):
    pk, _ = L.c1_plans(P, S, mi0, shift=k)
    q3["c_shift"][k] = S2(L.simulate(P, pk))
# (d) 集中度
tr = base["trades"].sort_values("pnl", ascending=False)
tot_closed = float(tr.pnl.sum()); tot_nav = float(base["nav"][E] - 1e6)
top10 = tr.head(10)
q3["d_conc"] = dict(total_closed_pnl=tot_closed, total_nav_gain=tot_nav, top10_pnl=float(top10.pnl.sum()),
                    top10_share_of_closed=float(top10.pnl.sum() / tot_closed), top10_share_of_nav_gain=float(top10.pnl.sum() / tot_nav),
                    top5_share_of_closed=float(tr.head(5).pnl.sum() / tot_closed),
                    top10=[dict(code=a, buy=b_, sell=c_, pnl=round(float(d_)), ret=pc(e_)) for a, b_, c_, d_, e_ in
                           zip(top10.code, top10.buy_date, top10.sell_date, top10.pnl, top10.ret)])
blk = set(zip(top10.code, top10.buy_date))
q3["d_block10_entry"] = S2(L.simulate(P, plans, block=blk))
span = {}
for c_, a_, b_ in zip(top10.code, top10.tb, top10.ts): span.setdefault(c_, []).append((int(a_), int(b_)))
q3["d_block10_span"] = S2(L.simulate(P, plans, block_span=span))
# 以報酬率（非金額）計的集中度：等額口徑
tr2 = base["trades"]; tr19 = tr2[tr2.tb > i19]
rs_ = np.sort(tr19.ret.values)[::-1]
q3["d_ret_conc"] = dict(n=int(len(rs_)), sum_ret=float(rs_.sum()), top10_ret_share=float(rs_[:10].sum() / rs_.sum()),
                        exp_all=pc(rs_.mean()), exp_ex_top10=pc(rs_[10:].mean()), exp_ex_top20=pc(rs_[20:].mean()), median=pc(np.median(rs_)))
tr2 = tr2.assign(sy=tr2.sell_date.str[:4], ind=tr2.code.str[:2])
q3["d_by_year"] = {y: dict(pnl=round(float(g_.pnl.sum())), share=round(float(g_.pnl.sum() / tot_closed) * 100, 1), n=int(len(g_)), mean_ret=pc(g_.ret.mean()), win=pc((g_.ret > 0).mean()))
                   for y, g_ in tr2.groupby("sy")}
gi = tr2.groupby("ind").agg(pnl=("pnl", "sum"), n=("pnl", "size"), mean_ret=("ret", "mean"), sum_ret=("ret", "sum")).sort_values("pnl", ascending=False)
q3["d_by_ind"] = {k: dict(pnl=round(float(r_.pnl)), share=round(float(r_.pnl / tot_closed) * 100, 1), n=int(r_.n), mean_ret=pc(r_.mean_ret),
                          ret_share=round(float(r_.sum_ret / tr2.ret.sum()) * 100, 1)) for k, r_ in gi.iterrows()}
# (e) 固定持有
q3["e_hold"] = {}
for hd in (10, 20):
    q3["e_hold"][hd] = S2(L.simulate(P, plans, hold_days=hd))
# (f) 前 20 檔
q3["f_top20"] = S2(L.simulate(P, plans, nmax=20))
# 附帶：2018-12-28 以 100 萬重新起跑（帳戶規模接近實際）、不設 5% 上限
t0_keep = P.t0; P.t0 = i19
r19 = L.simulate(P, plans); s19 = L.summary(P, r19)
q3["x_start_2019_1m"] = dict(ann_2019=pc(s19["ann_2019"]), n=s19["trades"]["n"], liq_capped=s19["cnt"]["liq_capped"], sell_split=s19["cnt"]["sell_split"],
                             final_nav=s19["final_nav"], mdd=pc(s19["mdd_2019"][0]))
P.t0 = t0_keep
q3["x_no_liq_cap"] = S2(L.simulate(P, plans, liq_cap=None))
q3["x_no_lot_rule"] = S2(L.simulate(P, plans, lot_rule=False))
# 月超額 t
ma = L.monthly(P, base["nav"], i19); me = L.monthly(P, e50, i19); xx = (ma - me).values
q3["x_month_excess_t_2019"] = dict(t=float(xx.mean() / (xx.std(ddof=1) / np.sqrt(len(xx)))), mean=float(xx.mean()), n=int(len(xx)),
                                   win_months=float((xx > 0).mean()))
out["q3"] = q3
print("q3b", q3["b_no2013_null"]["ann_2019"], q3["b_no2013_null"]["ann_full"], q3["b_no2013_null"]["lists_identical_from_2016_02"])
print("q3c", {k: (v["ann_2019"], v["ann_full"]) for k, v in q3["c_shift"].items()})
print("q3d", q3["d_conc"]["top10_share_of_closed"], q3["d_conc"]["top10_share_of_nav_gain"], q3["d_block10_entry"]["ann_2019"], q3["d_block10_span"]["ann_2019"], q3["d_ret_conc"])
print("q3d year", q3["d_by_year"]); print("q3d ind", dict(list(q3["d_by_ind"].items())[:10]))
print("q3e", {k: (v["ann_2019"], v["ann_full"], v["n"], v["n_2019"], v["exp"], v["exp_2019"], v["win"], v["hold"], v["avg_npos"]) for k, v in q3["e_hold"].items()})
print("q3f", q3["f_top20"]["ann_2019"], q3["f_top20"]["ann_full"], q3["f_top20"]["n"], q3["f_top20"]["exp"], q3["f_top20"]["mdd_2019"])
print("x", q3["x_start_2019_1m"], q3["x_no_liq_cap"]["ann_2019"], q3["x_no_lot_rule"]["ann_2019"], q3["x_no_lot_rule"]["ann_full"], q3["x_month_excess_t_2019"])
out["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
json.dump(out, open(os.path.join(HERE, "review_numbers.json"), "w"), ensure_ascii=False, indent=1, default=str)
print("peakRSS", out["peak_rss_mb"])
