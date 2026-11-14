# -*- coding: utf-8 -*-
"""S1 C1 獨立審查（補充，事後）：換股日位移 j=0..12 的中位數（主策略與 C1 同口徑）、分段對 0050。
用法：python3 -I -B rep_s1_review2.py <scratch_dir>     輸出：review2_numbers.json
背景：主策略正式數字 23.27%（V7X）是「13 個換股日 j 取中位數」，不是單一換股日；要比引擎樂觀程度必須同樣取 j 中位數。
j 的定義這裡假設為「基準訊號日往後 j 個交易日，j=0..12」（V7X 的確切定義未核）。"""
import sys, os, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import rep_s1_lib as L
P = L.build(sys.argv[1]); R, mi0 = L.load_rev(P); S, D = L.surge(R, mi0)
T = P.T; E = T - 1; i19 = L.didx(P, "2018-12-28"); e50 = P.e50.values
G = P.G; A_all = G / pd.DataFrame(G).rolling(252, min_periods=252).max().values
nm = R.shape[0]; Y = np.full_like(R, np.nan); Y[12:] = R[12:] / R[:-12] - 1.0
avail = np.array([L.rev_avail(P, mi0 + m)[2] for m in range(nm)])
fin = np.array([(c[:2] == "28") or c in ("5876", "5880", "6005", "6024") for c in P.codes])
ky = np.array([("KY" in P.names.get(c, "")) or ("DR" in P.names.get(c, "")) for c in P.codes])
codes_arr = np.array(P.codes)
base_days = []
for mth in sorted(set(d[:7] for d in P.dstr[P.t0:])):
    cand = [k for k in range(T) if P.dstr[k][:7] == mth and int(P.dstr[k][8:10]) >= 11]
    if cand: base_days.append(cand[0])


def main_plans(j):
    pl = {}
    for s0 in base_days:
        s = s0 + j
        if s >= T - 1 or s < P.t0: continue
        ms = np.where(avail <= s)[0]
        m = int(ms.max())
        b = (Y[m] + Y[m - 1] + Y[m - 2]) / 3.0
        idx = np.where(P.inbar[s] & P.is_twse[s] & np.isfinite(P.adv60[s]))[0]
        top = idx[np.argsort(-P.adv60[s, idx], kind="stable")[:300]]
        a = A_all[s]
        with np.errstate(invalid="ignore"):
            ok = (P.adv60[s, top] >= 1e7) & (P.C[s, top] >= 10) & P.traded[s, top] & ~fin[top] & ~ky[top] & np.isfinite(b[top]) & np.isfinite(a[top])
        pool = top[ok]
        if len(pool) < 20: continue
        sc = 0.5 * pd.Series(a[pool]).rank(pct=True).values + 0.5 * pd.Series(b[pool]).rank(pct=True).values
        pl[s] = pool[np.lexsort((codes_arr[pool], -sc))]
    return pl


out = dict(main={}, c1={}, main_x03={}, c1_x03={})
for j in range(13):
    for key, kw in (("main", {}), ("main_x03", dict(extra=0.003))):
        s = L.summary(P, L.simulate(P, main_plans(j), fallback=True, **kw))
        out[key][j] = (round(s["ann_2019"] * 100, 2), round(s["ann_full"] * 100, 2))
    pk, _ = L.c1_plans(P, S, mi0, shift=j)
    for key, kw in (("c1", {}), ("c1_x03", dict(extra=0.003))):
        s = L.summary(P, L.simulate(P, pk, **kw))
        out[key][j] = (round(s["ann_2019"] * 100, 2), round(s["ann_full"] * 100, 2))
for k in ("main", "c1", "main_x03", "c1_x03"):
    v = np.array([out[k][j] for j in range(13)])
    out[k + "_stats"] = dict(median_2019=float(np.median(v[:, 0])), min_2019=float(v[:, 0].min()), max_2019=float(v[:, 0].max()),
                             p10_2019=float(np.percentile(v[:, 0], 10)), median_full=float(np.median(v[:, 1])),
                             n_above_0050_2019=int((v[:, 0] > L.bench(P)["ann_2019"] * 100).sum()))
    print(k, [x[0] for x in v.tolist()], out[k + "_stats"])
# 分段對 0050
plans, info = L.c1_plans(P, S, mi0); base = L.simulate(P, plans); nav = base["nav"]
seg = {}
for lab, d0 in (("2016-02-17起", "2016-02-17"), ("2019起", "2018-12-28"), ("2021起", "2020-12-31"), ("2022起", "2021-12-30"), ("2024起", "2023-12-29")):
    i0 = L.didx(P, d0)
    seg[lab] = (P.dstr[i0], round(L.ann(P, nav, i0, E) * 100, 2), round(L.ann(P, e50, i0, E) * 100, 2))
for lab, a, b in (("2019-2023", "2018-12-28", "2023-12-29"), ("2019起扣掉2020", None, None)):
    if a:
        ia, ib = L.didx(P, a), L.didx(P, b)
        seg[lab] = (a, round(L.ann(P, nav, ia, ib) * 100, 2), round(L.ann(P, e50, ia, ib) * 100, 2))
ia, ib = L.didx(P, "2019-12-31"), L.didx(P, "2020-12-31")
yrs = (P.dates[E] - P.dates[i19]).days / 365.25 - (P.dates[ib] - P.dates[ia]).days / 365.25
seg["2019起扣掉2020"] = ("幾何", round(((nav[E] / nav[i19]) / (nav[ib] / nav[ia])) ** (1 / yrs) * 100 - 100, 2),
                       round(((e50[E] / e50[i19]) / (e50[ib] / e50[ia])) ** (1 / yrs) * 100 - 100, 2))
out["segments"] = seg; print(seg)
# 滾動 3 年：C1 贏 0050 的比例
w = 756; r_c = nav[P.t0 + w:] / nav[P.t0:-w]; r_e = e50[P.t0 + w:] / e50[P.t0:-w]
i0 = L.didx(P, "2016-02-17"); r_c = nav[i0 + w:] / nav[i0:-w]; r_e = e50[i0 + w:] / e50[i0:-w]
out["rolling3y_win_share"] = float((r_c > r_e).mean()); print("roll3y", out["rolling3y_win_share"])
json.dump(out, open(os.path.join(HERE, "review2_numbers.json"), "w"), ensure_ascii=False, indent=1)
