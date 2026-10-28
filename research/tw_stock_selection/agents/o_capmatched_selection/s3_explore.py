#!/usr/bin/env python3
"""步驟 3a：探索（只用挑選期 2019-01～2022-12 的訊號窗；最後一窗止於 2023-01 換股日開盤）。
  E1 機制拆解（主策略 V4 AB 已看過的期間，描述）：主策略−0050 ＝（同池等權−0050）＋（主策略−同池等權）
  E2 P1 層內選股力：市值層 2–10、11–50、51–300（以及 51–150、151–300），層內合格非金融股分數前 30% 減層內市值加權（扣一般成本：
     前 30% 籃子單邊周轉×來回成本）；另報等權版與「減全層（含金融、不合格）市值加權」版
  E3 引擎驗證：0050 自由流通複製（REP）經月換股引擎，對 0050 的追蹤誤差（資料檢查，不是策略）
  E4 P6 探索：股價 ≥ 500 與 < 500 的層內殘差差距，2020-10-26 盤中零股開放前後（挑選期內）
寫 explore.json 的 STEP3。"""
import sys, json, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import cmlib as L, cmstrat as T

SEL_A, SEL_B = pd.Timestamp('2019-01-01'), pd.Timestamp('2022-12-31')
days = [d for d in L.official_days('2018-12-01', '2023-01-31') if d >= SEL_A]
assert days[-1] < pd.Timestamp('2023-02-01')
G = L.grow_exfix()
b, s = L.rates('normal'); RT = b + s
out = {}

# E1
nv = pd.read_parquet(L.RAW / 'navs' / 'navs_v4.parquet')
m_, ew_ = nv['AB_V4|normal'], nv['AB_V4|EW']
xm = L.excess_series(m_, L.E50, days, SEL_A, SEL_B); xe = L.excess_series(ew_, L.E50, days, SEL_A, SEL_B)
xs = (L.win_ret(m_, days) - L.win_ret(ew_, days)).reindex(xm.index)
out['E1_decomp_selection'] = {'main_minus_0050_pp': float(xm.mean() * 100), 't': L.tstat(xm), 'ew_minus_0050_pp': float(xe.mean() * 100), 't_ew': L.tstat(xe),
                              'main_minus_ew_pp': float(xs.mean() * 100), 't_sel': L.tstat(xs), 'n': int(len(xm)),
                              'note': '已看過（o6 拆解）；只描述。'}

# E2 層內選股力
TIERS = {'T1_2_10': (2, 10), 'T2_11_50': (11, 50), 'T3_51_300': (51, 300), 'T3a_51_150': (51, 150), 'T3b_151_300': (151, 300)}
rows = []; prevw = {k: {} for k in TIERS}; prevw_eq = {k: {} for k in TIERS}
for i in range(len(days) - 1):
    d = days[i]; cs = T.cross_section(d); e0 = cs['e']; e1 = T.cross_section(days[i + 1])['e'] if days[i + 1] in L.IDX else None
    f = L.window_factors_all(e0, e1, G)
    rank, mc = cs['rank'], cs['mc']
    for k, (a, bb) in TIERS.items():
        Tm = (rank >= a) & (rank <= bb)
        el = Tm & cs['elig'] & np.isfinite(f)
        idx = T.score_order(cs, el)
        if len(idx) < 3: continue
        ntop = max(1, int(math.ceil(0.3 * len(idx)))); top = idx[:ntop]
        wt = mc[top] / mc[top].sum(); rtop = float((wt * (f[top] - 1)).sum())
        wb = mc[el] / mc[el].sum(); rel = float((wb * (f[el] - 1)).sum())
        allT = Tm & np.isfinite(f) & np.isfinite(mc)
        rall = float(((mc[allT] / mc[allT].sum()) * (f[allT] - 1)).sum())
        req = float((f[top] - 1).mean()); rel_eq = float((f[el] - 1).mean())
        nw = dict(zip(top.tolist(), wt.tolist())); tov = 0.5 * sum(abs(nw.get(j, 0) - prevw[k].get(j, 0)) for j in set(nw) | set(prevw[k]))
        nwe = {j: 1 / ntop for j in top.tolist()}; tove = 0.5 * sum(abs(nwe.get(j, 0) - prevw_eq[k].get(j, 0)) for j in set(nwe) | set(prevw_eq[k]))
        prevw[k] = nw; prevw_eq[k] = nwe
        hi = el & (L.Cn[cs['t']] >= 500); lo = el & (L.Cn[cs['t']] < 500)
        rows.append(dict(d=d, tier=k, n_el=int(el.sum()), ntop=ntop, top_cap=rtop - rel - (tov * RT if i > 0 else b), top_cap_gross=rtop - rel,
                         top_eq=req - rel_eq - (tove * RT if i > 0 else b), top_vs_alltier=rtop - rall - (tov * RT if i > 0 else b), tov=tov,
                         hi500_share=float(mc[hi].sum() / mc[el].sum()) if el.any() else np.nan))
D = pd.DataFrame(rows)
D.to_csv(L.DATA / 'explore_tier_windows.csv', index=False)
p1 = {}
for k in TIERS:
    z = D[D.tier == k]
    p1[k] = {c: {'mean_pp': float(z[c].mean() * 100), 't': L.tstat(z[c]), 'n': int(len(z))} for c in ['top_cap', 'top_cap_gross', 'top_eq', 'top_vs_alltier']}
    p1[k]['n_el_median'] = float(z.n_el.median()); p1[k]['turnover_oneway_mean'] = float(z.tov.iloc[1:].mean())
    # 半期
    for lab, a, bb in [('2019-2020', '2019-01-01', '2020-12-31'), ('2021-2022', '2021-01-01', '2022-12-31')]:
        zz = z[(z.d >= a) & (z.d <= bb)]
        p1[k][f'top_cap_{lab}'] = {'mean_pp': float(zz.top_cap.mean() * 100), 't': L.tstat(zz.top_cap)}
out['E2_P1_tier_selection_selection_period'] = p1

# E2b 補充（探索，非 P1 定義）：層內分數前 N 檔等權（N＝5、10、20）減層內市值加權、減層內等權（扣一般成本）
rows = []; prevN = {}
for i in range(len(days) - 1):
    d = days[i]; cs = T.cross_section(d); e0 = cs['e']; e1 = T.cross_section(days[i + 1])['e']
    f = L.window_factors_all(e0, e1, G); rank, mc = cs['rank'], cs['mc']
    for k, (a, bb) in [('T2_11_50', (11, 50)), ('T3_51_300', (51, 300)), ('T23_11_300', (11, 300)), ('T123_2_300', (2, 300))]:
        el = (rank >= a) & (rank <= bb) & cs['elig'] & np.isfinite(f)
        idx = T.score_order(cs, el)
        rel = float(((mc[el] / mc[el].sum()) * (f[el] - 1)).sum()); rew = float((f[el] - 1).mean())
        for n in (5, 10, 20):
            if len(idx) < 2 * n: continue
            top = idx[:n]; r = float((f[top] - 1).mean())
            key = (k, n); nw = set(top.tolist()); tov = len(nw - prevN.get(key, set())) / n if i > 0 else 1.0
            prevN[key] = nw
            c = tov * RT if i > 0 else b
            rows.append(dict(d=d, tier=k, n=n, vs_cap=r - rel - c, vs_ew=r - rew - c, tov=tov))
D2 = pd.DataFrame(rows)
out['E2b_topN_within_tier_exploratory'] = {f'{k}|N{n}': {'vs_cap_pp': float(z.vs_cap.mean() * 100), 't_cap': L.tstat(z.vs_cap), 'vs_ew_pp': float(z.vs_ew.mean() * 100), 't_ew': L.tstat(z.vs_ew), 'tov': float(z.tov.iloc[1:].mean())}
                                          for (k, n), z in D2.groupby(['tier', 'n'])}

# E3 引擎驗證：REP
nav, lg, plan, info = T.run(('REP', {}), days, days[-1], cost='normal', G=G)
out['E3_replica_engine'] = L.rel_stats(nav, L.E50, days, SEL_A, SEL_B)
out['E3_replica_engine']['turnover_oneway'] = lg['turnover']

# E4 P6 探索：≥500 元 vs <500 元的層內殘差（挑選期內，2020-10-26 前後）
rows6 = []
for i in range(len(days) - 1):
    d = days[i]; cs = T.cross_section(d); e0 = cs['e']; e1 = T.cross_section(days[i + 1])['e']
    f = L.window_factors_all(e0, e1, G)
    for k, (a, bb) in [('T2_11_50', (11, 50)), ('T3_51_300', (51, 300)), ('T1_2_10', (2, 10))]:
        Tm = (cs['rank'] >= a) & (cs['rank'] <= bb); el = Tm & cs['elig'] & np.isfinite(f)
        if el.sum() < 6: continue
        # 層內殘差：個股報酬 − 層內合格股市值加權；按分數高 30% 分；再按價位分
        bench = float(((cs['mc'][el] / cs['mc'][el].sum()) * (f[el] - 1)).sum())
        idx = T.score_order(cs, el); top = set(idx[:max(1, int(math.ceil(0.3 * len(idx))))].tolist())
        for j in np.where(el)[0]:
            rows6.append(dict(d=d, tier=k, hi=L.Cn[cs['t'], j] >= 500, top=j in top, res=f[j] - 1 - bench))
E6 = pd.DataFrame(rows6)
p6 = {}
for lab, a, bb in [('pre_2019-01_2020-10', '2019-01-01', '2020-10-25'), ('post_2020-10_2022-12', '2020-10-26', '2022-12-31')]:
    z = E6[(E6.d >= a) & (E6.d <= bb)]
    g = z.groupby(['d', 'hi', 'top']).res.mean().unstack(['hi', 'top'])
    sp_hi = (g.get((True, True)) - g.get((True, False))) if (True, True) in g and (True, False) in g else pd.Series(dtype=float)
    sp_lo = (g.get((False, True)) - g.get((False, False)))
    p6[lab] = {'spread_hi500_pp': float(sp_hi.mean() * 100) if len(sp_hi.dropna()) else None, 'n_hi_windows': int(sp_hi.notna().sum()),
               'spread_lo500_pp': float(sp_lo.mean() * 100), 't_lo': L.tstat(sp_lo.dropna()), 'n_hi500_stockwindows': int(z.hi.sum())}
out['E4_P6_exploratory'] = p6
o = json.loads((L.HERE / 'explore.json').read_text(encoding='utf-8')); o['STEP3'] = out
(L.HERE / 'explore.json').write_text(L.dumps(o), encoding='utf-8')
print(L.dumps(out))
