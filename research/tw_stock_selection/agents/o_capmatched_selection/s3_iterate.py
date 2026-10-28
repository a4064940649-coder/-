#!/usr/bin/env python3
"""步驟 3b：迭代（只用挑選期 2019-01～2022-12 的訊號窗；淨值止於 2023-01 換股日收盤，不碰 2023 年以後任何報酬）。
第 1 輪：事前列出的 12 個版本（cmstrat.VERSIONS：F1 8 個、F2 2 個、F3 2 個）全部跑。
挑選準則（事前）：扣一般成本後對 0050 的 IR（訊號窗）；IR 差 < 0.05 視為平手，選周轉較低者。
硬約束：台積電主動權重 |·| ≤ 2pp（相對複製）、單檔 |主動| ≤ 5pp、追蹤誤差 ≤ 10%／年（P2）。
寫 iterations.json。"""
import sys, json, math, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import cmlib as L, cmstrat as T

SEL_A, SEL_B = pd.Timestamp('2019-01-01'), pd.Timestamp('2022-12-31')
days = [d for d in L.official_days('2018-12-01', '2023-01-31') if d >= SEL_A]
assert days[-1] < pd.Timestamp('2023-02-01') and days[-1].year == 2023
G = L.grow_exfix()
nv = pd.read_parquet(L.RAW / 'navs' / 'navs_v4.parquet')
MAIN = nv['AB_V4|normal']
rep_nav, _, _, _ = T.run(('REP', {}), days, days[-1], 'normal', G)
r2330 = pd.Series(np.cumprod(G[:, L.J2330]), index=L.IDX)

def evaluate(v):
    o = {}
    for cost in ('normal', 'stress'):
        nav, lg, plan, info = T.run(v, days, days[-1], cost, G)
        rs = L.rel_stats(nav, L.E50, days, SEL_A, SEL_B)
        rs['turnover_oneway'] = lg['turnover']
        if cost == 'normal':
            xr = L.excess_series(nav, rep_nav, days, SEL_A, SEL_B)
            rs['vs_replica'] = {'mean_pp': float(xr.mean() * 100), 't': L.tstat(xr)}
            xm = (L.win_ret(nav, days) - L.win_ret(MAIN, days).reindex(L.win_ret(nav, days).index))
            xm = xm[(xm.index >= SEL_A) & (xm.index <= SEL_B)].dropna()
            rs['vs_main_AB_V4'] = {'mean_pp': float(xm.mean() * 100), 't': L.tstat(xm)}
            I = pd.DataFrame(info)
            rs['holdings_median'] = float(I.n_hold.median()); rs['active_share_mean'] = float(I.active_share.mean())
            rs['max_abs_active'] = float(I.max_abs_active.max()); rs['max_abs_act2330'] = float(I.act2330.abs().max())
            rs['n_blocked'] = int(I.n_blocked.sum())
            # 台積電主動權重貢獻：Σ 窗 主動權重 × (2330 窗報酬 − 0050 窗報酬)
            w2 = L.win_ret(r2330, days); w5 = L.win_ret(L.E50, days)
            ca = I.act2330.to_numpy()[:len(w2)] * (w2 - w5.reindex(w2.index)).to_numpy()[:len(I)]
            rs['tsmc_contrib_pp_m'] = float(np.nanmean(ca) * 100)
            out_nav = nav
        o[cost] = rs
    return o, out_nav

res = {}; navs = {}
t0 = time.time()
for v in T.VERSIONS:
    res[v], navs[v] = evaluate(v)
    n = res[v]['normal']
    print(f"{v:18s} IR {n['IR']:.2f} t {n['t']:.2f} ex {n['mean_pp']:.2f}pp TE {n['TE']:.1f} CAGR {n['CAGR']:.1f} (0050 {n['bench_CAGR']:.1f}) MDD {n['maxDD']:.1f} "
          f"tov {n['turnover_oneway']:.2f} AS {n['active_share_mean']:.2f} vsREP {n['vs_replica']['mean_pp']:.2f}/{n['vs_replica']['t']:.2f} "
          f"vsMAIN {n['vs_main_AB_V4']['mean_pp']:.2f}/{n['vs_main_AB_V4']['t']:.2f} stressIR {res[v]['stress']['IR']:.2f} maxact {n['max_abs_active']:.3f} act2330 {n['max_abs_act2330']:.4f}", flush=True)
print('elapsed', time.time() - t0)
ok = {v: r for v, r in res.items() if r['normal']['TE'] <= 10 and r['normal']['max_abs_act2330'] <= 0.02 and r['normal']['max_abs_active'] <= 0.05 + 1e-6}
best = max(ok, key=lambda v: ok[v]['normal']['IR'])
ties = [v for v in ok if ok[best]['normal']['IR'] - ok[v]['normal']['IR'] < 0.05]
chosen = min(ties, key=lambda v: ok[v]['normal']['turnover_oneway'])
pd.DataFrame(navs).to_parquet(L.DATA / 'navs_selection.parquet')
it = {'round1': {'versions': list(T.VERSIONS), 'reason': '事前列出的 12 個版本全部跑（F1 層替換 8、F2 分數傾斜 2、F3 層內中性 2）',
                 'results': res, 'feasible': list(ok), 'best_IR': best, 'ties_within_0.05': ties, 'chosen': chosen,
                 'criterion': '扣一般成本 IR（訊號窗、對 0050）；平手（差 < 0.05）選周轉最低；約束：TE ≤ 10%、|台積電主動| ≤ 2pp、|單檔主動| ≤ 5pp'},
      'n_versions_total': len(T.VERSIONS), 'rounds_run': 1,
      'note': '只用挑選期；淨值止於 2023-01 換股日收盤。第 2、3 輪只有在第 1 輪全部不可行或有實作錯誤時才跑（事前規則）。'}
(L.HERE / 'iterations.json').write_text(L.dumps(it), encoding='utf-8')
print('chosen', chosen, 'best', best, 'ties', ties)
