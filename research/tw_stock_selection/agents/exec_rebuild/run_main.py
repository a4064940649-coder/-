#!/usr/bin/env python3
"""A＋B：主策略變體 M0～M3 × n∈{30,10} × 成本 {滑價 0.1%、0.3%、壓力來回 1%}；日頻 NAV。輸出 results_main.json、main_nav.parquet。"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import exlib as L

OUT = L.HERE
res = {'_note': 'M0_period 為原版月報酬法（必須重現 49.0%）；其餘為日頻 NAV 模擬。t_*_m 為換股日收盤→下一換股日收盤的月窗 t。'}

# ---- M0 原版重現（月報酬法，複製 bt_final_rule.run）----
def run_orig(n=30, rt=0.005):
    el = L.t300_d & L.yoy3.notna() & L.s52_d.notna() & ~L.inD_d
    sc = 0.5 * L.s52_d.where(el).rank(axis=1, pct=True) + 0.5 * L.yoy3.where(el).rank(axis=1, pct=True)
    port, uni, to, prev = {}, {}, [], None
    for t in L.R.index:
        ok = el.loc[t] & L.R.loc[t].notna()
        if ok.sum() < 50: continue
        pick = list(sc.loc[t][ok].sort_values(ascending=False).index[:n])
        port[t] = L.R.loc[t][pick].mean(); uni[t] = L.R.loc[t][ok].mean()
        if prev is not None: to.append(len(set(pick) - prev) / n)
        prev = set(pick)
    g = pd.Series(port); tov = float(np.mean(to))
    return g - tov * rt, pd.Series(uni), tov
for n in (30, 10):
    net, uni, tov = run_orig(n)
    ex = net - L.B.reindex(net.index)
    o = {'all': {**L.ann(net, L.dates), 't_vs_0050': L.tstat(ex), 't_vs_EW': L.tstat(net - uni), 'turnover_names': round(tov, 3)}}
    for k, (a, b) in {'select': ('2021-03-01', '2023-12-31'), 'valid': ('2024-01-01', '2026-12-31'), 'valid_ex2026': ('2024-01-01', '2025-12-31')}.items():
        m = (net.index >= a) & (net.index <= b)
        o[k] = {**L.ann(net[m], L.dates), 't_vs_0050': L.tstat(ex[m]), 't_vs_EW': L.tstat((net - uni)[m])}
    o['yearly'] = L.yearly(net, L.dates)
    res[f'M0_period_n{n}'] = o
    print('M0 period n', n, o['all'], flush=True)

t0 = L.pos(L.dates[0]); t1 = L.pos(L.dates[-1])
navs = {}
holdcount = {}
for var in ('M0', 'M1', 'M2', 'M3'):
    for n in (30, 10):
        ev, pools = L.selection(var, n)
        ew, _ = L.sim_portfolio(pools, 0.0, 0.0, t0, t1)
        navs[f'EW_{var}'] = ew
        g, tov = L.sim_portfolio(ev, 0.0, 0.0, t0, t1)
        navs[f'{var}_n{n}_gross'] = g
        res[f'{var}_n{n}_gross'] = {'turnover_value_per_rebalance': round(tov, 3), 'windows': L.all_windows(g, L.E50_NAV, ew)}
        holdcount[f'{var}_n{n}'] = {'min': int(min(len(v[1]) for v in ev.values())), 'mean': round(float(np.mean([len(v[1]) for v in ev.values()])), 1),
                                    'rebalances': len(ev), 'skipped_limit_up_names_total': int(L.selection.nblk) if var != 'M0' else 0}
        for slip in (0.001, 0.003, 'stress'):
            b, s = L.cost_rates(slip)
            nav, tov = L.sim_portfolio(ev, b, s, t0, t1)
            key = f'{var}_n{n}_slip{slip}'
            navs[key] = nav
            res[key] = {'buy_rate': round(b, 5), 'sell_rate': round(s, 5), 'turnover_value_per_rebalance': round(tov, 3),
                        'windows': L.all_windows(nav, L.E50_NAV, ew)}
            w = res[key]['windows']
            print(key, {k: (v['CAGR'], v.get('t_vs_0050_m'), v.get('t_vs_EW_m')) for k, v in w.items() if v}, flush=True)
res['names_held_per_rebalance'] = holdcount
# 診斷：M1 的差異來自「前一日訊號」還是「排除鎖漲停」（滑價 0.1%）
for var in ('M1_lagonly', 'M1_exclonly'):
    for n in (30, 10):
        ev, pools = L.selection(var, n); nb = int(L.selection.nblk)
        ew, _ = L.sim_portfolio(pools, 0.0, 0.0, t0, t1)
        b, s = L.cost_rates(0.001); nav, tov = L.sim_portfolio(ev, b, s, t0, t1)
        w = L.all_windows(nav, L.E50_NAV, ew)
        res[f'DIAG_{var}_n{n}_slip0.001'] = {'skipped_limit_up_names_total': nb, 'windows': {k: {kk: v[kk] for kk in ('CAGR', 't_vs_0050_m', 't_vs_EW_m')} for k, v in w.items() if v}}
        print('DIAG', var, n, res[f'DIAG_{var}_n{n}_slip0.001']['windows']['full_2021-03_2026-10'], flush=True)
res['0050_same_span'] = L.all_windows(L.E50_NAV.iloc[t0:t1 + 1])
(OUT / 'results_main.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
pd.DataFrame(navs).to_parquet(OUT / 'main_nav.parquet')
