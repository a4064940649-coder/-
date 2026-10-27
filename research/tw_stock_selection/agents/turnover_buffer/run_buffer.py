#!/usr/bin/env python3
"""持股緩衝區檢定（事前固定，見 FINDINGS）。
(A) P1 52 週高點單獨：長歷史共同股池（同 momentum_family/run_long.py），2005-01～2019-04 挑選、2019-05～2026-10 驗證。
(B) 主策略 P1×月營收 3 個月年增率（同 run_combo.py 的 MAIN_P1xREV）：2021-03～2023-12 挑選、2024-01～2026-10 驗證。
N=30：K∈{30,45,60,90}；N=10：K∈{10,15,20,30}。成本 gross／滑價 0.1%／壓力。
事前選擇：挑選期、滑價 0.1%、對等權月窗 t 最高的 K；若比原版（K=N）高不到 0.3 → 維持原版。
輸出 results_buffer.json、buffer_nav.parquet。"""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import blib as B
M, L = B.M, B.L
T0 = time.time()
E50 = L.E50_NAV
KS = {30: [30, 45, 60, 90], 10: [10, 15, 20, 30]}
COSTS = ('gross', 0.001, 'stress')


from run_buffer_defs import build_A, build_B, seg  # noqa: E402


res = {'_meta': {'K_grid': KS, 'costs': [str(c) for c in COSTS],
                 'rule': '挑選期、滑價 0.1%、對等權月窗 t 最高的 K；若 t(K) − t(K=N) < 0.3 → 維持原版 K=N。',
                 'note': 'EW = 同股池等權 gross（同引擎、d+1 開盤換股；與 K 無關）。turnover_value = 每次換股賣出市值 ÷ 組合市值（漂移後權重）；'
                         'name_turnover = 每次換股新進名稱比例。cost_drag_pp_per_year = 扣成本前年化 − 扣成本後年化（百分點）。paired = 月窗（d 收盤→下一 d 收盤）報酬差，對原版 K=N。'}}
navs = {}
for strat, builder in (('A_P1only', build_A), ('B_MAIN_P1xREV', build_B)):
    D, sc, el, W, selw = builder()
    o = {'windows_def': W}
    ev0, pools = M.select(sc, el, D, 30)
    EW, _ = M.sim(pools, 'gross', D); navs[f'{strat}_EW'] = EW
    o['EW'] = {k: M.stats(EW, a, b, D, E50) for k, (a, b) in W.items()}
    for n, ks in KS.items():
        ref, _ = M.select(sc, el, D, n)
        for K in ks:
            ev, _ = B.select_buf(sc, el, D, n, K)
            if K == n:
                assert B.same_events(ev, ref), f'{strat} n{n} K=N 與原版不同'
            info = {'mean_names_kept': round(B.select_buf.mean_kept, 2), 'name_turnover': round(B.select_buf.name_turnover, 3),
                    'skipped_limit_up': int(B.select_buf.nblk)}
            for cost in COSTS:
                nav, tov = M.sim(ev, cost, D); navs[f'{strat}_n{n}_K{K}_{cost}'] = nav
                info[str(cost)] = {'turnover_value_per_rebalance': round(tov, 3),
                                   'windows': {k: M.stats(nav, a, b, D, E50, EW) for k, (a, b) in W.items()},
                                   'yearly_pct': M.yearly(nav)}
            for cost in (0.001, 'stress'):
                info[str(cost)]['cost_drag_pp_per_year'] = {k: round((info['gross']['windows'][k]['CAGR'] - info[str(cost)]['windows'][k]['CAGR']) * 100, 2)
                                                           for k in W if info[str(cost)]['windows'][k]}
            o[f'n{n}_K{K}'] = info
            w = info['0.001']['windows'][selw]
            print(strat, n, K, 'sel CAGR', w['CAGR'], 'tEW', w['t_vs_EW_m'], 't50', w['t_vs_0050_m'], 'tov', info['0.001']['turnover_value_per_rebalance'],
                  'kept', info['mean_names_kept'], round(time.time() - T0), flush=True)
        # 配對 t（對原版）
        for K in ks:
            for cost in COSTS:
                a_ = navs[f'{strat}_n{n}_K{K}_{cost}']; b_ = navs[f'{strat}_n{n}_K{n}_{cost}']
                x = M.month_rets(a_, D) - M.month_rets(b_, D); xd = a_.pct_change() - b_.pct_change()
                o[f'n{n}_K{K}'][str(cost)]['paired_vs_K{}'.format(n)] = {
                    k: {'diff_pct_m': round(float(seg(x, a, b).mean() * 100), 2), 't_m': L.tstat(seg(x, a, b)), 't_daily': L.tstat(seg(xd, a, b))}
                    for k, (a, b) in W.items()}
        # 事前選擇
        tsel = {K: o[f'n{n}_K{K}']['0.001']['windows'][selw]['t_vs_EW_m'] for K in ks}
        best = max(tsel, key=tsel.get)
        chosen = best if tsel[best] - tsel[n] >= 0.3 else n
        o[f'SELECTION_n{n}'] = {'select_window': selw, 't_vs_EW_by_K': tsel, 'best_K': best, 'gain_vs_orig': round(tsel[best] - tsel[n], 2), 'chosen_K': chosen,
                                'paired_chosen_vs_orig_slip0.001': o[f'n{n}_K{chosen}']['0.001'][f'paired_vs_K{n}']}
        print('SELECTION', strat, n, tsel, '->', chosen, flush=True)
    res[strat] = o

(HERE / 'results_buffer.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
pd.DataFrame(navs).to_parquet(HERE / 'buffer_nav.parquet')
print('done', round(time.time() - T0))
