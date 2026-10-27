#!/usr/bin/env python3
"""事後敏感度（不是事前檢定、不參與選擇）：更細的 K 網格，看選中的 K 是否落在平滑高原或只是單點。
輸出 results_sens.json。"""
import json, sys
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_buffer_defs as R  # noqa: E402
B, M, L = R.B, R.M, R.L
GRID = {('A_P1only', 30): [30, 35, 40, 45, 50, 60, 75, 90, 105, 120, 150, 200],
        ('A_P1only', 10): [10, 12, 15, 20, 25, 30, 40, 50, 60],
        ('B_MAIN_P1xREV', 30): [30, 33, 36, 40, 45, 60],
        ('B_MAIN_P1xREV', 10): [10, 11, 12, 15, 20, 30]}
res = {'_meta': {'note': '事後敏感度，非事前檢定；只供判斷穩定性。t_EW = 對等權月窗 t；paired = 對 K=N 月窗配對 t。'}}
for strat, builder in (('A_P1only', R.build_A), ('B_MAIN_P1xREV', R.build_B)):
    D, sc, el, W, selw = builder()
    _, pools = M.select(sc, el, D, 30); EW, _ = M.sim(pools, 'gross', D)
    for n in (30, 10):
        base = {}
        for K in GRID[(strat, n)]:
            ev, _ = B.select_buf(sc, el, D, n, K)
            row = {}
            for cost in ('gross', 0.001):
                nav, tov = M.sim(ev, cost, D)
                if K == n: base[cost] = nav
                x = M.month_rets(nav, D) - M.month_rets(base[cost], D)
                row[str(cost)] = {'turnover': round(tov, 3), **{k: {'CAGR': M.stats(nav, a, b, D, L.E50_NAV, EW)['CAGR'],
                                                               't_EW': M.stats(nav, a, b, D, L.E50_NAV, EW)['t_vs_EW_m'],
                                                               'paired_t': L.tstat(R.seg(x, a, b)),
                                                               'paired_pct_m': round(float(R.seg(x, a, b).mean() * 100), 2)} for k, (a, b) in W.items()}}
            res[f'{strat}_n{n}_K{K}'] = row
            s, v = list(W)[0], list(W)[1]
            print(strat, n, K, 'tov', row['0.001']['turnover'], 'sel tEW', row['0.001'][s]['t_EW'], 'p', row['0.001'][s]['paired_t'],
                  '| val tEW', row['0.001'][v]['t_EW'], 'p', row['0.001'][v]['paired_t'], '| gross sel p', row['gross'][s]['paired_t'], 'val p', row['gross'][v]['paired_t'], flush=True)
(HERE / 'results_sens.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
