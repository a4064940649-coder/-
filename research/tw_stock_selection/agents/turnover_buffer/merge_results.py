#!/usr/bin/env python3
"""合併 results_buffer.json、results_alloc_buffer.json、results_sens.json → results.json，並印出 FINDINGS 用的表格（summary.txt）。"""
import json
from pathlib import Path
HERE = Path(__file__).resolve().parent
rb = json.loads((HERE / 'results_buffer.json').read_text(encoding='utf-8'))
ra = json.loads((HERE / 'results_alloc_buffer.json').read_text(encoding='utf-8'))
rs = json.loads((HERE / 'results_sens.json').read_text(encoding='utf-8'))
(HERE / 'results.json').write_text(json.dumps({'buffer': rb, 'alloc': ra, 'sensitivity_post_hoc': rs}, ensure_ascii=False, indent=1), encoding='utf-8')
out = []
P = out.append
KS = {30: [30, 45, 60, 90], 10: [10, 15, 20, 30]}
pct = lambda x: f'{x * 100:.1f}%'
for s in ('A_P1only', 'B_MAIN_P1xREV'):
    r = rb[s]; W = list(r['windows_def'])
    P(f'\n=== {s} ===  EW: ' + '; '.join(f"{k} {pct(v['CAGR'])} DD {pct(v['maxDD'])}" for k, v in r['EW'].items()))
    for n, ks in KS.items():
        sel = r[f'SELECTION_n{n}']
        P(f'\n-- n{n} selection: {sel["t_vs_EW_by_K"]} best {sel["best_K"]} gain {sel["gain_vs_orig"]} chosen {sel["chosen_K"]}')
        for w in W:
            P(f'\n[{w}] | K | 年化 | 最大回撤 | 對0050 t | 對等權月超額 | 對等權 t | gross 對等權 t | 壓力 對等權 t | 周轉(市值) | 新進名稱比例 | 成本拖累 pp/年 | 配對差 %/月 | 配對 t(月) | 配對 t(日) | gross 配對 t |')
            for K in ks:
                o = r[f'n{n}_K{K}']; x = o['0.001']; ww = x['windows'][w]
                if ww is None: continue
                p = x[f'paired_vs_K{n}'][w]; pg = o['gross'][f'paired_vs_K{n}'][w]
                P(f"| {K} | {pct(ww['CAGR'])} | {pct(ww['maxDD'])} | {ww['t_vs_0050_m']} | {ww['excess_vs_EW_pct_m']}% | {ww['t_vs_EW_m']} | "
                  f"{o['gross']['windows'][w]['t_vs_EW_m']} | {o['stress']['windows'][w]['t_vs_EW_m']} | {x['turnover_value_per_rebalance']} | {o['name_turnover']} | "
                  f"{x['cost_drag_pp_per_year'][w]} | {p['diff_pct_m']} | {p['t_m']} | {p['t_daily']} | {pg['t_m']} |")
        P('\n逐年 %（滑價 0.1%）：')
        ys = sorted(r[f'n{n}_K{ks[0]}']['0.001']['yearly_pct'], key=int)
        P('| 年 | ' + ' | '.join(f'K{K}' for K in ks) + ' |')
        for y in ys:
            P(f'| {y} | ' + ' | '.join(str(r[f'n{n}_K{K}']['0.001']['yearly_pct'].get(y)) for K in ks) + ' |')
P('\n=== ALLOC ===')
for k, v in ra.items():
    if k.startswith('_'): continue
    f = v['windows']
    P(f"{k} | " + ' | '.join(f"{w}: {pct(f[w]['CAGR'])} DD {pct(f[w]['maxDD'])} t50 {f[w]['t_vs_0050_m']}" for w in ('full_2021-03_2026-10', 'select_2021-03_2023-12', 'valid_2024-01_2026-10', 'valid_ex2026_2024-01_2025-12') if f.get(w))
      + f" | 期末 {v['final_value_NTD']:,} | 重現 {v.get('reproduces_results_alloc2', '')}")
txt = '\n'.join(out); (HERE / 'summary.txt').write_text(txt, encoding='utf-8'); print(txt)
