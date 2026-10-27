#!/usr/bin/env python3
"""把 results.json 印成表（給 FINDINGS 用）。"""
import json, sys
from pathlib import Path
R = json.loads((Path(__file__).resolve().parent / 'results.json').read_text(encoding='utf-8'))
RULES = ['BASE', 'D1_K10', 'D1_K20', 'D1_K30', 'D2', 'D3', 'MIX_0.80']
def f(x, p=1): return '–' if x is None else f'{x*100:.{p}f}'
for u in ('ALL', 'EXFIN'):
    for blk, pers in (('RESULTS_SELVAL', ('SEL', 'VAL', 'SELVAL')), ('HOLDOUT', ('HOLD', 'HOLD_2024_2026'))):
        ru = R[blk][u]
        for p in pers:
            print(f'\n### {u} {p}')
            print('| 規則 | 放 | 年化 | MDD | Calmar | t對0050 | t對原版 | 月差% | 平均曝險 | 實際股票比重 | e<1月數 | 壓力年化 | 同MDD 0050前緣年化(w) | 同MDD 現金前緣年化(w) |')
            for rn in RULES:
                for alt in (('cash',) if rn == 'BASE' else ('cash', '0050')):
                    k = f'{rn}|{alt}|0.001'; o = ru[k][p]; st = ru.get(f'{rn}|{alt}|stress', {}).get(p, {})
                    m5 = o.get('matchDD_0050_CAGR'); w5 = o.get('matchDD_0050_w'); mc = o.get('matchDD_cash_CAGR'); wc = o.get('matchDD_cash_w')
                    print(f"| {rn} | {alt} | {f(o['CAGR'])} | {f(o['maxDD'])} | {o['Calmar']:.2f} | {o['t_0050']:.2f} | {o['t_vs_base'] if o['t_vs_base']==o['t_vs_base'] else float('nan'):.2f} | {o['diff_vs_base_m']*100:+.2f} | {o['avg_exposure_target']:.2f} | {o['avg_stock_weight_daily']:.2f} | {o['months_exposure_lt1']}/{o['n_months']} | {f(st.get('CAGR'))} | {f(m5)}({'' if w5 is None else w5}) | {f(mc)}({'' if wc is None else wc}) |")
    print(f'\n### {u} 空頭期 報酬／期內MDD')
    for rn in RULES:
        for alt in (('cash',) if rn == 'BASE' else ('cash', '0050')):
            k = f'{rn}|{alt}|0.001'
            b = dict(R['RESULTS_SELVAL'][u][k]['bears']); b.update(R['HOLDOUT'][u][k]['bears'])
            print(f'| {rn} | {alt} | ' + ' | '.join(f"{b[x]['ret']}／{b[x]['mdd']}" for x in ('2008', '2011', '2015', '2018', '2020-03', '2022')) + ' |')
    print(f'\n### {u} 逐年')
    for rn in RULES:
        for alt in (('cash',) if rn == 'BASE' else ('cash', '0050')):
            k = f'{rn}|{alt}|0.001'
            y = dict(R['RESULTS_SELVAL'][u][k]['SELVAL']['yearly']); y.update(R['HOLDOUT'][u][k]['HOLD']['yearly'])
            print(f'| {rn} | {alt} | ' + ' | '.join(f'{y[str(x)] if str(x) in y else y.get(x)}' for x in range(2006, 2027)) + ' |')
