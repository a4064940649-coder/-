#!/usr/bin/env python3
"""合併 results_long.json、results_combo.json → results.json，並印出 FINDINGS 用的摘要表。"""
import json
from pathlib import Path
HERE = Path(__file__).resolve().parent
lo = json.loads((HERE / 'results_long.json').read_text(encoding='utf-8')); co = json.loads((HERE / 'results_combo.json').read_text(encoding='utf-8'))
(HERE / 'results.json').write_text(json.dumps({'long_history_2005_2026': lo, 'combo_2021_03_2026_10': co}, ensure_ascii=False, indent=1), encoding='utf-8')
SIG = ['P1_52wHigh', 'P2_mom12_1', 'P3_mom6_1', 'P4_resid12_1_taiex', 'P5_indNeutral12_1', 'P6_volAdj12_1', 'P4e_resid12_1_EW']
WS = ['select_2005-01_2019-04', 'valid_2019-05_2026-10', 'main_2021-03_2026-10', 'valid_survivorOnly_2019-05_2023-12', 'valid_2024-01_2026-10', 'valid_2024-01_2025-12', 'full_2005-01_2026-10']
def f(v, p=False): return '—' if v is None else (f'{v*100:.1f}%' if p else f'{v}')
for key in ('n30_0.001', 'n10_0.001', 'n30_stress', 'n10_stress', 'n30_gross'):
    print(f'\n### {key}')
    for w in WS:
        print(f'\n#### {w}'); print('| 訊號 | 年化 | 最大回撤 | 對 0050 月超額 | t | 對等權月超額 | t | IC | IC t | 周轉 |'); print('|---|---|---|---|---|---|---|---|---|---|')
        e = lo['EW_common'][w]; print(f"| 等權（共同股池） | {f(e['CAGR'],1)} | {f(e['maxDD'],1)} | {e['excess_vs_0050_pct_m']}% | {e['t_vs_0050_m']} | — | — | — | — | — |")
        print(f"| 0050 | {f(e['0050_CAGR'],1)} | {f(e['0050_maxDD'],1)} | — | — | — | — | — | — | — |")
        for s in SIG:
            x = lo[s][key]['windows'][w]; ic = lo[s]['IC'][w]
            print(f"| {s} | {f(x['CAGR'],1)} | {f(x['maxDD'],1)} | {x['excess_vs_0050_pct_m']}% | {x['t_vs_0050_m']} | {x['excess_vs_EW_pct_m']}% | {x['t_vs_EW_m']} | {ic['mean']} | {ic['t']} | {lo[s][key]['turnover_value_per_rebalance']} |")
print('\nP4 vs own-pool EW4 (n30 0.001):', {w: lo['P4_resid12_1_taiex']['n30_0.001']['windows_vs_ownpool_EW4'][w]['t_vs_EW_m'] for w in WS})
print('P4 common window select 2006-02~2019-04 t_vs_EW:', {s: lo[s]['n30_0.001']['windows']['P4_common_select_2006-02_2019-04']['t_vs_EW_m'] for s in SIG})
print('\nSELECTION', lo['SELECTION'])
print('\nYEARLY n30 0.001')
yrs = sorted(lo['EW_yearly'], key=int)
print('| 年 | 0050 | 等權 | ' + ' | '.join(s.split('_')[0] for s in SIG) + ' |')
for y in yrs:
    print(f"| {y} | {lo['0050_yearly'].get(y)} | {lo['EW_yearly'][y]} | " + ' | '.join(str(lo[s]['n30_0.001']['yearly_pct'].get(y)) for s in SIG) + ' |')
print('\nREGIME'); [print(s, lo['REGIME'][s]) for s in SIG]
print('\nREGIME IC'); [print(s, {w: (lo[s]['IC'][w]['mean'], lo[s]['IC'][w]['t']) for w in lo[s]['IC'] if w.startswith('R_')}) for s in SIG]
print('\nBEAR'); [print(y, v) for y, v in lo['BEAR_YEARS'].items()]
print('\ncorr', lo['excess_corr_full'])
print('\nmeta', lo['_meta'])
print('\n### COMBO')
for key in ('n30_0.001', 'n30_stress', 'n10_0.001', 'n10_stress'):
    print(f'\n#### {key}'); print('| 規則 | 全期年化 | 回撤 | 對0050 t | 對等權 t | 2021-23 對0050 t | 2024-26 對0050 t | 2024-25 對0050 t | 2024-25 對等權 t | 配對差 %/月 | 配對 t(月) | 配對 t(日) | 2024-26 配對差 | t |'); print('|' + '---|' * 14)
    for k, v in co.items():
        if k.startswith('_') or k.startswith('DECOMP') or k.startswith('PRE'): continue
        w = v[key]['windows']; p = v[key]['paired_vs_MAIN']; a = w['full_2021-03_2026-10']
        print(f"| {k} | {f(a['CAGR'],1)} | {f(a['maxDD'],1)} | {a['t_vs_0050_m']} | {a['t_vs_EW_m']} | {w['select_2021-03_2023-12']['t_vs_0050_m']} | {w['valid_2024-01_2026-10']['t_vs_0050_m']} | {w['valid_ex2026_2024-01_2025-12']['t_vs_0050_m']} | {w['valid_ex2026_2024-01_2025-12']['t_vs_EW_m']} | {p['full_2021-03_2026-10']['diff_pct_m']} | {p['full_2021-03_2026-10']['t_m']} | {p['full_2021-03_2026-10']['t_daily']} | {p['valid_2024-01_2026-10']['diff_pct_m']} | {p['valid_2024-01_2026-10']['t_m']} |")
print('\nEW main', co['MAIN_P1xREV']['n30_0.001']['windows']['full_2021-03_2026-10'].get('EW_CAGR'))
print('DECOMP', co['DECOMP_n30_slip0.001'])
print('yearly main/P1/REV', {k: co[k]['n30_0.001']['yearly_pct'] for k in ['MAIN_P1xREV', 'ONLY_P1_52wHigh', 'ONLY_REV_yoy3']})
