#!/usr/bin/env python3
"""反方審查（資料與統計）o18：只讀 results.json，不改研究員檔案。python3 -I -B review_stat_o18.py"""
import json, math
from pathlib import Path
R = json.loads((Path(__file__).resolve().parent / 'results.json').read_text(encoding='utf-8'))
ST, TB = R['RESULTS_PAPER']['stats'], R['RESULTS_V7X']['table']
def P(k, c='normal', e='cons_main', ref='C0'): return ST[f'{k}|{c}|{e}|vs_{ref}']
print('== 1. V7X opt 是否為不利端（100 萬一般／300 萬／1,000 萬／壓力）')
for t, c in (('100萬', 'normal'), ('300萬', 'normal'), ('1000萬', 'normal'), ('100萬', 'stress')):
    for f in ('exc_med', 'exc_p10', 't_0050_med', 'exc_Y19on_med', 'exc_S1318_med'):
        xs = {v: TB[f'REAL_{v}|cons_main|{c}|{t}'][f] for v in ('opt', 'p90', 'carry')}
        w = min(xs, key=xs.get)
        print(f'  {c} {t} {f}: ' + ' '.join(f'{k}={x:+.3f}' for k, x in xs.items()) + f'  最不利={w}' + ('' if w == 'opt' else '  <-- opt 不是不利端'))
print('== 2. ALLPIT_REAL_opt 對 o17 ALLPIT opt 的月差 t（FINDINGS 寫 t +0.17）')
s = P('ALLPIT_REAL_opt', ref='ALLPIT_opt_o17')['ALL']; print('  ann_diff %.3f  t_diff %.3f' % (s['ann_diff_vs_ref'], s['t_diff_vs_ref']))
print('== 3. 分段 t（新正式紙上）')
m = P('ALLPIT_REAL_opt')
for k in ('SEG1', 'SEG2', 'SEG3', 'SEL', 'VAL', 'HOLD'):
    x = m[k]; se_m = x['mean_ex_pp'] / x['t_vs_0050'] if x['t_vs_0050'] else float('nan')
    print(f"  {k} n={x['n']} gex={x['gex']:+.2f} t={x['t_vs_0050']:.3f}  月均超額 {x['mean_ex_pp']:.3f}pp ±1.96SE→ 年化約 ±{1.96*se_m*12:.1f}pp")
print('== 4. 28.28→22.22 拆解（是否全為「2013 年後營收存活者偏誤」）')
c0 = P('C0', e='freeze')['ALL']['ann']
for k in ('U13_REAL_opt', 'UFULL_REAL_opt', 'ALLPIT_REAL_opt'):
    x = P(k); print(f"  {k}: ann {x['ALL']['ann']:.2f} (對 C0 {x['ALL']['ann']-c0:+.2f}) seg1 {x['SEG1']['gex']:.2f} seg2 {x['SEG2']['gex']:.2f} seg3 {x['SEG3']['gex']:.2f}")
u13, uf, al = (P(k)['ALL']['ann'] for k in ('U13_REAL_opt', 'UFULL_REAL_opt', 'ALLPIT_REAL_opt'))
print(f'  2013 後存活者 {u13-c0:+.2f}；2006～12 放回 {uf-u13:+.2f}；A_UB＋處置濾網 {al-uf:+.2f}（合計 {al-c0:+.2f}）')
print('== 5. P4 每槽傷害：real vs fill')
for k in ('U13_REAL_opt', 'ALLPIT_REAL_opt'):
    b = R['RESULTS_PAPER']['P4'][k]['by_how']; print(' ', k, {h: (v['n_slots'], round(v['mean_diff_pct'], 2), round(v['t_by_day'] or 0, 2)) for h, v in b.items()})
g = R['RESULTS_PAPER']['P4']['U13_REAL_opt']['by_grp']
a, b = g['le6'], g['gt12']
print('  le6 vs gt12 平均差 %.2f vs %.2f；t_by_day %.2f / %.2f（組間差未檢定）' % (a['mean_diff_pct'], b['mean_diff_pct'], a['t_by_day'], b['t_by_day']))
print('== 6. 拿掉最好 3 年', m['ALL']['best3'], 't', round(m['ALL']['t_drop_best3'], 3))
