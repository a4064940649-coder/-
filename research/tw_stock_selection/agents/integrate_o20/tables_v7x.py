#!/usr/bin/env python3
"""integrate_o20 V7X 表格輸出（只讀 results.json）。python3 -I -B tables_v7x.py"""
import json
from pathlib import Path
R = json.loads((Path(__file__).resolve().parent / 'results.json').read_text(encoding='utf-8')); X = R['RESULTS_V7X']
def f(x, n=2): return '—' if x is None else f'{x:.{n}f}'
print('jobs', X['n_jobs'], X['plans_changed'], 'rep max', max(X['reproduce_maxabs'].values()))
for cell in ('normal|100萬|std', 'stress|100萬|std', 'normal|300萬|std', 'normal|1000萬|std', 'normal|30萬|std', 'normal|30萬|wholelot', 'stress|30萬|std'):
    print('##', cell)
    for dv in ('base', 'M2'):
        for v in ('REAL_opt', 'REAL_p90', 'REAL_carry'):
            k = f'{dv}|{v}|{cell}'; T = X['table'][k]; Pp = T['periods']; C = X['vs_main'][k]
            print(f"| {dv} {v} | {f(T['CAGR_med'])}% | {f(T['exc_med'])} ({f(T['exc_p10'])}) | {f(T['t_0050_med'],3)} | {f(T['CAGR_Y19on_med'])}% ({f(T['exc_Y19on_med'])}, {f(T['t_Y19on_med'])}) | "
                  f"{f(Pp['S0612']['exc']['med'])}/{f(Pp['S1318']['exc']['med'])}/{f(Pp['S24on']['exc']['med'])} | {f(T['maxDD_med'],1)} | {f(C['ALL']['diff_pp']['med'],4)} ({f(C['ALL']['t']['med'])}) | {f(C['ALL']['corr']['med'],5)}/{f(C['daily_corr']['med'],5)} |")
print('## per-j diff M2-base')
for k, v in X['per_j_diff'].items():
    print(k, {f: (round(x['median'], 3), round(x['maxabs'], 3), x['n_nonzero']) for f, x in v.items() if f in ('exc', 't_0050', 'exc_Y19on')})
print('## periods M2 vs base opt 100萬')
for dv in ('base', 'M2'):
    Pp = X['table'][f'{dv}|REAL_opt|normal|100萬|std']['periods']
    print(dv, {p: (round(Pp[p]['exc']['med'], 2), round(Pp[p]['t']['med'], 2), round(Pp[p]['mdd']['med'], 1), round(Pp[p]['cagr']['med'], 2)) for p in Pp})
print('## blends')
for k, b in X['blends'].items():
    if '100萬|std' not in k and '30萬|std' not in k: continue
    s = b['stats']; vm = b['vs_main']
    print(f"| {k} | {f(s['CAGR_med'])}% | {f(s['exc_med'])} ({f(s['exc_p10'])}) | {f(s['t_0050_med'],3)} | {f(s['exc_Y19on_med'])} | {f(s['maxDD_med'],1)} | {f(vm['ALL']['diff_pp']['med'],4)} ({f(vm['ALL']['t']['med'])}) | {f(vm['ALL']['corr']['med'],5)} |")
cs = [b['vs_main']['ALL']['corr']['med'] for b in X['blends'].values()]; print('min blend corr', min(cs), len(cs))
