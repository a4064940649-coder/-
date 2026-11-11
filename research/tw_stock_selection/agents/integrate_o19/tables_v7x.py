#!/usr/bin/env python3
"""integrate_o19 V7X 表格輸出（只讀 results.json，不寫檔）。python3 -I -B tables_v7x.py"""
import json
from pathlib import Path
R = json.loads((Path(__file__).resolve().parent / 'results.json').read_text(encoding='utf-8'))['RESULTS_V7X']
T, C, B = R['table'], R['vs_main'], R['blends']; F = lambda x, n=2: '—' if x is None else f'{x:.{n}f}'
print('utc', R['utc'], 'runtime', R['runtime_s'], 'jobs', R['n_jobs'], R['plans_changed'])
print('reproduce', {k: round(v, 6) for k, v in R['reproduce_maxabs'].items()})
VS = ['REAL_opt', 'REAL_p90', 'REAL_carry']; DV = ['strict', 'posthoc', 'I3a', 'I3b', 'I3c']
CELLS = ['normal|100萬|std', 'stress|100萬|std', 'normal|300萬|std', 'normal|1000萬|std', 'normal|30萬|std', 'normal|30萬|wholelot', 'stress|30萬|std']
def per(t, k, f='exc'): return t['periods'][k][f]['med']
print('\n## per cell: CAGR | exc (p10) | t | Y19 CAGR (exc, t) | S0612/S1318/S24on exc | mdd | vs strict diff pp(t) corr dailycorr')
for cell in CELLS:
    for dv in DV:
        for v in VS:
            t = T[f'{dv}|{v}|{cell}']; c = C[f'{dv}|{v}|{cell}']
            print(f"|{cell}|{dv}|{v}|{F(t['CAGR_med'])}%|{F(t['exc_med'])}（{F(t['exc_p10'])}）|{F(t['t_0050_med'],3)}|{F(t['CAGR_Y19on_med'])}%（{F(t['exc_Y19on_med'])}，{F(t['t_Y19on_med'])}）|{F(per(t,'S0612'))}／{F(per(t,'S1318'))}／{F(per(t,'S24on'))}|{F(t['maxDD_med'],1)}|{F(c['ALL']['diff_pp']['med'],4)}（{F(c['ALL']['t']['med'])}）|{F(c['ALL']['corr']['med'],5)}|{F(c['daily_corr']['med'],5)}|")
    w = {f: min(T[f'strict|{v}|{cell}'][f] for v in VS) for f in ('exc_med', 'exc_p10', 't_0050_med', 'CAGR_Y19on_med', 'exc_Y19on_med')}
    print('WORST strict', cell, {k: round(x, 3) for k, x in w.items()})
    for dv in DV[1:]:
        wd = {f: round(min(T[f'{dv}|{v}|{cell}'][f] for v in VS) - w[f], 3) for f in w}
        print('  delta', dv, wd)
print('\n## per-period t (strict, 100萬一般)')
for v in VS:
    t = T[f'strict|{v}|normal|100萬|std']; print(v, {k: (F(x['exc']['med']), F(x['t']['med']), F(x['mdd']['med'], 1), F(x['cagr']['med'])) for k, x in t['periods'].items()})
print('\n## blends: CAGR | exc (p10) | t | Y19 exc | mdd | diff vs main pp (t) | corr')
for k, s in B.items():
    if '100萬|std' not in k and 'stress|100萬' not in k: continue
    st = s['stats']; vm = s['vs_main']['ALL']
    print(f"|{k}|{F(st['CAGR_med'])}%|{F(st['exc_med'])}（{F(st['exc_p10'])}）|{F(st['t_0050_med'],3)}|{F(st['exc_Y19on_med'])}|{F(st['maxDD_med'],1)}|{F(vm['diff_pp']['med'],4)}（{F(vm['t']['med'])}）|{F(vm['corr']['med'],5)}|")
print('min blend corr', min(s['vs_main']['ALL']['corr']['med'] for s in B.values() if s['vs_main']['ALL']['corr']['med'] is not None))
