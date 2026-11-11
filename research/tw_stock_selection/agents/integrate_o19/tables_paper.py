#!/usr/bin/env python3
"""integrate_o19 紙上表格輸出（只讀 results.json，不寫檔）。python3 -I -B tables_paper.py"""
import json
from pathlib import Path
R = json.loads((Path(__file__).resolve().parent / 'results.json').read_text(encoding='utf-8'))['RESULTS_PAPER']
S = R['stats']; F = lambda x, n=2: '—' if x is None else f'{x:.{n}f}'
DV = ['official', 'strict', 'posthoc', 'I3a', 'I3b', 'I3c']; KS = ['C0', 'ALLPIT_REAL_opt', 'ALLPIT_REAL_p90', 'ALLPIT_REAL_carry']
for c in ('normal', 'stress'):
    print(f'\n## paper {c}: ann | gex | t | mdd win/daily | diff vs same-fill official pp (t) | anndiff | corr | corr0050 | tdrop3 | SEG1 SEG2 SEG3 HOLD SEL VAL gex (t)')
    for k in KS:
        for dv in DV:
            s = S[f'{dv}|{k}|{c}']; a = s['ALL']
            print(f"|{dv}|{k}|{F(a['ann'])}%|{F(a['gex'])}|{F(a['t_vs_0050'],3)}|{F(a['mdd'],1)}／{F(a.get('mdd_daily'),1)}|{F(a['diff_vs_ref_pp'],4)}（{F(a['t_diff_vs_ref'])}）|{F(a['ann_diff_vs_ref'])}|{F(a['corr_ref'],4)}|{F(a['corr_0050'],3)}|{F(a['t_drop_best3'])}|"
                  + '|'.join(f"{F(s[p]['gex'])}（{F(s[p]['t_vs_0050'])}）" for p in ('SEG1', 'SEG2', 'SEG3', 'HOLD', 'SEL', 'VAL')) + '|')
    print('vs_C0 (official|*)', {k: {p: (round(v['diff_pp'], 3), None if v['t'] is None else round(v['t'], 2), None if v['corr'] is None else round(v['corr'], 4)) for p, v in S[f'official|{k}|{c}']['vs_C0'].items() if p in ('ALL',)} for k in KS})
print('\n## yearly abs diff vs official (nonzero)')
for k in KS:
    for dv in ('I3a', 'I3b', 'I3c', 'posthoc'):
        y = S[f'{dv}|{k}|normal']['yearly_abs_diff_vs_ref_pp']; nz = {a: round(b, 2) for a, b in y.items() if abs(b) > 1e-9}
        if nz: print(dv, k, nz, 'stress', {a: round(b, 2) for a, b in S[f'{dv}|{k}|stress']['yearly_abs_diff_vs_ref_pp'].items() if abs(b) > 1e-9})
print('\n## changed slots'); 
for kk, v in R['changed_slots'].items():
    if v: print(kk, v)
print('\n## blends: ann | gex | t | mdd | diff vs main pp (t) | corr | SEG1/2/3 gex | HOLD')
for kk, s in R['blends'].items():
    a = s['ALL']; print(f"|{kk}|{F(a['ann'])}%|{F(a['gex'])}|{F(a['t_vs_0050'],3)}|{F(a['mdd'],1)}|{F(a['diff_vs_ref_pp'],4)}（{F(a['t_diff_vs_ref'])}）|{F(a['corr_ref'],5)}|" + '／'.join(F(s[p]['gex']) for p in ('SEG1', 'SEG2', 'SEG3')) + f"|{F(s['HOLD']['gex'])}|")
print('\n## sampling', {k: (round(v['gex'], 2), round(v['lo'], 2), round(v['hi'], 2)) for k, v in R['sampling_interval'].items() if k.startswith(('strict', 'I3b', 'I3c'))})
