#!/usr/bin/env python3
"""integrate_o20 紙上表格輸出（只讀 results.json）。python3 -I -B tables_paper.py"""
import json
from pathlib import Path
R = json.loads((Path(__file__).resolve().parent / 'results.json').read_text(encoding='utf-8')); P = R['RESULTS_PAPER']; S = P['stats']
def f(x, n=2): return '—' if x is None else f'{x:.{n}f}'
print('## main'); 
for c in ('normal', 'stress'):
    for tag in ('M0', 'M1', 'M2'):
        for k in ('C0', 'ALLPIT_REAL_opt', 'ALLPIT_REAL_p90', 'ALLPIT_REAL_carry'):
            key = f'{tag}|{k}|{c}'
            if key not in S: continue
            a = S[key]['ALL']; vs0 = S[key].get('vs_M0', {}).get('ALL', {})
            print(f"| {c} {tag} {k} | {f(a['ann'])}% / {f(a['ann_cal'])}% | {f(a['gex'])} | {f(a['t_vs_0050'],3)} | {f(a['mdd'],1)}/{f(a['mdd_daily'],1)} | {f(a['diff_vs_ref_pp'],4)} ({f(a['t_diff_vs_ref'])}) | {f(a['corr_ref'],4)} | {f(a['ann_diff_vs_ref'],3)} | {f(a['t_drop_best3'])} | vsM0 {f(vs0.get('diff_pp'),4)}({f(vs0.get('t'))}) | BE {f(S[key].get('breakeven_extra_oneway_pct_from_topic'),3)} |")
print('## segs')
for c in ('normal', 'stress'):
    for tag in ('M0', 'M1', 'M2'):
        for k in ('C0', 'ALLPIT_REAL_opt', 'ALLPIT_REAL_p90', 'ALLPIT_REAL_carry'):
            key = f'{tag}|{k}|{c}'
            if key not in S: continue
            s = S[key]
            print(f"| {c} {tag} {k} | " + ' | '.join(f"{f(s[p]['ann'])}/{f(s[p]['gex'])}({f(s[p]['t_vs_0050'])}) mdd{f(s[p]['mdd'],1)} d{f(s[p]['diff_vs_ref_pp'],3)}({f(s[p]['t_diff_vs_ref'])}) r{f(s[p]['corr_ref'],4)}" for p in ('SEG1', 'SEG2', 'SEG3', 'HOLD', 'SEL', 'VAL')) + ' |')
print('## yearly diff M2 vs M1 (normal)')
for k in ('ALLPIT_REAL_opt', 'ALLPIT_REAL_p90', 'ALLPIT_REAL_carry'):
    y = S[f'M2|{k}|normal']['yearly_abs_diff_vs_ref_pp']; print(k, {a: round(b, 2) for a, b in y.items() if abs(b) > 1e-9})
print('## blends')
for key, s in P['blends'].items():
    a = s['ALL']; print(f"| {key} | {f(a['ann'])}% / {f(a['ann_cal'])}% | {f(a['gex'])} | {f(a['t_vs_0050'],3)} | {f(a['mdd'],1)} | {f(s['SEG3']['gex'])} | {f(a['diff_vs_ref_pp'],4)} ({f(a['t_diff_vs_ref'])}) | {f(a['corr_ref'],5)} |")
print('## sampling', {k: [round(v['gex'], 2), round(v['lo'], 2), round(v['hi'], 2)] for k, v in P['sampling_interval'].items()})
