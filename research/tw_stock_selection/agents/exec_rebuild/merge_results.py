#!/usr/bin/env python3
"""合併 results_main/disposal/alloc → results.json，並加上事前決策規則的判定。"""
import json
from pathlib import Path
H = Path(__file__).resolve().parent
m = json.loads((H / 'results_main.json').read_text()); d = json.loads((H / 'results_disposal.json').read_text()); a = json.loads((H / 'results_alloc.json').read_text())
dec = {}
for n in (30, 10):
    for slip in ('0.001', '0.003', 'stress'):
        w = m[f'M2_n{n}_slip{slip}']['windows']['valid_ex2026_2024-01_2025-12']
        trig = (w['t_vs_0050_m'] < 1) and (w['t_vs_EW_m'] < 2)
        dec[f'M2_n{n}_slip{slip}'] = {'t_vs_0050_m': w['t_vs_0050_m'], 't_vs_EW_m': w['t_vs_EW_m'], 'downgrade_to_watch': bool(trig)}
out = {'design': '見 FINDINGS（事前固定）；程式 exlib.py、dlib.py、run_main.py、run_disposal.py、run_alloc.py',
       'decision_rule_M2_valid_ex2026': dec, 'main': m, 'disposal': d, 'alloc': a}
(H / 'results.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
print(json.dumps(dec, ensure_ascii=False, indent=1))
