#!/usr/bin/env python3
"""合併 results_rules / results_composite / results_main_filter → results.json，並輸出逐年表（全期 2005–2026，含 0050、上市等權）。"""
import sys, json, pickle
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import tlib as L

D = L.load()
r1 = json.loads((L.OUT / 'results_rules.json').read_text())
r2 = json.loads((L.OUT / 'results_composite.json').read_text())
r3 = json.loads((L.OUT / 'results_main_filter.json').read_text())
with open(L.PKL, 'rb') as f:
    daily = pickle.load(f)['daily']
m = D['idx'] >= pd.Timestamp('2005-01-01')
yr = {k: L.yearly((pf['gross'] - pf['cost'])[m]) for k, pf in daily.items()}
yr['0050'] = L.yearly(D['r0050'][m]); yr['EW_pool'] = L.yearly(D['r_ew'][m])
out = {'rules': r1['rules'], 'tables_rules': r1['tables'], 'candidates_pass_S': r1['candidates_pass_S'],
       'composite': r2, 'main_strategy_filters': r3, 'yearly_net_pct_full_2005_2026': yr}
L.save('results.json', out)
print(pd.DataFrame(yr).to_string())
