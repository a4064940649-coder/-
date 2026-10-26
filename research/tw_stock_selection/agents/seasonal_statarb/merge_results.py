#!/usr/bin/env python3
"""把 s1～s5 的 results_sN.json 合併成 results.json（執行順序：s1..s5 → merge_results.py）。"""
import json
from pathlib import Path
D = Path(__file__).resolve().parent
names = {'s1': '1_stock_same_month_seasonality', 's2': '2_industry_reversal_longonly', 's3': '3_pairs_longonly',
         's4': '4_market_calendar_0050', 's5': '5_52w_high_by_month'}
out = {v: json.loads((D / f'results_{k}.json').read_text(encoding='utf-8')) for k, v in names.items()}
(D / 'results.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
for k in names: (D / f'results_{k}.json').unlink()
print('ok')
