#!/usr/bin/env python3
"""合併 results_t1..t5.json → results.json（精簡版：只留摘要欄位）。"""
import json
from pathlib import Path
D = Path(__file__).resolve().parent
out = {}
for i in range(1, 6):
    r = json.loads((D / f'results_t{i}.json').read_text(encoding='utf-8'))
    if i == 1: r = {k: r[k] for k in ['summary', 'spreads', 'selected_in_sample', 'selected_validation', 'latest_picks']}
    if i == 2: r = {k: v for k, v in r.items() if k not in ()}
    if i == 3: r = {k: v for k, v in r.items()}; r['trading'] = {k: {kk: vv for kk, vv in v.items() if kk != 'validation_full'} for k, v in r['trading'].items()}
    if i == 4: r = {k: r[k] for k in ['summary', 'spread', 'selected']} | {'sub_2021_2026': {k: v['sub_2021_2026'] for k, v in r['configs'].items()}}
    out[f't{i}'] = r
(D / 'results.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
print('ok', (D / 'results.json').stat().st_size)
