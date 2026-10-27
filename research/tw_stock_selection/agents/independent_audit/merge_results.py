#!/usr/bin/env python3
"""合併各結果檔成 results.json。"""
import json
from pathlib import Path
H = Path(__file__).resolve().parent
out = {}
for f in ['results_main.json', 'results_checks.json', 'results_tiebreak.json', 'results_oldvalue.json', 'results_extra.json', 'results_cmp_ra.json',
          'results_lenient_diag.json', 'results_tcheck.json', 'results_handcalc.json', 'results_ra_oldvalue.json']:
    p = H / f
    if p.exists(): out[f[:-5]] = json.loads(p.read_text(encoding='utf-8'))
(H / 'results.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
print(list(out))
