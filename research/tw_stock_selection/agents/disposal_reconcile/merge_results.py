#!/usr/bin/env python3
"""合併 results_recon.json、results_tpex.json → results.json。"""
import json
from pathlib import Path
H = Path(__file__).resolve().parent
out = {'_files': {'recon': 'recon.py → results_recon.json', 'tpex': 'tpex_combo.py → results_tpex.json', 'lib': 'rlib.py'},
       'recon': json.loads((H / 'results_recon.json').read_text(encoding='utf-8')),
       'tpex_and_portfolio': json.loads((H / 'results_tpex.json').read_text(encoding='utf-8'))}
(H / 'results.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
