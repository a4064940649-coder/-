#!/usr/bin/env python3
"""把 robust、tpex、explore、data_freeze、iterations 併進 results.json：python3 -I -B merge_results.py"""
import json, hashlib
from pathlib import Path
HERE = Path(__file__).resolve().parent
R = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))
for k, f in (('ROBUST', 'results_robust.json'), ('TPEX', 'tpex_scan.json'), ('explore', 'explore.json'), ('data_freeze', 'data_freeze.json'), ('iterations', 'iterations.json'),
             ('delist_prices_meta', 'delist_prices_meta.json'), ('disposal_hist_meta', 'disposal_hist_meta.json')):
    R[k] = json.loads((HERE / f).read_text(encoding='utf-8'))
R['navs_files'] = {f: hashlib.sha256((HERE / f).read_bytes()).hexdigest() for f in ('navs_P0.parquet', 'navs_P1.parquet')}
R['V3_nav_column'] = {'file': 'navs_P1.parquet', 'column': R['RESULTS']['V3_key'], 'EW_column': R['RESULTS']['V3_key'].rsplit('|', 2)[0] + '|EW'}
(HERE / 'results.json').write_text(json.dumps(R, ensure_ascii=False, indent=1, default=float), encoding='utf-8')
print(R['V3_nav_column'], R['navs_files'])
