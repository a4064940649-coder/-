#!/usr/bin/env python3
"""把 results_oos.json、results_gate.json、results_live_check.json、explore／iterations 摘要併進 results.json（PREREG 不動）。"""
import json
from pathlib import Path
HERE = Path(__file__).resolve().parent
R = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))
oos = json.loads((HERE / 'results_oos.json').read_text(encoding='utf-8'))
R['RESULTS'] = oos
R['GATE'] = json.loads((HERE / 'results_gate.json').read_text(encoding='utf-8'))
R['LIVE_CHECK'] = json.loads((HERE / 'results_live_check.json').read_text(encoding='utf-8'))
R['ITERATIONS_SUMMARY'] = {k: v['normal'] for k, v in json.loads((HERE / 'iterations.json').read_text(encoding='utf-8'))['round1']['results'].items()}
R['VERDICT'] = {k: v['verdict'] for k, v in oos['judgement'].items()}
(HERE / 'results.json').write_text(json.dumps(R, ensure_ascii=False, indent=1), encoding='utf-8')
print(R['VERDICT'])
