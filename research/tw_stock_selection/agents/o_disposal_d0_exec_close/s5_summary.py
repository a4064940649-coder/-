#!/usr/bin/env python3
"""把 results.json 的 RESULTS 整理成精簡摘要（只讀，不重算）。用法：python3 -I -B s5_summary.py"""
import json
from pathlib import Path
R = json.loads((Path(__file__).resolve().parent / 'results.json').read_text())['RESULTS']
pr = lambda *a: print(*a)
pr('DECISION', json.dumps(R['DECISION'], ensure_ascii=False))
for k in ('100萬|normal', '100萬|stress', '30萬|normal', '300萬|normal', '1000萬|normal', '30萬|stress', '300萬|stress', '100萬|normal|vs0050', '100萬|normal|gross_ideal'):
    t = R['TRADES'][k]
    pr('TRADE', k, {p: (v['median_of_means_pct'], v['p10_of_means_pct'], v['median_t_month'], v['n_median']) for p, v in t.items()})
pr('C1xCL by period', {p: v['by_timing']['C1xCL'] for p, v in R['TRADES']['100萬|normal'].items() if p in ('FULL', 'VAL', 'SEL', 'NEWREG')})
for tag, s in R['SAT'].items():
    pr('SAT', tag, {p: (s[p]['median_t'], s[p]['p10_t'], s[p]['median_mean_pp'], s[p]['median_series'].get('t')) for p in ('SEL', 'VAL', 'FULL', '2019+', 'P1_2006_2012', 'P2_2013_2018', 'P3_2019_2026')}, s['ledger'])
pr('VAL by timing', R['SAT']['MAIN']['VAL']['by_timing_t'])
pr('P2', json.dumps({k: (v['FULL']['o9_V7X'].get('mean_pct'), v['FULL']['exec_C1xCL'].get('mean_pct'), v['FULL']['reduction_pp'], v['VAL']['reduction_pp']) for k, v in R['P2'].items() if isinstance(v, dict)}, ensure_ascii=False))
pr('PRED', json.dumps(R['PRED'], ensure_ascii=False, default=str))
pr('COMPARE', json.dumps(R['COMPARE'], ensure_ascii=False, default=str))
pr('G', json.dumps(R['G'], ensure_ascii=False, default=str)[:6000])
pr('ROBUST', json.dumps(R['ROBUST'], ensure_ascii=False, default=str)[:3000])
pr('n_variants', R['n_variants_this_topic'])
