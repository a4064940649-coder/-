# -*- coding: utf-8 -*-
"""Append the verdict block to results.json (does not touch PREREG / OOS)."""
import json, hashlib
from pathlib import Path
HERE = Path(__file__).resolve().parent
r = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))
assert hashlib.sha256(json.dumps(r['PREREG'], ensure_ascii=False, sort_keys=True).encode()).hexdigest() == r['PREREG_sha256']
o = r['OOS']['strategy']
r['VERDICT'] = dict(
    verdict='不可用', mechanism='部分支持', data_tier='第 1 層（2007～2026 CB 完整；現增有 2009Q4、2010H2、2013-12、2016、2025-08 以後缺口）',
    event_gate_selection_passed=r['OOS']['gate_event_level_selection']['passed'], strategy_gates_n10=r['OOS']['gates_n10'],
    n10=dict(sel_paired_t=o['n10']['normal|SEL']['paired']['paired_t'], val_paired_mean_pp=o['n10']['normal|VAL']['paired']['paired_mean_pp'],
             val_paired_t=o['n10']['normal|VAL']['paired']['paired_t'], all_dCAGR_pp=o['n10']['normal|ALL']['dCAGR_pp']),
    n30=dict(sel_paired_t=o['n30']['normal|SEL']['paired']['paired_t'], val_paired_mean_pp=o['n30']['normal|VAL']['paired']['paired_mean_pp'],
             val_paired_t=o['n30']['normal|VAL']['paired']['paired_t'], all_dCAGR_pp=o['n30']['normal|ALL']['dCAGR_pp']),
    versions_tried=8, sop_change='無')
(HERE / 'results.json').write_text(json.dumps(r, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
print(json.dumps(r['VERDICT'], ensure_ascii=False, indent=1, default=str))
