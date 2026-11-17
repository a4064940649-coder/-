#!/usr/bin/env python3
"""S54 事後描述（未事前登記，不改判定）：CASH 版濾網格 vs S47 的『空位常態留現金、不設濾網』（S70_D10_CASH，同引擎、同 20 條路徑），
用來分開『留現金』與『擇時』的效果。只讀兩題的 results.json，不跑回測。輸出 results_posthoc.json。用法：python3 -I -B posthoc.py"""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
r = json.loads((HERE / 'results.json').read_text())
s47 = json.loads((HERE.parent / 's47_allocation_family' / 'results.json').read_text())['cells']
ref = s47['S70_D10_CASH']['paths']; base47 = s47['S70_D10']['paths']
keys = r['path_keys']
assert all(base47[k]['since2019'] == r['cells']['NONE']['paths'][k]['since2019'] and base47[k]['maxdd'] == r['cells']['NONE']['paths'][k]['maxdd_full'] for k in keys), 'S47 現行路徑與本題不同'
out = dict(note='事後描述，未事前登記，不影響判定', ref='S47 S70_D10_CASH（空位常態留現金）',
           ref_path_median=dict(since2019=float(np.median([ref[k]['since2019'] for k in keys])), maxdd=float(np.median([ref[k]['maxdd'] for k in keys])),
                                calmar=float(np.median([ref[k]['calmar'] for k in keys]))),
           ref_vs_current=dict(since2019_pp=float(np.median([(ref[k]['since2019'] - base47[k]['since2019']) * 100 for k in keys])),
                               maxdd_pp=float(np.median([(ref[k]['maxdd'] - base47[k]['maxdd']) * 100 for k in keys])),
                               calmar=float(np.median([ref[k]['calmar'] - base47[k]['calmar'] for k in keys]))), cells={})
for c in ('MA60_CASH', 'MA120_CASH', 'MA200_CASH', 'R60_CASH'):
    p = r['cells'][c]['paths']
    dR = np.array([(p[k]['since2019'] - ref[k]['since2019']) * 100 for k in keys])
    dD = np.array([(p[k]['maxdd_full'] - ref[k]['maxdd']) * 100 for k in keys])
    dC = np.array([p[k]['calmar'] - ref[k]['calmar'] for k in keys])
    v = r['cells'][c]['vs_current']
    out['cells'][c] = dict(vs_always_cash=dict(since2019_pp=float(np.median(dR)), since2019_n_pos=int((dR > 0).sum()),
                                               maxdd_pp=float(np.median(dD)), maxdd_n_shallower=int((dD > 0).sum()),
                                               calmar=float(np.median(dC)), calmar_n_pos=int((dC > 0).sum())),
                           cost_per_dd_pp=float(-v['since2019_median_pp'] / v['maxdd_full_median_pp']))
out['ref_cost_per_dd_pp'] = float(-out['ref_vs_current']['since2019_pp'] / out['ref_vs_current']['maxdd_pp'])
(HERE / 'results_posthoc.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
print(json.dumps(out, ensure_ascii=False, indent=1))
