#!/usr/bin/env python3
"""第 2 步（迭代）：只用挑選期 2021-10～2023，只調執行面，6 個版本全部報告 → iterations.json
選版準則（寫在跑之前，存進 iterations.json 的 criterion）：
  以 v1（PLAN 原設定）為預設；其他版本只有在「挑選期 100 萬 FIX 衛星對 0050 月超額 t 比 v1 高 ≥ 0.3，
  且逐筆可執行淨超額（每槽 10 萬）平均 ≥ v1，且 1,000 萬 GROW 對 0050 月超額 ≥ v1」三項同時成立時才取代 v1；
  多個符合時取 t 最高者。"""
import sys, json, datetime
sys.dont_write_bytecode = True
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import core as K

SELP = 'SEL_2021-10_2023'; A, B = K.PER[SELP]
CRIT = __doc__.split('選版準則')[1].strip()
VERS = {
    'v1_cap20_shrink_defer': dict(build={}, cap=0.2, min_fill=0.0, defer=True),
    'v2_cap10': dict(build={}, cap=0.1, min_fill=0.0, defer=True),
    'v3_cap30': dict(build={}, cap=0.3, min_fill=0.0, defer=True),
    'v4_first_only': dict(build={'first_only': True}, cap=0.2, min_fill=0.0, defer=True),
    'v5_skip_if_fill_lt50': dict(build={}, cap=0.2, min_fill=0.5, defer=True),
    'v6_no_exit_defer': dict(build={}, cap=0.2, min_fill=0.0, defer=False),
}
U = K.universe()
out = {'time_utc': datetime.datetime.utcnow().isoformat(timespec='seconds') + 'Z', 'criterion': CRIT, 'period': [A, B], 'versions': {}}
for name, v in VERS.items():
    df, st = K.build(v['build'], U, SELP)
    X = K.exec_cost_event(df, 1e5, cap=v['cap'], defer=v['defer'])
    r = {'cfg': {k: vv for k, vv in v.items()}, 'n': int(len(df)), 'event_flat': K.S(df), 'event_exec_slot10萬': K.S(X, 'net_x'),
         'event_exec_slot100萬': K.S(K.exec_cost_event(df, 1e6, cap=v['cap'], defer=v['defer']), 'net_x'), 'sleeve': {}}
    for tier in (1e6, 3e6, 1e7):
        for mode in ('FIX', 'GROW'):
            nav, s = K.sleeve(df, tier, mode, cap=v['cap'], t_start=A, t_end=B, min_fill=v['min_fill'], defer=v['defer'])
            r['sleeve'][f'{int(tier/1e4)}萬|{mode}'] = dict(K.vs0050(nav), **{k: s[k] for k in ('entered', 'capped', 'zero_cap', 'mean_invested')})
    out['versions'][name] = r
    print(name, r['event_flat'].get('mean_pct'), r['event_exec_slot10萬'].get('mean_pct'), r['sleeve']['100萬|FIX'].get('t_vs_0050'),
          r['sleeve']['1000萬|GROW'].get('ex_month_mean_pp'))
b = out['versions']['v1_cap20_shrink_defer']
cands = []
for name, r in out['versions'].items():
    if name.startswith('v1'): continue
    ok = ((r['sleeve']['100萬|FIX']['t_vs_0050'] or -9) >= (b['sleeve']['100萬|FIX']['t_vs_0050'] or -9) + 0.3
          and r['event_exec_slot10萬']['mean_pct'] >= b['event_exec_slot10萬']['mean_pct']
          and r['sleeve']['1000萬|GROW']['ex_month_mean_pp'] >= b['sleeve']['1000萬|GROW']['ex_month_mean_pp'])
    if ok: cands.append((r['sleeve']['100萬|FIX']['t_vs_0050'], name))
out['selected'] = max(cands)[1] if cands else 'v1_cap20_shrink_defer'
out['n_versions'] = len(VERS)
(HERE / 'iterations.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
print('selected', out['selected'])
