#!/usr/bin/env python3
"""第 3 步補：選定版本 v3 的鄰域（槽數 10/30、持有 40）— 只用來檢查「不翻號」，不參與選擇；併入 iterations.json 與 SPA 家族。"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from s3_iterate import VARIANTS, run_variant, summ, E0, HERE
it = json.load(open(HERE / 'iterations.json'))
base = dict(VARIANTS['v3'])
nb = {}
for name, ch in {'v3_S10': dict(S=10), 'v3_S30': dict(S=30), 'v3_H40': dict(H=40)}.items():
    v = dict(base, **ch, note='v3 鄰域：' + str(ch))
    res, _ = run_variant(E0, v)
    s = summ(res); nb[name] = dict(spec=v, summary=s, per_j={int(k): r for k, r in res.items()})
    print(name, s)
it['neighbors_of_choice'] = nb
it['choice'] = 'v3'
it['choice_check'] = {'neighbors_sign_same_as_choice': all(n['summary']['ex_med'] > 0 for n in nb.values()),
                      'n_variants_selection': len(it['variants']), 'n_neighbor_runs': len(nb)}
json.dump(it, open(HERE / 'iterations.json', 'w'), ensure_ascii=False, indent=1, default=float)
print(it['choice_check'])
