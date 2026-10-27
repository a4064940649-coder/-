#!/usr/bin/env python3
"""資料關卡用：列出「持有中永久停止交易」的部位（凍結事件），不計算任何報酬統計。
python3 -I -B probe_dead.py [patch]
輸出 probe_dead[_patch].json：每個版本（ALL/EXFIN × n10/n30/EW）的凍結事件、其中永久停止交易者的代碼與最後交易日。"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import o3_core as O
V, K = O.V, O.K
PATCH = len(sys.argv) > 1 and sys.argv[1] == 'patch'
if PATCH: print('patched cells', O.apply_patch(require_freeze=False))
YL, Y5, _ = V.load_revenue()
lt = O.last_trade_idx()
out = {}
for univ in ('ALL', 'EXFIN'):
    S = V.V2('gapfix', YL, Y5, exfin=(univ == 'EXFIN'))
    t0, t1, sd = S.bounds()
    for n in (10, 30, 'EW'):
        pl = S.plan(10 if n == 'EW' else n, 'official')
        plan = [(p[0], p[2]) for p in pl] if n == 'EW' else [(p[0], p[1]) for p in pl]
        nav, lg = O.simulate(plan, 0, 0, t0, t1)
        dead = {}
        for d, codes, sh in lg['frozen']:
            t = K.IDX.get_loc(pd.Timestamp(d))
            for c in codes:
                j = K.CI[c]
                if lt[j] < t: dead.setdefault(c, {'last_trade': str(K.IDX[lt[j]].date()), 'first_frozen': d, 'n_frozen_rebal': 0})['n_frozen_rebal'] += 1
        out[f'{univ}|{n}'] = {'n_frozen_events': len(lg['frozen']), 'max_frozen_share': max([x[2] for x in lg['frozen']], default=0),
                              'n_dead_codes': len(dead), 'dead': dead}
        print(univ, n, len(lg['frozen']), round(out[f'{univ}|{n}']['max_frozen_share'], 3), len(dead), sorted(dead)[:60], flush=True)
(Path(__file__).resolve().parent / f'probe_dead{"_patch" if PATCH else ""}.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
