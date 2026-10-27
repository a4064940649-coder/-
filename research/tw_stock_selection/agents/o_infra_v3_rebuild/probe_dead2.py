#!/usr/bin/env python3
"""資料關卡用（不計算報酬統計）：R_DEL 營收＋價格修補＋缺營收界限（pfix 0.5／1.0）下，哪些「永久停止交易」的股票曾在持有中凍結。
python3 -I -B probe_dead2.py → probe_dead2.json（決定 merger_consideration.csv 必須涵蓋的事件）"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import o3_core as O
V, K = O.V, O.K
O.apply_patch(require_freeze=False)
YR = O.load_rdel(); _, Y5, _ = V.load_revenue()
GC = O.delist_codes()
FOREIGN = set(json.loads((O.ORS / 'data_freeze.json').read_text(encoding='utf-8'))['foreign_codes'])
lt = O.last_trade_idx()
out = {}; allc = {}
for pf in (None, 0.5, 1.0):
    for univ in ('ALL', 'EXFIN'):
        for dm in ('v2', 'none'):
            O.set_disposal(dm)
            S = O.V3('gapfix', YR, Y5, exfin=(univ == 'EXFIN'), gapcodes=GC, pfix=pf, foreign=FOREIGN)
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
                        if lt[j] < t:
                            x = dead.setdefault(c, {'last_trade': str(K.IDX[lt[j]].date()), 'first_frozen': d, 'n': 0, 'max_share': 0}); x['n'] += 1; x['max_share'] = max(x['max_share'], sh)
                            allc.setdefault(c, set()).add(f'{pf}|{univ}|{dm}|{n}')
                key = f'pfix{pf}|{univ}|disp_{dm}|{n}'
                out[key] = {'n_frozen_events': len(lg['frozen']), 'n_dead_events': sum(v['n'] for v in dead.values()), 'dead': dead}
                print(key, len(lg['frozen']), out[key]['n_dead_events'], sorted(dead), flush=True)
out['_all_dead_codes'] = {c: sorted(v) for c, v in sorted(allc.items())}
(Path(__file__).resolve().parent / 'probe_dead2.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
print(sorted(allc))
