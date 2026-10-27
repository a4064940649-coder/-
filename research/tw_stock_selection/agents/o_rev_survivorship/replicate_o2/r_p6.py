#!/usr/bin/env python3
"""複製員 P6：用 o_capacity_tsmc_gap 的凍結核心（只讀 import ocore），把長期營收換成我自建的版本，重算
(1) 名次歸因 n10 驗證期「中型減大型」（101～300 減 1～100，對同池等權、扣成本）；(2) 台積電 30% 驗證期對 0050 t。"""
import sys, json
from pathlib import Path
H = Path(__file__).resolve().parent
sys.path.insert(0, str(H.parent.parent / 'o_capacity_tsmc_gap'))
import ocore as O
sys.path.pop(0)
import numpy as np, pandas as pd
K = O.K
out = {}
for ver in sys.argv[1:] or ['ORIG', 'R_DEL']:
    Y = pd.read_parquet(H / f'my_yoy_{ver}.parquet'); Y.index = pd.PeriodIndex(Y.index, freq='M')
    O.YL = Y
    S = O.V2R(1, 300)
    t0, t1, sd = S.bounds()
    pl = S.plan(10, 'official')
    ew, _ = K.simulate([(p[0], p[2]) for p in pl], 0, 0, t0, t1)
    df = O.event_table(S, pl)
    P = O.periods(sd)
    a, b = P['VAL'][0], P['VAL'][1]
    sa, sb = P['SEL'][0], P['SEL'][1]
    r = {'mml_VAL': O.mml_stats(O.mid_minus_large(df, 'ex_pool', a, b)), 'mml_SEL': O.mml_stats(O.mid_minus_large(df, 'ex_pool', sa, sb))}
    for w in (0.0, 0.3):
        nav, _ = O.simulate_w(O.tsmc_plan(pl, w), *O.NORMAL, t0, t1)
        r[f'w{int(w*100)}'] = {k: {kk: v[kk] for kk in ('CAGR', 't_0050', 'maxDD')} for k, v in ((k, O.segstat(nav, ew, sd, *P[k])) for k in ('SEL', 'VAL', 'HOLD', 'ALL'))}
    out[ver] = r
    print(ver, json.dumps(r, ensure_ascii=False), flush=True)
(H / 'my_p6.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
