import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
import json
import numpy as np, pandas as pd
import rlib as R
S = R.main_strategy(); pl = R.plan(10); t0, t1, sd = S.bounds()
IDX = R.IDX
skip = {p[0] for p in pl if pd.Timestamp('2024-02-15') <= IDX[p[0]] <= pd.Timestamp('2025-09-23')}
ew = R.ew_nav(pl, t0, t1)
SEGS = {'ALL': ('2006-04-01', '2026-10-12', '2006-04-01', '2026-10-12'),
        'SEL_to0513': ('2006-04-01', '2019-05-13', '2006-04-01', '2019-04-30'),
        'SEL_to0430': ('2006-04-01', '2019-04-30', '2006-04-01', '2019-04-30'),
        'VAL': ('2019-05-01', '2026-10-12', '2019-05-01', '2026-10-12')}
out = {}
for cost in ('normal', 'stress'):
    base = R.sim(pl, t0, t1, cost=cost)
    for cont in ('rebal', 'hold'):
        for sk in ('exsrc3', 'all'):
            for lu in (False, True):
                nv = R.sim(pl, t0, t1, cost=cost, buy_px='close', cont=cont, skip_t=skip if sk == 'exsrc3' else None, lu_next=lu)
                key = f'{cost}|{cont}|{sk}|lu{int(lu)}'
                o = {}
                for k, (a, b, wa, wb) in SEGS.items():
                    s0 = R.seg(base, sd, a, b, ew, wa, wb); s1 = R.seg(nv, sd, a, b, ew, wa, wb)
                    o[k] = {'base': {x: s0[x] for x in ('CAGR', 'maxDD', 't_0050')}, 'p4b': {x: s1[x] for x in ('CAGR', 'maxDD', 't_0050')},
                            'paired': R.paired(nv, base, sd, wa, wb)}
                out[key] = o
                print(key, {k: (v['base']['CAGR'], v['p4b']['CAGR'], v['p4b']['maxDD'], v['base']['t_0050'], v['p4b']['t_0050'], v['paired']['mean_pct'], v['paired']['t']) for k, v in o.items()}, flush=True)
json.dump(out, open(R.HERE / 'p4b.json', 'w'), ensure_ascii=False, indent=1)
