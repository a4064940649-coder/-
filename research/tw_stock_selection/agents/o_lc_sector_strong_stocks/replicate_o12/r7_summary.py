#!/usr/bin/env python3
"""Q1/Q2/Q3/挑選期格點彙總（13 個 j 中位數）。"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from r3_stats import sl, st, med, P
OUT = Path(__file__).resolve().parent / 'out'
Q = {j: pd.read_parquet(OUT / f'q_J{j}.parquet') for j in range(13)}
def ser_stat(get, pers=('SEL', 'VAL', 'POST19', 'PIT')):
    out = {}
    for p in pers:
        rows = []
        for j in range(13):
            x = get(j, p).dropna().to_numpy(); rows.append(dict(mean_pp=100 * x.mean(), t=x.mean() / x.std(ddof=1) * np.sqrt(len(x)), n=len(x)))
        out[p] = med(rows)
    return out
R = {}
for n in (1, 2, 3):
    R[f'Q2_n{n}'] = ser_stat(lambda j, p, n=n: sl(Q[j], p)[f'Q2_n{n}'])
    R[f'Q2eq_n{n}'] = ser_stat(lambda j, p, n=n: sl(Q[j], p)[f'Q2eq_n{n}'])
R['Q1'] = ser_stat(lambda j, p: sl(Q[j], p)['Q1'])
d = lambda a, b, c='normal': (lambda j, p: sl(P[j], p)[f'{a}|{c}'] - sl(P[j], p)[f'{b}|{c}'])
R['Q3_ind_effect IND_ONLY-B'] = ser_stat(d('IND_ONLY_PURE', 'B'))
R['Q3_within IND-IND_ONLY'] = ser_stat(d('IND_n2_cap_c15_PURE', 'IND_ONLY_PURE'))
R['Q3_CTRL-B'] = ser_stat(d('CTRL_n2_cap_c15_PURE', 'B'))
R['Q3_IND-CTRL PURE'] = ser_stat(d('IND_n2_cap_c15_PURE', 'CTRL_n2_cap_c15_PURE'))
R['Q3_IND-CTRL SAT'] = ser_stat(d('IND_n2_cap_c15_SAT', 'CTRL_n2_cap_c15_SAT'))
# 挑選期格點（v1、T1）：IR 對 0050（紙上一般成本）
g = {}
for kind in ('IND', 'CTRL'):
    for n in (1, 2, 3):
        for wt in ('cap', 'eq'):
            for cap in (15, 25):
                for car in ('PURE', 'SAT'):
                    c = f'{kind}_n{n}_{wt}_c{cap}_{car}'
                    g[c] = float(np.median([st(sl(P[j], 'SEL')[c + '|normal'].to_numpy(), sl(P[j], 'SEL')['0050'].to_numpy())['IR'] for j in range(13)]))
R['SEL_grid_IR'] = g
json.dump(R, open(OUT / 'q_stats.json', 'w'), ensure_ascii=False, indent=1)
for k, v in R.items():
    if k == 'SEL_grid_IR':
        print(k); [print('   ', a, round(b, 3)) for a, b in sorted(v.items(), key=lambda x: -x[1])]
    else:
        print(k, {p: {a: round(b, 3) for a, b in s.items()} for p, s in v.items()})
