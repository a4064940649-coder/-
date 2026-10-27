#!/usr/bin/env python3
"""診斷：把 exlib M2 的選股規則（依其原始碼語意）在本檔重寫一次，和 ia_core 的選股逐期比較，找出差異來源。"""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd, ia_core as K
Y = K.yoy_from_rev5y() if (len(sys.argv) < 2 or sys.argv[1] == 'rev5y') else K.yoy_from_long()
start = '2021-01-01' if (len(sys.argv) < 2 or sys.argv[1] == 'rev5y') else '2005-01-01'
n = int(sys.argv[2]) if len(sys.argv) > 2 else 30
S = K.Strategy(Y, start)
adj = K.ADJ; C = K.C; O = K.O
AC = adj.ffill().to_numpy(); fac = (adj / C).to_numpy(); AOr = K.On * fac
ACp = np.vstack([np.full((1, K.N), np.nan), AC[:-1]])
LIM = np.where(K.IDX < pd.Timestamp('2015-06-01'), 0.07, 0.10)[:, None]
GU_ex = (AOr / ACp - 1) >= LIM - 0.006
pl = S.plan(n)
held_r = set(); rows = []
for (e, pk, pool, d) in pl:
    cs = S.cross_section(d)
    # 他們：同一池、同分數；差別只在漲停判定
    cand = np.argsort(-np.nan_to_num(cs['sc'], nan=-9), kind='stable')[:int(cs['el'].sum())]
    gu_me = K.gap_up_open(e)
    pick = []
    for j in cand:
        if GU_ex[e, j] and j not in held_r: continue
        pick.append(j)
        if len(pick) == n: break
    held_r = set(pick)
    a, b = set(pk), set(pick)
    if a != b:
        rows.append((str(d.date()), [K.COLS[j] for j in a - b], [K.COLS[j] for j in b - a],
                     [(K.COLS[j], round(float(AOr[e, j] / ACp[e, j] - 1), 4), bool(gu_me[j]), bool(GU_ex[e, j])) for j in (a ^ b)]))
for r in rows: print(r)
print('n diff periods', len(rows))
