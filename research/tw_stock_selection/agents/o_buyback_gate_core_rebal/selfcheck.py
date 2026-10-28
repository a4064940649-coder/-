# -*- coding: utf-8 -*-
"""自我檢查：(1) NONE＝(1−w)A＋wB 的封閉解；(2) 每月版用另一種寫法（逐窗連乘）重算，與 rlib.sim 比對；(3) 無成本每月版的訊號窗報酬＝(1−w)rA＋w rB。"""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import rlib as L
out = {}
for P in ('SEL', 'POST', 'ALL'):
    a, b = L.PER[P]
    A = L.bench('MAIN|normal', a, b); B = L.bench('E50', a, b)
    n = L.sim(0.3, 'NONE', a, b)['nav']
    out[f'NONE_{P}_maxabs'] = float(((0.7 * A + 0.3 * B) - n).abs().max())
    # 逐窗：訊號日收盤回到目標（扣成本）
    D = [d for d in L.SIG if A.index[0] < d <= A.index[-1]]
    pts = [A.index[0]] + D + ([A.index[-1]] if A.index[-1] not in D else [])
    c = L.C['normal']; v = 1.0; wa, wb = 0.7, 0.3; vals = []
    for i in range(1, len(pts)):
        p0, p1 = pts[i - 1], pts[i]
        ga = A[p1] / A[p0]; gb = B[p1] / B[p0]
        va, vb = v * wa * ga, v * wb * gb; tot = va + vb
        if p1 in D:
            tb = 0.3 * tot
            cost = (vb - tb) * (c['etf_sell'] + c['stock_buy']) if vb > tb else (tb - vb) * (c['stock_sell'] + c['etf_buy'])
            tot -= cost; wa, wb = 0.7, 0.3
        else:
            wa, wb = va / tot, vb / tot
        v = tot; vals.append(v)
    m = L.sim(0.3, 'M', a, b)['nav']
    out[f'M_{P}_end_rel_diff'] = float(vals[-1] / m.iloc[-1] - 1)
print(json.dumps(out, indent=1))
assert all(abs(v) < 1e-9 for v in out.values())
(HERE / 'selfcheck.json').write_text(json.dumps(out, indent=1), encoding='utf-8')
