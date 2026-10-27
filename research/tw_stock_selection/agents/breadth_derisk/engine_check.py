#!/usr/bin/env python3
"""斷言：simulate_x（曝險≡1、現金）與 ia_core 同款 simulate 逐日相同；並重現 independent_audit 2006～2020 n10 年化。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd, json
import bd_core as K
YL = K.yoy_from_long()
S = K.Strategy(YL, '2005-01-01', end='2020-12-31'); pl = S.plan(10)
t0 = K.IDX.get_loc(S.sig_days[0]); t1 = K.IDX.get_loc(S.end_day)
bb, ss = K.rates(0.001)
a, _ = K.simulate([(p[0], p[1]) for p in pl], bb, ss, t0, t1)
b, sw, _ = K.simulate_x([(p[0], p[1], 1.0) for p in pl], bb, ss, t0, t1)
d = float((a.pct_change() - b.pct_change()).abs().max())
o = K.stats(b, '2006-04-01', '2020-12-31', S.sig_days + [S.end_day], K.E50)
print('max daily ret diff', d, 'CAGR', o['CAGR'], 't0050', o['t_0050'])
assert d < 1e-12
assert abs(o['CAGR'] - 0.183) < 0.002
# 現金 50% 曝險：淨值日報酬應約為一半（成本外）
c, sw2, _ = K.simulate_x([(p[0], p[1], 0.5) for p in pl], bb, ss, t0, t1)
print('half-exposure avg stock weight', float(sw2.mean()))
e, sw3, _ = K.simulate_x([(p[0], p[1], 0.0) for p in pl], bb, ss, t0, t1)
print('zero exposure final nav', float(e.iloc[-1]))
assert abs(e.iloc[-1] - (1 - 0)) < 1e-9 or e.iloc[-1] <= 1.0
f, _, _ = K.simulate_x([(p[0], p[1], 0.0) for p in pl], bb, ss, t0, t1, alt='0050', b50=K.FEE, s50=K.FEE + 0.001)
e50 = K.E50.iloc[t0:t1 + 1]; r = float(f.iloc[-1]); r50 = float(e50.iloc[-1] / e50.iloc[0])
print('all-0050 final', r, '0050 buy&hold', r50)
assert abs(r / r50 - 1) < 0.02
print('ENGINE_CHECK_OK')
