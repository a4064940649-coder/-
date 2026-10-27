#!/usr/bin/env python3
"""引擎核對：主策略排除金融 n10 全期（2006-04～2026-10-12），對照 main_v2 的 28.15% / t 2.64。"""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import rlib as L
END = pd.Timestamp('2026-10-12')
lmL, lm5 = L.last_full(L.YOY), L.last_full(L.Y5)
days = []
for d in L.reb_days('2005-01-01', '2026-10-11'):
    m = pd.Period(d, 'M'); Y, lm = (L.Y5, lm5) if d >= L.SWITCH else (L.YOY, lmL)
    if (m - 1) <= lm and np.isfinite(L.yoy3(m, Y)).sum() > 100: days.append(d)
t0 = L.IDX.get_loc(days[0]); t1 = L.IDX.get_loc(END)
pl = L.make_plan(days, 10, lambda cs, d: L.main_score(cs))
nav, _ = L.simulate_w(L.to_w(pl), *L.rates('normal'), t0, t1)
cg, dd = L.cagr_dd(nav)
WD = days + [END]
e50 = L.E50.iloc[t0:t1 + 1]
o = {'first': str(days[0].date()), 'n_sig': len(days), 'CAGR': cg * 100, 'maxDD': dd * 100, 't_0050': L.tstat(L.win_ret(nav, WD) - L.win_ret(e50, WD))}
json.dump(o, open(HERE / 'engine_check.json', 'w'), indent=1, default=float); print(o)
