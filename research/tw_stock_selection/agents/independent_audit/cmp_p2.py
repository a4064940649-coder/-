#!/usr/bin/env python3
"""逐期比對：自建引擎 vs exec_rebuild main_nav（只讀其輸出 parquet）。"""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd, ia_core as K
Y5 = K.yoy_from_rev5y()
S = K.Strategy(Y5, '2021-01-01')
t0 = K.IDX.get_loc(S.sig_days[0]); t1 = K.IDX.get_loc(S.end_day)
ex = pd.read_parquet(K.HERE.parent / 'exec_rebuild/main_nav.parquet')
for n in (30, 10):
    pl = S.plan(n)
    b, s = K.rates(0.001)
    nav, _ = K.simulate([(p[0], p[1]) for p in pl], b, s, t0, t1)
    g, _ = K.simulate([(p[0], p[1]) for p in pl], 0, 0, t0, t1)
    ref = ex[f'M2_n{n}_slip0.001']; refg = ex[f'M2_n{n}_gross']
    sd = pd.DatetimeIndex(S.sig_days + [S.end_day])
    a = nav.reindex(sd); r = ref.reindex(sd)
    pr = (a.shift(-1) / a - 1); rr = (r.shift(-1) / r - 1)
    d = (pr - rr).dropna()
    print(n, 'end nav mine', nav.iloc[-1], 'ref', ref.iloc[-1], 'gross mine', g.iloc[-1], 'ref', refg.iloc[-1])
    print(d[abs(d) > 0.002].round(4))
