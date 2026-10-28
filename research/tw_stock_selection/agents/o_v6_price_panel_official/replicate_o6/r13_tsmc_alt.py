#!/usr/bin/env python3
"""台積電缺口：權重口徑敏感度（窗起點權重 vs 逐日權重；校準差平均 vs 不校準）。"""
import os, sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import r4_env
L = r4_env.load(os.environ.get('O6R_PANEL', 'compat')); K = L.K
OUT = HERE / 'out_compat'
NV = pd.read_parquet(OUT / 'navs.parquet'); sd = pd.DatetimeIndex(json.load(open(OUT / 'plans.json'))['sig_days'])
ex = pd.read_csv(K.X / 'ex_tsmc.csv', parse_dates=['date']).set_index('date')
sh = pd.read_csv(K.X / 'shares_hist.csv', dtype={'code': str}).sort_values('ym').groupby('code')['shares'].last()
cols = [c for c in K.C.columns if c in sh.index and not c.startswith('0')]
cap = K.C[cols].mul(sh[cols], axis=1)
def west(d):
    c = cap.loc[d].dropna().sort_values(ascending=False).iloc[:50]; return float(c.get('2330', np.nan) / c.sum())
ovl = [d for d in ex.index if d in K.IDX]
calv = [ex.loc[d, 'w2330'] / 100 - west(d) for d in ovl]
cal = float(np.mean(calv))
E50 = K.E50; A = K.MYADJ['2330'].ffill(); nav = NV['AB_V6|normal']
g = [d for d in sd if d >= pd.Timestamp('2023-12-11')]
a, b = g[0], g[-1]
days = K.IDX[(K.IDX > a) & (K.IDX <= b)]
out = {'cal_mean': cal, 'cal_sd': float(np.std(calv))}
for calib in (True, False):
    w = {}
    for d in K.IDX[(K.IDX >= a) & (K.IDX <= b)]:
        w[d] = ex.loc[d, 'w2330'] / 100 if (d >= pd.Timestamp('2025-09-26') and d in ex.index) else west(d) + (cal if calib else 0)
    w = pd.Series(w)
    r50 = E50.pct_change().reindex(days); r33 = A.pct_change().reindex(days)
    wprev = w.shift(1).reindex(days)
    rex = (r50 - wprev * r33) / (1 - wprev)
    lW50 = np.log(E50.loc[b] / E50.loc[a]); lWm = np.log(nav.loc[b] / nav.loc[a]); lWex = np.log1p(rex).sum()
    out[f'daily|calib={calib}'] = dict(gap=float(lW50 - lWex), lag=float(lW50 - lWm), share=float((lW50 - lWex) / (lW50 - lWm)), lead=float(lWm - lWex))
print(json.dumps(out, indent=1))
