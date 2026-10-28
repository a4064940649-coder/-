#!/usr/bin/env python3
"""複製員 o6：j 分布統計（r8_jscan 的輸出）＋ B10 可執行版 j 中位數（實盤預期）。
t：對 0050、j 自己的換股窗（訊號日收盤→下一訊號日收盤）、窗起點在 2006-04-01～2026-10-12；年化：v4lib.seg。"""
import os, sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import r4_env
PANEL = os.environ.get('O6R_PANEL', 'compat')
L = r4_env.load(PANEL); K = L.K
OUT = HERE / f'out_{PANEL}'
NJ = pd.read_parquet(OUT / 'navs_jscan.parquet')
TJ = json.load(open(OUT / 'jscan_seed_t.json'))
sys.path.insert(0, str(L.AG / 'o_stagger_tranche'))
import st_core as ST
sys.path.pop(0)
PJ = json.load(open(OUT / 'plans.json')); ANCH = [pd.Period(d, 'M') for d in pd.DatetimeIndex(PJ['sig_days'][:-1])]
def jdays(j):
    days = [ST.day_for(m, j) for m in ANCH]; days = [d for d in days if d is not None]
    de = ST.day_for(pd.Period('2026-10', 'M'), j)
    if de is None or de > K.IDX[-1]: de = K.IDX[-1]
    return [d for d in days if d < de] + [de]
def r(x, k=4): return round(float(x), k)
res = {}
drags = {f'{t}|{f}': pd.read_parquet(OUT / f'execdrag_{t}_{f}.parquet')['drag'] for t in ('30萬', '100萬', '300萬', '1000萬') for f in ('FIX', 'GROW')}
for k in ('V6', 'V6_B10'):
    for c in ('normal', 'stress'):
        ts, cg = [], []
        for j in range(13):
            sd = jdays(j); nav = NJ[f'AB_{k}|j{j}|{c}'].dropna()
            s = L.seg(nav, nav, sd, '2006-04-01', '2026-10-12'); ts.append(s['t_0050']); cg.append(s['CAGR'])
        st = np.array([TJ[f'{k}|{c}|j{j}|s{s}'] for j in range(13) for s in range(40)])
        res[f'AB_{k}|{c}'] = dict(t_by_j=[r(x, 3) for x in ts], cagr_by_j=[r(x, 2) for x in cg],
                                  t_median=r(np.median(ts), 3), t_p10=r(np.percentile(ts, 10), 3), cagr_median=r(np.median(cg), 2),
                                  seedj_t_median=r(np.median(st), 3), seedj_t_p10=r(np.percentile(st, 10), 3))
# 可執行版（B10，一般成本）：j 版逐日報酬＋官方換股日的可執行拖累（o_exec − AB_V4）
for tf, dr in drags.items():
    ts, cg = [], []
    for j in range(13):
        sd = jdays(j); nav = NJ[f'AB_V6_B10|j{j}|normal'].dropna(); rr = nav.pct_change().fillna(0.0) + dr.reindex(nav.index).fillna(0.0)
        en = (1 + rr).cumprod() * nav.iloc[0]
        s = L.seg(en, en, sd, '2006-04-01', '2026-10-12'); ts.append(s['t_0050']); cg.append(s['CAGR'])
    res[f'exec_B10|{tf}'] = dict(cagr_median=r(np.median(cg), 2), t_median=r(np.median(ts), 3), cagr_by_j=[r(x, 2) for x in cg])
json.dump(res, open(OUT / 'jstats.json', 'w'), ensure_ascii=False, indent=1)
print(json.dumps({k: {kk: vv for kk, vv in v.items() if not kk.endswith('by_j')} for k, v in res.items()}, ensure_ascii=False, indent=1))
