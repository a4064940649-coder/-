#!/usr/bin/env python3
"""複製員 o6：SPA 族群的 14 條 o_v4 成員換成 V6 版（只算一般成本與各自 EW）。另存 AB_V6 的 EW。
用法：O6R_PANEL=compat python3 -I -B r6_family.py"""
import os, sys, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import r4_env
PANEL = os.environ.get('O6R_PANEL', 'compat')
L = r4_env.load(PANEL); K, V = L.K, L.V
OUT = HERE / f'out_{PANEL}'; OUT.mkdir(exist_ok=True)
T0 = time.time()
def log(*a): print(f'[{time.time() - T0:6.0f}s]', *a, flush=True)
XR = L.setup_px()
YFX, Y5F = L.load_rfix(); YRD = L.load_rdel(); Y5 = L.y5_orig()
GAP = L.gap_dict(); QP = L.qproxy_fn()
DMT, DMS = L.disp_mask_fn('trade'), L.disp_mask_fn('signal')
def mk(**kw):
    YL = kw.pop('YL', YFX); Y5x = kw.pop('Y5', Y5F); dm = kw.pop('dm', DMT)
    return L.V4(YL, Y5x, dm, **kw)
SPECS = {'V6': dict(), 'V6_B05': dict(gap=GAP, pfix=0.5), 'V6_B10': dict(gap=GAP, pfix=1.0), 'V6_BQ': dict(gap=GAP, yproxy=QP),
         'V6_SIG': dict(dm=DMS), 'V6_RDEL': dict(YL=YRD, Y5=Y5), 'V6_SPEC0': dict(fin2=False, exclude_foreign=False),
         'O3REP': dict(YL=YRD, Y5=Y5, dm=DMS, fin2=False, exclude_foreign=False)}
navs = {}; t0 = t1 = None
for k, kw in SPECS.items():
    S = mk(**kw); ta, tb, sd = S.bounds(); pl = S.plan(10, 'official')
    if t0 is None: t0, t1 = ta, tb
    navs[f'{k}|normal'], _ = L.sim_plan(pl, ta, tb, 'normal', XR)
    navs[f'{k}|EW'], _ = K.simulate([(p[0], p[2]) for p in pl], 0, 0, ta, tb)
    log(k)
mean_nav = lambda L_: (1 + pd.concat(L_, axis=1).mean(axis=1).fillna(0)).cumprod()
for k in ('V6', 'V6_B05', 'V6_B10', 'V6_BQ'):
    rr = []; ee = []
    for seed in range(40):
        S = mk(unavail=V.make_unavail('mc', seed), **SPECS[k]); pl = S.plan(10, 'official')
        nav, _ = L.sim_plan(pl, t0, t1, 'normal', XR); rr.append(nav.pct_change())
        ew, _ = K.simulate([(p[0], p[2]) for p in pl], 0, 0, t0, t1); ee.append(ew.pct_change())
    navs[f'AB_{k}|normal'] = mean_nav(rr); navs[f'AB_{k}|EW'] = mean_nav(ee)
    log('AB', k)
for k in ('V6', 'V6_B10'):
    S = mk(unavail=V.make_unavail('worst'), **SPECS[k]); pl = S.plan(10, 'official')
    navs[f'ABW_{k}|normal'], _ = L.sim_plan(pl, t0, t1, 'normal', XR)
    navs[f'ABW_{k}|EW'], _ = K.simulate([(p[0], p[2]) for p in pl], 0, 0, t0, t1)
pd.DataFrame(navs).to_parquet(OUT / 'navs_family14.parquet')
log('done')
