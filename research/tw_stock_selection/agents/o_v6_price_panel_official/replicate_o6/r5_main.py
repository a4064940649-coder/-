#!/usr/bin/env python3
"""複製員 o6：V6 正式數字（跨循環、分段、逐 seed、端點、成本損益兩平、壓力、ABW、全額手續費）。
用法：O6R_PANEL=compat|spec python3 -I -B r5_main.py
輸出：out_<panel>/navs.parquet、seed_t.json、main.json"""
import os, sys, json, math, time
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
YFX, Y5F = L.load_rfix(); GAP = L.gap_dict(); DMT = L.disp_mask_fn('trade')
A_, B_ = '2006-04-01', '2026-10-12'
SPECS = {'V6': dict(), 'V6_B10': dict(gap=GAP, pfix=1.0)}
def mk(**kw): return L.V4(YFX, Y5F, DMT, **kw)
SLIPS = [round(0.0010 + 0.0002 * i, 4) for i in range(16)]

navs = {}; plans = {}
t0 = t1 = sd = None
for k, kw in SPECS.items():
    S = mk(**kw); ta, tb, sdx = S.bounds(); pl = S.plan(10, 'official')
    if t0 is None: t0, t1, sd = ta, tb, sdx
    assert (ta, tb) == (t0, t1) and sdx == sd
    for c in ('normal', 'stress'):
        navs[f'{k}|{c}'], _ = L.sim_plan(pl, t0, t1, c, XR)
    navs[f'{k}|fullfee'], _ = L.sim_plan(pl, t0, t1, ('fullfee', 0.001), XR)
    navs[f'{k}|EW'], _ = K.simulate([(p[0], p[2]) for p in pl], 0, 0, t0, t1)
    plans[k] = [(str(p[3].date()), str(K.IDX[p[0]].date()), [K.COLS[j] for j in p[1]]) for p in pl]
    log(k, 'nblk', S.nblk, round(L.seg(navs[f'{k}|normal'], navs[f'{k}|EW'], sd, A_, B_)['CAGR'], 3))

seed_t = {}
for k, kw in SPECS.items():
    rets = {c: [] for c in ('normal', 'stress')}; rs = {s: [] for s in SLIPS}; ewr = []; st = {'normal': [], 'stress': []}
    for seed in range(40):
        S = mk(unavail=V.make_unavail('mc', seed), **kw); pl = S.plan(10, 'official')
        if seed == 0: plans[f'AB_{k}_s0'] = [(str(p[3].date()), str(K.IDX[p[0]].date()), [K.COLS[j] for j in p[1]]) for p in pl]
        ew, _ = K.simulate([(p[0], p[2]) for p in pl], 0, 0, t0, t1); ewr.append(ew.pct_change())
        for c in ('normal', 'stress'):
            nav, _ = L.sim_plan(pl, t0, t1, c, XR); rets[c].append(nav.pct_change())
            st[c].append(L.seg(nav, ew, sd, A_, B_)['t_0050'])
        for s in SLIPS:
            nav, _ = L.sim_plan(pl, t0, t1, ('slip', s), XR); rs[s].append(nav.pct_change())
    mean_nav = lambda L_: (1 + pd.concat(L_, axis=1).mean(axis=1).fillna(0)).cumprod()
    for c in rets: navs[f'AB_{k}|{c}'] = mean_nav(rets[c])
    for s in SLIPS: navs[f'AB_{k}|slip{s}'] = mean_nav(rs[s])
    navs[f'AB_{k}|EW'] = mean_nav(ewr)
    seed_t[f'AB_{k}'] = st
    pd.concat(rets['normal'], axis=1).to_parquet(OUT / f'seedret_AB_{k}_normal.parquet')
    pd.concat(rets['stress'], axis=1).to_parquet(OUT / f'seedret_AB_{k}_stress.parquet')
    log('AB', k, round(L.seg(navs[f'AB_{k}|normal'], navs[f'AB_{k}|EW'], sd, A_, B_)['CAGR'], 3))
    S = mk(unavail=V.make_unavail('worst'), **kw); pl = S.plan(10, 'official')
    for c in ('normal', 'stress'): navs[f'ABW_{k}|{c}'], _ = L.sim_plan(pl, t0, t1, c, XR)
    navs[f'ABW_{k}|EW'], _ = K.simulate([(p[0], p[2]) for p in pl], 0, 0, t0, t1)

N = pd.DataFrame(navs); N.to_parquet(OUT / 'navs.parquet')
json.dump({'sig_days': [str(d.date()) for d in sd], 't0': str(K.IDX[t0].date()), 't1': str(K.IDX[t1].date()), 'plans': plans},
          open(OUT / 'plans.json', 'w'), ensure_ascii=False)
json.dump(seed_t, open(OUT / 'seed_t.json', 'w'))
log('done')
