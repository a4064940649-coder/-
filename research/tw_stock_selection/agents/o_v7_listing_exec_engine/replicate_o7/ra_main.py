#!/usr/bin/env python3
"""複製員 o7 軌 A：凍結引擎（一字不改）＋V7 面板。V7、V7_B10 官方日；AB（40 seeds）一般／壓力／滑價格點。
輸出 out/navs_a.parquet、out/seed_t.json、out/plans_a.json"""
import sys, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import r_env
L = r_env.load(); K, V = L.K, L.V
OUT = HERE / 'out'
T0 = time.time()
def log(*a): print(f'[{time.time() - T0:6.0f}s]', *a, flush=True)
XR = L.setup_px(); YFX, Y5F = L.load_rfix(); GAP = L.gap_dict(); DMT = L.disp_mask_fn('trade')
A_, B_ = '2006-04-01', '2026-10-12'
SPECS = {'V7': dict(), 'V7_B10': dict(gap=GAP, pfix=1.0)}
SLIPS = [round(0.0010 + 0.00025 * i, 5) for i in range(17)]   # 0.10%～0.50%
def mk(**kw): return L.V4(YFX, Y5F, DMT, **kw)
mean_nav = lambda L_: (1 + pd.concat(L_, axis=1).mean(axis=1).fillna(0)).cumprod()
navs = {}; plans = {}; sd0 = None
for k, kw in SPECS.items():
    S = mk(**kw); t0, t1, sd = S.bounds(); pl = S.plan(10, 'official'); sd0 = sd
    for c in ('normal', 'stress'): navs[f'{k}|{c}'], _ = L.sim_plan(pl, t0, t1, c, XR)
    for s in SLIPS: navs[f'{k}|slip{s}'], _ = L.sim_plan(pl, t0, t1, ('slip', s), XR)
    navs[f'{k}|EW'], _ = K.simulate([(p[0], p[2]) for p in pl], 0, 0, t0, t1)
    plans[k] = [(str(p[3].date()), str(K.IDX[p[0]].date()), [K.COLS[j] for j in p[1]]) for p in pl]
    log(k, L.seg(navs[f'{k}|normal'], navs[f'{k}|EW'], sd, A_, B_))
seed_t = {}
for k, kw in SPECS.items():
    rets = {c: [] for c in ('normal', 'stress')}; rs = {s: [] for s in SLIPS}; ewr = []; st = {'normal': [], 'stress': []}
    for seed in range(40):
        S = mk(unavail=V.make_unavail('mc', seed), **kw); pl = S.plan(10, 'official')
        plans[f'AB_{k}_s{seed}'] = [(str(p[3].date()), str(K.IDX[p[0]].date()), [K.COLS[j] for j in p[1]]) for p in pl]
        ew, _ = K.simulate([(p[0], p[2]) for p in pl], 0, 0, t0, t1); ewr.append(ew.pct_change())
        for c in ('normal', 'stress'):
            nav, _ = L.sim_plan(pl, t0, t1, c, XR); rets[c].append(nav.pct_change())
            st[c].append(L.seg(nav, ew, sd, A_, B_)['t_0050'])
        for s in SLIPS:
            nav, _ = L.sim_plan(pl, t0, t1, ('slip', s), XR); rs[s].append(nav.pct_change())
    for c in rets:
        navs[f'AB_{k}|{c}'] = mean_nav(rets[c])
        pd.concat(rets[c], axis=1).to_parquet(OUT / f'seedret_AB_{k}_{c}.parquet')
    for s in SLIPS: navs[f'AB_{k}|slip{s}'] = mean_nav(rs[s])
    navs[f'AB_{k}|EW'] = mean_nav(ewr)
    seed_t[f'AB_{k}'] = st
    log('AB', k, L.seg(navs[f'AB_{k}|normal'], navs[f'AB_{k}|EW'], sd, A_, B_))
pd.DataFrame(navs).to_parquet(OUT / 'navs_a.parquet')
json.dump({'sig_days': [str(d.date()) for d in sd0], 'plans': plans}, open(OUT / 'plans_a.json', 'w'), ensure_ascii=False)
json.dump(seed_t, open(OUT / 'seed_t.json', 'w'))
log('done')
