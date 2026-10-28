#!/usr/bin/env python3
"""步驟 0：重建 AB_V4 的逐 seed 名單（v2_core.make_unavail('mc', seed)，seed 0～39）與 V4 主名單，
並用 O3.simulate（零衝擊、一般成本）重現 navs_v4.parquet 的 AB_V4|normal 與 V4|normal（斷言）。
只讀 import o_v4_revenue_patch_official/v4lib.py（再 import exlib/v2_core/o3_core）；不寫其他資料夾。
輸出：data/plans_ab.csv（seed,signal,exec,exec_idx,picks）、data/plan_v4.csv、data/zero_impact_seed_navs.parquet、data/pool_sizes.csv
用法：python3 -I -B gen_plans.py
"""
import sys, json, hashlib, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
V4D = HERE.parent / 'o_v4_revenue_patch_official'
sys.path.insert(0, str(V4D))
import numpy as np, pandas as pd
import v4lib as L
sys.path.pop(0)
K, V = L.K, L.V
T0 = time.time()
def log(*a): print(f'[{time.time() - T0:6.0f}s]', *a, flush=True)

XR = L.setup_px()
YFX, Y5F = L.load_rfix()
DMT = L.disp_mask_fn('trade')
NV = pd.read_parquet(V4D / 'navs_v4.parquet')

S = L.V4(YFX, Y5F, DMT)
t0, t1, sd = S.bounds()
pl = S.plan(10, 'official')
nav, _ = L.sim_plan(pl, t0, t1, 'normal', XR)
d = float((nav / NV['V4|normal'].reindex(nav.index) - 1).abs().max())
log('V4|normal 重現最大相對差', d); assert d < 1e-9
pc = pd.read_csv(V4D / 'plan_V4.csv', dtype=str)
assert all(','.join(K.COLS[j] for j in p[1]) == pc.picks.iloc[i] for i, p in enumerate(pl))
rows = [(-1, str(p[3].date()), str(K.IDX[p[0]].date()), int(p[0]), ','.join(K.COLS[j] for j in p[1])) for p in pl]
pd.DataFrame(rows, columns=['seed', 'signal', 'exec', 'exec_idx', 'picks']).to_csv(HERE / 'data' / 'plan_v4.csv', index=False)

rows = []; rets = []; seednav = {}; pool = []
for seed in range(40):
    Sx = L.V4(YFX, Y5F, DMT, unavail=V.make_unavail('mc', seed))
    plx = Sx.plan(10, 'official')
    assert Sx.bounds()[:2] == (t0, t1)
    nv, _ = L.sim_plan(plx, t0, t1, 'normal', XR)
    rets.append(nv.pct_change()); seednav[f's{seed:02d}'] = nv
    for p in plx:
        rows.append((seed, str(p[3].date()), str(K.IDX[p[0]].date()), int(p[0]), ','.join(K.COLS[j] for j in p[1])))
        if seed == 0: pool.append((str(p[3].date()), int(len(p[2]))))
    log('seed', seed, 'nav_end', round(float(nv.iloc[-1]), 3))
mean_nav = (1 + pd.concat(rets, axis=1).mean(axis=1).fillna(0)).cumprod()
d = float((mean_nav / NV['AB_V4|normal'].reindex(mean_nav.index) - 1).abs().max())
log('AB_V4|normal 重現最大相對差', d); assert d < 1e-9
pd.DataFrame(rows, columns=['seed', 'signal', 'exec', 'exec_idx', 'picks']).to_csv(HERE / 'data' / 'plans_ab.csv', index=False)
pd.DataFrame(seednav).to_parquet(HERE / 'data' / 'zero_impact_seed_navs.parquet')
pd.DataFrame(pool, columns=['signal', 'n_pool']).to_csv(HERE / 'data' / 'pool_sizes.csv', index=False)
json.dump({'V4_repro_maxrel': 0.0, 'AB_V4_repro_maxrel': d, 't0': str(K.IDX[t0].date()), 't1': str(K.IDX[t1].date()),
           'n_reb': len(pl), 'n_patched_P': L._STATE['npatch'], 'n_exit_rules': len(XR)},
          open(HERE / 'data' / 'gen_plans_check.json', 'w'), ensure_ascii=False, indent=1)
log('done')
