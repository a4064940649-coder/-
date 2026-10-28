#!/usr/bin/env python3
"""煙霧測試（凍結前，只看 2006～2023）：用 V6 面板跑 V4 引擎，日淨值在 ≤ 2023-12-29 必須與 navs_v4 逐位相同（差 ≤ 1e-12）。
程式只輸出 ≤ 2023-12-29 的差異統計，不輸出、不存任何 2024 年以後的淨值。"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import v6env, v6common as C
L = v6env.load('v6')
K, V = L.K, L.V
CUT = pd.Timestamp('2023-12-29')
XR = L.setup_px()
YFX, Y5F = L.load_rfix(); DMT = L.disp_mask_fn('trade'); GAP = L.gap_dict()
N4 = pd.read_parquet(C.V4DIR / 'navs_v4.parquet')
out = {'cut': str(CUT.date()), 'n_dates_v6': len(K.IDX)}
def mk(**kw): return L.V4(YFX, Y5F, DMT, **kw)
def cmp(name, nav):
    a = nav[nav.index <= CUT]; b = N4[name].dropna(); b = b[b.index <= CUT]
    j = a.index.intersection(b.index)
    d = float(np.max(np.abs(a.reindex(j) / b.reindex(j) - 1)))
    return {'n': len(j), 'n_v4': len(b), 'max_rel_diff': d, 'identical_1e12': bool(d <= 1e-12 and len(j) == len(b))}
specs = {'V4': dict(), 'V4_B10': dict(gap=GAP, pfix=1.0)}
for k, kw in specs.items():
    S = mk(**kw); ta, tb, sd = S.bounds(); pl = S.plan(10, 'official')
    for c in ('normal', 'stress'):
        nav, _ = L.sim_plan(pl, ta, tb, c, XR)
        out[f'{k}|{c}'] = cmp(f'{k}|{c}', nav)
    ew, _ = K.simulate([(p[0], p[2]) for p in pl], 0, 0, ta, tb)
    out[f'{k}|EW'] = cmp(f'{k}|EW', ew)
    out[f'{k}_plan_pre2024_identical'] = bool(pd.read_csv(C.V4DIR / f'plan_{k}.csv').query("signal <= '2023-12-29'")[['signal', 'picks']].values.tolist() ==
                                              [[str(p[3].date()), ','.join(K.COLS[j] for j in p[1])] for p in pl if p[3] <= CUT])
    print(k, {kk: v for kk, v in out.items() if kk.startswith(k)}, flush=True)
# AB（40 seeds）
for k in ('V4', 'V4_B10'):
    rets = []
    for seed in range(40):
        S = mk(unavail=V.make_unavail('mc', seed), **specs[k]); ta, tb, sd = S.bounds(); pl = S.plan(10, 'official')
        nav, _ = L.sim_plan(pl, ta, tb, 'normal', XR); rets.append(nav.pct_change())
    nav = (1 + pd.concat(rets, axis=1).mean(axis=1).fillna(0)).cumprod()
    out[f'AB_{k}|normal'] = cmp(f'AB_{k}|normal', nav)
    print('AB', k, out[f'AB_{k}|normal'], flush=True)
out['all_identical'] = all(v['identical_1e12'] for kk, v in out.items() if isinstance(v, dict)) and out['V4_plan_pre2024_identical'] and out['V4_B10_plan_pre2024_identical']
C.jdump(out, C.HERE / 'smoke.json')
print('ALL IDENTICAL:', out['all_identical'])
