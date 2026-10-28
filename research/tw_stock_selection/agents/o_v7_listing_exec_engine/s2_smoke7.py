#!/usr/bin/env python3
"""煙霧測試（凍結前，只看 2006～2023）：用 V7 面板跑 V4 引擎，≤ 2023-12-29 的日淨值必須與 V6（navs_v6.parquet）逐位相同（差 ≤ 1e-12）；
另列 V7 與 V6 面板逐格差異的位置（必須全部落在替換窗或窗後 adj 連乘，且都能歸到上市身分、颱風除權息或 2024-01 前視修補）。
程式只輸出 ≤ 2023-12-29 的淨值差異統計，不輸出、不存任何 2024 年以後的淨值。"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import v7env, v7common as C
L = v7env.load('v7')
K, V = L.K, L.V
CUT = pd.Timestamp('2023-12-29')
XR = L.setup_px()
YFX, Y5F = L.load_rfix(); DMT = L.disp_mask_fn('trade'); GAP = L.gap_dict()
N6 = pd.read_parquet(C.V6DIR / 'navs_v6.parquet')
out = {'cut': str(CUT.date()), 'n_dates_v7': len(K.IDX)}
def mk(**kw): return L.V4(YFX, Y5F, DMT, **kw)
def cmp(name, nav):
    a = nav[nav.index <= CUT]; b = N6[name].dropna(); b = b[b.index <= CUT]
    j = a.index.intersection(b.index)
    d = float(np.max(np.abs(a.reindex(j) / b.reindex(j) - 1)))
    return {'n': len(j), 'n_v6': len(b), 'max_rel_diff': d, 'identical_1e12': bool(d <= 1e-12 and len(j) == len(b))}
specs = {'V6': dict(), 'V6_B10': dict(gap=GAP, pfix=1.0)}
P6 = pd.read_csv(C.V6DIR / 'plan_V6.csv')
for k, kw in specs.items():
    S = mk(**kw); ta, tb, sd = S.bounds(); pl = S.plan(10, 'official')
    for c in ('normal', 'stress'):
        nav, _ = L.sim_plan(pl, ta, tb, c, XR)
        out[f'{k}|{c}'] = cmp(f'{k}|{c}', nav)
    ew, _ = K.simulate([(p[0], p[2]) for p in pl], 0, 0, ta, tb)
    out[f'{k}|EW'] = cmp(f'{k}|EW', ew)
    p6 = pd.read_csv(C.V6DIR / f'plan_{k}.csv')
    out[f'{k}_plan_pre2024_identical'] = bool(p6.query("signal <= '2023-12-29'")[['signal', 'picks']].values.tolist() ==
                                              [[str(p[3].date()), ','.join(K.COLS[j] for j in p[1])] for p in pl if p[3] <= CUT])
    out[f'{k}_signal_days_identical'] = [str(d.date()) for d in sd[:-1]] == list(p6.signal)
    print(k, {kk: v for kk, v in out.items() if kk.startswith(k)}, flush=True)
for k in ('V6', 'V6_B10'):
    rets = []
    for seed in range(40):
        S = mk(unavail=V.make_unavail('mc', seed), **specs[k]); ta, tb, sd = S.bounds(); pl = S.plan(10, 'official')
        nav, _ = L.sim_plan(pl, ta, tb, 'normal', XR); rets.append(nav.pct_change())
    nav = (1 + pd.concat(rets, axis=1).mean(axis=1).fillna(0)).cumprod()
    out[f'AB_{k}|normal'] = cmp(f'AB_{k}|normal', nav)
    print('AB', k, out[f'AB_{k}|normal'], flush=True)
# 面板逐格差異的位置（V7 − V6）
diff = {}
for f in ('close_raw', 'open', 'ret', 'adj', 'value', 'flag'):
    a = pd.read_parquet(C.DATA / f'{f}.parquet'); b = pd.read_parquet(C.V6DATA / f'{f}.parquet').reindex(index=a.index, columns=a.columns)
    m = ~((a == b) | (a.isna() & b.isna()))
    dd = m.sum(axis=1); dd = dd[dd > 0]
    diff[f] = {'n_cells': int(m.to_numpy().sum()), 'first_date': str(dd.index.min().date()) if len(dd) else None, 'last_date': str(dd.index.max().date()) if len(dd) else None,
               'n_cells_before_2024': int(m.loc[:'2023-12-31'].to_numpy().sum()), 'n_codes': int((m.sum() > 0).sum())}
out['panel_diff_v7_minus_v6'] = diff
# (c) 2024-01-02～02-14 的上市判定差異
a = pd.read_parquet(C.DATA / 'close_raw.parquet').loc['2024-01-02':'2024-02-14']; b = pd.read_parquet(C.V6DATA / 'close_raw.parquet').loc['2024-01-02':'2024-02-14']
ha, hb = a.notna().any(), b.notna().any()
out['gap_2024_01_codes_v7_only'] = sorted(ha[ha & ~hb.reindex(ha.index).fillna(False)].index)
out['gap_2024_01_codes_v6_only'] = sorted(hb[hb & ~ha.reindex(hb.index).fillna(False)].index)
out['all_identical'] = all(v['identical_1e12'] for kk, v in out.items() if isinstance(v, dict) and 'identical_1e12' in v) and \
    out['V6_plan_pre2024_identical'] and out['V6_B10_plan_pre2024_identical']
C.jdump(out, C.HERE / 'smoke.json')
print('ALL IDENTICAL:', out['all_identical'], diff, out['gap_2024_01_codes_v7_only'], out['gap_2024_01_codes_v6_only'])
