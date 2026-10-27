import sys; sys.path.insert(0, "/home/user/-/research/tw_stock_selection/agents/o_disposal_rejudge/replicate_o3")
import json, pickle
import numpy as np, pandas as pd
import r_core as R
D = pickle.load(open(R.HERE / 'my_runs.pkl', 'rb')); PL, NAV = D['PL'], D['NAV']
IDX = R.IDX
# 受影響名額
def affected(n, a, b):
    out = []
    for p in PL[('E0', n)]:
        if not (pd.Timestamp(a) <= p['d'] <= pd.Timestamp(b)): continue
        for j in p['pick']:
            if j in p['st']: out.append((p['d'], R.COLS[j], p['st'][j]))
    return out
for n, a, b in ((10, '2006-04-01', '2021-08-31'), (30, '2006-04-01', '2021-08-31'), (10, '2006-04-01', '2026-10-12')):
    x = affected(n, a, b)
    print('affected n', n, a, b, len(x), 'in 2021-01..08:', sum(1 for d, *_ in x if d.year == 2021))
for d, c, s in affected(10, '2006-04-01', '2021-08-31'): print('  ', d.date(), c, s[:3])
print('--- paired vs E1, n10 normal')
for v in ['E0', 'E2', 'E3', 'E3@0.5', 'E3@0.7', 'E5']:
    print(v, {p: R.paired(NAV[(v, 10, 'normal')], NAV[('E1', 10, 'normal')], p) for p in ('sel', 'val', 'hold', 'oos')})
print('n30 E0', R.paired(NAV[('E0', 30, 'normal')], NAV[('E1', 30, 'normal')], 'sel'), 'E5', R.paired(NAV[('E5', 30, 'normal')], NAV[('E1', 30, 'normal')], 'sel'))
# 剔除 2021
x = (R.win_ret(NAV[('E0', 10, 'normal')]) - R.win_ret(NAV[('E1', 10, 'normal')]))
xs = R.sub(x, 'sel'); xs2 = xs[xs.index.year != 2021]
print('E0 sel ex2021', round(xs2.mean() * 100, 3), round(R.tstat(xs2), 2), 'max month', xs.idxmax().date(), round(xs.max() * 100, 2))
print('stressX E5-E1 sel', R.paired(NAV[('E5', 10, 'stressX')], NAV[('E1', 10, 'stressX')], 'sel'))
print('--- CAGR')
for v in ['E1', 'E0', 'E5']:
    for p in ('all', 'sel', 'val', 'hold'):
        print(v, p, R.cagr(NAV[(v, 10, 'normal')], p))
for v in ['E1', 'E0', 'E5']: print('v2stress', v, R.cagr(NAV[(v, 10, 'v2stress')], 'all'))
