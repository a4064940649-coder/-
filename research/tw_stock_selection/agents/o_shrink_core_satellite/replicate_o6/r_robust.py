"""Independent replication o6: robustness subsets, period splits, history."""
import json
import sys
sys.path.insert(0, "/home/user/-/research/tw_stock_selection/agents/o_shrink_core_satellite/replicate_o6")
import numpy as np
import pandas as pd
from r_lib import TH, hac, decompose, fit_g2, post_grid, summ
from r_alloc import sb_index, draw_alpha, simulate, stats

OUT = '/home/user/-/research/tw_stock_selection/agents/o_shrink_core_satellite/replicate_o6/'
A = '/home/user/-/research/tw_stock_selection/agents/'
SDC = 0.03624025193680153
foc = pd.read_parquet(OUT + 'focus.parquet')
X = pd.read_parquet(OUT + 'flong.parquet') * 100
ref = json.load(open(A + 'o_shrink_core_satellite/data/families.json'))
Am = X[ref['MAIN']].values
yrs = foc.index.year.values


def post_sub(mask, f='EXEC_J'):
    x, s2, m, vP = decompose(Am[mask])
    g2, _ = fit_g2(x, s2)
    e = foc[f].values[mask]
    s2f = hac(e - m) + (SDC ** 2 if f.startswith('EXEC') else 0)
    return post_grid(g2, e.mean(), s2f, vP)


def xstar(mask, p, f='EXEC_J', B=20000):
    n = mask.sum()
    idx = sb_index(n, 120, 6, B, 20261029)
    e = foc[f].values[mask] / 100
    res = e - e.mean()
    r50 = foc['r0050'].values[mask][idx]
    rm = r50 + draw_alpha(p, B)[:, None] + res[idx]
    el = {}
    for x in np.arange(0, 1.01, 0.1):
        Wp, W5 = simulate(r50, rm, x)
        el[round(x, 1)] = np.mean(np.log(Wp[:, -1] / W5[:, -1]))
    return max(el, key=el.get)


rows = []
for y in sorted(set(yrs)):
    mk = yrs != y
    p = post_sub(mk)
    s = summ(p)
    rows.append((y, s['mean'], s['P']))
    print('drop', y, 'mean %.3f P %.3f' % (s['mean'], s['P']), flush=True)
r = np.array([[a, b, c] for a, b, c in rows])
print('EXEC_J drop-year range mean %.3f..%.3f P %.2f..%.2f' % (r[:, 1].min(), r[:, 1].max(), r[:, 2].min(), r[:, 2].max()))
for y in [2021]:
    mk = yrs != y
    print('drop2021 x*', xstar(mk, post_sub(mk)))
mk = yrs >= 2008
p = post_sub(mk)
print('start2008', summ(p), 'x*', xstar(mk, p))
ej = foc['EXEC_J'].values
mk = np.ones(len(ej), bool)
mk[np.argsort(ej)[-10:]] = False
p = post_sub(mk)
print('drop best10', summ(p), 'x*', xstar(mk, p))
abP = [summ(post_sub(yrs != y, 'AB'))['P'] for y in sorted(set(yrs))]
print('AB drop-year min P', min(abP))

# periods
d = foc.index
SEL = d <= '2019-04-30'
VAL = (d > '2019-04-30') & (d < '2024-01-01')
HOLD = d >= '2024-01-01'
print('SEL/VAL/HOLD windows', SEL.sum(), VAL.sum(), HOLD.sum())
for nm, mk in [('VAL', VAL), ('HOLD', HOLD)]:
    print(nm, 'AB realized %.3f' % foc['AB'].values[mk].mean())

# history (x=100%, no rebalancing cost needed)
yrs_len = (pd.Timestamp('2026-10-12') - pd.Timestamp('2006-04-11')).days / 365.25
r50 = foc['r0050'].values
for f in ['EXEC_J', 'AB']:
    rm = r50 + foc[f].values / 100
    W = np.cumprod(1 + rm)
    print(f, 'hist ann %.3f vs 0050 %.3f' % (W[-1] ** (1 / yrs_len) - 1, np.prod(1 + r50) ** (1 / yrs_len) - 1),
          'window MDD %.3f' % (np.min(np.concatenate([[1], W]) / np.maximum.accumulate(np.concatenate([[1], W])) - 1)))
