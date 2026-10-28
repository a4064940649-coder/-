"""Independent replication o6: OOS check of family shrinkage (SEL prior -> VAL/HOLD)."""
import json, sys
sys.path.insert(0, "/home/user/-/research/tw_stock_selection/agents/o_shrink_core_satellite/replicate_o6")
import numpy as np, pandas as pd
from r_lib import TH, hac, decompose, fit_g2, post_grid, summ
OUT = '/home/user/-/research/tw_stock_selection/agents/o_shrink_core_satellite/replicate_o6/'
A = '/home/user/-/research/tw_stock_selection/agents/'
foc = pd.read_parquet(OUT + 'focus.parquet')
X = pd.read_parquet(OUT + 'flong.parquet') * 100
ref = json.load(open(A + 'o_shrink_core_satellite/data/families.json'))
Am = X[ref['MAIN']].values
d = foc.index
SEL = np.asarray(d <= '2019-04-30'); VAL = np.asarray((d > '2019-04-30') & (d < '2024-01-01')); HOLD = np.asarray(d >= '2024-01-01')
x, s2, m, vP = decompose(Am[SEL])
g2, _ = fit_g2(x, s2)
ps = [post_grid(g2, x[k], s2[k], vP) for k in range(len(x))]
sh = np.array([p @ TH for p in ps])
cov = lambda p, y: np.interp(.1, np.cumsum(p), TH) <= y <= np.interp(.9, np.cumsum(p), TH)
for nm, mk in [('VAL', VAL), ('HOLD', HOLD)]:
    y = Am[mk].mean(0)
    b_raw = np.polyfit(x, y, 1)[0]; b_sh = np.polyfit(sh, y, 1)[0]
    print(nm, 'MSE raw %.3f shrunk %.3f slope raw %.2f shrunk %.2f cov80 shrunk %.2f' % (
        np.mean((x - y) ** 2), np.mean((sh - y) ** 2), b_raw, b_sh, np.mean([cov(p, yy) for p, yy in zip(ps, y)])))
e = foc['AB'].values
pa = post_grid(g2, e[SEL].mean(), hac(e[SEL] - m), vP)
for nm, mk in [('VAL', VAL), ('HOLD', HOLD)]:
    y = e[mk].mean(); print('AB', nm, 'realized %.2f pct %.2f' % (y, np.interp(y, TH, np.cumsum(pa))))
# predictive version: add sampling variance of the out-of-sample mean
from scipy.stats import norm
def pred(p, v):
    sd = np.sqrt(v); k = norm.pdf(np.arange(-300, 301) * 0.02 / sd); k /= k.sum()
    q = np.convolve(p, k, 'same'); return q / q.sum()
for nm, mk in [('VAL', VAL), ('HOLD', HOLD)]:
    y = Am[mk].mean(0); _, s2o, mo, vPo = decompose(Am[mk])
    vo = np.array([hac(Am[mk][:, k]) for k in range(Am.shape[1])])
    c_sh = np.mean([cov(pred(p, v), yy) for p, v, yy in zip(ps, vo, y)])
    c_raw = np.mean(np.abs(y - x) <= 1.2816 * np.sqrt(s2 + vP + vo))
    print(nm, 'pred cov80 raw %.2f shrunk %.2f' % (c_raw, c_sh))
    ya = e[mk].mean(); print(' AB pred pct %.2f' % np.interp(ya, TH, np.cumsum(pred(pa, hac(e[mk])))))
