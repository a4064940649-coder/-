import sys, json; sys.path.insert(0, '.')
from breaks import *
from scipy import stats as st
v4 = pd.read_parquet(AG / 'o_v4_revenue_patch_official/navs_v4.parquet')
x = win_ret(v4['AB_V4|normal']) - win_ret(e0050())
z = st.norm.ppf(0.95) + st.norm.ppf(0.8)
a, b = x[:213], x[213:]
print('MDE 2s separate HAC', z * math.sqrt(hac_lr(a - a.mean()) / 213 + hac_lr(b - b.mean()) / 32) * 100)
nullH = null_supF(NW, True); crit = np.quantile(nullH, 0.95)
sel = x[:157]; mu = sel.mean(); n0 = 157
out = {}
for demean in (False, True):
    for drop in (0.25, 0.5, 1.0):
        rng = np.random.default_rng(4242); hf = hc = 0
        for r in range(400):
            ii = np.empty(NW, int); ii[0] = rng.integers(0, n0)
            for t in range(1, NW): ii[t] = rng.integers(0, n0) if rng.random() < 1 / 6 else (ii[t - 1] + 1) % n0
            src = sel - mu if demean else sel
            y = src[ii] + (mu if demean else 0); y[157:] -= drop * mu
            hf += supF(y, True)[0] > crit; hc += chow(y, 157)['p'] < 0.05
        out[f'{demean}|{drop}'] = (hf / 400, hc / 400); print('demean', demean, drop, hf / 400, hc / 400, flush=True)
json.dump(out, open(OUT / 'power.json', 'w'))
