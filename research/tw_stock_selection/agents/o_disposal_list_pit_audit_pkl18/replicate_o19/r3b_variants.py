"""v1～v4 迭代版本的 AIC 與 2019 點預測（假設區間＝v2～v4 點預測）。"""
import json, math, sys
import numpy as np, pandas as pd
from scipy import optimize, special
sys.path.insert(0, '/home/user/-/research/tw_stock_selection/agents/o_disposal_list_pit_audit_pkl18/replicate_o19')
OUT = '/home/user/-/research/tw_stock_selection/agents/o_disposal_list_pit_audit_pkl18/replicate_o19/'
D = pd.read_csv(OUT + 'r3_monthly.csv', index_col=0)
D.index = pd.PeriodIndex(D.index, freq='M'); ts = D.index.to_timestamp()
D['lS'] = np.log1p(D.S)
fit = ((ts >= '2006-01-01') & (ts <= '2018-12-01')) | ((ts >= '2020-01-01') & (ts <= '2023-12-01'))
m19 = ts.year == 2019
base = ['const', 'R14', 'R15', 'R17', 'COVID']
def nb(p, X, y):
    a = math.exp(p[-1]); mu = np.exp(X @ p[:-1]); r = 1 / a
    return -np.sum(special.gammaln(y + r) - special.gammaln(r) - special.gammaln(y + 1) + r * np.log(r / (r + mu)) + y * np.log(mu / (r + mu)))
def po(b, X, y):
    mu = np.exp(X @ b); return -np.sum(y * np.log(mu) - mu - special.gammaln(y + 1))
out = {}
for name, cols, fam in [('v1', ['lS'], 'P'), ('v2', ['lS'], 'NB'), ('v3', ['lK'], 'NB'), ('v4', ['lS', 'lK'], 'NB')]:
    cc = ['const'] + cols + base[1:]
    X = D.loc[fit, cc].values; y = D.loc[fit, 'Y'].values
    b0 = np.linalg.lstsq(X, np.log(y + 0.5), rcond=None)[0]
    rp = optimize.minimize(po, b0, args=(X, y), method='BFGS')
    if fam == 'P':
        p = rp.x; ll = rp.fun; k = len(p); beta = p
    else:
        r = optimize.minimize(nb, np.r_[rp.x, -1.6], args=(X, y), method='BFGS')
        r = optimize.minimize(nb, r.x, args=(X, y), method='Nelder-Mead', options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 40000})
        p = r.x; ll = r.fun; k = len(p); beta = p[:-1]
    pred = float(np.exp(D.loc[m19, cc].values @ beta).sum())
    out[name] = dict(aic=round(2 * ll + 2 * k, 1), pred2019=round(pred, 2))
print(json.dumps(out, indent=1))
json.dump(out, open(OUT + 'r3b_variants.json', 'w'), indent=1)
