import sys, os, json
import numpy as np, pandas as pd
adj = pd.read_parquet('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel/adj.parquet')
cnt = adj.notna().sum(1)
days = adj.index[cnt >= 300]
days = pd.DatetimeIndex(days)
pos = {d: i for i, d in enumerate(days)}
def J(y, m):
    d0 = pd.Timestamp(y, m, 10)
    i0 = days.searchsorted(d0)
    return [i0 + j for j in range(13)]
months = pd.period_range('2006-04', '2019-06', freq='M')
Js = [J(p.year, p.month) for p in months]
def sets(k):  # windows of month index k: (J_j(k), J_j(k+1)]
    return [set(range(Js[k][j] + 1, Js[k+1][j] + 1)) for j in range(13)]
rhos = []; l12 = []; gap = []
for k in range(len(months) - 2):
    A = sets(k); B = sets(k+1)
    va = sum(len(a & b) for a in A for b in A); vb = sum(len(a & b) for a in B for b in B)
    cv = sum(len(a & b) for a in A for b in B)
    rhos.append(cv / np.sqrt(va * vb)); l12.append(len(A[12])); gap.append(Js[k+1][12] - Js[k+1][0])
rhos = np.array(rhos[:157])
r = float(np.mean(rhos)); rm = float(np.median(rhos))
th = lambda r: (1 - np.sqrt(1 - 4 * r * r)) / (2 * r)
out = dict(n_pairs=len(rhos), months=f"{months[0]}..{months[156]}", rho1_mean=round(r, 4), rho1_median=round(rm, 4),
           p10_p90=[round(float(np.quantile(rhos, .1)), 4), round(float(np.quantile(rhos, .9)), 4)],
           theta_mean=round(th(r), 4), theta_median=round(th(rm), 4), theta_from_max=round(th(rhos.max()), 4),
           j12_len_median=float(np.median(l12[:157])), j12end_minus_nextJ0_median=float(np.median(gap[:157])))
print(json.dumps(out))
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), 'rep_e2.json'), 'w'))
