#!/usr/bin/env python3
"""彙整：python3 -I -B r3_stats.py <npz> ..."""
import sys, json, math
import numpy as np
def tstat(x):
    x = np.asarray(x, float); return float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x))))
SETS = ['OFF'] + [f'J{j}' for j in range(13)]
res = {}
for f in sys.argv[1:]:
    z = np.load(f)
    row = {}
    for k in SETS:
        a = z[f'{k}|act']; r = z[f'{k}|rnd']; e = z[f'{k}|e50']; x = a - r.mean(axis=0)
        mr = (r - e).mean(axis=1); ma = (a - e).mean()
        row[k] = dict(n=len(x), pp=x.mean() * 100, t=tstat(x), perm_p=float(np.mean(mr >= ma)), v50_pp=(a - e).mean() * 100, v50_t=tstat(a - e))
    tj = np.array([row[f'J{j}']['t'] for j in range(13)]); pj = np.array([row[f'J{j}']['pp'] for j in range(13)])
    v5 = np.array([row[f'J{j}']['v50_t'] for j in range(13)]); v5p = np.array([row[f'J{j}']['v50_pp'] for j in range(13)])
    s = dict(OFF_t=row['OFF']['t'], OFF_pp=row['OFF']['pp'], OFF_perm_p=row['OFF']['perm_p'], jmed_t=float(np.median(tj)), jp10_t=float(np.percentile(tj, 10)),
             jmin=float(tj.min()), jmax=float(tj.max()), jmed_pp=float(np.median(pj)), n=[row[k]['n'] for k in SETS],
             perm_p_max=max(row[k]['perm_p'] for k in SETS), v50_jmed_t=float(np.median(v5)), v50_jp10_t=float(np.percentile(v5, 10)), v50_jmed_pp=float(np.median(v5p)),
             v50_OFF_t=row['OFF']['v50_t'], tj=[round(v, 3) for v in tj])
    res[f] = s
    print(f, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()}, ensure_ascii=False))
