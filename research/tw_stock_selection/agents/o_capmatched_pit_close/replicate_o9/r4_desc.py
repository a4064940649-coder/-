#!/usr/bin/env python3
"""描述性複核（版本 a＝car_a2）：去最好 1 月／1 年、分段、逐年、NW(3)、合併 2014～2026、安慰劑校準。"""
import json, math
import numpy as np, pandas as pd
def tstat(x):
    x = np.asarray(x, float); return float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x))))
def nw_t(s, L_=3):
    s = np.asarray(s, float); n = len(s); u = s - s.mean(); v = u @ u / n
    for l in range(1, L_ + 1): v += 2 * (1 - l / (L_ + 1)) * (u[l:] @ u[:-l]) / n
    return s.mean() / math.sqrt(v / n)
M = np.load('out_car_a2_MAIN_normal_1000.npz'); P = np.load('out_car_a2_PRE_normal_1000.npz')
Js = [f'J{j}' for j in range(13)]
def ser(z, k):
    a = z[f'{k}|act']; r = z[f'{k}|rnd']; return pd.Series(a - r.mean(axis=0), index=pd.to_datetime(z[f'{k}|days']))
out = {}
X = {k: ser(M, k) for k in Js}
out['drop_best_month'] = float(np.median([tstat(x.drop(x.idxmax())) for x in X.values()]))
def dby(x):
    y = x.groupby(x.index.year).sum().idxmax(); return tstat(x[x.index.year != y]), int(y)
r = [dby(x) for x in X.values()]; out['drop_best_year'] = float(np.median([a for a, _ in r])); out['best_year_mode'] = pd.Series([b for _, b in r]).mode().tolist()
out['seg_2019_05_2022'] = (float(np.median([x[x.index < '2023-01-01'].mean() * 100 for x in X.values()])), float(np.median([tstat(x[x.index < '2023-01-01']) for x in X.values()])))
out['seg_2023_2026'] = (float(np.median([x[x.index >= '2023-01-01'].mean() * 100 for x in X.values()])), float(np.median([tstat(x[x.index >= '2023-01-01']) for x in X.values()])))
out['seg_2024_2026_t'] = float(np.median([tstat(x[x.index >= '2024-01-01']) for x in X.values()]))
yr = pd.DataFrame({k: x.groupby(x.index.year).mean() * 100 for k, x in X.items()}).median(axis=1)
out['yearly_pp_jmed'] = {int(k): round(float(v), 3) for k, v in yr.items()}
out['NW3_jmed'] = float(np.median([nw_t(x) for x in X.values()]))
out['drop_year_min_jmed'] = min((float(np.median([tstat(x[x.index.year != y]) for x in X.values()])), y) for y in range(2019, 2027))
Pp = {k: ser(P, k) for k in Js}
out['combined_2014_2026_jmed_t'] = float(np.median([tstat(pd.concat([Pp[k], X[k]])) for k in Js]))
cor = pd.DataFrame(X).corr().to_numpy(); out['corr_between_j_median'] = float(np.median(cor[np.triu_indices(13, 1)]))
# 安慰劑校準：拿第 s 次隨機當「實際」，對其餘 999 次平均
tt = []
for k in Js:
    rr = M[f'{k}|rnd']; tot = rr.sum(axis=0)
    for s in range(0, 1000, 5):
        tt.append(tstat(rr[s] - (tot - rr[s]) / 999))
tt = np.array(tt); out['placebo_t'] = dict(mean=float(tt.mean()), sd=float(tt.std()), p95=float(np.percentile(tt, 95)), frac_ge2=float((tt >= 2).mean()))
print(json.dumps(out, ensure_ascii=False, indent=1))
json.dump(out, open('out_desc_a.json', 'w'), ensure_ascii=False, indent=1)
