#!/usr/bin/env python3
"""看完主結果後加的描述性檢查（不是事前登記，不改判定）：寫入 results.json 的 POSTHOC_not_preregistered。"""
import json, math
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
res = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))
E = pd.read_csv(HERE / 'events_grouped.csv.gz', parse_dates=['date'], dtype={'code': str})
E['m'] = E['date'].dt.to_period('M')
def tstat(x):
    x = pd.Series(x).dropna()
    return round(float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))), 2) if len(x) > 2 and x.std() > 0 else None
def st(e):
    if len(e) == 0: return {'n': 0}
    x = e.net; mm = x.groupby(e.m).mean()
    return {'n': int(len(e)), 'mean_net_pct': round(x.mean() * 100, 2), 'median_net_pct': round(x.median() * 100, 2), 'win': round(float((x > 0).mean()), 3),
            't_month': tstat(mm), 'mean_net_ex_top5pct': round(x[x <= x.quantile(.95)].mean() * 100, 2) if len(e) >= 5 else None}
def dif(e):
    A = e[e.grp == 'G1']; B = e[e.grp != 'G1']
    dd = (A.groupby('m').net.mean() - B.groupby('m').net.mean()).dropna()
    return {'month_diff_mean_pct': round(dd.mean() * 100, 2), 'month_diff_t': tstat(dd), 'months': int(len(dd))}
M = E[E.date <= '2021-09-30']; X = M[M.date <= '2020-12-31']
ph = {'note': '看完主結果後才算，只是描述，不是證據'}
ph['excl_2021_2006-04_2020-12'] = {g: st(X[X.grp == g]) for g in ('G1', 'G2', 'G3')} | {'diff': dif(X)}
ph['only_2021-01_2021-09'] = {g: st(M[(M.date >= '2021-01-01') & (M.grp == g)]) for g in ('G1', 'G2', 'G3')}
b = pd.cut(M.main_rank, [0, 30, 100, 200, 400], labels=['rank1-30', 'rank31-100', 'rank101-200', 'rank>200'])
ph['by_rank_bucket_main'] = {str(k): st(M[b == k]) for k in b.cat.categories} | {'not_in_pool(G3)': st(M[M.grp == 'G3'])}
ph['G1_share_of_G1_ab_from_2021'] = round(float(M[(M.grp == 'G1') & (M.date >= '2021-01-01')].ab.sum() / M[M.grp == 'G1'].ab.sum()), 2)
ph['G1_events_by_year'] = M[M.grp == 'G1'].groupby(M.date.dt.year).size().astype(int).to_dict()
res['POSTHOC_not_preregistered'] = ph
(HERE / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=lambda x: None if isinstance(x, float) and not np.isfinite(x) else str(x)), encoding='utf-8')
print(json.dumps(ph, ensure_ascii=False, indent=0, default=str))
