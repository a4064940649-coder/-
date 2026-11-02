"""j 分布統計：軌 A（凍結引擎）與軌 B（自寫引擎、自寫 day_for／unavail_for）；AB 40 seeds 平均。"""
import sys, json; from pathlib import Path; HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd, rb_core as B
OUT = HERE / 'out'; res = {}
JA = json.load(open(OUT / 'jdays.json')) if (OUT / 'jdays.json').exists() else {}; NA = pd.read_parquet(OUT / 'navs_jscan_a.parquet')
fb = OUT / 'navs_jscan_b.parquet'
NB = pd.read_parquet(fb) if fb.exists() else None; JB = json.load(open(OUT / 'jscan_b.json'))['jdays'] if NB is not None else None
R = pd.read_parquet(HERE.parent / 'navs_jscan_v7.parquet')
def st(nav, sd, a='2006-04-01'):
    return B.seg(nav.dropna(), pd.DatetimeIndex(sd), a, '2026-10-12')
for k in ('V7', 'V7_B10'):
    for c in ('normal', 'stress'):
        for trk in ('A', 'B'):
            if trk == 'B' and NB is None: continue
            if trk == 'A' and f'AB_{k}|j12|{c}' not in NA: continue
            ts, cg, t19, dmax = [], [], [], 0.0
            for j in range(13):
                if trk == 'A': nav = NA[f'AB_{k}|j{j}|{c}'].dropna(); sd = JA[f'j{j}']
                else: nav = B.avg_nav([NB[f'{k}|j{j}|{c}|s{s}'].dropna() for s in range(40)]); sd = JB[f'j{j}']
                s = st(nav, sd); ts.append(s['t']); cg.append(s['CAGR']); t19.append(st(nav, sd, '2019-01-01')['t'])
                rc = f'AB_{k}|{c}|J{j}'
                if rc in R: x = R[rc].dropna(); dmax = max(dmax, float((x - nav.reindex(x.index)).abs().max()))
            res[f'{trk}|AB_{k}|{c}'] = dict(t_median=round(float(np.median(ts)), 3), t_p10=round(float(np.percentile(ts, 10)), 3), t_min=round(min(ts), 3), t_max=round(max(ts), 3),
                                            cagr_median=round(float(np.median(cg)), 2), t2019_median=round(float(np.median(t19)), 3), maxdiff_vs_researcher_nav=dmax,
                                            t_by_j=[round(x, 3) for x in ts])
json.dump(res, open(OUT / 'jstats.json', 'w'), indent=1); print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 't_by_j'} for k, v in res.items()}, indent=1))
