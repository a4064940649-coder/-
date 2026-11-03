#!/usr/bin/env python3
"""可執行統計（13 個 j 中位數）。"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from r3_stats import sl, st, med, PER, anchor_month
OUT = Path(__file__).resolve().parent / 'out'
E = {j: pd.read_parquet(OUT / f'exec_J{j}.parquet') for j in range(13)}
P = {j: pd.read_parquet(OUT / f'paper_J{j}.parquet') for j in range(13)}
for j in range(13): E[j]['0050'] = P[j]['0050'].reindex(E[j].index)

def tab(col, pers=('SEL', 'VAL', 'POST19', 'PIT', 'ALL')):
    out = {}
    for p in pers:
        rows = []
        for j in range(13):
            d = sl(E[j], p); rows.append(st(d[col].to_numpy(), d['0050'].to_numpy()))
        out[p] = med(rows)
    return out

def geo_excess(col, a, b):
    vs = []
    for j in range(13):
        d = E[j]; am = anchor_month(d.index); y = am.year
        m = (y >= a) & (y <= b); x = d[col][m]; z = d['0050'][m]; n = m.sum()
        vs.append(100 * ((np.prod(1 + x) ** (12 / n)) - (np.prod(1 + z) ** (12 / n))))
    return float(np.median(vs))

def drop_best3(col, p):
    ts = []
    for j in range(13):
        d = sl(E[j], p); ex = d[col] - d['0050']; yr = anchor_month(d.index).year
        best = ex.groupby(yr).sum().sort_values().index[-3:]
        e2 = ex[~np.isin(yr, best)]; ts.append(e2.mean() / e2.std(ddof=1) * np.sqrt(len(e2)))
    return float(np.median(ts))

if __name__ == '__main__':
    R = {}
    cols = ['SEL|100萬|GROW|normal', 'SEL|100萬|GROW|stress', 'SEL|30萬|FIX|normal', 'SEL|300萬|GROW|normal', 'SEL|1000萬|GROW|normal',
            'B|100萬|GROW|normal', 'SEL_PURE|100萬|GROW|normal', 'CTRL|100萬|GROW|normal', 'CTRL|100萬|GROW|stress', 'CTRL_PURE|100萬|GROW|normal', 'CTRL_PURE|100萬|GROW|stress']
    for c in cols: R[c] = tab(c)
    k = 'SEL|100萬|GROW|normal'
    R['geo3'] = {f'{a}-{b}': geo_excess(k, a, b) for a, b in ((2006, 2012), (2013, 2018), (2019, 2026))}
    R['drop_best3_t'] = {p: drop_best3(k, p) for p in ('PIT', 'ALL')}
    R['cost_pp_month_PIT'] = float(np.median([100 * sl(E[j], 'PIT')[k + '|cost'].mean() for j in range(13)]))
    R['odd_share_PIT'] = float(np.median([sl(E[j], 'PIT')[k + '|odd'].mean() for j in range(13)]))
    R['stress_ALL_t'] = R['SEL|100萬|GROW|stress']['ALL']['t']
    json.dump(R, open(OUT / 'exec_stats.json', 'w'), ensure_ascii=False, indent=1)
    for c, v in R.items():
        if isinstance(v, dict) and all(isinstance(x, dict) for x in v.values()):
            print(c)
            for p, s in v.items(): print('   ', p, {a: round(b, 4) for a, b in s.items()})
        else:
            print(c, v)
