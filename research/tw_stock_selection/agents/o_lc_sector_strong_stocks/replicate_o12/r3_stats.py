#!/usr/bin/env python3
"""紙上統計（13 個 j 中位數）＋與研究員窗報酬逐窗比對。"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
OUT = Path(__file__).resolve().parent / 'out'
PER = {'SEL': ('2014-01', '2016-12'), 'VAL': ('2017-01', '2026-10'), 'POST19': ('2019-01', '2026-10'),
       'PIT': ('2014-01', '2026-10'), 'ALL': ('2006-04', '2026-10')}
def anchor_month(ix):
    # 錨定月：訊號日 d 的月份，但跨月者（d 日 < 10 且屬下月）歸上一月
    m = ix.to_period('M')
    return pd.PeriodIndex([mm - 1 if d.day < 10 else mm for d, mm in zip(ix, m)])
def sl(df, p):
    a, b = PER[p]; am = anchor_month(df.index)
    return df[(am >= pd.Period(a, 'M')) & (am <= pd.Period(b, 'M'))]
def st(x, b=None):
    x = np.asarray(x, float); n = len(x)
    ex = x - (b if b is not None else 0)
    m = ex.mean(); sd = ex.std(ddof=1)
    out = dict(n=n, mean_pp=100 * m, t=m / sd * np.sqrt(n), IR=m / sd * np.sqrt(12), TE=sd * np.sqrt(12))
    out['ann'] = np.prod(1 + x) ** (12 / n) - 1
    if b is not None: out['ann_b'] = np.prod(1 + b) ** (12 / n) - 1
    nav = np.cumprod(1 + x); out['mdd'] = (nav / np.maximum.accumulate(np.concatenate([[1], nav]))[1:] - 1).min()
    return out
def med(rows):
    keys = rows[0].keys(); return {k: float(np.median([r[k] for r in rows])) for k in keys}

P = {j: pd.read_parquet(OUT / f'paper_J{j}.parquet') for j in range(13)}
def table(col, bench='0050', cost='normal', pers=('SEL', 'VAL', 'POST19', 'PIT', 'ALL')):
    res = {}
    for p in pers:
        rows = []
        for j in range(13):
            d = sl(P[j], p)
            b = d[bench] if bench == '0050' else d[f'{bench}|{cost}']
            rows.append(st(d[f'{col}|{cost}'], b.to_numpy()))
        res[p] = med(rows)
    return res

if __name__ == '__main__':
    R = {}
    for col in ['IND_n2_cap_c15_SAT', 'IND_n2_cap_c15_PURE', 'CTRL_n2_cap_c15_SAT', 'CTRL_n2_cap_c15_PURE', 'IND_ONLY_PURE', 'IND_ONLY_SAT']:
        for bench in ('0050', 'B'):
            R[f'{col} vs {bench}'] = table(col, bench)
    R['B vs 0050'] = table('B', '0050')
    R['SAT stress vs 0050'] = table('IND_n2_cap_c15_SAT', '0050', 'stress')
    json.dump(R, open(OUT / 'paper_stats.json', 'w'), ensure_ascii=False, indent=1)
    for k, v in R.items():
        print(k)
        for p, s in v.items(): print('   ', p, {a: round(b, 4) for a, b in s.items()})
