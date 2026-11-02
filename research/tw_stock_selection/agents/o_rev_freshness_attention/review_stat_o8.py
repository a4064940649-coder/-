#!/usr/bin/env python3
"""反方審查（資料與統計，o8）：只讀研究員的 data/alpha_all_v1.parquet、contrib_all_v1.parquet、results.json，不改任何檔案。
檢查：
  R1 重算 b_高、b_低、I（前期／後期／VAL／HOLD），與 results.json 對照
  R2 相鄰錨定月持有窗重疊 → Newey-West（lag 1..3）t；月斜率的一階自相關
  R3 去掉 2020～2021、去掉 2020-03／2020-04 等極端月；中位數與截尾平均
  R4 低組 VAL 斜率的 j 端點依賴：只用 j0..j6、j6..j12；首尾差
  R5 多重檢定：後期 b_低 / VAL b_低 在已報告檢定數下的 Bonferroni／Holm
  R6 每月 j 數、各 j 的月數（缺 j 是否偏向某些月份）
  R7 R2 by-j：配對差的正值是否只在後期（前期配對差）
"""
import json, math, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
A = pd.read_parquet(HERE / 'data' / 'alpha_all_v1.parquet')
R = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))
P = lambda s: pd.Period(s, 'M')
PER = {'PRE': (P('2006-04'), P('2018-12')), 'POST': (P('2019-05'), P('2026-10')), 'VAL': (P('2019-05'), P('2023-12')), 'HOLD': (P('2024-01'), P('2026-10'))}
JS = list(range(13))

def slope(d):
    js = np.array(sorted(d), float)
    if len(js) < 10: return np.nan
    y = np.array([d[int(j)] for j in js]); jc = js - js.mean()
    return float((jc * y).sum() / (jc ** 2).sum())

def mslopes(scheme, leg, g, col='alpha', per=None, jsub=None):
    x = A[(A.scheme == scheme) & (A.leg == leg) & (A.g == g)]
    if jsub is not None: x = x[x.j.isin(jsub)]
    mm = pd.PeriodIndex(x.m, freq='M')
    if per: x = x[(mm >= per[0]) & (mm <= per[1])]
    out = {}
    for m, gg in x.groupby('m'):
        d = dict(zip(gg.j.astype(int), gg[col]))
        if jsub is not None:
            if len(d) < len(jsub) - 1: continue
            js = np.array(sorted(d), float); y = np.array([d[int(j)] for j in js]); jc = js - js.mean()
            out[m] = float((jc * y).sum() / (jc ** 2).sum())
        else:
            out[m] = slope(d)
    s = pd.Series(out).dropna(); s.index = pd.PeriodIndex(s.index, freq='M'); return s.sort_index()

def t(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return float(x.mean() / x.std(ddof=1) * math.sqrt(len(x)))

def nw_t(x, L):
    x = np.asarray(x, float); x = x[np.isfinite(x)]; n = len(x); e = x - x.mean()
    v = (e @ e) / n
    for l in range(1, L + 1):
        w = 1 - l / (L + 1); v += 2 * w * (e[l:] @ e[:-l]) / n
    return float(x.mean() / math.sqrt(v / n))

def ac1(x): x = np.asarray(x, float); return float(np.corrcoef(x[1:], x[:-1])[0, 1])

O = {}
# R1／R2
for per in PER:
    bh = mslopes('A', 'main', 2, per=PER[per]); bl = mslopes('A', 'main', 0, per=PER[per])
    ix = bh.index.intersection(bl.index); I = bh[ix] - bl[ix]
    O[per] = {}
    for k, s in (('b_hi', bh), ('b_lo', bl), ('I', I)):
        O[per][k] = dict(mean_pp=round(s.mean() * 100, 4), n=len(s), t=round(t(s), 2), nw1=round(nw_t(s, 1), 2), nw3=round(nw_t(s, 3), 2), ac1=round(ac1(s), 3),
                         median_pp=round(s.median() * 100, 4))
print('R1/R2', json.dumps(O, ensure_ascii=False, indent=0))
rp = R['RESULTS_OOS']['primary']
print('reported post', {k: (round(rp['post'][k]['mean_pp'], 4), round(rp['post'][k]['t'], 2)) for k in ('b_hi', 'b_lo', 'I')})

# R3
bl_post = mslopes('A', 'main', 0, per=PER['POST']); bl_val = mslopes('A', 'main', 0, per=PER['VAL'])
avg_post = (mslopes('A', 'main', 0, per=PER['POST']) + mslopes('A', 'main', 1, per=PER['POST']) + mslopes('A', 'main', 2, per=PER['POST'])).dropna() / 3
def drop_years(s, ys): return s[[str(m)[:4] not in ys for m in s.index]]
print('R3 b_lo post by year mean_pp', (bl_post.groupby(bl_post.index.year).mean() * 100).round(3).to_dict())
print('R3 b_lo post by year n', bl_post.groupby(bl_post.index.year).size().to_dict())
for lab, s in (('b_lo_post', bl_post), ('b_lo_val', bl_val), ('avg3_post', avg_post)):
    s2 = drop_years(s, {'2020', '2021'})
    lo, hi = s.quantile(0.05), s.quantile(0.95); tr = s[(s >= lo) & (s <= hi)]
    worst = s.sort_values().head(5)
    print('R3', lab, 'drop2020-21', round(s2.mean() * 100, 4), round(t(s2), 2), 'n', len(s2), '| trim5%', round(tr.mean() * 100, 4), round(t(tr), 2),
          '| worst5', {str(k): round(v * 100, 3) for k, v in worst.items()}, '| share of sum from worst5', round(worst.sum() / s.sum(), 2))
# 符號檢定
for lab, s in (('b_lo_post', bl_post), ('b_lo_val', bl_val)):
    k = int((s < 0).sum()); n = len(s)
    from math import comb
    p = sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n
    print('R3 sign', lab, f'{k}/{n} negative, one-sided p={p:.3f}')

# R4 j 子區間
for lab, js in (('j0-6', range(0, 7)), ('j6-12', range(6, 13)), ('j1-12', range(1, 13))):
    s = mslopes('A', 'main', 0, per=PER['VAL'], jsub=list(js)); s2 = mslopes('A', 'main', 0, per=PER['POST'], jsub=list(js))
    print('R4 b_lo', lab, 'VAL', round(s.mean() * 100, 4), round(t(s), 2), '| POST', round(s2.mean() * 100, 4), round(t(s2), 2))

# R6 coverage
x = A[(A.scheme == 'A') & (A.leg == 'main') & (A.g == 0)]
cnt = x.groupby('m').j.nunique(); print('R6 months with <10 j', (cnt < 10).sum(), 'of', len(cnt), list(cnt[cnt < 10].index)[:20])
print('R6 months per j', x.groupby('j').m.nunique().to_dict())

# R5 multiple testing for b_lo post (two-sided p)
from math import erf
def p2(tv): return 2 * (1 - 0.5 * (1 + erf(abs(tv) / math.sqrt(2))))
for lab, s in (('b_lo_post', bl_post), ('b_lo_val', bl_val)):
    tv = nw_t(s, 3); print('R5', lab, 'NW3 t', round(tv, 2), 'p', round(p2(tv), 4), 'Bonferroni x14 tests', round(min(1, p2(tv) * 14), 3))

# R7 R2
c = R['RESULTS_R2']['cells']
for ck in ('K5T|normal|100萬|FIX', 'zero|normal'):
    bj = c[ck]['by_j']
    pre = [bj[str(j)]['diff_pre_mean_pp'] for j in range(1, 13)]; post = [bj[str(j)]['diff_post_mean_pp'] for j in range(1, 13)]
    print('R7', ck, 'pre diff median', round(float(np.median(pre)), 3), 'post diff median', round(float(np.median(post)), 3),
          'n_j pre>0', sum(v > 0 for v in pre), 'n_j post>0', sum(v > 0 for v in post),
          'median-of-CAGR diff vs diff-of-median', round(c[ck]['median_j']['CAGR_diff'], 2), round(c[ck]['median_j']['CAGR_R2'] - c[ck]['median_j']['CAGR_B10'], 2))
