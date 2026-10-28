#!/usr/bin/env python3
"""反方審查（資料與統計，o4）：只讀研究員輸出，不改任何研究員檔案。
用 navs_oos.parquet 與 explore_rows.csv 重算配對差、年份集中、區塊 bootstrap、P2a 對照組敏感度。"""
import sys, json, math
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
nav = pd.read_parquet(HERE / 'navs_oos.parquet')
idx = nav.index
# 訊號日＝每月第一個日 >= 11 的交易日
sd = pd.Series(idx, index=idx).groupby([idx.year, idx.month]).apply(lambda s: s[s.dt.day >= 11].iloc[0] if (s.dt.day >= 11).any() else pd.NaT).dropna()
sd = pd.DatetimeIndex(sd.values)
ex = pd.read_csv(HERE / 'explore_rows.csv', parse_dates=['d'])
seld = pd.DatetimeIndex(sorted(ex.d.unique()))
print('signal days SEL match explore:', set(seld) <= set(sd), len(seld), 'sd total', len(sd), sd[0], sd[-1])

def win(n):
    x = n.reindex(sd); return (x.shift(-1) / x - 1).iloc[:-1]
def t(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x))))
def nw_t(x, L=3):
    x = np.asarray(x, float); x = x[np.isfinite(x)]; n = len(x); m = x.mean(); e = x - m
    g0 = (e @ e) / n; s = g0
    for l in range(1, L + 1): s += 2 * (1 - l / (L + 1)) * (e[l:] @ e[:-l]) / n
    return float(m / math.sqrt(s / n))
def boot(x, B=5000, blk=6, seed=0):
    x = np.asarray(x, float); n = len(x); rng = np.random.default_rng(seed); ms = []
    for _ in range(B):
        st = rng.integers(0, n - blk + 1, size=n // blk + 1)
        ms.append(np.concatenate([x[s:s + blk] for s in st])[:n].mean())
    ms = np.array(ms); return dict(p_gt0=float((ms > 0).mean()), ci90=[float(np.percentile(ms, 5) * 100), float(np.percentile(ms, 95) * 100)])
PER = {'SEL': ('2006-01-01', '2018-12-31'), 'VAL': ('2019-01-01', '2023-12-31'), 'HOLD': ('2024-01-01', '2026-10-12'),
       'POST': ('2019-01-01', '2026-10-12'), 'ALL': ('2006-01-01', '2026-10-12')}
out = {}
for c in ('gross', 'normal', 'stress', 'cap10m'):
    d = (win(nav[f'A20|{c}']) - win(nav[f'BASE|{c}'])).dropna()
    for p, (a, b) in PER.items():
        x = d[(d.index >= a) & (d.index <= b)]
        out[f'{c}|{p}'] = dict(n=len(x), mean_pp=round(x.mean() * 100, 4), t=round(t(x), 3), nw3=round(nw_t(x), 3), **boot(x.to_numpy()))
for k, v in out.items(): print('paired A20-BASE', k, v)
# 年份集中：一般成本配對差逐年加總（pp）
d = (win(nav['A20|normal']) - win(nav['BASE|normal'])).dropna()
ys = d.groupby(d.index.year).sum() * 100
print('yearly paired sum pp (normal):', ys.round(2).to_dict())
xs = d[(d.index <= '2018-12-31')]
yy = xs.groupby(xs.index.year)
print('SEL drop-one-year t:', {int(y): round(t(xs[xs.index.year != y]), 2) for y in sorted(set(xs.index.year))})
print('SEL ex 2006-2007 mean/t:', round(xs[xs.index.year >= 2008].mean() * 100, 3), round(t(xs[xs.index.year >= 2008]), 2))
# 回撤主要來自哪一段
for nm in ('BASE|normal', 'A20|normal'):
    s = nav[nm]; dd = s / s.cummax() - 1; i = dd.idxmin(); pk = s[:i].idxmax()
    print(nm, 'maxDD', round(dd.min() * 100, 1), 'peak', pk.date(), 'trough', i.date())
    s2 = s[s.index >= '2009-06-01']; dd2 = s2 / s2.cummax() - 1; print('   maxDD after 2009-06', round(dd2.min() * 100, 1), dd2.idxmin().date())

# ---------- P2a 對照組敏感度（explore_rows：扣成本前、未套 X 出清） ----------
ex['rk'] = ex['rank']
new = ex[ex.grp == 'new']; oo = ex[ex.grp == 'old_out']
p2 = oo[oo.rk.between(11, 30) & (oo.s52 >= 0.90) & oo.r.notna()]
def cmpg(g_obs, ref):
    g = g_obs.groupby('i')['r'].mean(); r = ref.groupby('i')['r'].mean(); dd = (g - r.reindex(g.index)).dropna()
    return dict(n_months=len(dd), mean_pp=round(dd.mean() * 100, 3), t=round(t(dd), 2), nw3=round(nw_t(dd), 2))
print('P2a vs all new (replicate):', cmpg(p2, new))
# A20 實際被擠掉的是當月「排名最差的新進股」：對照組改成每月新進中排名最差的 k 檔
for k in (1, 2, 3):
    marg = new.sort_values(['i', 'rk']).groupby('i').tail(k)
    print(f'P2a vs worst-{k} ranked new:', cmpg(p2, marg))
# 新進股排名與報酬
print('new by rank (mean pp per obs):', (new.groupby(pd.cut(new.rk, [0, 3, 6, 8, 10, 200]))['r'].mean() * 100).round(2).to_dict())
# 股票層級：P2a 的貢獻集中度
p2m = p2.merge(new.groupby('i')['r'].mean().rename('rn'), left_on='i', right_index=True)
p2m['ex'] = p2m.r - p2m.rn
print('P2a stock-level n', len(p2m), 'mean ex pp', round(p2m.ex.mean() * 100, 3), 'median', round(p2m.ex.median() * 100, 3))
cs = p2m.groupby('j')['ex'].sum().sort_values()
print('P2a worst 5 stocks sum pp', (cs.head(5) * 100).round(1).to_dict(), 'total', round(cs.sum() * 100, 1))
tot_months = p2m.groupby('i')['ex'].mean()
g = tot_months.sort_values()
print('P2a drop worst 1/3/5 months t:', [round(t(g.iloc[k:]), 2) for k in (1, 3, 5)])
yrs = pd.Series([pd.Timestamp(seld[i]).year for i in tot_months.index], index=tot_months.index)
print('P2a by year mean pp / n:', {int(y): (round(v.mean() * 100, 2), len(v)) for y, v in tot_months.groupby(yrs)})
# 持有月數 2 的檢定
ag2 = ex[(ex.grp == 'cont') & (ex.age == 2)]
print('age2 vs new:', cmpg(ag2, new))
cont = ex[ex.grp == 'cont']
print('cont vs new:', cmpg(cont, new))
# P2a 是預先登記的檢定，但在同一份探索裡同時看了多少格？
print('number of cell tests in explore.json:', len(json.loads((HERE / 'explore.json').read_text())['bucket_vs_new']))
