#!/usr/bin/env python3
"""反方審查（資料與統計）：只讀研究員檔案，不改寫。"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; DATA = HERE / 'data'
def tstat(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return x.mean() / (x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 2 else np.nan
E = pd.read_parquet(DATA / 'events.parquet')
pz = np.load(DATA / 'event_paths.npz'); PATH, EWP = pz['PATH'], pz['EWP']
m = ((E['k'] >= 2019 * 4 + 1) & E['SUEP'].notna()).to_numpy()
E = E[m].copy(); PATH = PATH[m]; EWP = EWP[m]
E['grp'] = np.where(E['in0050'], '0050', np.where(E['att'] == 1, 'low', np.where(E['att'] == 2, 'mid', 'high')))
E['ab60'] = E['r60_j0'] - E.groupby(['k', 'grp'])['r60_j0'].transform('mean')
# 1. 同日（日曆對齊）基準：PATH[60] − EWP[60]（EWP 起點 e−1 收盤，近似）
E['abd'] = PATH[:, 60] - EWP[:, 60]
nf = ~E['in0050']; top = E['top_SUEP']
def rep(name, d, col):
    s = d.groupby('k')[col].mean(); dd = d.groupby('A')[col].mean()
    print(f'{name:45s} mean {d[col].mean():+.4f} seasonmean {s.mean():+.4f} t_season {tstat(s):5.2f} t_date {tstat(dd):5.2f} n {d[col].notna().sum()}')
rep('nonfifty top ab60 (k,grp mean)', E[nf & top], 'ab60')
rep('nonfifty top abd (same-date pool EW)', E[nf & top], 'abd')
rep('nonfifty ALL abd (same-date pool EW)', E[nf], 'abd')
E['abd_dm'] = E['abd'] - E.groupby(['k'])['abd'].transform('mean')
rep('nonfifty top abd demeaned by season', E[nf & top], 'abd_dm')
# 時間差：top 是否較早公告
E['dpos'] = E.groupby('k')['A'].rank(pct=True)
print('announcement-date pct rank: top', E.loc[nf & top, 'dpos'].mean().round(3), 'rest', E.loc[nf & ~top, 'dpos'].mean().round(3))
# 2. 依 A_method
for mth, g in E[nf & top].groupby('A_method'):
    print('A_method', mth, len(g), 'ab60 mean %.4f' % g['ab60'].mean(), 't_season %.2f' % tstat(g.groupby('k')['ab60'].mean()))
print('A_method share top vs all:', (E.loc[nf & top, 'A_method'].value_counts(normalize=True).round(3)).to_dict(), (E.loc[nf, 'A_method'].value_counts(normalize=True).round(3)).to_dict())
rep('nonfifty top ab60 excl deadline-method', E[nf & top & (E['A_method'] != 'deadline')], 'ab60')
rep('nonfifty top ab60 fsq only', E[nf & top & (E['A_method'] == 'fsq')], 'ab60')
# 3. 重疊：top − bottom 季差 & 子期
for nm, mm in [('VAL', E['k'] <= 2023 * 4 + 3), ('HOLD', E['k'] >= 2024 * 4)]:
    rep('nonfifty top ab60 ' + nm, E[nf & top & mm], 'ab60')
# 4. 季別依賴
s = E[nf & top].groupby('k')['ab60'].mean()
print('season means sorted:', s.sort_values().round(4).to_dict())
for drop in [1, 2, 3]:
    ss = s.sort_values()[:-drop]; print(f'drop best {drop} seasons: mean {ss.mean():+.4f} t {tstat(ss):.2f}')
# 5. 0050 組（事後新題 A）
g = E[(E['grp'] == '0050') & top]
print('0050 top n', len(g), 'seasons', g['k'].nunique(), 'unique codes', g['code'].nunique())
c = g.groupby('code')['ab60'].sum().sort_values()
print('0050 top contrib top5', c.tail(5).round(3).to_dict())
for n in [1, 3, 5]:
    gg = g[~g['code'].isin(c.index[-n:])]; sk = gg.groupby('k')['ab60'].mean()
    print(f'0050 drop top{n}: mean {sk.mean():+.4f} t {tstat(sk):.2f}  date-t {tstat(gg.groupby("A")["ab60"].mean()):.2f}')
rep('0050 top abd same-date', g, 'abd')
# 6. 低 − 高 用同日基準
a = E[(E['grp'] == 'low') & top].groupby('k')['abd'].mean(); b = E[E['grp'].isin(['high', '0050']) & top].groupby('k')['abd'].mean()
d = (a - b).dropna(); print('low − high (same-date abd)', round(d.mean(), 4), round(tstat(d), 2))
# 7. 下市/無出場
print('nonfifty top r60 NaN share', E.loc[nf & top, 'r60_j0'].isna().mean().round(4))
# 8. 缺值來源
t8 = E[nf & top]
print('NaN r60 by k (share):', t8.groupby('k')['r60_j0'].apply(lambda s: round(s.isna().mean(), 3)).to_dict())
print('e_j0<0 share', (t8['e_j0'] < 0).mean().round(4))
# 9. 上市 vs 上櫃 × 方法
for (mk, mt), g in t8.groupby(['mkt', 'A_method']):
    if len(g) > 50: print(mk, mt, len(g), 'ab60 %.4f' % g['ab60'].mean(), 't %.2f' % tstat(g.groupby('k')['ab60'].mean()))
