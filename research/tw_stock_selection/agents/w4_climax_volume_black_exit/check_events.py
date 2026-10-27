#!/usr/bin/env python3
"""事後抽樣核對（不影響判定）：PRIMARY 出場事件逐筆重算觸發條件；看極端值。"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
P = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel')
ev = pd.read_csv('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/w4_climax_events.csv.gz', dtype={'code': str}, parse_dates=['trig', 'sell'])
ev = ev[ev['tag'] == 'n30_V20xBLK_slip0.001_cash']
adj = pd.read_parquet(P / 'adj.parquet'); C = pd.read_parquet(P / 'close_raw.parquet'); O = pd.read_parquet(P / 'open.parquet')
V = pd.read_parquet(P / 'value.parquet'); R = pd.read_parquet(P / 'ret.parquet'); S = pd.read_parquet(P / 'src.parquet')
good = adj.notna().sum(axis=1) >= 300
rng = np.random.default_rng(0)
samp = ev.iloc[rng.choice(len(ev), 8, replace=False)]
for _, e in samp.iterrows():
    c = e['code']; t = e['trig']
    a = adj.loc[good, c].dropna(); vol = (V.loc[good, c] / C.loc[good, c]).dropna()
    i = a.index.get_loc(t); iv = vol.index.get_loc(t)
    mn = a.iloc[max(0, i - 60):i].min(); av = vol.iloc[max(0, iv - 20):iv].mean()
    print(c, t.date(), 'sell', e['sell'].date(), f"open {O.at[t,c]} close {C.at[t,c]} ret {R.at[t,c]:.3%} up_from_min60 {a.iloc[i]/mn-1:.1%} vol/avg20 {vol.iloc[iv]/av:.2f} src {S.at[t,c]} r_stock {e['r_stock']:.2%}")
print('\nr_stock 分位數', ev['r_stock'].quantile([0, .05, .25, .5, .75, .95, 1]).round(3).to_dict())
print('最大 5 筆', ev.nlargest(5, 'r_stock')[['code', 'trig', 'sell', 'r_stock', 'repicked']].to_string())
print('最小 5 筆', ev.nsmallest(5, 'r_stock')[['code', 'trig', 'sell', 'r_stock', 'repicked']].to_string())
for nm, a, b in (('select', '2006-01-01', '2019-04-30'), ('valid', '2019-05-01', '2026-12-31')):
    e = ev[(ev['trig'] >= a) & (ev['trig'] <= b)]
    w = e['r_stock'].clip(e['r_stock'].quantile(.05), e['r_stock'].quantile(.95))
    print(nm, 'winsor5% r_stock mean', round(w.mean() * 100, 2), 'excl top5%', round(e['r_stock'][e['r_stock'] < e['r_stock'].quantile(.95)].mean() * 100, 2))
