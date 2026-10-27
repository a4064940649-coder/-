#!/usr/bin/env python3
"""抽樣手算核對 VR／PSY 與主規格事件。"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
P = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel')
adj = pd.read_parquet(P / 'adj.parquet'); C = pd.read_parquet(P / 'close_raw.parquet'); V = pd.read_parquet(P / 'value.parquet'); O = pd.read_parquet(P / 'open.parquet')
good = adj.notna().sum(axis=1) >= 300
adj, C, V, O = adj[good], C[good], V[good], O[good]
ev = pd.read_csv(P.parent / 'w4_vr_VR_L60_none_X350_H20_events.csv.gz', dtype={'code': str}, parse_dates=['sig', 'entry'])
print('events', len(ev), 'select', (ev.sig <= '2019-04-30').sum(), 'valid', (ev.sig >= '2019-05-01').sum())
print(ev.describe().T[['mean', '50%', 'min', 'max']])
rng = np.random.default_rng(1)
for k in rng.choice(len(ev), 3, replace=False):
    r = ev.iloc[k]; c = r.code; i = adj.index.get_loc(r.sig)
    def vr(ii):
        a = adj[c].iloc[ii - 26:ii + 1]; ret = a.pct_change().iloc[1:]; vol = (V[c] / C[c]).iloc[ii - 25:ii + 1]
        u = vol[ret > 1e-9].sum(); d = vol[ret < -1e-9].sum(); f = vol[(ret.abs() <= 1e-9)].sum()
        return (u + 0.5 * f) / (d + 0.5 * f) * 100
    e = adj.index.get_loc(r.entry)
    p0 = O[c].iloc[e] * adj[c].iloc[e] / C[c].iloc[e]
    print(c, r.sig.date(), 'VR t-1 %.1f  t %.1f' % (vr(i - 1), vr(i)), 'entry', r.entry.date(), 'hold', r.hold, r.why,
          'p0 %.4f' % p0, 'exit open', adj.index[e + r.hold].date(), 'raw %.4f' % r.raw,
          'hand raw %.4f' % (O[c].iloc[e + r.hold] * adj[c].iloc[e + r.hold] / C[c].iloc[e + r.hold] / p0 - 1))
# 極端值
print('top raw'); print(ev.nlargest(5, 'raw')[['code', 'sig', 'raw', 'ew']])
print('bottom raw'); print(ev.nsmallest(5, 'raw')[['code', 'sig', 'raw', 'ew']])
# 去極端值（winsorize 1%）後的平均淨異常
for a, b in [('2005-01-01', '2019-04-30'), ('2019-05-01', '2026-10-31')]:
    s = ev[(ev.sig >= a) & (ev.sig <= b)]
    w = s['ab'].clip(s['ab'].quantile(0.01), s['ab'].quantile(0.99))
    print(a, 'ab mean %.3f%% winsor %.3f%% median %.3f%%' % (s.ab.mean() * 100, w.mean() * 100, s.ab.median() * 100),
          'net-of-EW gross %.3f%%' % (s.ab_gross.mean() * 100))
