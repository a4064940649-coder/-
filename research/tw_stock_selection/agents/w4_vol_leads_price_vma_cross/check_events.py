#!/usr/bin/env python3
"""抽樣核對主規格事件＋診斷（持有天數分布、訊號日收盤相對 MA20 位置）。"""
import sys; sys.path.insert(0, '/home/user/-/research/tw_stock_selection/agents/w4_vol_leads_price_vma_cross')
import run as R, numpy as np, pandas as pd
df = pd.read_csv(R.L.P.parent / 'w4_vlp_VL_H20_events.csv.gz', parse_dates=['sig', 'entry'], dtype={'code': str})
print('hold dist', df['hold'].value_counts().sort_index().head(8).to_dict())
pos = []
for _, r in df.iterrows():
    j = R.cols.index(r['code']); t = R.IDX.get_loc(r['sig'])
    pos.append(R.ADJn[t, j] / R.MA20n[t, j] - 1)
df['c_vs_ma20'] = pos
print('signal-day close below MA20 share', round(float((df['c_vs_ma20'] < 0).mean()), 3), 'median dist %', round(df['c_vs_ma20'].median() * 100, 2))
for a, b in [('2005-01-01', '2019-04-30'), ('2019-05-01', '2026-10-31')]:
    s = df[(df.sig >= a) & (df.sig <= b)]
    for nm, g in [('close<MA20', s[s.c_vs_ma20 < 0]), ('close>=MA20', s[s.c_vs_ma20 >= 0])]:
        gm = g.groupby(g['entry'].dt.to_period('M'))['ab'].mean()
        print(a, nm, len(g), 'ab_net %.3f%%' % (g['ab'].mean() * 100), 't', R.tstat(gm), 'hold', round(g['hold'].mean(), 1))
row = df.sample(1, random_state=7).iloc[0]; print(row.to_dict())
c = row['code']; j = R.cols.index(c); t = R.IDX.get_loc(row['sig'])
v5, v20, m5, m20, m60 = R.VMA5[c], R.VMA20[c], R.MA5[c], R.MA20[c], R.MA60[c]
print('VMA5 t-1,t', v5.iloc[t-1], v5.iloc[t], 'VMA20 t-1,t', v20.iloc[t-1], v20.iloc[t])
print('manual VMA5_t', (R.V[c] / R.CR[c]).iloc[t-4:t+1].mean(), 'MA5 t-1,t', m5.iloc[t-1], m5.iloc[t], 'MA20', m20.iloc[t], 'MA20 t-5', m20.iloc[t-5], 'MA60', m60.iloc[t], m60.iloc[t-20])
print('V60', R.V60[c].iloc[t], 'close', R.CR[c].iloc[t])
e = t + 1; p0 = R.AOn[e, j]; x = e + int(row['hold'])
seg = pd.DataFrame({'adj': R.ADJn[e:x, j], 'ma20': R.MA20n[e:x, j], 'stop': p0 * 0.93}, index=R.IDX[e:x]); print(seg)
print('entry', p0, 'exit px', R.PX[x, j], R.IDX[x], 'raw', R.PX[x, j] / p0 - 1, 'csv raw', row['raw'])
