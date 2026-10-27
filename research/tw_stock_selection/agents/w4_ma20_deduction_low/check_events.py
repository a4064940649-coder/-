#!/usr/bin/env python3
"""抽樣核對主規格事件＋診斷（訊號日後 MA20 是否續揚、隔夜跳空、逐年）。"""
import sys; sys.path.insert(0, '/home/user/-/research/tw_stock_selection/agents/w4_ma20_deduction_low')
import run as R, numpy as np, pandas as pd
df = pd.read_csv(R.L.P.parent / 'w4_ded_K5_none_MA20_events.csv.gz', parse_dates=['sig', 'entry'], dtype={'code': str})
print('hold dist', df['hold'].value_counts().sort_index().head(8).to_dict(), 'mean', round(df.hold.mean(), 1))
# MA20 在訊號後 5 日是否真的續揚（扣抵低的機械含義）
up5 = []
for _, r in df.sample(3000, random_state=1).iterrows():
    j = R.cols.index(r['code']); t = R.IDX.get_loc(r['sig'])
    if t + 5 < R.T: up5.append(R.MA20n[t + 5, j] > R.MA20n[t, j])
print('share MA20_{t+5} > MA20_t', round(float(np.mean(up5)), 3))
# 進場用訊號日收盤（做不到）的上限診斷
for a, b in [('2005-01-01', '2019-04-30'), ('2019-05-01', '2026-10-31')]:
    s = df[(df.sig >= a) & (df.sig <= b)]
    ab_close = (1 + s['raw']) * (1 + s['gap']) - 1 - s['ew']
    gm = ab_close.groupby(s['entry'].dt.to_period('M')).mean()
    print(a, 'gross ab if entered at signal close (infeasible) %.3f%%' % (ab_close.mean() * 100), 't', R.tstat(gm))
row = df.sample(1, random_state=11).iloc[0]; print(row.to_dict())
c = row['code']; j = R.cols.index(c); t = R.IDX.get_loc(row['sig'])
a = R.ADJ[c]
print('MA20 t-2,t-1,t', a.iloc[t-21:t-1].mean(), a.iloc[t-20:t].mean(), a.iloc[t-19:t+1].mean(), 'code MA20', R.MA20[c].iloc[t-2:t+1].tolist())
print('deduct t-19..t-15', a.iloc[t-19:t-14].tolist(), 'close_t', a.iloc[t], 'F60 window max', a.iloc[t-59:t-54].max())
print('V60', R.V60[c].iloc[t], 'close raw', R.CR[c].iloc[t], 'flag', R.F[c].iloc[t])
e = t + 1; p0 = R.AOn[e, j]; x = e + int(row['hold'])
seg = pd.DataFrame({'adj': R.ADJn[e:x, j], 'ma20': R.MA20n[e:x, j]}, index=R.IDX[e:x]); print(seg)
print('entry', p0, 'exit px', R.PX[x, j], R.IDX[x], 'raw', R.PX[x, j] / p0 - 1, 'csv raw', row['raw'])
