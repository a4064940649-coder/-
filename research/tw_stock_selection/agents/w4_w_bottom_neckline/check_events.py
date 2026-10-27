#!/usr/bin/env python3
"""診斷（不影響判定）：抽樣核對型態、前視檢查、組合與事件平均差異的來源。"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import run as R

pats = R.find_patterns(); brk = R.triggers(pats, True)
rng = np.random.default_rng(0)
print('--- 抽樣 5 個帶量突破，逐項核對 ---')
pm = {}
for p in pats: pm.setdefault(p[0], []).append(p)
for k in rng.choice(len(brk), 5, replace=False):
    j, t, Nk, L2 = brk[k]
    cand = [p for p in pm[j] if p[6] == Nk and p[5] == L2 and p[3] < t <= p[3] + 30][0]
    _, p1, p2, c2, L1, L2_, _ = cand
    a = R.ADJn[:, j]
    print(R.cols[j], 'L1', R.IDX[p1].date(), round(L1, 2), 'L2', R.IDX[p2].date(), round(L2, 2), 'gap', p2 - p1, 'L2/L1', round(L2 / L1, 3),
          'N', round(Nk, 2), 'N/L1', round(Nk / L1, 3), 'pre', round(R.PREMAX[p1, j] / L1, 2), 'conf', R.IDX[c2].date(), 'trig', R.IDX[t].date(),
          'prev<=N', a[t - 1] <= Nk, 'close>N', a[t] > Nk, 'vol/vma', round(float(R.VOL.iloc[t, j] / R.VMA20p.iloc[t, j]), 2),
          'pivot1 ok', np.nanmin(a[p1 - 5:p1 + 6]) == L1, 'pivot2 ok', np.nanmin(a[p2 - 5:p2 + 6]) == L2, 'trig>conf', t > c2)
print('--- 前視檢查：所有突破日都在右腳確認日之後 ---')
ok = all(any(p[3] < t <= p[3] + 30 for p in pm[j] if p[6] == Nk) for j, t, Nk, L2 in brk)
print('all trig > confirm:', ok)
# 組合 vs 事件平均
ev, *_ = R.build_events(brk); df = R.event_table(ev)
sel = df[(df.sig >= '2005-01-01') & (df.sig <= '2019-04-30')]
print('--- 事件層（挑選期） ---')
print('mean raw', round(sel.raw.mean() * 100, 2), 'median raw', round(sel.raw.median() * 100, 2), 'mean log', round(np.log1p(sel.raw).mean() * 100, 2),
      'sum raw / sum hold (per day %)', round(sel.raw.sum() / sel.hold.sum() * 100, 3))
r, npos = R.portfolio(ev, 0.0, 0.0)
m = (r.index >= '2005-01-01') & (r.index <= '2019-04-30')
print('組合（毛）日均報酬 %', round(r[m].mean() * 100, 3), '日波動', round(r[m].std() * 100, 2))
# 時間加權 vs 事件加權：每日部位數與當日平均報酬的關係
x = pd.DataFrame({'r': r[m], 'n': npos[m]}); x = x[x.n > 0]
x['nb'] = pd.qcut(x.n, 4, duplicates='drop')
print(x.groupby('nb', observed=True).r.agg(['mean', 'count']).assign(mean=lambda d: d['mean'] * 100))
print('事件加權（以部位數加權）日均 %', round((x.r * x.n).sum() / x.n.sum() * 100, 3))
# 出場類別
print(sel.groupby('why')[['raw', 'hold']].agg(['mean', 'count']))
# 極端值
print('top 5 raw'); print(df.nlargest(5, 'raw')[['code', 'sig', 'hold', 'why', 'raw']])
