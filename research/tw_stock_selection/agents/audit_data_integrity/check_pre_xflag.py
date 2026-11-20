#!/usr/bin/env python3
"""補充（B4）：2015 前來源檔的 X 記號（漲跌價差不比價）不只除權息。量化「當日報酬被記 0」的格數，以及 S75 已發布交易持有期間碰到幾格。只讀。"""
import sys; from pathlib import Path as _P; sys.path.insert(0, str(_P(__file__).resolve().parent))
import json
import numpy as np, pandas as pd
from common import *                                    # noqa: E402

d = pd.read_csv(PRE, dtype={'code': str}); d['date'] = pd.to_datetime(d['date'])
d = d[d['date'] < '2015-01-05'].sort_values(['code', 'date'])
d['pc'] = d.groupby('code')['close'].transform(lambda s: s.where(s > 0).ffill().shift(1))
d['v60'] = d.groupby('code')['amount'].transform(lambda s: s.fillna(0).rolling(60, min_periods=60).mean())
x = d[(d['xflag'] > 0) & (d['close'] > 0) & d['pc'].notna()].copy()
x['gap'] = x['close'] / x['pc'] - 1
x['y'] = x['date'].dt.year
x['n_code_year'] = x.groupby(['code', 'y'])['gap'].transform('size')
x['cls'] = np.where(x['n_code_year'] <= 3, '每年 ≤3 次（像除權息／減資）', '每年 ≥4 次（不像除權息）')
x['in_pool_like'] = (x['close'] >= 10) & (x['v60'] >= 1e7)
R = dict(rule='X 記號＝來源檔漲跌價差欄是 X（不比價）；面板把這些日子的漲跌價差設 0 → 當日報酬記 0。分類只是啟發式：同一代號同一年出現 ≤3 次視為像除權息',
         by_class={k: dict(n=len(g), n_codes=int(g['code'].nunique()), mean_gap_pct=round(float(g['gap'].mean() * 100), 3), mean_abs_gap_pct=round(float(g['gap'].abs().mean() * 100), 3),
                           n_pool_like=int(g['in_pool_like'].sum()), pool_like_mean_abs_gap_pct=round(float(g.loc[g['in_pool_like'], 'gap'].abs().mean() * 100), 3) if g['in_pool_like'].any() else None,
                           by_year={int(y): int(n) for y, n in g.groupby('y').size().items()}) for k, g in x.groupby('cls')},
         pool_like_def='當日收盤 ≥ 10 元且前 60 日平均成交金額 ≥ 1,000 萬（近似策略股池門檻）')
t = pd.read_csv(AG / 's75_continuous_2008_2026/trades.csv.gz', dtype={'code': str}, parse_dates=['ent_date', 'exit_date'])
out = {}
for ver, tv in t.groupby('version'):
    tv = tv[tv['ent_date'] < '2015-01-05']
    hit = []
    for r in tv.itertuples():
        m = x[(x['code'] == r.code) & (x['date'] > r.ent_date) & (x['date'] <= r.exit_date)]
        for q in m.itertuples():
            hit.append(dict(code=r.code, ent=str(r.ent_date.date()), exit=str(r.exit_date.date()), x_date=str(q.date.date()), gap_pct=round(q.gap * 100, 2), cls=q.cls))
    h = pd.DataFrame(hit)
    out[ver] = dict(n_trades_pre2015=len(tv), n_trades_touching_x=int(h[['code', 'ent']].drop_duplicates().shape[0]) if len(h) else 0, n_x_cells_held=len(h),
                    by_class={k: dict(n=len(g), sum_gap_pct=round(float(g['gap_pct'].sum()), 2), mean_gap_pct=round(float(g['gap_pct'].mean()), 3)) for k, g in h.groupby('cls')} if len(h) else {},
                    non_exright_like=[q for q in hit if '≥4' in q['cls']][:30])
R['s75_trades_overlap'] = out
R['note'] = '交易損益用實際成交價算（cost／proceeds），不受當日報酬記 0 影響；受影響的是逐日淨值、還原價訊號（季線、近高）與持有期間除息少算股利。是否改變 S75／R75 數字要由重跑引擎的稽核員判定'
jdump(R, HERE / 'parts/pre_xflag.json'); print(json.dumps(R, ensure_ascii=False)[:3500])
