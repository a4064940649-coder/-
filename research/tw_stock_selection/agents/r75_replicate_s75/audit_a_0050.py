#!/usr/bin/env python3
"""R75 稽核 (a)：0050 含息淨值（jonatw etf_0050.csv 的 adj）2008～2014 的正確性。
本地沒有第二份 2015 年以前的 0050 含息或原始收盤（Asoul 長表、data_panel 都只有四碼普通股；其他 etf_0050.csv 都是同一檔的複本），
所以用三個間接核對：(1) 對加權指數（taiex_twii.csv，不含息）逐年、逐日；(2) 已知除息日當天「0050 含息 − 加權指數」日報酬是否沒有掉一個殖利率；
(3) 2015 起對官方日 K 的 0050 原始收盤：adj÷原始收盤 的比值只該在除息日跳，跳幅＝殖利率；(4) 連續相同值（疑似補值）天數。"""
import json
from pathlib import Path
import numpy as np, pandas as pd
TMP = Path('/tmp/claude-0/-home-user/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/r75')
e = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/etf_0050.csv', parse_dates=['date']).set_index('date')['adj'].sort_index()
t = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/taiex_twii.csv', parse_dates=['date']).set_index('date')['close'].sort_index()
out = {}
ye = e.groupby(e.index.year).last(); yt = t.groupby(t.index.year).last()
out['yearly_calendar'] = {int(y): dict(e0050=round(float(ye.pct_change()[y] * 100), 2), taiex_price=round(float(yt.pct_change()[y] * 100), 2),
                                       diff=round(float((ye.pct_change()[y] - yt.pct_change()[y]) * 100), 2)) for y in range(2004, 2027)}
out['from_2008_01_02'] = dict(e0050=float(e['2008-12-31'] / e['2008-01-02'] - 1), taiex=float(t['2008-12-31'] / t['2008-01-02'] - 1),
                              e0050_2008_01_02_day=float(e['2008-01-02'] / e['2007-12-31'] - 1), e0050_2008_01_03_day=float(e['2008-01-03'] / e['2008-01-02'] - 1))
re_, rt = e.pct_change(), t.pct_change().reindex(e.index)
d = (re_ - rt)
# 已知 0050 除息日與每單位配息（元）；資料為稽核員記憶中的公開資訊，僅作方向核對
EX = {'2008-10-24': 2.0, '2009-10-23': 1.0, '2010-10-25': 2.2, '2011-10-26': 1.95, '2012-10-24': 1.85, '2013-10-24': 1.35, '2014-10-24': 1.55, '2015-10-26': 2.0,
      '2016-07-28': 0.85, '2017-02-08': 1.7, '2017-07-31': 0.7, '2018-01-29': 2.2, '2018-07-23': 0.7}
out['exdiv_day_gap_vs_taiex_pp'] = {k: round(float(d.get(pd.Timestamp(k), np.nan) * 100), 2) for k in EX}
out['daily_gap_std_pp_by_year'] = {int(y): round(float(d[str(y)].std() * 100), 3) for y in range(2004, 2027)}
out['daily_corr_by_year'] = {int(y): round(float(np.corrcoef(re_[str(y)].dropna(), rt[str(y)].reindex(re_[str(y)].dropna().index).fillna(0))[0, 1]), 4) for y in range(2004, 2027)}
big = d[(d.abs() > 0.02)]
out['abs_gap_gt_2pp'] = {str(k.date()): round(float(v * 100), 2) for k, v in big.loc['2007':'2015'].items()}
flat = (re_ == 0)
out['flat_days_by_year'] = {int(y): int(flat[str(y)].sum()) for y in range(2004, 2027)}
out['taiex_flat_days_by_year'] = {int(y): int((t.pct_change()[str(y)] == 0).sum()) for y in range(2004, 2027)}
out['flat_while_taiex_moved_gt_0.3pp_2007_2014'] = [str(k.date()) for k in flat.index[flat & (rt.abs() > 0.003)] if 2007 <= k.year <= 2014]
# 月報酬對加權指數：月差距超過 4 個百分點的月份
me = e.groupby(e.index.to_period('M')).last().pct_change(); mt = t.groupby(t.index.to_period('M')).last().pct_change()
md = (me - mt).loc['2008-01':'2014-12']
out['monthly_gap_abs_gt_4pp_2008_2014'] = {str(k): round(float(v * 100), 2) for k, v in md[md.abs() > 0.04].items()}
# 2015 起：adj ÷ 官方原始收盤
o = pd.read_csv(TMP / 'cache_is/e50_ohlc.csv', parse_dates=['date']).set_index('date')['close']
r = (e.reindex(o.index) / o).dropna()
st = r.pct_change().dropna()
steps = st[st.abs() > 0.003]
out['ratio_steps_2015_on'] = {str(k.date()): round(float(v * 100), 3) for k, v in steps.items()}
out['ratio_first_last'] = [float(r.iloc[0]), float(r.iloc[-1])]
(TMP / 'audit_a.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
print(json.dumps(out, ensure_ascii=False, indent=1))
