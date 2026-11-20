#!/usr/bin/env python3
"""補充 2：(a) TWT49U 事件落在休市日的 119 筆，實際除權息日（下一交易日）引擎怎麼算；(b) 大贏家 SURGE 視窗不符的格子是哪個來源；(c) 兩來源不同的 20 格營收。"""
import sys, json, glob; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import alib as A
out = {}
off = pd.read_parquet(A.TMP / 'off.parquet'); off = off[off['date'] <= A.END]
cal = pd.DatetimeIndex(sorted(off['date'].unique()))
ev = A.twt49u(); ev = ev[ev['code'].str.fullmatch(r'[1-9]\d{3}') & (ev['date'] >= cal[0]) & (ev['date'] <= A.END) & ~ev['date'].isin(cal)]
o = off.set_index(['code', 'date']).sort_index()
res = []
for e in ev.itertuples():
    nd = cal[cal > e.date][0]
    if (e.code, nd) not in o.index: res.append('無日K'); continue
    r = o.loc[(e.code, nd)]; ref = r['close'] - r['change']
    if not (r['close'] > 0): res.append('無成交')
    elif abs(ref - e.ref_price) < 0.06: res.append('原漲跌價差已是對除權息參考價（正確）')
    elif r['change'] == 0 and abs(r['close'] - e.prev_close) > 1e-9: res.append('漲跌價差＝0 → 當日報酬記 0（少算）')
    elif abs(ref - e.prev_close) < 0.011: res.append('漲跌價差對前收（把除權息缺口當下跌，低估）')
    else: res.append('其他')
out['twt49u_on_closed_days'] = dict(n=int(len(ev)), outcome=pd.Series(res).value_counts().to_dict())
r = json.load(open(A.HERE / 'results_rev.json'))
out['census_same_month_diffs'] = r['census']['官方倉庫revenue_hist']['n_diff_gt_1e6']
(A.HERE / 'results_misc2.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
print(json.dumps(out, ensure_ascii=False, indent=1))
