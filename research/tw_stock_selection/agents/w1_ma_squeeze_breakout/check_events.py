#!/usr/bin/env python3
"""抽樣核對與診斷：糾結條件寬鬆度、每年事件數、一筆事件手算。"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run as R
U = R.UNIV
ok5 = (R.SPREAD <= 0.05).astype(float).where(R.SPREAD.notna()).rolling(20, min_periods=20).min() == 1
ok3 = (R.SPREAD <= 0.03).astype(float).where(R.SPREAD.notna()).rolling(10, min_periods=10).min() == 1
o = {}
o['share_univ_days_squeeze_5_20'] = round(float((ok5 & U).sum().sum() / U.sum().sum()), 3)
o['share_univ_days_squeeze_3_10'] = round(float((ok3 & U).sum().sum() / U.sum().sum()), 3)
o['median_spread_univ'] = round(float(R.SPREAD.where(U).stack().median()), 4)
df = pd.read_csv(Path(R.L.P).parent / 'w1_ma_squeeze_primary_events.csv.gz', parse_dates=['sig', 'entry'], dtype={'code': str})
o['events_per_year'] = df.groupby(df['sig'].dt.year).size().to_dict()
o['mean_gap_sig_close_to_entry_open_pct'] = round(df['gap'].mean() * 100, 3)
# 用訊號日收盤進場（做不到）的毛異常：raw_close = (1+raw)*(1+gap)-1 ，EW 仍用開盤對開盤（近似）
rc = (1 + df['raw']) * (1 + df['gap']) - 1
o['hypothetical_sig_close_entry_ab_gross_pct_select'] = round(float((rc - df['ew'])[df['sig'] < '2019-05-01'].mean() * 100), 3)
o['hypothetical_sig_close_entry_ab_gross_pct_valid'] = round(float((rc - df['ew'])[df['sig'] >= '2019-05-01'].mean() * 100), 3)
# 手算一筆
row = df[df['sig'] >= '2023-01-01'].iloc[0]
c = row['code']; t = R.IDX.get_loc(row['sig'])
a = R.ADJ[c]
ma = {n: a.iloc[t - n + 1:t + 1].mean() for n in (5, 10, 20, 60)}
sp = [(max(a.iloc[s - n + 1:s + 1].mean() for n in (5, 10, 20)) - min(a.iloc[s - n + 1:s + 1].mean() for n in (5, 10, 20))) / a.iloc[s] for s in range(t - 20, t)]
vol = (R.V[c] / R.CR[c])
chk = {'code': c, 'sig': str(row['sig'].date()), 'max_spread_prev20': round(max(sp), 4), 'ma60_dist': round(a.iloc[t] / ma[60] - 1, 4),
       'close_gt_maxMA': bool(a.iloc[t] > max(ma[5], ma[10], ma[20])), 'dret': round(a.iloc[t] / a.iloc[t - 1] - 1, 4),
       'red': bool(R.CR[c].iloc[t] > R.OP[c].iloc[t]), 'vol_ratio': round(vol.iloc[t] / vol.iloc[t - 5:t].mean(), 2),
       'v60': round(float(R.V60[c].iloc[t]) / 1e6, 1), 'close_raw': float(R.CR[c].iloc[t]),
       'entry_open_adj': float(R.AO_raw[c].iloc[t + 1]), 'exit_open_adj': float(R.AO_raw[c].iloc[t + 21]),
       'raw_manual': round(R.AO_raw[c].iloc[t + 21] / R.AO_raw[c].iloc[t + 1] - 1, 5), 'raw_prog': round(row['raw'], 5)}
o['spot_check'] = chk
print(json.dumps(o, ensure_ascii=False, indent=1, default=str))
f = Path(__file__).resolve().parent / 'results.json'
d = json.loads(f.read_text(encoding='utf-8')); d['diagnostics'] = o
f.write_text(json.dumps(d, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
