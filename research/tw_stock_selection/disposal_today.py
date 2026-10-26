#!/usr/bin/env python3
"""處置股策略每日清單：今天要買（處置第一天）／今天要賣（處置最後一天）／持有中。
規則：處置第一天收盤買進、處置最後一天收盤賣出；只做上市股；等金額分配。
用法：python3 disposal_today.py --data <tw-stock-screener>/data [--date 2026-10-27]
資料為第三方整理，下單前請以證交所「處置有價證券」公告核對：https://www.twse.com.tw/zh/announcement/punish.html
處置股需預收款券（現金先在帳戶），撮合間隔較長，請先確認券商規定。
"""
import argparse
from pathlib import Path
import pandas as pd

ap = argparse.ArgumentParser(); ap.add_argument('--data', required=True); ap.add_argument('--date'); a = ap.parse_args()
D = Path(a.data)
dp = pd.read_csv(D / 'extras/disposal_5y.csv.gz', dtype={'code': str}, parse_dates=['date', 'start', 'end'])
lst = pd.read_csv(D / 'stock_list.csv', dtype=str); twse = lst[lst['market'] == 'TWSE'].set_index('code')['name']
dp = dp[dp['code'].isin(twse.index)].copy(); dp['name'] = dp['code'].map(twse)
today = pd.Timestamp(a.date) if a.date else pd.Timestamp.today().normalize()
buy = dp[dp['start'] == today]; sell = dp[dp['end'] == today]; holding = dp[(dp['start'] < today) & (dp['end'] > today)]
print(f'日期 {today.date()}（資料最後公告日 {dp["date"].max().date()}）')
for title, df in [('今天收盤買進（處置第一天）', buy), ('今天收盤賣出（處置最後一天）', sell), ('持有中', holding)]:
    print(f'\n{title}：{len(df)} 檔')
    if len(df): print(df[['code', 'name', 'start', 'end']].to_string(index=False))
