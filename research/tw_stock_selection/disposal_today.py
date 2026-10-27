#!/usr/bin/env python3
"""處置股策略每日清單（第 1 輪審查後的正式版規則）：今天要買／今天要賣／持有中。
規則（見 agents/exec_rebuild/FINDINGS.md、agents/disposal_bear_longhistory/FINDINGS.md）：
  - 處置第一天收盤買、處置最後一天收盤賣；只做上市股（--tpex 另列上櫃，樣本只有約 13 個月）。
  - 固定槽位：每檔金額＝處置部位淨值 ÷ 槽數（預設 10）；持有已滿就跳過新事件。
  - 同一檔處置期間重疊或首尾相接（再處置）只做第一筆，持有到原本那筆的最後一天。
  - 進場日收盤漲幅 ≥ 9%（接近漲停、多半買不到）不買；出場日收盤跌幅 ≤ −9% 延到第一個非跌停日收盤賣。
  - 空頭開關：加權指數前一日收盤低於 200 日均線時不開新倉（已持有的照常到期賣）。
  - 單筆金額不超過進場日成交值的 5%。
用法：python3 -I disposal_today.py --data <tw-stock-screener>/data [--date 2026-10-27] [--equity 200000] [--slots 10] [--taiex <taiex_twii.csv>] [--tpex]
資料為第三方整理，下單前請以證交所「處置有價證券」公告核對：https://www.twse.com.tw/zh/announcement/punish.html
處置股需預收款券（現金先在帳戶），撮合間隔較長，請先確認券商規定。
"""
import argparse
from pathlib import Path
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('--data', required=True); ap.add_argument('--date')
ap.add_argument('--equity', type=float, help='處置部位目前淨值（元），用來算每檔金額')
ap.add_argument('--slots', type=int, default=10)
ap.add_argument('--held', type=int, help='你目前實際持有的處置股檔數（今天要賣的不算）；不給就用名單推算')
ap.add_argument('--taiex', default='/home/user/jonatw/finlab-txd/data/raw/taiex_twii.csv')
ap.add_argument('--tpex', action='store_true', help='一併列出上櫃處置股')
a = ap.parse_args()
D = Path(a.data)
dp = pd.read_csv(D / 'extras/disposal_5y.csv.gz', dtype={'code': str}, parse_dates=['date', 'start', 'end'])
lst = pd.read_csv(D / 'stock_list.csv', dtype=str)
mk = ['TWSE', 'TPEX'] if a.tpex else ['TWSE']
names = lst[lst['market'].isin(mk)].set_index('code')
dp = dp[dp['code'].isin(names.index)].copy()
dp['name'] = dp['code'].map(names['name']); dp['market'] = dp['code'].map(names['market'])
today = pd.Timestamp(a.date) if a.date else pd.Timestamp.today().normalize()

# 同一檔重疊或首尾相接（30 日內）只留第一筆；被併入的事件標為「再處置，不另開倉」
dp = dp.sort_values(['code', 'start']).reset_index(drop=True)
dp['take'] = True; dp['hold_end'] = dp['end']
for code, g in dp.groupby('code'):
    last_end = None
    for i in g.index:
        if last_end is not None and dp.at[i, 'start'] <= last_end + pd.Timedelta(days=3):
            dp.at[i, 'take'] = False
        else:
            last_end = dp.at[i, 'end']
ev = dp[dp['take']]

bear = None
try:
    tx = pd.read_csv(a.taiex, parse_dates=['date']).set_index('date')['close'].sort_index()
    tx = tx[tx.index < today]
    if len(tx) >= 200:
        bear = bool(tx.iloc[-1] < tx.iloc[-200:].mean())
        tx_info = f'加權指數 {tx.index[-1].date()} 收盤 {tx.iloc[-1]:.0f}，200 日均線 {tx.iloc[-200:].mean():.0f}'
except (OSError, KeyError, ValueError):
    pass

buy = ev[ev['start'] == today]; sell = ev[ev['hold_end'] == today]
holding = ev[(ev['start'] < today) & (ev['hold_end'] > today)]
print(f'日期 {today.date()}（資料最後公告日 {dp["date"].max().date()}）')
if bear is None:
    print('空頭開關：找不到加權指數資料，請自行確認指數是否在 200 日均線之上')
else:
    print(f'空頭開關：{tx_info} → {"空頭，今天不開新倉" if bear else "多頭，可以開新倉"}')
held = a.held if a.held is not None else len(holding)
print(f'持有 {held} 檔（{"自填" if a.held is not None else "依名單推算，假設每筆都有買到"}；槽數 {a.slots}，剩 {max(a.slots - held, 0)} 槽）')
if a.equity: print(f'每檔金額 ≈ {a.equity / a.slots:,.0f} 元（不超過該股當日成交值 5%）')

cols = ['code', 'name', 'market', 'start', 'end']
print(f'\n今天收盤賣出（處置最後一天）：{len(sell)} 檔' + ('；收盤跌幅 ≤ −9% 時延到第一個非跌停日收盤賣' if len(sell) else ''))
if len(sell): print(sell[cols].to_string(index=False))
if bear:
    print(f'\n今天收盤買進：0 檔（空頭開關，原本有 {len(buy)} 檔）')
else:
    room = max(a.slots - held, 0)
    print(f'\n今天收盤買進（處置第一天，最多 {room} 檔）：{len(buy)} 檔；收盤漲幅 ≥ 9% 的不買' + ('；超過空槽時依公告順序取前幾檔' if len(buy) > room else ''))
    if len(buy): print(buy[cols].to_string(index=False))
skipped = dp[(dp['start'] == today) & ~dp['take']]
if len(skipped): print(f'\n再處置（已持有，不另開倉）：{len(skipped)} 檔'); print(skipped[cols].to_string(index=False))
print(f'\n持有中：{len(holding)} 檔')
if len(holding): print(holding[cols].to_string(index=False))
