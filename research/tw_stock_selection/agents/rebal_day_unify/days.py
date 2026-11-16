#!/usr/bin/env python3
"""換股日兩種定義的逐月日期表（2019-01～2026-12）。交易日＝官方日 K 倉庫（tsd）的 data/universe/daily/ 檔名。
A10（回測、make_list.py、cslib.rev_avail_days）：每月 10 日（含）起第一個交易日的「下一個交易日」。
B11（舊 XQ 20／21／22 號 IsRebal）：每月第一個「日期 ≥ 11」的交易日。
用法：python3 -I -B days.py <tsd 倉庫> [<研究中心 raw 封存>] → days.csv（同資料夾）；有第二個參數時另核對兩個倉庫的交易日是否一致。"""
import sys, subprocess
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent


def tdays(repo):
    out = subprocess.run(['git', '-C', repo, 'ls-tree', '--name-only', 'HEAD', 'data/universe/daily/'], capture_output=True, text=True, check=True).stdout
    return pd.DatetimeIndex(sorted(pd.Timestamp(x.rsplit('/', 1)[1][:10]) for x in out.split() if x.endswith('.csv')))


def first_on_or_after(D, ts):
    i = D.searchsorted(ts, side='left')
    return i if i < len(D) else None


D = tdays(sys.argv[1])
if len(sys.argv) > 2:
    R = tdays(sys.argv[2])
    a, b = D[(D >= '2019-01-01') & (D <= R[-1])], R[R >= '2019-01-01']
    print('交易日核對（2019 起、到 raw 封存最後一天）：tsd', len(a), 'raw', len(b), '相同' if a.equals(b) else '不同', sorted(set(a) ^ set(b))[:10])
rows = []
for m in pd.period_range('2019-01', '2026-12', freq='M'):
    s = m.to_timestamp()
    d10 = s + pd.Timedelta(days=9); d11 = s + pd.Timedelta(days=10)
    i10 = first_on_or_after(D, d10); i11 = first_on_or_after(D, d11)
    ok = i10 is not None and i10 + 1 < len(D) and D[i10].to_period('M') == m
    if not ok:
        rows.append(dict(month=str(m), day10_weekday='一二三四五六日'[d10.weekday()], deadline_td='', A10='', B11='', exec_A10='', exec_B11='', same='', diff_td='', note=f'tsd 只到 {D[-1].date()}'))
        continue
    a10 = i10 + 1; b11 = i11
    rows.append(dict(month=str(m), day10_weekday='一二三四五六日'[d10.weekday()], day10_trading=bool(D[i10] == d10), deadline_td=str(D[i10].date()),
                     A10=str(D[a10].date()), B11=str(D[b11].date()), exec_A10=str(D[a10 + 1].date()) if a10 + 1 < len(D) else '',
                     exec_B11=str(D[b11 + 1].date()) if b11 + 1 < len(D) else '', same=bool(a10 == b11), diff_td=int(b11 - a10), note=''))
df = pd.DataFrame(rows)
df.to_csv(HERE / 'days.csv', index=False, encoding='utf-8')
v = df[df['note'] == '']
print('月數', len(v), '相同', int(v['same'].sum()), '不同', int((~v['same'].astype(bool)).sum()), 'B11−A10（交易日）分布', v['diff_td'].value_counts().to_dict())
print('10 日是交易日 → 都相同：', bool(v.loc[v['day10_trading'].astype(bool), 'same'].astype(bool).all()),
      '；10 日非交易日 → 都不同：', bool((~v.loc[~v['day10_trading'].astype(bool), 'same'].astype(bool)).all()))
print(v.loc[~v['same'].astype(bool), ['month', 'day10_weekday', 'deadline_td', 'B11', 'A10', 'diff_td']].to_string(index=False))
