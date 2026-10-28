#!/usr/bin/env python3
"""官方日 K（成交量、OHLC）長表：2023-10～2026-10，只取上市普通股（4 碼、非 0 開頭）。來源一律只當 CSV 讀：
  hschi：raw/official/hschi_dayK（FinMind TaiwanStockPrice，2020-01～2025-05-28；Trading_Volume＝股）
  bacardi：raw/official/bacardi/prices_2025.csv、prices_2026.csv（2025-04-01～2026-10-16；market＝TWSE；volume＝股）
  history：raw/screener/history.csv.gz（2025-09-25 起；volume＝股）
優先序：hschi（到 2025-05-28）→ bacardi → history。重疊期逐格比對量與開收盤，寫進 data/official_check.json。
輸出 data/official_daily.parquet（date, code, open, high, low, close, vol, src）。
另做步驟 0 清點：各來源的跳日（全市場連續 > 5 個交易日沒有資料）與重複列。
用法：python3 -I -B build_official.py
"""
import glob, json
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
RAW = HERE / 'raw'
A0 = pd.Timestamp('2023-10-01')
def ok(c): return c.str.len().eq(4) & ~c.str.startswith('0')

def hschi():
    out = []
    for f in sorted(glob.glob(str(RAW / 'official' / 'hschi_dayK' / '*.csv'))):
        x = pd.read_csv(f, dtype={'stock_id': str})
        if 'stock_id' not in x: continue
        x = x.rename(columns={'stock_id': 'code', 'max': 'high', 'min': 'low', 'Trading_Volume': 'vol', 'Trading_money': 'value'})
        out.append(x[['date', 'code', 'open', 'high', 'low', 'close', 'vol', 'value']])
    d = pd.concat(out); d['date'] = pd.to_datetime(d.date); return d

def bacardi():
    out = []
    for f in sorted(glob.glob(str(RAW / 'official' / 'bacardi' / 'prices_*.csv'))):
        x = pd.read_csv(f, dtype={'stock_id': str}, encoding='utf-8-sig')
        x = x[x.market == 'TWSE'].rename(columns={'stock_id': 'code', 'volume': 'vol', 'turnover': 'value'})
        out.append(x[['date', 'code', 'open', 'high', 'low', 'close', 'vol', 'value']])
    d = pd.concat(out); d['date'] = pd.to_datetime(d.date); return d

def history():
    x = pd.read_csv(RAW / 'screener' / 'history.csv.gz', dtype={'code': str})
    sl = pd.read_csv(RAW / 'screener' / 'stock_list.csv', dtype=str)
    x = x[x.code.isin(set(sl[sl.market == 'TWSE'].code))].rename(columns={'volume': 'vol'})
    x['value'] = np.nan; x['date'] = pd.to_datetime(x.date); return x[['date', 'code', 'open', 'high', 'low', 'close', 'vol', 'value']]

def clean(d, name, info):
    d = d[ok(d.code.astype(str)) & (d.date >= A0)].copy()
    for c in ('open', 'high', 'low', 'close', 'vol', 'value'): d[c] = pd.to_numeric(d[c], errors='coerce')
    n0 = len(d); dup = int(d.duplicated(['date', 'code']).sum())
    d = d.drop_duplicates(['date', 'code'], keep='last')
    d = d[(d.vol > 0) & (d.close > 0) & (d.open > 0)]
    days = pd.DatetimeIndex(sorted(d.date.unique()))
    gaps = [(str(a.date()), str(b.date()), int(np.busday_count(a.date(), b.date()) - 1)) for a, b in zip(days[:-1], days[1:]) if np.busday_count(a.date(), b.date()) - 1 > 5]
    info[name] = dict(rows=n0, dup_rows=dup, first=str(days[0].date()), last=str(days[-1].date()), n_days=len(days), n_codes=int(d.code.nunique()), gaps_gt5_busdays=gaps)
    d['src'] = name
    return d

if __name__ == '__main__':
    info = {}
    H = clean(hschi(), 'hschi', info); B = clean(bacardi(), 'bacardi', info); Y = clean(history(), 'history', info)
    chk = {}
    for (a, b, na, nb) in ((H, B, 'hschi', 'bacardi'), (B, Y, 'bacardi', 'history')):
        m = a.merge(b, on=['date', 'code'], suffixes=('_a', '_b'))
        chk[f'{na}_vs_{nb}'] = dict(n=len(m), vol_exact=float((m.vol_a == m.vol_b).mean()), vol_ratio_median=float((m.vol_a / m.vol_b).median()),
                                     open_exact=float((m.open_a == m.open_b).mean()), close_exact=float((m.close_a == m.close_b).mean()),
                                     range=[str(m.date.min().date()), str(m.date.max().date())] if len(m) else None)
    Hc = H[H.date <= '2025-05-28']
    Bc = B[(B.date > '2025-05-28')]
    Yc = Y[Y.date > B.date.max()]
    O = pd.concat([Hc, Bc, Yc], ignore_index=True).sort_values(['code', 'date'])
    days = pd.DatetimeIndex(sorted(O.date.unique()))
    cnt = O.groupby('date').size()
    gaps = [(str(a.date()), str(b.date()), int(np.busday_count(a.date(), b.date()) - 1)) for a, b in zip(days[:-1], days[1:]) if np.busday_count(a.date(), b.date()) - 1 > 5]
    info['combined'] = dict(rows=len(O), n_days=len(days), first=str(days[0].date()), last=str(days[-1].date()), gaps_gt5_busdays=gaps,
                            days_with_lt_600_codes=[str(d.date()) for d, v in cnt.items() if v < 600])
    O[['date', 'code', 'open', 'high', 'low', 'close', 'vol', 'src']].to_parquet(HERE / 'data' / 'official_daily.parquet', index=False)
    json.dump(dict(sources=info, overlap=chk), open(HERE / 'data' / 'official_check.json', 'w'), ensure_ascii=False, indent=1)
    print(json.dumps(dict(sources={k: {kk: vv for kk, vv in v.items() if kk != 'days_with_lt_600_codes'} for k, v in info.items()}, overlap=chk), ensure_ascii=False, indent=1)[:4000])
