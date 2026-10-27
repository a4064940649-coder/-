"""把 voidful/tw_stocker 原始 5 分 K（含重複列）彙總成每日統計：
原始加總量（= 面板 src 3 用的量）、依時間戳去重後的量（first / last / max 三種）、列數、唯一時間戳數、最後一根時間、去重後收盤。
用法：python3 -I -B agg_raw5m.py <raw_csv> <out_parquet>"""
import sys
import pandas as pd
src, out = sys.argv[1], sys.argv[2]
r = pd.read_csv(src)
r['ts'] = pd.to_datetime(r['Datetime'], utc=True).dt.tz_convert('Asia/Taipei')
r['d'] = r['ts'].dt.tz_localize(None).dt.normalize()
g = r.groupby('d')
o = pd.DataFrame({'n_rows': g.size(), 'n_ts': g['ts'].nunique(), 'vol_raw': g['Volume'].sum(),
                  'last_ts': g['ts'].max().dt.strftime('%H:%M'), 'close_last_row': g['Close'].last(), 'open_first_row': g['Open'].first()})
gt = r.groupby(['d', 'ts'])['Volume']
for nm, s in [('vol_first', gt.first()), ('vol_last', gt.last()), ('vol_max', gt.max())]:
    o[nm] = s.groupby(level=0).sum()
o['n_ts_diffvals'] = r.groupby(['d', 'ts'])[['Open', 'High', 'Low', 'Close', 'Volume']].nunique().max(axis=1).gt(1).groupby(level=0).sum()
dd = r.sort_values('ts', kind='stable').drop_duplicates('ts', keep='first')
o['close_dedup'] = dd.groupby('d')['Close'].last()
o.index.name = 'date'
o.to_parquet(out)
