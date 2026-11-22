#!/usr/bin/env python3
"""S97 抽查：連續引擎名單（V_BRIDGE，2008～2026）口徑 A 判為開盤鎖漲停的 (日期, 代號)，rng(97) 不放回抽 10 筆。
引擎側的值從 out_cont.json 的 D1 rows 取日期與代號；價格一律從第二份來源重讀（2015 起 raw/tsd_extra/daily；之前 raw/pre2015/asoul），
漲停價在這裡用純 Python 小數手算（不呼叫引擎）。python3 -I -B spot_check.py → spot_check.csv"""
import json, sys
from pathlib import Path
from decimal import Decimal as D, ROUND_FLOOR
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
RAW = HERE.parents[1] / 'raw'
rows = json.loads((HERE / 'out_cont.json').read_text())['desc']['D1']['2008-01～2026-10']['rows']
rng = np.random.default_rng(97)
pick = sorted(rng.choice(len(rows), size=min(10, len(rows)), replace=False).tolist())
sel = [rows[i] for i in pick]
pre = None


def tick(p):
    for lim, t in ((10, '0.01'), (50, '0.05'), (100, '0.1'), (500, '0.5'), (1000, '1')):
        if p < lim:
            return D(t)
    return D('5')


def bars(code, date):
    """回傳（前一個有成交日的列, 當日列）。"""
    global pre
    d = pd.Timestamp(date)
    if d >= pd.Timestamp('2015-01-05'):
        fr = []
        for m in (d - pd.offsets.MonthBegin(2), d - pd.offsets.MonthBegin(1) if d.day > 1 else d, d):
            f = RAW / 'tsd_extra' / 'daily' / f'{m.year}' / f'{m.year}-{m.month:02d}.parquet'
            if f.exists():
                x = pd.read_parquet(f); fr.append(x[x['code'] == code])
        x = pd.concat(fr).drop_duplicates('date'); x['date'] = pd.to_datetime(x['date']); src = 'tsd_extra/daily'
    else:
        if pre is None:
            pre = pd.read_csv(RAW / 'pre2015' / 'asoul_twse_2007_2015H1.csv.gz', dtype={'code': str}, parse_dates=['date'])
        x = pre[pre['code'] == code]; src = 'pre2015/asoul'
    x = x.sort_values('date'); cur = x[x['date'] == d].iloc[0]
    pv = x[(x['date'] < d) & (x['close'] > 0)].iloc[-1]
    return pv, cur, src


out = []
for r in sel:
    pv, cur, src = bars(r['code'], r['date'])
    pct = D('0.07') if pd.Timestamp(r['date']) < pd.Timestamp('2015-06-01') else D('0.10')
    chg = cur['change'] if pd.notna(cur['change']) else 0.0
    ref = D(str(round(float(cur['close']) - float(chg), 4)))            # 參考價＝收盤−漲跌價差
    prevc = D(str(pv['close']))
    up = ref * (1 + pct); t = tick(up); lu = (up / t).to_integral_value(rounding=ROUND_FLOOR) * t
    o, h, l, c = (float(cur[k]) for k in ('open', 'high', 'low', 'close'))
    A = o >= float(lu) - 1e-9; B = A and not (l < float(lu) - 1e-9)
    out.append(dict(code=r['code'], date=r['date'], source=src, prev_date=str(pv['date'].date()), prev_close=float(prevc), ref_price=float(ref), ref_equals_prev_close=bool(ref == prevc),
                    limit_pct=float(pct), limit_up_hand=float(lu), open=o, high=h, low=l, close=c, src_limit_flag=str(cur['limit']) if 'limit' in cur.index else '',
                    A_open_at_limit=bool(A), B_locked_all_day=bool(B), engine_A=True, engine_B=bool(r['B']), match=bool(A and (B == r['B'])),
                    open_vs_ref_pct=round((o / float(ref) - 1) * 100, 3)))
df = pd.DataFrame(out); df.to_csv(HERE / 'spot_check.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
print(df.to_string()); print('全部相符', bool(df['match'].all()), '抽到', len(df), '／母體', len(rows))
