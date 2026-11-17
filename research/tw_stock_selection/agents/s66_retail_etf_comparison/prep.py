"""S66 第 0 步：資料盤點（不算報酬、不算任何績效）。輸出 data_check.json。"""
import json, sys
import pandas as pd
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
import s66lib as L
d = L.load_daily(); e = L.load_div()
cal = sorted(d.date.unique())
nav = pd.read_csv(L.NAV_CSV, parse_dates=['date'])
long_ = pd.read_csv(L.EXTRAS + 'etf_long.csv.gz', dtype={'code': str}); trm = pd.read_csv(L.EXTRAS + 'etf_tr_m.csv.gz', dtype={'code': str})
out = {'官方日K': {'來源': 'raw/jb7k6295r7-bot_tw-stock-data data/universe/daily（git cat-file 唯讀）', '起': str(cal[0].date()), '迄': str(cal[-1].date()), '交易日數': len(cal), '00 開頭代號數': int(d.stock_id.nunique())},
       '正式淨值': {'檔': L.NAV_CSV, 'sha256': L.sha(L.NAV_CSV), '起': str(nav.date.iloc[0].date()), '迄': str(nav.date.iloc[-1].date()), '列數': len(nav), '欄': list(nav.columns)},
       'twt49u': {'起': str(e.date.min().date()), '迄': str(e.date.max().date()), '說明': '含 ETF（非四碼代號）除息事件；只有上市'},
       'data_panel': {'panel/*.parquet': '只有四碼普通股，沒有 ETF', 'scr_extra/etf.csv.gz': '全部 ETF 日 K，但只有 2025-09-24～2026-10-22（約 1 年），不含配息'},
       '其他本地來源': {'finlab etf_0050.csv／etf_0056.csv': '第三方還原收盤（0050 自 2003-07-01、0056 自 2007-12-26）', 'etf_long.csv.gz': {c: [g.date.min(), g.date.max()] for c, g in long_.groupby('code')}, 'etf_tr_m.csv.gz': '月底總報酬指數，2020-08 起（只供核對）'},
       '各檔': {}}
for c in L.MAIN + L.SUPP:
    s = d[d.stock_id == c].sort_values('date').set_index('date')
    P = s.close.ffill(); pc = P.shift(1)
    ev = e[(e.code == c) & (e.date >= s.index[0])]
    imp = (pc - (s.close - s.change)).round(3); imp = imp[(imp.abs() > 0.005) & s.close.notna() & (s.close / pc - 1).abs().lt(0.3)]
    evd = [s.index[min(s.index.searchsorted(t), len(s) - 1)] for t in ev.date]
    j = ev.assign(td=evd).set_index('td')
    first = s.index[0]
    out['各檔'][c] = {'名稱': s.name.iloc[-1], '類別': L.CAT[c], '主表或補充': '主表' if c in L.MAIN else '補充',
        '官方日K首日': str(first.date()), '首日即資料起點（上市早於 2015-01-05）': bool(first == cal[0]), '官方日K末日': str(s.index[-1].date()),
        '有K棒日數': len(s), '應有日數': sum(1 for t in cal if t >= first), '無成交日數': int(s.close.isna().sum()),
        '至 2026-10-22 上市年數（以資料首日計）': round((L.END - first).days / 365.25, 2), '上市不到 5 年': bool((L.END - first).days / 365.25 < 5) if first != cal[0] else False,
        'twt49u 除息次數': len(ev), 'twt49u 息值合計': round(float(ev.value.sum()), 3), '種類': sorted(set(ev.type)),
        'twt49u 前收與官方日K不符筆數': int(((j.prev_close - pc.reindex(j.index).to_numpy()).abs() > 0.011).sum()),
        '官方漲跌價差隱含除息日數（獨立核對）': len(imp), '隱含有但 twt49u 沒有（多為無成交日後的基準差）': [(str(t.date()), float(imp[t])) for t in imp.index.difference(pd.DatetimeIndex(evd))][:10],
        '疑似分割日（|收盤÷前收−1|>30%）': [str(t.date()) for t in s.index[(s.close / pc - 1).abs() > 0.3]],
        '能否算含息還原': '可（2015-01-05 或上市日起；官方收盤＋twt49u 除息）',
        '2015 以前': ({'0050': '可接 finlab 第三方還原價（2003-07-01 起）', '0056': '可接 finlab 第三方還原價（2007-12-26 起）', '006208': '可接 etf_long 第三方還原價（2012-06-22 起）', '0051': '資料不足（本地沒有 2015 以前的日 K 或還原價）'}.get(c, '不適用'))}
json.dump(out, open(L.HERE / 'data_check.json', 'w'), ensure_ascii=False, indent=1, default=str)
for c, v in out['各檔'].items():
    print(c, v['名稱'], v['官方日K首日'], v['有K棒日數'], v['應有日數'], v['無成交日數'], v['twt49u 除息次數'], v['官方漲跌價差隱含除息日數（獨立核對）'], v['疑似分割日（|收盤÷前收−1|>30%）'], v['上市不到 5 年'])
