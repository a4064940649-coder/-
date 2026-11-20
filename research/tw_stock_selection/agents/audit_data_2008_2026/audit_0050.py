#!/usr/bin/env python3
"""項目 3：0050 逐年含息報酬，多來源交叉核對。python3 -I -B audit_0050.py"""
import sys, json; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import alib as A
out = {}
# A 引擎用：jonatw etf_0050.csv adj
a = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/etf_0050.csv', parse_dates=['date']).set_index('date')['adj'].sort_index()
# S75 輸出
nav = pd.read_csv(A.S75 / 'nav_2008_2026.csv.gz', parse_dates=['date']).set_index('date')
yr = pd.read_csv(A.S75 / 'yearly.csv').set_index('year')
# B 自建：官方日 K 原始價 ＋ TWT49U 除息 ＋ 分割（2015-01-05 起）
off = pd.read_parquet(A.TMP / 'off.parquet'); e = off[off['code'] == '0050'].set_index('date').sort_index()
ev = A.twt49u(); e49 = ev[ev['code'] == '0050'].set_index('date').sort_index()
C = e['close']; prev = C.shift(1)
fac = pd.Series(1.0, index=C.index); notes = []
for d, x in e49.iterrows():
    if d in C.index:
        ok = abs(x['prev_close'] - prev[d]) < 0.011
        fac[d] = x['prev_close'] / x['ref_price']; notes.append(dict(date=str(d.date()), type=x['type'], prev_close=float(x['prev_close']), ref=float(x['ref_price']), value=float(x['value']), prev_matches_daily=bool(ok)))
    else:
        notes.append(dict(date=str(d.date()), note='事件日不在日 K', type=x['type']))
# 分割：找收盤對前收 < 0.5 且非除息
gap = C / prev
split = gap[(gap < 0.6) | (gap > 1.6)]
sp = []
for d in split.index:
    i = C.index.get_loc(d); ratio = prev[d] / (C[d] - e['change'][d])
    sp.append(dict(date=str(d.date()), prev_close=float(prev[d]), prev_date=str(C.index[i - 1].date()), close=float(C[d]), change=float(e['change'][d]), implied_ref=float(C[d] - e['change'][d]), ratio=float(ratio),
                   calendar_gap_days=int((d - C.index[i - 1]).days)))
    fac[d] = round(ratio)                                     # 1 拆 4 → 持有股數 ×4
trB = (C * fac.cumprod()); trB = trB / trB.iloc[0]
out['B_events'] = dict(dividends=notes, splits=sp, n_div=len(notes))
# C 第三方：stevenhsieh etf_long（2009 起）、bench_long（2020-10 起）
el = pd.read_csv('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/extras/etf_long.csv.gz', dtype={'code': str}, parse_dates=['date']); el = el[el['code'] == '0050'].set_index('date').sort_index()
bl = pd.read_csv('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/extras/bench_long.csv', dtype={'code': str}, parse_dates=['date']); bl = bl[bl['code'] == '0050'].set_index('date').sort_index()
tw = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/taiex_twii.csv', parse_dates=['date']).set_index('date')['close'].sort_index()
src = {'A_jonatw_adj（引擎用）': a, 'S75_nav檔0050欄': nav['0050'], 'B_官方日K＋TWT49U自建': trB, 'C_stevenhsieh_etf_long_adj': el['adj'], 'C2_stevenhsieh_bench_long': bl['close_adj'], '加權指數（不含息，僅供參考）': tw}
def yearly(s, y0=2008):
    s = s.dropna(); res = {}
    for y in range(y0, 2027):
        cur = s[(s.index.year == y) & (s.index <= A.END)]
        base = s[s.index.year == y - 1]
        if y == 2008: base = s[s.index == pd.Timestamp('2008-01-02')]
        if len(cur) and len(base):
            res[y] = float(cur.iloc[-1] / base.iloc[-1] - 1)
    return res
Y = pd.DataFrame({k: yearly(v) for k, v in src.items()})
Y['yearly.csv_e0050'] = yr['e0050']
Y.index.name = 'year'
Y.to_csv(A.HERE / 'e0050_yearly.csv')
pd.set_option('display.width', 250)
print((Y * 100).round(2).to_string())
k = 'A_jonatw_adj（引擎用）'
cmp = {}
for c in Y.columns:
    if c == k or c.startswith('加權'): continue
    d = ((Y[c] - Y[k]) * 100).dropna()
    cmp[c] = dict(years=[int(d.index.min()), int(d.index.max())], n=int(len(d)), max_abs_diff_pp=float(d.abs().max()), year_of_max=int(d.abs().idxmax()), n_gt_0p3pp=int((d.abs() > 0.3).sum()), diffs_pp={int(i): round(float(v), 3) for i, v in d.items()})
out['yearly'] = {int(y): {c: (None if pd.isna(v) else round(float(v), 6)) for c, v in r.items()} for y, r in Y.iterrows()}
out['vs_engine_source'] = cmp
# 日資料層級：A 對 B 的日報酬差（2015 起）、A 對 C
def daily_cmp(x, y, a0=None):
    rx = x.dropna().pct_change(); ry = y.dropna().pct_change()
    j = rx.index.intersection(ry.index); j = j[(j <= A.END)]
    if a0: j = j[j >= a0]
    d = (rx[j] - ry[j]).dropna()
    big = d[d.abs() > 0.002]
    return dict(n=int(len(d)), n_absdiff_gt_0p2pp=int(len(big)), max_abs=float(d.abs().max()), big={str(i.date()): round(float(v) * 100, 3) for i, v in big.items()} if len(big) <= 60 else f'{len(big)} 天',
                first=str(j[0].date()), last=str(j[-1].date()))
out['daily_A_vs_B'] = daily_cmp(a, trB)
out['daily_A_vs_C'] = daily_cmp(a, el['adj'])
out['daily_A_vs_B_from_2026_08_12'] = daily_cmp(a, trB, pd.Timestamp('2026-08-12'))
# 2025-06 分割附近
w = e.loc['2025-06-05':'2025-06-20', ['open', 'close', 'change']].copy(); w['A_adj'] = a.reindex(w.index); w['C_close'] = el['close'].reindex(w.index); w['C_adj'] = el['adj'].reindex(w.index)
out['split_window'] = {str(i.date()): {c: (None if pd.isna(v) else float(v)) for c, v in r.items()} for i, r in w.iterrows()}
print(w.to_string())
aa = a.reindex(pd.date_range('2025-06-09', '2025-06-19')).dropna(); print(aa.to_string())
out['A_file'] = dict(first=str(a.index[0].date()), last=str(a.index[-1].date()), n=int(len(a)), dup_dates=int(a.index.duplicated().sum()),
                     missing_on_s75_days=int(a.reindex(nav.index).isna().sum()), nav_0050_matches_A=float(((nav['0050'] / 1e6) / (a.reindex(nav.index).ffill() / a.loc['2008-01-02']) - 1).abs().max()))
# 全期
out['full'] = {k2: dict(mult=float(v.dropna()[:A.END].iloc[-1] / v.loc['2008-01-02'])) for k2, v in src.items() if pd.Timestamp('2008-01-02') in v.index}
out['from_2015'] = {k2: float(v.dropna()[:A.END].iloc[-1] / v.dropna()[:'2015-01-05'].iloc[-1]) for k2, v in src.items() if v.dropna().index[0] <= pd.Timestamp('2015-01-05')}
(A.HERE / 'results_0050.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
print(json.dumps({k2: out[k2] for k2 in ('vs_engine_source', 'daily_A_vs_B', 'daily_A_vs_C', 'daily_A_vs_B_from_2026_08_12', 'A_file', 'full', 'from_2015')}, ensure_ascii=False, indent=1)[:7000])
print(json.dumps(out['B_events'], ensure_ascii=False)[:3000])

# ---------------------------------------------------------------- 補充
# (1) 2009～2015：A 的日報酬對 C 的「原始收盤」日報酬——應只在除息日不同（差＝殖利率）
rc = el['close'].pct_change(); ra = a.reindex(el.index).pct_change()
dd = (ra - rc).loc['2009-01-06':'2015-12-31'].dropna()
big = dd[dd.abs() > 0.002]
imp = [dict(date=str(i.date()), diff_pp=round(float(v) * 100, 3), C_prev_close=float(el['close'].shift(1)[i]), implied_cash_div=round(float(v * el['close'].shift(1)[i]), 3)) for i, v in big.items()]
out['A_vs_C_rawclose_2009_2015'] = dict(n_days=int(len(dd)), n_diff_gt_0p2pp=int(len(big)), days=imp, note='C 的 close 欄在 2025-06 分割前的日子已除以 4（回溯調整），不影響日報酬比較')
# (2) 2015：B 從 2015-01-05 起，同窗比 A
out['y2015_from_0105'] = dict(A=float(a[:'2015-12-31'].iloc[-1] / a['2015-01-05'] - 1), B=float(trB[:'2015-12-31'].iloc[-1] / trB['2015-01-05'] - 1))
# (3) A 檔的非交易日列、2026-08-12 起的漂移
cal = set(off['date'].unique()); a15 = a['2015-01-05':'2026-10-22']
junk = [str(d.date()) for d in a15.index if d not in cal]
ratio = (a / e['close']).dropna()
out['A_rows_on_non_trading_days_2015on'] = junk
seg = ratio['2026-07-22':'2026-10-22']
chg = (seg / seg.shift(1) - 1).dropna(); chg = chg[chg.abs() > 1e-5]
out['A_drift_2026'] = dict(ratio_2026_08_11=float(ratio['2026-08-11']), ratio_2026_10_22=float(ratio['2026-10-22']), drift=float(ratio['2026-10-22'] / ratio['2026-08-11'] - 1),
                           n_days_ratio_changed=int(len(chg)), first=str(chg.index[0].date()), last=str(chg.index[-1].date()), twt49u_0050_events_in_window=[str(d.date()) for d in e49.index if pd.Timestamp('2026-08-11') < d <= A.END],
                           days={str(i.date()): round(float(v) * 100, 4) for i, v in chg.items()})
# 2015～2026-08-11 之間，比例只該在除息日／分割日變動
r0 = ratio['2015-01-05':'2026-08-11']; c0 = (r0 / r0.shift(1) - 1).dropna(); c0 = c0[c0.abs() > 2e-4]
evd = set(e49.index) | {pd.Timestamp('2025-06-18')}
out['A_ratio_changes_2015_to_2026_08_11'] = dict(n=int(len(c0)), n_on_event_days=int(sum(d in evd for d in c0.index)), non_event_days={str(d.date()): round(float(v) * 100, 3) for d, v in c0.items() if d not in evd})
bm = nav['0050']
out['bench_effect'] = dict(e0050_2026_A=float(Y.loc[2026, k]), e0050_2026_B=float(Y.loc[2026, 'B_官方日K＋TWT49U自建']), final_0050_nav=float(bm.iloc[-1]), final_0050_nav_if_B=float(bm.iloc[-1] / (1 + out['A_drift_2026']['drift'])),
                           cagr_0050_A=float((bm.iloc[-1] / 1e6) ** (365.25 / (bm.index[-1] - bm.index[0]).days) - 1),
                           cagr_0050_if_B=float((bm.iloc[-1] / (1 + out['A_drift_2026']['drift']) / 1e6) ** (365.25 / (bm.index[-1] - bm.index[0]).days) - 1))
(A.HERE / 'results_0050.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
print(json.dumps({k2: out[k2] for k2 in ('A_vs_C_rawclose_2009_2015', 'y2015_from_0105', 'A_rows_on_non_trading_days_2015on', 'A_drift_2026', 'A_ratio_changes_2015_to_2026_08_11', 'bench_effect')}, ensure_ascii=False, indent=1))
