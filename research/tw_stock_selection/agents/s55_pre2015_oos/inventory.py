#!/usr/bin/env python3
"""S55 第一步：資料盤點（不算任何報酬）→ data_inventory.json。用法：python3 -I -B inventory.py（需先跑 build_pre_panel.py）"""
import json, glob
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = Path('/home/user/-/research/tw_stock_selection')
REV = '/home/user/niklavsd/bottleneck-signals/data/processed/tw_revenue.csv'
PRE = Path('/tmp/claude-0/-home-user/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s55/cache_pre')
OFF = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/eng/cache')
pc = json.loads((HERE / 'panel_check.json').read_text())

# 營收
r = pd.read_csv(REV, dtype={'code': str}, parse_dates=['month'])
r4 = r[r['code'].str.fullmatch(r'[1-9]\d{3}')]
sii = r4[r4['market'] == 'sii']
by_year = sii.groupby(sii['month'].dt.year)['code'].nunique().to_dict()
lastm = sii.groupby('code')['month'].max()
stop_before_2013 = lastm[(lastm < pd.Timestamp('2012-12-01'))]
stop_2012_12 = lastm[lastm == pd.Timestamp('2012-12-01')]
# 價格面板（上市，含下市）與營收的對照
z = np.load(PRE / 'panel_cs.npz', allow_pickle=False)
IDX = pd.DatetimeIndex(z['IDX']); COLS = list(z['COLS']); has = np.isfinite(z['C'])
mon = IDX.to_period('M')
pr_codes_by_m = {}
for m in pd.period_range('2008-01', '2015-06', freq='M'):
    rows = mon == m
    pr_codes_by_m[m] = set(np.array(COLS)[has[rows].any(axis=0)])
cov = {}
for y in range(2008, 2016):
    a = b = c = 0
    for m in pd.period_range(f'{y}-01', f'{y}-12' if y < 2015 else '2015-06', freq='M'):
        rev_c = set(sii.loc[sii['month'].dt.to_period('M') == m, 'code'])
        px = pr_codes_by_m.get(m, set())
        a += len(px); b += len(px & rev_c); c += len(rev_c - px)
    cov[str(y)] = dict(price_stock_months=a, with_revenue=b, revenue_coverage_of_priced=round(b / a, 4) if a else None, revenue_without_price=c)
# 2013～2015-06 下市股的營收缺口（存活者偏誤）
lastp = pd.DataFrame(has, index=IDX, columns=COLS).apply(lambda s: s[s].index.max())
dl = lastp[(lastp >= pd.Timestamp('2008-01-01')) & (lastp < pd.Timestamp('2015-06-01'))]
dl_info = []
for c, d in dl.items():
    lm = lastm.get(c)
    dl_info.append(dict(code=c, last_trade=str(d.date()), last_rev_month=str(lm.date())[:7] if lm is not None else None))
dl_df = pd.DataFrame(dl_info)
dl_pre13 = dl_df[pd.to_datetime(dl_df['last_trade']) < pd.Timestamp('2013-01-01')]
dl_post13 = dl_df[pd.to_datetime(dl_df['last_trade']) >= pd.Timestamp('2013-01-01')]
gaps = [(pd.Timestamp(x.last_trade).to_period('M') - pd.Period(x.last_rev_month, 'M')).n for x in dl_df.itertuples()
        if x.last_rev_month and pd.Period(x.last_rev_month, 'M') <= pd.Timestamp(x.last_trade).to_period('M')]
rev_ok = lambda d: int(sum(1 for x in d.itertuples() if x.last_rev_month and pd.Period(x.last_rev_month) >= pd.Timestamp(x.last_trade).to_period('M') - 3))

e50 = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/etf_0050.csv', parse_dates=['date'])
tw = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/taiex_twii.csv', parse_dates=['date'])
eo = pd.read_csv(OFF / 'e50_ohlc.csv', parse_dates=['date'])
t49 = sorted(glob.glob(str(ROOT / 'raw/twt49u/twse-exright/*/*.csv.gz')))
t49_first = pd.read_csv(t49[0])['date'].min()
fl = json.loads((ROOT / 'raw/pre2015/fetch_log.json').read_text())

inv = {
    'written_at': pd.Timestamp.now('UTC').isoformat(),
    'purpose': 'S55 把凍結的現行規則往 2015-07 以前延伸：盤點 2015-07 以前的本地資料與公開來源（本檔不含任何報酬數字）',
    'engine_panel_now': {
        'stlib.load / s2lib.load / s10lib.load / s17lib.load / s20lib.load / s47lib.load': '全部經 stlib.build_panel → cslib.Panel：讀 cslib.TMP/panel_cs.npz（快取；不存在時 cslib.build_panel 用 git cat-file 從 raw/jb7k6295r7-bot_tw-stock-data 的 data/universe/daily/YYYY-MM-DD.csv 建）',
        'daily_source_first_date': '2015-01-05（官方日 K 倉庫 data/universe/daily 第一天；上市＋上櫃，含成交股數、成交金額、漲跌價差、逐日市場別）',
        'panel_end': 'cslib.END＝2026-10-22（P.T＝2876 列、P.N＝2142 欄）',
        'backtest_start': 'stlib.START＝2015-07-01（P.t0）；訊號月份另有 s17lib.FIRST_MONTH＝2016-01（σ24 從 2013 IFRS 斷點後要 24 個月才算得出）',
        'other_inputs': {'月營收': 'stlib.REV_CSV＝/home/user/niklavsd/bottleneck-signals/data/processed/tw_revenue.csv（s17lib.rev_family 讀；本身從 2005-01 起）',
                         '0050 還原收盤': 'cslib.E50_CSV＝/home/user/jonatw/finlab-txd/data/raw/etf_0050.csv',
                         '0050 原始開高低收': 'S1/e50_ohlc.csv（stlib.build_panel 從同一官方日 K 倉庫抽 0050；2015-01-05 起）',
                         '加權指數': 'cslib.TWII_CSV＝/home/user/jonatw/finlab-txd/data/raw/taiex_twii.csv',
                         '除權息': 'cslib.TWT49U_DIR＝raw/twt49u/twse-exright（EXRIGHT_FIX）',
                         '營收期限例外／保險 15 日': 'raw/rev_deadline_exceptions（DEADLINE_FIX／INS15_MASK）',
                         '創新板': 's4lib.innov_panel：官方日 K name 欄（2021 起才有創新板）'}},
    'datasets': {
        'monthly_revenue': {
            'local': REV, 'first_month': str(r['month'].min().date())[:7], 'last_month': str(r['month'].max().date())[:7],
            'twse_sii_codes_by_year': {str(k): int(v) for k, v in by_year.items()},
            'pre2015_usable': '可用：2005-01 起全市場（MOPS t21sc03 彙總頁解析，revenue_longhistory 已驗證與 rev_5y 重疊期 99.6% 相同）',
            'defects': ['2013 IFRS 口徑斷點（2012 個體→2013 合併）：引擎把 2013 年的年增量 D 一律設缺值（凍結規則的一部分）→ σ24 在 2013-01～2015-12 全缺，凍結規則在這 36 個月沒有訊號、也不出場',
                        '2013 起只剩當時仍上市的公司（2013～2015 下市者營收只到 2012）；但這段本來就沒有訊號，不影響選股'],
            'revenue_coverage_of_priced_twse_stocks': cov,
            'twse_codes_last_revenue_before_2012_12': int(len(stop_before_2013)), 'twse_codes_last_revenue_2012_12': int(len(stop_2012_12)),
            'survivorship': {'twse_delisted_2008_to_2012_in_price_panel': int(len(dl_pre13)), 'of_which_revenue_until_within_3m_of_last_trade': rev_ok(dl_pre13),
                             'twse_delisted_2013_to_2015H1_in_price_panel': int(len(dl_post13)), 'of_which_revenue_until_within_3m_of_last_trade_2013_15': rev_ok(dl_post13),
                             'n_delisted_with_revenue': len(gaps), 'months_last_trade_minus_last_revenue_median': float(np.median(gaps)),
                             'months_gap_quartiles': [float(np.percentile(gaps, q)) for q in (25, 50, 75)],
                             'reading': '下市股的營收大多在最後交易前約 12～14 個月就斷了（2009-11、2011-12 兩個斷點特別多，像是 MOPS 彙總頁在事後重製時刪掉已下市公司）→ 下市前最後一年 SURGE 缺值、不可能入選＝存活者偏誤（方向多半對策略有利：下市前一年多為營運惡化或被併購）；TDR（91xx）本來就沒有營收',
                             'delisted_list': dl_info}},
        'daily_k': {
            'engine_official_cache': '2015-01-05 起（上市＋上櫃）',
            'data_panel/panel': '2004-01-02～2026-10-22，上市普通股＋已下市上市股（1,249 檔；來源 Asoul/tsec ≤2019-05、k66 STOCK_DAY 2010～2023 等；有收、開、高、成交金額、還原、報酬，沒有低、成交股數、漲跌價差）',
            'asoul_tsec_local': '/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/tsec_all（1,921 檔，1992～2019-05-16，證交所原始行情含開高低收、股數、金額、漲跌價差（除權息日標 X）、含已下市股；在別的工作階段 scratchpad，可能被清掉 → S55 已把用到的部分另存 raw/pre2015/asoul_twse_2007_2015H1.csv.gz＋sha256 清單）',
            'built_pre_panel': {k: pc[k] for k in ('range', 'T', 'N', 'n_asoul_rows', 'n_x_rows', 'n_cells_filled_from_data_panel', 'close_match_vs_data_panel', 'per_year', 'vs_official_2015H1', 'n_codes_last_trade_before_2015_06')},
            'adjusted_price': '引擎自己用漲跌價差算含息報酬 R（參考價＝收盤−漲跌價差）；Asoul 除權息日漲跌價差標 X → 記 0（同官方 2015 起格式，當日報酬＝0）',
            'turnover_amount': '有（Asoul 成交金額；股池 60 日均額 ≥ 1,000 萬用）'},
        'ex_rights_twt49u': {
            'local_first_date': str(t49_first), 'local_months': len(t49),
            'pre2014': '本地沒有。證交所 TWT49U（rwd/zh/exRight/TWT49U）2026-11-17 第一次測試可連（2008-07 回 267 筆，HTTP 200），之後同主機請求被斷線（connection reset）→ 依紀律停止、沒有抓',
            'impact': '2007-01～2013-12 除權息日（Asoul 9,214 個 X 列中的大部分）無法用參考價修正，當日報酬記 0（＝EXRIGHT_FIX 關閉時的舊行為）；data_exright_fix 量過這個修正對 2019 起年化 +0.86 個百分點 → 對策略偏保守'},
        'listing_list': {'historical': '沒有逐日上市清單檔；上市與否用 data_panel（已去掉轉上市前的櫃買期）＋ Asoul 列判斷；現行清單 stock_list.csv 只有現存股',
                         'delisted_coverage': f"價格面板 2008-01～2015-05 最後交易的上市股 {len(dl)} 檔（Asoul 含已下市股）"},
        '0050': {'adj_close': f"{e50['date'].min().date()}～{e50['date'].max().date()}（jonatw/finlab-txd etf_0050.csv，含息還原）",
                 'raw_ohlc': f"{eo['date'].min().date()} 起（官方日 K 快取）；2007～2014 沒有 → 引擎內建退路：開盤／收盤比＝1（0050 換手用當日還原收盤代替開盤）、跳動點用 100 元的 0.05",
                 'twse_stock_day': 'STOCK_DAY 只提供 2010-01-04 以後（2007-01 試抓回「查詢日期小於99年1月4日」）；2010-01 那次請求被斷線，停止'},
        'twii': f"{tw['date'].min().date()}～{tw['date'].max().date()}（本地）",
        'rev_deadline_exceptions': '例外表只有 2019-01、2020-01（不套用）、2023-03；2015 以前的特例延長（若有）沒有登錄 → 用法定 10 日；保險 15 日名單從 2026-01 起，2015 以前不影響',
        'price_limit': '2015-06-01 以前漲跌幅 7%（TWT49U 2008-07 回應的漲停價也證實 7%）；引擎固定用 10% 算漲跌停價'},
    'public_sources_probe': [
        dict(host='mopsov.twse.com.tw', what='月營收彙總 t21sc03', result='代理 403（CONNECT tunnel failed）→ 停止', time='2026-11-17'),
        dict(host='api.finmindtrade.com', what='FinMind 公開 API', result='代理 403 → 停止', time='2026-11-17'),
        dict(host='www.twse.com.tw', what='STOCK_DAY／TWT49U', result='前 3 次 HTTP 200（STOCK_DAY 2010-01 0050、TWT49U 2008-07 有資料）；第 4 次（fetch_twse.py 第二輪第一個請求）connection reset（代理紀錄：tunnel closed after 3s）→ 視為被擋，停止，不換方法',
             log=fl)],
    'earliest_signal': {'revenue_first': '2005-01', 'D_first（年增量需要 12 個月前）': '2006-01', 'sigma24_first_signal_month（需要前 24 個 D）': '2008-01',
                        'first_signal_day': '2008-01 營收的可得日（2008-02-10 期限，順延後的下一個交易日）', 'last_signal_month_before_IFRS_gap': '2012-12（訊號日約 2013-01-11）',
                        'gap': '2013-01～2015-12 營收月份 SURGE 全缺（凍結規則），訊號恢復於 2016-01（＝現行正式回測的第一個訊號月）'},
    'verdict': {'sufficient': True,
                'why': '跑凍結規則需要的四樣核心資料都在本地且含已下市股：上市日 K（Asoul 證交所原始，含開高低收、成交金額、漲跌價差）2007～2015、月營收 2005 起、0050 含息還原 2003 起、加權指數。2015 上半年與官方日 K 收盤、開盤、成交金額、漲跌價差 100% 相同。',
                'approximations': ['2014 以前除權息日報酬記 0（沒有 TWT49U 參考價；偏保守）', '2007～2014 0050 用當日還原收盤代替開盤成交（只影響底倉與空位 0050 的進出價，量小）',
                                   '漲跌停價固定 10%（2015-06 以前實際 7%）：可能把 7%～10% 之間的開盤當成可買／可賣；事前登記照凍結引擎不改，事後報告受影響筆數',
                                   '72,224 個格子（data_panel 有、Asoul 沒有）沒有低價、股數、漲跌價差（引擎改用收盤對前收）'],
                'missing_and_how_to_get': ['TWT49U 2007-01～2013-12：證交所 rwd/zh/exRight/TWT49U 每月一次共 84 次請求（約 10 分鐘）；本環境被斷線，需研究中心另找可連的環境或第三方鏡像',
                                           '0050 原始開盤 2007-01～2014-12：證交所 STOCK_DAY 只到 2010-01；2007～2009 需 MI_INDEX 逐日（約 750 次請求）或第三方資料']}}
(HERE / 'data_inventory.json').write_text(json.dumps(inv, ensure_ascii=False, indent=1, default=str))
print(json.dumps(inv['datasets']['monthly_revenue']['revenue_coverage_of_priced_twse_stocks'], ensure_ascii=False))
print(json.dumps({k: v for k, v in inv['datasets']['monthly_revenue']['survivorship'].items() if k != 'delisted_list'}, ensure_ascii=False))
