#!/usr/bin/env python3
"""彙整 parts/*.json → results.json（含 C 總結表）。python3 -I -B assemble.py。只讀其他資料夾。"""
import sys; from pathlib import Path as _P; sys.path.insert(0, str(_P(__file__).resolve().parent))
import json
import pandas as pd
from common import *                                    # noqa: E402

P = {k: json.loads((HERE / 'parts' / f'{k}.json').read_text()) for k in ('prices', 'exright', 'revenue_delist', 'e50_chain', 'pre_xflag')}
man = json.loads((HERE / 'data_manifest.json').read_text())
meta = json.loads((TMP / 'off_meta.json').read_text()) if (TMP / 'off_meta.json').exists() else None
b1, b2, b3 = P['prices']['B1_calendar'], P['prices']['B2_coverage'], P['prices']['B3_sanity']
ex = P['exright']; b5, b6 = P['revenue_delist']['B5_revenue'], P['revenue_delist']['B6_delisted']; b7, b8 = P['e50_chain']['B7_0050'], P['e50_chain']['B8_daily_chain']; px = P['pre_xflag']
vk = ex['vs_daily_k_official_twse']; fs = b8['official_rev_fetch_summary']
n_x = sum(v['n'] for v in px['by_class'].values())
V = {}
V['B1_交易日曆'] = dict(result='通過', numbers=dict(n_days=b1['calendar']['n'], first=b1['calendar']['first'], last=b1['calendar']['last'],
    panel_days_missing_vs_twii=b1['vs_twii_forward_copy']['n_in_panel_not_ref'], twii_rows_not_trading_days=b1['vs_twii_forward_copy']['in_ref_not_panel'],
    panel_days_missing_in_0050=b1['vs_0050_forward_copy']['in_panel_not_ref'], e50_rows_not_trading_days=b1['vs_0050_forward_copy']['in_ref_not_panel'],
    data_panel_missing_days=b1['vs_data_panel']['n_in_panel_not_ref'], official_panel_has_2024_gap_days=b1['data_panel_gap_2024']['official_days_2024_01_02_to_02_14'],
    engine_cache_cells_diff=b1['engine_cache_vs_rebuild']['n_cells_diff']),
    note='官方面板與加權指數檔逐日比對：面板的 4,883 個交易日加權指數全有；加權指數檔多 1 列非交易日（2026-09-20 週日）。官方面板不缺 2024-01-02～02-14（25 天全有），缺的是舊研究面板 data_panel。0050 檔缺 2025-06-11～17（分割停牌，引擎已處理）、多 3 列非交易日。')
V['B2_個股日K涵蓋'] = dict(result='通過', numbers=dict(drop_days_twse_listed=len(b2['drops_listed_twse']), drop_days_twse_priced=len(b2['drops_priced_twse']), drop_days_tpex_priced=b2['drops_priced_tpex'],
    by_year={y: dict(n_codes=v.get('official_n_codes_twse', v.get('n_codes_twse')), median_daily_priced=v.get('official_median_daily_priced', v.get('median_daily_priced')),
                     miss_close_pct=round(v.get('official_miss_close_pct', v.get('miss_close_pct')), 3), miss_open_pct=round(v.get('official_miss_open_pct', v.get('miss_open_pct')), 3),
                     zero_amount_pct=round(v.get('official_zero_amount_pct', v.get('zero_amount_pct')), 3), miss_change_pct=round(v.get('official_miss_change_pct', v.get('miss_change_pct')), 3)) for y, v in b2['by_year'].items()}),
    note='上市檔數沒有任何一天突然掉（門檻：低於前 20 日中位數 3%）。上櫃有 2 天（2018-11-05、2018-12-22）有價檔數少約 4.5%～4.8%，不在正式股池。收盤／開盤缺值 0.18%～1.45%（幾乎都是當日無成交）；成交金額缺值 0%。2015 前漲跌價差空白 8%～12%，138,307 格中 138,304 格收盤＝前收（空白＝平盤），不是缺資料。')
V['B3_價格合理性'] = dict(result='通過（上市）；上櫃有問題（不在正式股池）', numbers=dict(
    twse_over_limit_vs_prev_close=dict(pre2015=b3['pre2015_twse']['by_cat'], official=b3['official_twse']['by_cat']),
    twse_over_limit_vs_ref_price_excl_first5=dict(pre2015=b3['pre2015_twse']['n_over_limit_vs_ref_excl_first5'], official=b3['official_twse']['n_over_limit_vs_ref_excl_first5']),
    tpex=dict(by_cat=b3['official_tpex']['by_cat'], over_limit_vs_ref_excl_first5=b3['official_tpex']['n_over_limit_vs_ref_excl_first5'], abs_ref_ret_gt_50pct=b3['official_tpex']['n_abs_ref_ret_gt_50pct']),
    impossible=dict(official_twse=b3['impossible_official_twse'], official_tpex=b3['impossible_official_tpex'], pre2015_twse=b3['impossible_pre2015_twse'])),
    note='以引擎實際用的口徑（收盤 ÷ 參考價，參考價＝收盤 − 漲跌價差）：上市 2007～2026 超過漲跌停且非掛牌前 5 日的筆數＝0。以「收盤 ÷ 前一成交收盤」看，上市未解釋 24 筆，全部落在颱風停市後第一個交易日（除權息日順延，事件檔記在停市日），漲跌價差本身正確。收盤 ≤ 0、成交金額為負、最高 < 最低、收盤在高低之外：全部 0 筆。有量但無收盤價（只有盤後／鉅額成交）上市 2015 起 5,304 格、2015 前 3,079 格，引擎當無成交。')
V['B4_除權息涵蓋'] = dict(result='有問題', numbers=dict(twt49u=dict(first=ex['twt49u']['first'], last=ex['twt49u']['last'], by_year_4digit=ex['twt49u']['by_year_4digit'], sha_mismatch=len(ex['twt49u']['sha_mismatch'])),
    repo_exright=dict(first=ex['repo_exright']['tsd']['first'], last=ex['repo_exright']['tsd']['last'], by_year_4digit=ex['repo_exright']['by_year_4digit'], unreadable_days=len(ex['repo_exright']['unreadable_days'])),
    cross=dict(window=ex['cross_twt49u_vs_repo']['window'], n_both=ex['cross_twt49u_vs_repo']['n_both'], n_only_twt49u=ex['cross_twt49u_vs_repo']['n_only_twt49u'], n_only_repo=ex['cross_twt49u_vs_repo']['n_only_repo'], ref_price_equal_pct=ex['cross_twt49u_vs_repo']['ref_price_equal_pct']),
    vs_daily_k=dict(n_zero_change_but_moved=vk['n_zero_change_but_close_moved'], by_category=vk['by_category'], explained_pct=vk['explained_incl_prev_untraded_pct'],
                    events_traded=vk['events_traded_twse'], events_zero_change_in_k=vk['events_with_zero_change_in_k'], events_k_already_ref=vk['events_k_change_already_uses_ref'], events_other=vk['events_other']),
    after_archive=dict(n_trading_days=ex['after_twt49u_archive']['n_trading_days_uncovered'], n_events_twse_4digit=ex['after_twt49u_archive']['n_events_4digit_twse'], events=ex['after_twt49u_archive']['events'], hit_holdings=ex['after_twt49u_archive'].get('hit_current_holdings')),
    pre2014=dict(x_cells_2007_2014=n_x, by_class={k: dict(n=v['n'], n_pool_like=v['n_pool_like'], mean_gap_pct=v['mean_gap_pct']) for k, v in px['by_class'].items()}, check_2014=ex['pre2015_xflag']['check_2014'],
                 s75_trades_overlap={k: dict(n_trades_pre2015=v['n_trades_pre2015'], n_x_cells_held=v['n_x_cells_held'], by_class=v['by_class']) for k, v in px['s75_trades_overlap'].items()}),
    events_on_non_trading_days=dict(n=ex['events_on_non_trading_days_repo_exright']['n'], days=ex['events_on_non_trading_days_repo_exright']['days'])),
    note='兩個除權息來源在 2015-07-02～2026-10-29 重疊期 9,870 筆逐筆相同（參考價 100%）。問題三項：① TWT49U 封存停在 2026-10-29，之後 15 個交易日有 6 個上市事件，回測引擎與名單產生器都只讀這份封存；② 2014-01-16 以前沒有任何參考價，2007～2014 共 8,759 格 X 記號日報酬記 0；③ 官方倉庫 exright 有 42 天本機讀不到（2015 上半年 40 天）＝無法驗證。')
cy = b5['coverage_twse_by_year']
V['B5_月營收涵蓋'] = dict(result='有問題（已知缺口，數字如下）', numbers=dict(first=b5['first'], last=b5['last'], coverage_twse_median_by_year={y: v['median'] for y, v in cy.items()}, coverage_tpex_by_year=b5['coverage_tpex_by_year'],
    hole_months_twse=b5['hole_months_twse'], hole_months_tpex=b5['hole_months_tpex'], zero_or_negative=dict(n=b5['zero_or_negative_4digit']['n'], n_zero=b5['zero_or_negative_4digit']['n_zero'], n_negative=b5['zero_or_negative_4digit']['n_negative'], n_codes=b5['zero_or_negative_4digit']['n_codes']),
    duplicates=dict(same=b5['duplicates']['same_month_market_code_rows'], cross_market=b5['duplicates']['cross_market_month_code_pairs']), inner_missing_twse_listed_by_year=b5['inner_missing_code_months']['twse_listed_at_that_month_by_year'],
    cross_two_sources=dict(n_pairs=b5['cross_tw_revenue_vs_revenue_hist']['n_pairs'], exact_pct=b5['cross_tw_revenue_vs_revenue_hist']['exact_pct'], diff_gt_10pct=b5['cross_tw_revenue_vs_revenue_hist']['diff_gt_10pct'],
                           diff_ge_1pct_by_year={y: v['diff_gt_1pct'] for y, v in b5['cross_tw_revenue_vs_revenue_hist']['by_year'].items() if v['diff_gt_1pct']}),
    latest_month=b5['latest_months_from_revenue_hist'], first_print=b5['first_print_vs_revised']),
    note='上市涵蓋率 2007～2012 約 95%～98%，2013 起掉到 87%～91%：缺的是 KY／外國企業（營收檔沒有）與 2013 後下市公司。缺洞月：上市 2011-01（89.97%）、2011-02（91.21%）、2012-01（91.26%）；上櫃 2018-10 只有 16.93%。營收 ≤ 0 共 2,101 筆（引擎當缺值）；同公司同月多筆 0 筆。與官方倉庫彙總表交叉 129,279 筆，99.987% 相同；2015～2024 完全相同，差異 17 筆都在 2025～2026。')
V['B6_下市股'] = dict(result='有問題', numbers=dict(list=b6['official_list'], panel=b6['panel'], list_vs_panel={k: v for k, v in b6['list_vs_panel'].items()},
    price_gap_trading_days=b6['price_coverage_before_delist']['last_row'], n_price_gap_gt_5=b6['price_coverage_before_delist']['n_last_row_gap_gt_5'],
    revenue_gap=b6['revenue_coverage_before_last_trade'], survivorship=b6['survivorship_existing_research'] + [dict(source='agents/s1_shortterm_leaderboard/FINDINGS.md 註記 7（現行 SURGE 策略）',
        quote='已下市 58 檔（多數是被收購或合併，下市前一年平均 −2.7%）選不到，估計偏誤約 1.3 個百分點（鬆上界 8.8）；另外 82 檔 KY 股也沒有營收資料')]),
    note='下市清單只到 2024-09-02（之後面板推得 7 檔離開上市）；清單內 9 檔 2007～2013 下市股在價格面板整檔沒有；另有 2 檔面板推得離開但清單沒有（2358、2443）。價格：清單與面板都有的 120 檔，面板最後一列距官方下市日中位數 6 個交易日（73 檔超過 5 天，多為先停止買賣）。營收：129 檔離開上市的股票沒有任何一檔營收涵蓋到最後交易前 3 個月內。')
V['B7_0050'] = dict(result='通過（2015 以前配息明細無法驗證）', numbers=dict(n_nan=b7['n_nan'], missing_trading_days=b7['missing_trading_days'], extra_non_trading_rows=b7['extra_non_trading_rows'], max_abs_daily_ret_pct=b7['max_abs_daily_ret_pct'],
    split=b7['split']['detected'], dividend_count_by_year_twt49u=b7['twt49u_0050_count_by_year'], dividend_count_by_year_implied_from_adj=b7['implied_dividend_count_by_year'],
    twt49u_vs_implied=dict(n=b7['twt49u_vs_implied']['n_twt49u_2015_plus'], n_matched=b7['twt49u_vs_implied']['n_matched'], mismatches=[r for r in b7['twt49u_vs_implied']['rows'] if not r['match']]),
    site_nav_non_trading_dates=b7['site_nav_json'].get('dates_not_trading_days') if isinstance(b7['site_nav_json'], dict) else None, navs_official=b7['navs_official']),
    note='含息還原價沒有缺值；缺的 5 天是 2025-06-11～17 分割停牌。2025-06-18 一拆四，還原價連續（當日 +0.86%）。2015 起 22 次配息，用原始價對還原價推回的息值和 TWT49U 逐次比對，21 次相同、1 次差 0.023 元（2026-07-21：0.577 對 0.6）。多出的 3 列非交易日中 2026-09-20（週日）數值異常，但引擎與網站都只取交易日，不受影響。')
V['B8_每日更新鏈'] = dict(result='有問題', numbers=dict(now_taipei=b8['now_taipei'], latest_price_day=b8['tsd']['daily_last'], expected=b8['expected_last_trading_day_weekday_rule'], price_day_ok=b8['price_day_is_latest_trading_day'],
    site_data_last_day=b8['site']['status']['data_last_day'], exright_repo_last=b8['tsd']['exright_last_file'], exright_covers=b8['exright_covers_latest_price_day'], twt49u_archive_last=b8['twt49u_archive_last'],
    official_rev_fetch=fs, forward_log_days_missing=b8['forward_log_days_missing'], forward_rc_nonzero=[dict(log=l['log'], rc=l['rc']) for l in b8['forward_logs'] if l['rc'] not in (0,)]),
    note='最新股價日 2026-11-19＝最近交易日；網站資料最後一天 2026-11-19；官方倉庫除權息檔到 2026-11-20。問題：官方月營收抓取自 2026-11-03 起 14 次、11 個不同日全部失敗（代理 403，三個官方主機都不在允許清單），從未成功過；前瞻紀錄檔缺 2026-11-11（官方訊號日）與 11-13 兩天；TWT49U 封存落後。')
summary = [
 dict(rank=1, item='最新月份營收只有一個來源，而且放在暫存區', result='有問題', impact='2026-12-11 的名單需要 2026-10、2026-11 兩個月營收，只能從 /tmp 底下的 tsd（官方倉庫 clone）讀；官方直接抓取 14 次全失敗。tsd 被清掉且當晚 GitHub 連不上＝出不了名單。', action='把 tsd 搬到固定路徑；請產品擁有者決定是否開放三個官方主機；12/11 前演練一次備援（公開資訊觀測站人工下載彙總表）'),
 dict(rank=2, item='TWT49U 封存停在 2026-10-29', result='有問題', impact='之後 15 個交易日有 6 個上市除權息事件沒被修正（例：6165 在 11-02 配息 5 元、前收 47.9，還原價會被少算約 10.4%）；目前持股與 2026-11 名單前 20 名沒有碰到。12/11 名單的季線判斷會用到這段。', action='12/11 前補 2026-10-30 之後的月檔並重算 SHA256SUMS（README 既有規定）；或讓名單產生器比照 live_tr.py 先讀官方倉庫 exright'),
 dict(rank=3, item='2013 起上市股營收涵蓋率只有 87%～91%（KY 與下市公司沒有營收）', result='有問題（已知）', impact='回測與實盤一致地選不到這些股票。既有估計：下市 58 檔的偏誤約 1.3 個百分點／年（鬆上界 8.8，S1 註記 7）；2026-09 月底有 98 檔上市股沒有營收（多數是 KY／外國企業與新掛牌），其中 46 檔過收盤 ≥ 10 元、60 日均額 ≥ 1,000 萬的門檻；這部分對報酬的影響未量化。', action='補 KY 與下市公司營收後重跑（BACKLOG 已列）；對外文件維持加註'),
 dict(rank=4, item='回測用更正後營收、實盤用首次公布值', result='有問題（已量化）', impact='S39：2024-07～2026-10 年化 61.95% → 61.17%（−0.78 個百分點）；2024-06 以前無法量化。本稽核交叉核對：2015～2024 兩來源 100% 相同，差異 17 筆都在 2025～2026。', action='每月名單定稿時存當月彙總表快照，累積首次公布值'),
 dict(rank=5, item='2014-01-16 以前沒有除權息參考價', result='有問題（只影響 2008～2014 那一段）', impact='2007～2014 有 8,759 格 X 記號日報酬記 0；S75 已發布交易持有期間碰到 24 格（V_BRIDGE）／26 格（V_MASK），都是像除權息的那一類，當日價差合計 −93.93／−96.03 個百分點。對 2015 起的正式數字沒有影響。', action='由重跑引擎的稽核員（R75）判定是否需要補 2014 前除權息表'),
 dict(rank=6, item='正式資料大多不在版本控制內', result='有問題', impact='官方日 K 封存（.gitignore）、月營收檔、0050 檔都在主倉庫之外或被排除；容器回收後要靠上游 GitHub 倉庫仍在且沒改寫歷史才能重現 48.76%。', action='見清冊「建議搬到哪裡」欄；至少把 sha256 寫進正式結果檔'),
 dict(rank=7, item='下市清單不完整', result='有問題', impact='清單停在 2024-09-02；9 檔 2007～2013 下市股沒有價格。只影響 2008～2014 那一段的存活者偏誤與之後的存活者研究。', action='補清單到最新；缺的 9 檔註記在 S75'),
 dict(rank=8, item='官方倉庫 exright 有 42 天讀不到', result='無法驗證', impact='2015 上半年 40 天無法和 TWT49U 交叉；正式引擎讀的是 TWT49U，不受影響。', action='下次連網時補抓並存快照'),
 dict(rank=9, item='交易日曆、個股日 K 涵蓋、價格合理性（上市）、0050', result='通過', impact='官方面板不缺 2024-01-02～02-14；引擎快取與封存重建逐格相同；上市沒有不可能值。', action='無'),
]
out = dict(written_at=pd.Timestamp.now(tz='UTC').isoformat(), scope='資料完整性與資料設定清冊（不含引擎重跑、逐筆交易核對、已發布數字對帳）',
           discipline=dict(network='未連外網（git 設 GIT_NO_LAZY_FETCH=1＋protocol.allow=never；本機沒有的 blob 記為無法驗證）', python='python3 -I -B',
                           peak_rss_mb={k: v.get('maxrss_mb') for k, v in P.items() if isinstance(v, dict) and 'maxrss_mb' in v}, build_cache=meta and dict(runtime_s=meta['runtime_s'], maxrss_mb=meta['maxrss_mb'])),
           manifest_summary=dict(n_items=len(man['items']), single_points=man['single_points'], in_scratch=man['in_scratch'], not_version_controlled=man['not_version_controlled']),
           verdicts=V, summary_table=summary, detail=dict(B1=b1, B2=b2, B3=b3, B4=ex, B4_pre_xflag=px, B5=b5, B6=b6, B7=b7, B8=b8))
jdump(out, HERE / 'results.json')
for k, v in V.items():
    print(k, '→', v['result'])
print('peak', out['discipline']['peak_rss_mb'], out['discipline']['build_cache'])
