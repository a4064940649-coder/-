#!/usr/bin/env python3
"""合併各項結果 → results.json、findings_table.csv。python3 -I -B merge_results.py"""
import json, csv
from pathlib import Path
H = Path(__file__).resolve().parent
L = lambda f: json.load(open(H / f))
t, x, e, s, r, m, m2 = L('results_trades.json'), L('results_extra.json'), L('results_0050.json'), L('results_site.json'), L('results_rev.json'), L('results_misc.json'), L('results_misc2.json')
i1, i6, ra = t['item1'], t['item6'], t['recalc_all']
rows = [
 ('1 價格交叉核對（進出場日開盤價）', '通過（2008～2009 無法獨立驗證）',
  f"成交點 {i1['n_fills']} 個；有獨立第二來源 {i1['covered']} 個（{i1['coverage']*100:.1f}%），其中兩個以上獨立來源 {i1['two_plus_indep']} 個；只有同源副本 {i1['same_origin_only']} 個（2008～2009 的 132 個＋2010～2011 的 9 個）。"
  f"不符 {i1['n_mismatch']} 個、最大差 {i1['max_abs_diff']:.2f} 元，全部是第二來源自己的問題（Yahoo 5 分 K 除權息前回溯還原 2 個、首根 5 分 K 非開盤價 1 個、FinMind 9921 還原序列 4 個），引擎用的官方值都有另一來源或除權息表佐證。",
  '不影響'),
 ('2 最賺／最賠各 15 筆逐筆重算', '通過',
  f"獨立重算全部 {ra['closed_n']} 筆已平倉交易：買進成本 {ra['n']} 筆最大相對差 {ra['cost_max_rel_diff']:.1e}；報酬 {ra['ret_absdiff_le_1e9']} 筆差 ≤1e-9，其餘 {ra['ret_absdiff_gt_1e4']} 筆全是單日成交值 5% 上限造成的分批賣出（{ra['n_partial_explained']}/{ra['ret_absdiff_gt_1e4']} 筆已驗證）。"
  f"最賺／最賠（金額、報酬率各 15）＋指名 6 檔共 {t['item2_summary']['n_unique']} 筆：{t['item2_summary']['n_unique'] - t['item2_summary']['n_ret_diff_gt_1e6']} 筆差 <1e-6，1 筆（3003）為分批賣出。2408、3653、3017、2344、2059、1325 持有期間沒有分割、面額變更、減資；只有現金股利（3017 9.93 元、2059 51 元、1325 0.3 元、2344 2025-11 那筆 0.5 元）。",
  '不影響'),
 ('2b 減資恢復買賣日報酬記 0', '有問題（小、偏保守）', '8249（2024-09-12→2024-10-21）恢復買賣日官方日 K 漲跌價差＝0，引擎當日報酬記 0；第二來源（TWTAUU）參考價 53.27，開盤 58.5，這筆實際約多 9.8%（V_BRIDGE 約 +15 萬元）。3545（2015-12-15，只在 V_MASK）同型，本地沒有減資參考價 → 無法驗證方向。', '這筆少算約 15 萬元，約當時淨值的 0.8%（偏保守）'),
 ('3 0050 基準逐年含息報酬', '有問題（小）；2008 無法驗證',
  f"2016～2025 引擎來源對「官方日 K＋證交所除息表＋分割」自建含息淨值，逐年差 ≤{max(abs(v) for k, v in e['vs_engine_source']['B_官方日K＋TWT49U自建']['diffs_pp'].items() if k != '2026'):.3f} 個百分點；2025-06 一拆四正確（比例 {e['B_events']['splits'][0]['ratio']:.4f}）。"
  f"2026：引擎來源 {e['bench_effect']['e0050_2026_A']*100:.2f}%、官方自建 {e['bench_effect']['e0050_2026_B']*100:.2f}%，差 {(e['bench_effect']['e0050_2026_A']-e['bench_effect']['e0050_2026_B'])*100:.2f} 個百分點——引擎來源自 2026-08-12 起有 {e['A_drift_2026']['n_days_ratio_changed']} 天與官方收盤的日報酬不同，累計多 {e['A_drift_2026']['drift']*100:.2f}%，期間沒有除息。"
  "2009-03-10～2015 對第三方原始收盤：只在每年除息日不同，推算配息 1.0／2.2／1.95／1.85／1.35／1.55／2.0 元。2008-01～2009-03 本地沒有可用的第二來源。",
  f"0050 期末 {e['bench_effect']['final_0050_nav']/1e4:.0f} 萬 → {e['bench_effect']['final_0050_nav_if_B']/1e4:.0f} 萬、年化 {e['bench_effect']['cagr_0050_A']*100:.2f}% → {e['bench_effect']['cagr_0050_if_B']*100:.2f}%；策略淨值因持有 0050 也高估，上限 0.58%。結論不變"),
 ('4 月營收兩來源＋SURGE 重算', '通過（2008～2013 無法驗證）',
  f"抽 60 個（每年 3 個＋3 個）：通過 {r['sample60']['verdict'].get('通過', 0)}、無法驗證 {r['sample60']['verdict'].get('無法驗證（本地沒有第二來源）', 0)}（2008～2013 各 3 個，本地只有一個來源）、不一致 {r['sample60']['verdict'].get('不一致', 0)}。"
  f"普查 2015-01～2026-09：{r['census']['官方倉庫revenue_hist']['n_both']} 格對官方倉庫當期值，{r['census']['官方倉庫revenue_hist']['exact']} 格相同（{r['census']['官方倉庫revenue_hist']['match_rate']*100:.3f}%），不同 {r['census']['官方倉庫revenue_hist']['n_diff_gt_1e6']} 格都在 2025～2026。"
  f"SURGE：全部 {r['surge_recalc']['n']} 筆交易由原始營收重算，最大差 {r['surge_recalc']['max_abs_diff']:.1e}（含 V_BRIDGE 2013～2015 的 {r['surge_recalc']['b2_rows']} 筆 B2 校正）。交易用到的營收格（當月）{r['trade_rev_cells']['month_m']['checked']} 格有第二來源、{r['trade_rev_cells']['month_m']['ok']} 格相同，不同的 3 格是 2014 年只有次年重編值可比（差 ≤0.07%）。",
  '不影響'),
 ('5 前視檢查', '通過',
  f"全部 {r['item5_dates_all']['n']} 個訊號月：訊號日都是「期限日（次月 10 日，遇休市順延；2019-01、2023-03 用例外表）後第一個交易日」。隨機 40 個月、{r['item5']['n_stock_rows']} 檔入選股：SURGE 只用到當月（含）以前營收、訊號日收盤站上 60 日均線、20 日報酬 >10%、60 日均成交值 ≥1,000 萬、收盤 ≥10 元、當日上市，{r['item5']['stock_rows_ok']}/{r['item5']['n_stock_rows']} 全部成立（只用訊號日以前的價格重算）；其中有買進的 {r['item5']['bought']} 筆，分數相符 {r['item5']['score_match']}、成交在次一交易日開盤 {r['item5']['ent_is_next_day']}。全部 {ra['n']} 筆交易的進場日都是訊號日的下一個交易日。",
  '不影響（營收是事後更正版而非首次公布值，屬已知限制，本稽核無法以本地資料驗證 2024-06 以前）'),
 ('6 不可能的交易', '通過（有 6 筆已揭露的漲停買進）',
  f"進場日無成交量 {i6['ent_no_volume']['unique']} 筆（其中 10 筆進場日來自 data_panel 補格、沒有成交股數欄，改以成交金額 > 0 判定）；出場日無成交量 {i6['exit_no_volume']['unique']} 筆；開盤即漲停仍買進：照引擎 10% 規則 {i6['buy_at_open_limup_10pct_rule']['unique']} 筆，照當時實際 7% 漲停（2015-06 前）{i6['buy_at_open_limup_real']['unique']} 筆（與 S75 自己揭露的 6 筆相同，V_BRIDGE 這 6 筆損益合計 {i6['limup_pnl_sum_by_version']['V_BRIDGE']:.0f} 元）；開盤即跌停仍賣出 {i6['sell_at_open_limdn_real']['unique']} 筆；訊號日／進場日／出場日不在上市名單 {i6['not_twse_on_signal']}／{i6['not_twse_on_entry']}／{i6['not_twse_on_exit']} 筆；創新板 {i6['innov_on_entry']} 筆；下市後仍持有 {i6['held_after_last_data']} 筆；零股時代前買零股 {i6['lot_violation']} 筆；買進超過當日成交值 5% {i6['n_buy_share_gt_5pct']} 筆。持有期間有停牌的 3 筆（2457、3545、8249：停牌 7～13 個交易日後恢復、參考價跳升；本地只有 8249 有第二來源證實是減資）。",
  '不影響（拿掉 6 筆漲停買進，損益差約 1.4 萬元）'),
 ('7 網站數字對帳', '通過',
  f"對 navs_official.csv.gz 重算 {len(s['checks'])} 項全部相符：2019 起 {s['recalc']['cagr_since2019']*100:.2f}%、全期 {s['recalc']['cagr_full']*100:.2f}%、最大回撤 {s['recalc']['maxdd'][0]*100:.1f}%（{s['recalc']['maxdd'][1]}→{s['recalc']['maxdd'][2]}）、0050 回撤 {s['recalc']['maxdd_0050'][0]*100:.1f}%、落後 0050 最多 {-s['recalc']['roll252']['worst']:.1f}、落後時間 {s['recalc']['roll252']['frac_behind']:.1f}%；yearly.json 12 年 × 3 欄差 ≤0.05；nav.json 584 列最大差 0.5 元。",
  '不影響（網站 0050 的 2026 年 67.2% 沿用同一個有漂移的來源，官方日 K 算是 66.3%）'),
]
with open(H / 'findings_table.csv', 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.writer(f); w.writerow(['項目', '結果', '證據', '影響']); w.writerows(rows)
out = dict(task='S75 連續帳戶 2008～2026 資料面稽核', date='2026-11-20', table=[dict(item=a, verdict=b, evidence=c, impact=d) for a, b, c, d in rows],
           item1=i1, item2=dict(summary=t['item2_summary'], recalc_all=ra, lists=t['item2'], events_check=x, xday=t['xday']), item3=e, item4=dict(sample60=r['sample60'], census=r['census'], surge=r['surge_recalc'], trade_cells=r['trade_rev_cells'], winners=r['winners_surge']),
           item5=dict(dates=r['item5_dates_all'], sample40=r['item5'], seed=r['seed']), item6=i6, item7=s, misc=m, misc2=m2,
           memory_mb=dict(audit_trades=t['maxrss_mb'], audit_rev=1594, audit_extra=1457, audit_misc=1765, audit_0050=1179, audit_site=108))
(H / 'results.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
for a in rows: print(a[0], '|', a[1], '|', a[2][:400], '|', a[3])
