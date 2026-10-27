#!/usr/bin/env python3
"""事前設定（在看任何結果前寫入 results.json 的 PREREG；之後不改）。"""
import json, datetime
from pathlib import Path
HERE = Path(__file__).resolve().parent
PREREG = {
 'written_at': datetime.datetime.now().isoformat(timespec='seconds'),
 'rule_name': '蕭明道 凹洞量後攻擊量（量縮整理後放量），量縮不見高價／跌破 MA20／停損 −10%／最長 60 日出場',
 'sources': ['https://www.moneyweekly.com.tw/_Article?AID=128558', 'https://www.moneyweekly.com.tw/_Article?AID=53120',
             'https://www.businesstoday.com.tw/article/category/80401/post/201409190013/',
             'https://www.mirrormedia.mg/story/20200305fin002/index.html'],
 'type': '事件型（訊號觸發、移動出場）：主報告＝事件報酬（扣成本、對同池等權異常報酬、t 按進場月份分群）；另報對全上市普通股等權的異常報酬、對 0050 差額，以及日曆時間組合（所有在場事件等權、每日再平衡）的年化、日頻最大回撤、對 0050／同池等權的月超額 t、逐年',
 'data': {
   'panel': 'scratchpad/panel 上市日線（含已下市；2019–2023 只有存活者）；「有值檔數 < 300」的日子（2018-01-13）整列剔除後才做任何 rolling；flag 5 的價格視為缺值',
   'universe': '流動性池：上市普通股（4 碼數字、排除 91xx）；t 日 60 日均成交值（上市後停牌日記 0）>= NT$1,000 萬、原始收盤 >= 10 元、flag ∈ {0,1,2}',
   'price': '收盤、MA20、整理區間一律用 adj（含息還原收盤），避免除息假訊號；還原開盤＝open×adj/close_raw、還原最高＝high×adj/close_raw',
   'volume': '量 ＝ value / close_raw（股數）；value<=0 或停牌日量為缺值；任何視窗內有缺值，該視窗條件不成立（rolling min_periods＝窗長）',
 },
 'signal_t_close': {
   'notation': '「前 N 日」＝ t−N..t−1，不含 t；V20 ＝ mean(vol_{t−20..t−1})；V60 ＝ mean(vol_{t−60..t−1})',
   '1_dimple_each_day': 'vol_{t−k} < V20，k = 1..5（前 5 日每天都低於前 20 日均量）',
   '2_dimple_depth': 'mean(vol_{t−5..t−1}) <= 0.6 × V60（0.6 為研究員預設）',
   '3_consolidation': 'max(adj_{t−20..t−1}) / min(adj_{t−20..t−1}) − 1 <= 15%（15% 為研究員預設）',
   '4_attack_vs_prev5': 'vol_t > max(vol_{t−5..t−1})（量比前面 5 根大）',
   '5_attack_size': 'vol_t >= 1.5 × V20（1.5 為研究員預設）',
   '6_up_close': 'adj_t > adj_{t−1}',
   '7_above_ma20': 'adj_t >= MA20_t（MA20 含 t，adj 簡單均線）',
 },
 'execution': {
   'entry': 't+1 開盤（還原開盤）；t+1 無開盤價或開盤漲停（開盤/前收−1 >= 漲跌幅−0.6%，2015-06-01 前 7%、之後 10%）→ 放棄該訊號（不遞補）',
   'dedupe': '同一檔持有中（尚未到出場日）出現的新訊號忽略；出場日當天收盤可再觸發，於隔日開盤進場',
   'exit_rules': {
     'a_vol_no_new_high': '持有期間（s >= 進場日 e）若某日 m 的量 = 含 m 的 20 日最大量（vol_m >= max(vol_{m−19..m})），以 m 為參考日、參考價＝m 日還原最高；若 m+1..m+5 有任一日收盤 > 參考價 → 該參考解除；若到 m+5 收盤仍沒有 → m+6 開盤出場。參考日期間又出現新的 20 日最大量日，參考改為最新那天（重新計 5 日）。',
     'b_below_ma20': 'adj_s < MA20_s → s+1 開盤出場',
     'c_stop': 'adj_s <= 進場價 × 0.90（收盤判斷，面板無最低價）→ s+1 開盤出場（−10% 為研究員預設）',
     'd_time': '都沒發生 → 進場後第 60 個交易日（e+60）開盤出場',
   },
   'exit_price': '出場日有開盤價用開盤；停牌則用之後第一個開盤；已下市用最後收盤；跌停賣不掉不另處理（已知限制）',
   'cost': 'exlib.cost_rates：買＝0.1425%×0.6＋滑價 0.1%；賣＝0.1425%×0.6＋稅 0.3%＋滑價 0.1%（來回約 0.67%）；壓力情境 買 0.5%＋賣 0.5%（來回 1%）',
 },
 'variants': {
   'PRIMARY': '條件 1~7；出場 (a)(b)(c)(d)，(a) 的參考日只限持有期間（s >= e）',
   'ANCHOR_T': '同 PRIMARY，但 (a) 的參考日也可以是攻擊量當天 t（若 vol_t 為含 t 的 20 日最大量，t 即為第一個參考日，參考價＝t 日還原最高，從 t+1 起算 5 日）',
   'NO_A': '診斷：拿掉出場 (a)，只用 (b)(c)(d)',
   'HOLD20': '診斷：訊號同 PRIMARY，固定持有 20 個交易日（e+20 開盤出場），衡量進場訊號本身',
   'NO_DIMPLE': '診斷：拿掉條件 1、2（凹洞量），其餘同 PRIMARY，看凹洞量是否加分',
   'all_reported': '5 組全部報告；判定只看 PRIMARY，其他只作穩健性參考；不做任何事後調參',
 },
 'event_stats': {
   'raw': '出場價/進場價 − 1（還原）；net＝(1+raw)×(1−賣)/(1+買) − 1',
   'abnormal_EW': 'net − 同期間（進場開盤→出場開盤）同池等權報酬；同池＝前一日符合股池條件的上市普通股，每日等權、開盤對開盤',
   'abnormal_ALL': 'net − 同期間全上市普通股（不設流動性門檻，flag ∈ {0,1,2}）等權報酬（參考）',
   'vs_0050': 'net − 0050 含息（t 收盤→出場前一日收盤，0050 只有收盤）',
   't': '事件先按進場月份平均，再對月平均做 t（btlib.tstat）；t_date 僅參考',
 },
 'portfolio_stats': '日曆時間組合：每日持有所有在場事件、等權（每日再平衡），進場日報酬＝收/開，出場日＝出場價/前收，進出各按 1/n 權重扣成本；無持股日現金 0 報酬。報年化、日頻最大回撤、對 0050 與同池等權（收對收）的月超額 t、逐年、曝險天數比例',
 'windows': {'select': '2005-01-01~2019-04-30（以訊號日歸屬）', 'valid': '2019-05-01~2026-10-22', 'valid_2024_2025': '2024-01-01~2025-12-31',
             'valid_ex_src3': '驗證期剔除「訊號 60 日回看視窗或持有期」碰到 src 3（2024-02-15~2025-09-23）的事件（僅事件層）'},
 'src3_check': '另做資料抽查：src 3 期間 value/close_raw 的量與官方期間（2023、2025-10 起）的全市場量比例、src 切換日前後量的跳動、src 3 期間觸發數相對其他年份是否異常；history.csv.gz 與面板在 2025-09-24 起重疊日抽樣比對',
 'verdict_rule': 'PRIMARY 在挑選期與驗證期都要：事件淨異常報酬（對同池等權）> 0 且 t_month >= 2，且日曆組合淨年化 >= 0050 同期年化 → 可用；只有一段過 → 待觀察；否則不可用；某段事件 < 30 筆或事件月份 < 24 → 無法測試。',
 'known_limits': ['2019–2023 面板只有存活者（驗證期偏樂觀）', '2024-01-02~2024-02-14 無資料（rolling 跨缺口）',
                  'src 3 期間收盤為 13:20 近似、最高價與成交值亦為近似', '面板無最低價，停損以收盤判斷',
                  '成交量用 value/close_raw 推算，減資／分割日前後股數口徑會跳動', '出場不考慮跌停賣不掉'],
}
if __name__ == '__main__':
    f = HERE / 'results.json'
    d = json.loads(f.read_text(encoding='utf-8')) if f.exists() else {}
    assert 'PREREG' not in d, 'PREREG 已存在，不覆寫'
    d = {'PREREG': PREREG, **d}
    f.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding='utf-8')
    print('PREREG written')
