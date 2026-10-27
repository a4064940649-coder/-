#!/usr/bin/env python3
"""事前設定（在看任何結果前寫入 results.json 的 PREREG；之後不改）。"""
import json, datetime
from pathlib import Path
HERE = Path(__file__).resolve().parent
PREREG = {
 'written_at': datetime.datetime.now().isoformat(timespec='seconds'),
 'rule_name': '朱家泓 多頭排列＋漲幅 3.5% 長紅過前高，跌破 5 日線出場',
 'sources': ['https://www.cmoney.tw/notes/note-detail.aspx?nid=68237',
             'https://smart.businessweekly.com.tw/Magazine/detail.aspx?id=7010779&p=2',
             'https://www.ctee.com.tw/news/20190721700238-430298',
             'https://www.mirrormedia.mg/story/20180719fin018', 'https://hao.cnyes.com/ch/172825'],
 'type': '事件型（訊號觸發、移動出場）：主報告＝事件報酬（扣成本、對同池等權異常報酬、t 按進場月份分群）；另報對全上市普通股等權的異常報酬，以及日曆時間組合（所有在場事件每筆等額、每日再平衡）的年化、日頻最大回撤、對 0050／同池等權的月超額 t、逐年',
 'data': {
   'panel': 'scratchpad/panel 上市日線（含已下市；2019–2023 只有存活者）；「有值檔數 < 300」的日子（2018-01-13）整列剔除後才做任何 rolling；flag 5 的價格視為缺值',
   'universe': '流動性池：上市普通股（4 碼數字、排除 91xx）；t 日 60 日均成交值（上市後停牌日記 0）>= NT$1,000 萬、原始收盤 >= 10 元、flag ∈ {0,1,2}',
   'price': '均線、漲幅一律用 adj（含息還原收盤），避免除息造成假訊號；還原開盤＝open×adj/close_raw、還原最高＝high×adj/close_raw',
   'volume': '量 ＝ value / close_raw（股數）；停牌日量為缺值',
 },
 'signal_t_close': {
   '1_bull_alignment': 'MA5_t > MA10_t > MA20_t > MA60_t（adj 簡單均線，min_periods＝窗長）',
   '2_ma20_rising': 'MA20_t > MA20_{t-5}',
   '3_long_red': 'adj_t / adj_{t-1} − 1 >= 3.5%',
   '4_break_prev_high': 'adj_t > 還原最高_{t-1}（收盤過前一日最高價）',
   '5_volume_up': 'vol_t >= 1.5 × mean(vol_{t-5..t-1})（教學只說量增；1.5 倍為研究員預設）',
 },
 'execution': {
   'entry': 't+1 開盤（還原開盤）；t+1 無開盤價或開盤漲停（開盤/前收−1 >= 漲跌幅−0.6%，2015-06-01 前 7%、之後 10%）→ 放棄該訊號（不遞補）',
   'dedupe': '同一檔持有中（尚未到出場日）出現的新訊號忽略；出場日收盤後可再觸發',
   'exit_order': '從進場日 e 起逐日 s 檢查：(c) 若為獲利目標版本，s 日還原最高 >= 進場價×(1+目標) → s 日以 max(還原開盤_s, 目標價) 出場（盤中觸價，優先於收盤條件）；'
                 '(a) adj_s < MA5_s 或 (b) adj_s < 訊號日 t 的還原開盤（停損，教學用長紅 K 低點，面板無最低價以開盤代替）→ s+1 開盤賣；'
                 '(d) 都沒發生 → 進場後第 60 個交易日（e+60）開盤賣',
   'exit_price': '出場日有開盤價用開盤；停牌則用之後第一個開盤；已下市用最後收盤；跌停賣不掉不另處理（已知限制）',
   'cost': 'exlib.cost_rates：買＝0.1425%×0.6＋滑價 0.1%；賣＝0.1425%×0.6＋稅 0.3%＋滑價 0.1%（來回約 0.67%）；壓力情境 買 0.5%＋賣 0.5%（來回 1%）',
 },
 'variants': {
   'PRIMARY': '條件 1~5；出場 (a)(b)(d)（主要賣訊跌破 MA5）',
   'TP8': 'PRIMARY 再加 (c) 短線 +8% 觸價出場',
   'TP15': 'PRIMARY 再加 (c) 中線 +15% 觸價出場',
   'NOVOL': '診斷：拿掉條件 5（量增），其餘同 PRIMARY',
   'all_reported': '4 組全部報告；判定只看 PRIMARY，其他只作穩健性參考；不做任何事後調參',
 },
 'event_stats': {
   'raw': '出場價/進場價 − 1（還原）；net＝(1+raw)×(1−賣)/(1+買) − 1',
   'abnormal_EW': 'net − 同期間（進場開盤→出場開盤；盤中觸價出場則延伸到出場日收盤）同池等權報酬；同池＝前一日符合股池條件的上市普通股，每日等權、開盤對開盤',
   'abnormal_ALL': 'net − 同期間全上市普通股（不設流動性門檻，flag ∈ {0,1,2}）等權報酬（參考）',
   'vs_0050': 'net − 0050 含息（t 收盤→出場前一日收盤，0050 只有收盤）',
   't': '事件先按進場月份平均，再對月平均做 t（btlib.tstat）；t_date 僅參考',
 },
 'portfolio_stats': '日曆時間組合：每日持有所有在場事件、等權（每日再平衡），進場日報酬＝收/開，出場日＝出場價/前收，進出各按 1/n 權重扣成本；無持股日現金 0 報酬。報年化、日頻最大回撤、對 0050 與同池等權（收對收）的月超額 t、逐年、曝險天數比例',
 'windows': {'select': '2005-01-01~2019-04-30（以訊號日歸屬）', 'valid': '2019-05-01~2026-10-22', 'valid_2024_2025': '2024-01-01~2025-12-31',
             'valid_ex_src3': '驗證期剔除持有期碰到 src 3（2024-02-15~2025-09-23）的事件（僅事件層）'},
 'verdict_rule': 'PRIMARY 在挑選期與驗證期都要：事件淨異常報酬（對同池等權）> 0 且 t_month >= 2，且日曆組合淨年化 >= 0050 同期年化 → 可用；只有一段過 → 待觀察；否則不可用；某段事件 < 30 筆或事件月份 < 24 → 無法測試。',
 'known_limits': ['2019–2023 面板只有存活者（驗證期偏樂觀）', '2024-01-02~2024-02-14 無資料（rolling 跨缺口）',
                  'src 3 期間收盤為 13:20 近似、最高價與成交值亦為近似', '面板無最低價，停損以訊號日開盤代替長紅 K 低點，並以收盤判斷',
                  '成交量用 value/close_raw 推算，減資／分割日前後股數口徑會跳動', '出場不考慮跌停賣不掉'],
}
if __name__ == '__main__':
    f = HERE / 'results.json'
    d = json.loads(f.read_text(encoding='utf-8')) if f.exists() else {}
    assert 'PREREG' not in d, 'PREREG 已存在，不覆寫'
    d = {'PREREG': PREREG, **d}
    f.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding='utf-8')
    print('PREREG written')
