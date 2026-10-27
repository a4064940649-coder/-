#!/usr/bin/env python3
"""事前設定（在看任何結果前寫入 results.json 的 PREREG；之後不改）。"""
import json, datetime
from pathlib import Path
HERE = Path(__file__).resolve().parent
PREREG = {
 'written_at': datetime.datetime.now().isoformat(timespec='seconds'),
 'rule_name': '低檔多頭吞噬（紅 K 吞噬黑 K）＋放量＋確認 K',
 'sources': ['https://www.esunsec.com.tw/article/post/56', 'https://www.xq.com.tw/xstrader/?p=10568'],
 'type': '事件型（訊號觸發）：主報告＝事件報酬（扣成本、對同池等權異常報酬、t 按進場月份分群）；另報日曆時間組合（所有在場事件每筆等額、每日再平衡）',
 'data': {
   'panel': 'scratchpad/panel 上市日線（2004 起，含已下市；2019–2023 只有存活者）；「有值檔數 < 300」的日子（2018-01-13）整列剔除後才做任何 rolling／shift；flag 5 價格視為缺值',
   'universe': '上市普通股（4 碼數字、排除 91xx）；t 日 60 日均成交值（上市後停牌日記 0）>= NT$1,000 萬、原始收盤 >= 10 元；t-1 與 t 的 flag 不得為 3/4/5（flag 缺值視為 0）',
   'volume': '量 ＝ value / close_raw；停牌日為缺值；Vol_MA5[t-1] ＝ vol_{t-5..t-1} 簡單平均（min_periods 5）',
   'price': '實體（同日 C 對 O）用原始 open、close_raw；跨日比較（吞噬、跌幅、MA20、確認 K、停損）用還原尺度：還原開盤 AO＝open×adj/close_raw、收盤＝adj，無公司行動時與原始價完全相同',
 },
 'signal_t_close_BULL': {
   '1_black_t-1': '(open_{t-1} − close_raw_{t-1}) / open_{t-1} >= 1.5%',
   '2_engulf_t': 'close_raw_t > open_t；AO_t <= adj_{t-1} 且 adj_t >= AO_{t-1}；紅 K 實體 (close_raw_t − open_t)/open_t >= 1.5 × 黑 K 實體',
   '3_low_position': 'adj_{t-1}/adj_{t-11} − 1 <= −5% 且 adj_t < MA20_t（adj 20 日簡單均線，含 t）',
   '4_volume': 'vol_t >= 1.5 × Vol_MA5_{t-1}',
   '5_confirm': 'adj_{t+1} > adj_t（確認 K），t+2 開盤買；無確認版 NOCONF：t+1 開盤買',
 },
 'execution': {
   'entry': '進場日 e（主規格 t+2、NOCONF t+1）開盤（還原開盤）；e 無開盤價或開盤漲停（開盤/前收−1 >= 漲跌幅−0.6%，2015-06-01 前 7%、之後 10%）→ 放棄（不遞補）',
   'dedupe': '同一檔持有中（尚未到出場開盤日）出現的新訊號忽略',
   'exit': '從進場日 e 起第一個收盤日 s 滿足 adj_s < AO_t（吞噬紅 K 開盤）→ s+1 開盤停損賣；否則持有 10 個交易日，e+10 開盤賣',
   'exit_price': '出場日有開盤價用開盤；停牌則用之後第一個開盤；已下市用最後收盤',
   'cost': 'exlib.cost_rates：買＝0.1425%×0.6＋滑價 0.1%；賣＝0.1425%×0.6＋稅 0.3%＋滑價 0.1%（來回約 0.67%）；壓力情境 買 0.5%＋賣 0.5%',
 },
 'variants': {
   'PRIMARY': 'CONF：條件 1~5，t+2 開盤買',
   'NOCONF': '條件 1~4，t+1 開盤買（教學的無確認版）',
   'CONF_NOVOL': '條件 1~3＋5（拿掉放量），只作穩健性',
   'BEAR_CONF（對照）': '高檔空頭吞噬：t-1 紅 K 實體 >= 1.5%；t 黑 K：AO_t >= adj_{t-1} 且 adj_t <= AO_{t-1}、實體 >= 1.5×；adj_{t-1}/adj_{t-11} − 1 >= +5% 且 adj_t > MA20_t；放量同條件 4；確認 adj_{t+1} < adj_t；t+2 開盤「當作賣出」：報持有該股 10 日（收盤站回 AO_t 以上隔日開盤回補）的異常報酬，預期為負才代表賣出訊號有效',
   'BEAR_NOCONF（對照）': '同上無確認、t+1 開盤',
   'all_reported': '全部報告；判定只看 PRIMARY，其他僅參考；不做事後調參',
 },
 'event_stats': {
   'raw': '出場價/進場價 − 1（還原）；net＝(1+raw)×(1−賣)/(1+買) − 1',
   'abnormal_EW': 'net − 同期間（進場開盤→出場開盤）同池等權報酬；同池＝前一日符合流動性股池的上市普通股，每日等權、開盤對開盤',
   'vs_0050': 'net − 0050 含息（進場前一日收盤→出場前一日收盤）',
   't': '事件先按進場月份平均，再對月平均做 t（btlib.tstat）；t_date 僅參考',
 },
 'portfolio_stats': '日曆時間組合：每日持有所有在場事件等權（每日再平衡），進場日報酬＝收/開，出場日＝出場價/前收，進出各按 1/n 權重扣成本；無持股日現金 0。報年化、日頻最大回撤、對 0050 與同池等權（收對收）的月超額 t、逐年、曝險天數比例',
 'windows': {'select': '2005-01-01~2019-04-30（以訊號日歸屬）', 'valid': '2019-05-01~2026-10-22', 'valid_2024_2025': '2024-01-01~2025-12-31',
             'src3': '2024-02-15~2025-09-23 另報；驗證期另報剔除持有期碰到 src 3 的事件', 'pre2005': '2004 只做 rolling 暖機，不計入'},
 'verdict_rule': 'PRIMARY 在挑選期與驗證期都要：事件淨異常報酬（對同池等權）> 0 且 t_month >= 2，且日曆組合淨年化 >= 0050 同期年化 → 可用；只有一段過 → 待觀察；否則不可用；某段事件 < 30 筆或事件月份 < 24 → 無法測試。',
 'known_limits': ['2019–2023 面板只有存活者（驗證期偏樂觀）', '2024-01-02~2024-02-14 無資料（rolling 跨缺口）',
                  'src 3 期間收盤為 13:20 近似（實體、確認 K 判斷皆受影響）', '面板無最低價：教學常以紅 K 低點停損，這裡依任務以 O[t] 收盤跌破停損',
                  '成交量用 value/close_raw 推算'],
}
if __name__ == '__main__':
    f = HERE / 'results.json'
    d = json.loads(f.read_text(encoding='utf-8')) if f.exists() else {}
    assert 'PREREG' not in d, 'PREREG 已存在，不覆寫'
    d = {'PREREG': PREREG, **d}
    f.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding='utf-8')
    print('PREREG written')
