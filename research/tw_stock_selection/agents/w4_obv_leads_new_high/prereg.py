#!/usr/bin/env python3
"""事前設定（在看任何結果前寫入 results.json 的 PREREG；之後不改）。"""
import json, datetime
from pathlib import Path
HERE = Path(__file__).resolve().parent
PREREG = {
 'written_at': datetime.datetime.now().isoformat(timespec='seconds'),
 'rule_name': 'OBV 能量潮先創新高、股價還沒創新高（主版本）；OBV 均線黃金交叉（版本 B）',
 'sources': ['https://www.stockfeel.com.tw/obv-%E8%83%BD%E9%87%8F%E6%BD%AE-%E6%8C%87%E6%A8%99/', 'https://www.pocket.tw/school/articles/5358/'],
 'type': '事件型（訊號觸發）：主報告＝事件報酬（扣成本、對同池等權異常報酬、t 按進場月份分群）；另報日曆時間組合（所有在場事件每筆等額）',
 'data': {
   'panel': 'scratchpad/panel 上市日線（含已下市；2019–2023 只有存活者）；「有值檔數 < 300」的日子（2018-01-13）整列剔除後才做任何 rolling；flag 5 的價格視為缺值',
   'universe': '上市普通股（4 碼數字、排除 91xx）；t 日 60 日均成交值（上市後停牌日記 0）>= NT$1,000 萬、收盤（原始）>= 10 元、flag ∈ {0,1,2}',
   'segments': '面板分兩段：段 0＝~2023-12-29、段 1＝2024-02-15~。OBV 在每段從 0 重新累加；OBV 與價格的所有 rolling（OBV 60 日高、OBV MA5/MA20、價格 MA20/MA60、60 日最高收盤）都只在同一段內計算（min_periods＝窗長），缺口後要累積滿窗長才有訊號。60 日均成交值（股池門檻）照常跨缺口。',
   'obv': 'OBV_t = OBV_{t-1} + sign(adj_t − adj_{t-1}) × vol_t；vol＝value/close_raw（股數）。漲跌方向用 adj（含息還原收盤，對前一個有值日），避免除息日被誤判為下跌；當日無成交（停牌）增量 0；只用 OBV 的滾動相對位置（60 日高、均線），不比較絕對水準。',
   'price': '收盤＝adj（含息還原收盤）；MA20/MA60＝adj 簡單平均（含當日）；60 日最高收盤＝max(adj_{t-59..t})',
 },
 'signal_t_close': {
   'MAIN_OBV_lead': 'OBV_t > max(OBV_{t-59..t-1})（OBV 創 60 日新高）且 adj_t < 0.95 × max(adj_{t-59..t})（股價離 60 日高仍 > 5%）且 adj_t > MA60_t',
   'B_OBV_cross': 'OMA5_{t-1} <= OMA20_{t-1} 且 OMA5_t > OMA20_t（OBV 的 5 日均線上穿 20 日均線）且 adj_t > MA20_t',
 },
 'execution': {
   'entry': 't+1 開盤（還原開盤＝open×adj/close_raw）；t+1 無開盤價或開盤漲停（開盤/前收−1 >= 漲跌幅−0.6%，2015-06-01 前 7%、之後 10%）→ 該訊號放棄（不遞補）',
   'dedupe': '同一檔持有中（尚未到出場開盤日）出現的新訊號忽略；出場當日收盤後可再觸發',
   'exit': 's>=e（e＝進場日）第一個收盤 adj_s < MA20_s 或 OBV_s < OMA20_s → s+1 開盤賣；都沒發生則第 e+40 日開盤賣（最長 40 交易日）',
   'exit_price': '出場日有開盤價用開盤；停牌則用之後第一個開盤；已下市用最後收盤',
   'cost': 'exlib.cost_rates：買＝0.1425%×0.6＋滑價 0.1%；賣＝0.1425%×0.6＋稅 0.3%＋滑價 0.1%（來回約 0.67%）；壓力情境 買 0.5%＋賣 0.5%',
 },
 'variants': {
   'PRIMARY': 'MAIN：OBV 先創 60 日新高、價格未創高、價 > MA60；教學出場（MA20 或 OBV<OMA20）、最長 40 日',
   'B': 'B_OBV_cross：OBV MA5 上穿 MA20 且價 > MA20；同一出場',
   'aux_fixed20': 'MAIN_FIX20、B_FIX20：同訊號、固定持有 20 日（e → e+20 開盤），只作診斷，不進判定',
   'contrast_topdiv': '對照：股價創 60 日新高（adj_t > max(adj_{t-59..t-1})）但 OBV 沒有（OBV_t < max(OBV_{t-59..t-1})）＝頂背離；以及股價與 OBV 同創 60 日新高＝量價確認。都用 t+1 開盤進、固定 20 日（e+20 開盤）出、同一檔 20 日內不重複，報扣成本事件報酬與對等權異常',
   'all_reported': '全部報告；判定只看 PRIMARY，B 另報是否過關',
 },
 'event_stats': {
   'raw': '出場價/進場價 − 1（還原）；net＝(1+raw)×(1−賣)/(1+買) − 1',
   'abnormal_EW': 'net − 同期間（進場開盤→出場開盤）同池等權報酬；同池＝前一日符合股池條件的上市普通股，每日等權、開盤對開盤',
   'vs_0050': 'net − 0050 含息（t 收盤→出場前一日收盤）',
   't': '事件先按進場月份平均，再對月平均做 t（btlib.tstat）；t_date 僅供參考',
 },
 'portfolio_stats': '日曆時間組合：每日持有所有在場事件、等權（每日再平衡），進場日報酬＝收/開，出場日＝出場價/前收，進出各按 1/n 權重扣成本；無持股日現金 0 報酬。報年化、日頻最大回撤、對 0050 與同池等權（收對收）的月超額 t、逐年、曝險天數比例',
 'windows': {'select': '2005-01-01~2019-04-30（以訊號日歸屬）', 'valid': '2019-05-01~2026-10-22', 'valid_2024_2025': '2024-01-01~2025-12-31',
             'src3_only': '訊號日在 2024-02-15~2025-09-23（src 3 近似收盤）', 'valid_ex_src3': '驗證期剔除持有期碰到 src 3 的事件'},
 'verdict_rule': 'PRIMARY 在挑選期與驗證期都要：事件淨異常報酬（對同池等權）> 0 且 t_month >= 2，且日曆組合淨年化 >= 0050 同期年化 → 可用；只有一段過 → 待觀察；否則不可用。',
 'known_limits': ['2019–2023 面板只有存活者（驗證期偏樂觀）', '2024-01-02~2024-02-14 無資料（OBV 與均線在缺口後重算）',
                  'src 3 期間收盤為 13:20 近似、成交值亦為近似（OBV 方向與量都受影響）', '面板無最低價，出場以收盤判斷'],
}
if __name__ == '__main__':
    f = HERE / 'results.json'
    d = json.loads(f.read_text(encoding='utf-8')) if f.exists() else {}
    assert 'PREREG' not in d, 'PREREG 已存在，不覆寫'
    d = {'PREREG': PREREG, **d}
    f.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding='utf-8')
    print('PREREG written')
