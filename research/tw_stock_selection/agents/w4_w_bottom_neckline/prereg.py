#!/usr/bin/env python3
"""事前設定（在看任何結果前寫入 results.json 的 PREREG；之後不改）。"""
import json, datetime
from pathlib import Path
HERE = Path(__file__).resolve().parent
PREREG = {
 'written_at': datetime.datetime.now().isoformat(timespec='seconds'),
 'rule_name': 'W 底突破頸線（右腳不破左腳、帶量突破、等幅目標、跌破頸線停損）',
 'sources': ['https://smart.businessweekly.com.tw/Reading/IndepArticle.aspx?ID=36820', 'https://www.cmoney.tw/notes/note-detail.aspx?nid=893325'],
 'type': '事件型（訊號觸發）：主報告＝事件報酬（扣成本、對同池等權異常報酬、t 按進場月份分群）；另報日曆時間組合（所有在場事件每筆等權、每日再平衡）的年化、日頻 MDD、對 0050／同池等權月超額 t、逐年',
 'data': {
   'panel': 'scratchpad/panel 上市日線（含已下市；2019–2023 只有存活者）；「有值檔數 < 300」的日子（2018-01-13）整列剔除後才做任何 rolling；flag 5 的價格視為缺值',
   'universe': '上市普通股（4 碼數字、排除 91xx 存託憑證）；觸發日 t 的 60 日均成交值（上市後停牌日記 0）>= NT$1,000 萬、原始收盤 >= 10 元、flag ∈ {0,1,2}',
   'price': '收盤一律用 adj（含息還原收盤）判斷轉折、頸線、目標、停損（比率不受除息影響）；進出場用還原開盤＝open×adj/close_raw',
   'volume': '量＝value/close_raw（股數）；VMA20＝t 之前 20 日（t-20..t-1）簡單平均量，min_periods=20（教學「均量」常見口徑為前 20 日均量）',
 },
 'pattern': {
   'pivot_low': '日 p 為轉折低點 ⇔ adj_p = min(adj_{p-5..p+5})（11 日全有值）；在 p+5 收盤後才確認、才可使用',
   'W': '右腳 L2（日 p2，確認日 c2=p2+5）出現時，往前找左腳 L1（日 p1，也是轉折低點）：15 <= p2-p1 <= 60 交易日；L1×0.97 <= L2 <= L1×1.05；'
        '頸線 N = max(adj_{p1+1..p2-1})，N/L1 − 1 >= 10%；前段跌勢 max(adj_{p1-60..p1-1})/L1 >= 1.2（min_periods 48）。多個 L1 合格時取最靠近 L2 的那一個；每個 L2 至多一個 W。',
   'trigger_primary': '確認後 30 日內：t ∈ [c2+1, c2+30]，adj_{t-1} <= N 且 adj_t > N（第一次由下往上穿越），且 vol_t >= 1.5 × VMA20_{t}（前 20 日均量）',
   'trigger_novol': '同上但不要求量（另報）',
   'retest_variant': '先依主版本（帶量）突破於 t0；之後 t ∈ [t0+1, t0+20]，第一個滿足 N <= adj_t <= 1.02N 且 close_raw_t > open_t（收紅）的日子為訊號；若在此之前已有 adj < 0.97N 則該型態作廢；20 日內沒回測則不進場',
   'retest_novol': '同回測變體，但突破不要求量（另報）',
 },
 'execution': {
   'entry': '訊號日 t 收盤後，t+1 開盤（還原開盤）買；t+1 無開盤價或開盤漲停（開盤/前收−1 >= 漲跌幅−0.6%，2015-06-01 前 7%、之後 10%）→ 放棄該訊號',
   'dedupe': '同一檔持有中出現的新訊號忽略；出場開盤日之後可再進場',
   'exit': '從進場日 e 起第一個 s（e <= s < e+60）：adj_s >= N + (N − L2) 停利；或 adj_s < 0.97N 停損 → s+1 開盤賣；都沒發生則 e+60 日開盤賣（最長 60 個交易日）',
   'exit_price': '出場日有開盤價用開盤；停牌則用之後第一個開盤；已下市用最後收盤',
   'cost': 'exlib.cost_rates：買＝0.1425%×0.6＋滑價 0.1%；賣＝0.1425%×0.6＋稅 0.3%＋滑價 0.1%（來回約 0.67%）；壓力情境 買 0.5%＋賣 0.5%',
 },
 'variants': {
   'PRIMARY': 'VOL：帶量突破（量 >= 1.5×VMA20），t+1 開盤買',
   'NOVOL': '不要求量的突破',
   'RETEST': '帶量突破後回測頸線收紅再買',
   'RETEST_NOVOL': '不帶量突破後回測頸線收紅再買',
   'all_reported': '4 組全部報告；判定只看 PRIMARY，其他只作穩健性參考；不做任何事後調參',
 },
 'event_stats': {
   'raw': '出場價/進場價 − 1（還原）；net＝(1+raw)×(1−賣)/(1+買) − 1',
   'abnormal_EW': 'net − 同期間（進場開盤→出場開盤）同池等權報酬；同池＝前一日符合流動性股池條件的上市普通股，每日等權、開盤對開盤',
   'vs_0050': 'net − 0050 含息（t 收盤→出場前一日收盤，0050 只有收盤）',
   't': '事件先按進場月份平均，再對月平均做 t（btlib.tstat）；t_date 僅參考',
 },
 'portfolio_stats': '日曆時間組合：每日持有所有在場事件、等權（每日再平衡），進場日報酬＝收/開，出場日＝出場價/前收，進出各按 1/n 權重扣成本；無持股日現金 0 報酬。報年化、日頻 MDD、對 0050 與同池等權（收對收）的月超額 t、逐年、曝險天數比例',
 'windows': {'select': '2005-01-01~2019-04-30（以訊號日歸屬）', 'valid': '2019-05-01~2026-10-22', 'valid_2024_2025': '2024-01-01~2025-12-31',
             'valid_ex_src3': '驗證期剔除持有期碰到 src 3（2024-02-15~2025-09-23）的事件（僅事件層）'},
 'verdict_rule': 'PRIMARY 在挑選期與驗證期都要：事件淨異常報酬（對同池等權）> 0 且 t_month >= 2，且日曆組合淨年化 >= 0050 同期年化 → 可用；只有一段過 → 待觀察；否則不可用；某段事件 < 30 筆或事件月份 < 24 → 無法測試。',
 'known_limits': ['2019–2023 面板只有存活者（驗證期偏樂觀）', '2024-01-02~2024-02-14 無資料（rolling 跨缺口）',
                  'src 3 期間收盤為 13:20 近似、成交值亦為近似', '面板無最低價，轉折低點用收盤近似（教學通常用 K 線低點）',
                  '成交量用 value/close_raw 推算，減資／分割日前後股數口徑會跳動'],
}
if __name__ == '__main__':
    f = HERE / 'results.json'
    d = json.loads(f.read_text(encoding='utf-8')) if f.exists() else {}
    assert 'PREREG' not in d, 'PREREG 已存在，不覆寫'
    d = {'PREREG': PREREG, **d}
    f.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding='utf-8')
    print('PREREG written')
