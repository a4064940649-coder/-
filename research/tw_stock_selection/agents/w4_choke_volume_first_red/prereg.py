#!/usr/bin/env python3
"""事前設定（在看任何結果前寫入 results.json 的 PREREG；之後不改）。"""
import json, datetime
from pathlib import Path
HERE = Path(__file__).resolve().parent
PREREG = {
 'written_at': datetime.datetime.now().isoformat(timespec='seconds'),
 'rule_name': '窒息量後的第一根表態紅 K（量縮整理後量增價漲）',
 'sources': ['https://www.cmoney.tw/notes/note-detail.aspx?nid=210766', 'https://www.cmoney.tw/notes/note-detail.aspx?nid=171894'],
 'type': '事件型（訊號觸發）：主報告＝事件報酬（扣成本、對同池等權異常報酬、t 按進場月份分群）；另報日曆時間組合（所有在場事件每筆等額、每日再平衡）',
 'data': {
   'panel': 'scratchpad/panel 上市日線（含已下市；2019–2023 只有存活者）；「有值檔數 < 300」的日子（2018-01-13）整列剔除後才做任何 rolling；flag 5 的價格視為缺值',
   'universe': '上市普通股（4 碼數字、排除 91xx 存託憑證）；t 日 60 日均成交值（上市後停牌日記 0）>= NT$1,000 萬、原始收盤 >= 10 元、flag ∈ {0,1,2}',
   'volume': '量 ＝ value / close_raw（股數）；停牌日量為缺值（不當作窒息量）；VMA5／VMA20 為含當日簡單平均，min_periods＝窗長',
   'price': 'close 一律用 adj（含息還原收盤），避免除息造成假破底；開盤價換成還原開盤＝open×adj/close_raw',
 },
 'signal_t_close': {
   '1_prior_run': 'HH ＝ max(adj_{t-60..t-1})、LL ＝ min(adj_{t-120..t-1})（起漲低點）；HH/LL >= 1.25（逐字照教學，不另要求低點早於高點；事後只做診斷）',
   '2_choke_volume': '存在 d ∈ [t-10, t-1]：vol_d > 0 且 vol_d <= k × max(vol_{d-60..d-1}) 且 vol_d <= 0.5 × VMA20_d；主規格 k＝0.35，另測 k＝0.25、0.50',
   '3_no_break': 'min(adj_{t-10..t-1}) >= LL + 0.382 × (HH − LL)',
   '4_red_k': 'close_raw_t > open_t 且 adj_t/adj_{t-1} − 1 >= 2% 且 vol_t >= 1.5 × VMA5_{t-1}（前 5 日均量，不含 t；教學「量增」＝相對前幾日）',
   'variant_NH20': '主規格再加：adj_t > max(adj_{t-20..t-1})（收盤創 20 日新高）且 VMA5_t > VMA20_t',
 },
 'execution': {
   'entry': 't+1 開盤（還原開盤）；t+1 無開盤價或開盤漲停（開盤/前收−1 >= 漲跌幅−0.6%，2015-06-01 前 7%、之後 10%）→ 放棄該訊號（不遞補）',
   'dedupe': '同一檔持有中（尚未到出場開盤日）出現的新訊號忽略；出場當日收盤後可再觸發',
   'exit': '從進場日 e 起第一個 s 滿足：adj_s < 觸發日 t 的還原開盤（停損；教學用紅 K 低點，面板無最低價以開盤代替）或 adj_s < MA10_s（adj 10 日簡單均線）→ s+1 開盤賣；都沒發生則 e+40 日開盤賣',
   'exit_price': '出場日有開盤價用開盤；停牌則用之後第一個開盤；已下市用最後收盤',
   'cost': 'exlib.cost_rates：買＝0.1425%×0.6＋滑價 0.1%；賣＝0.1425%×0.6＋稅 0.3%＋滑價 0.1%（來回約 0.67%）；壓力情境 買 0.5%＋賣 0.5%',
 },
 'variants': {
   'PRIMARY': 'CHOKE35：條件 1~4、k＝0.35',
   'CHOKE25': '條件 1~4、k＝0.25',
   'CHOKE50': '條件 1~4、k＝0.50',
   'NH20': '條件 1~4（k＝0.35）＋收盤創 20 日新高＋VMA5>VMA20',
   'all_reported': '4 組全部報告；判定只看 PRIMARY，其他只作穩健性參考；不做任何事後調參',
 },
 'event_stats': {
   'raw': '出場價/進場價 − 1（還原）；net＝(1+raw)×(1−賣)/(1+買) − 1',
   'abnormal_EW': 'net − 同期間（進場開盤→出場開盤）同池等權報酬；同池＝前一日符合股池條件的上市普通股，每日等權、開盤對開盤',
   'vs_0050': 'net − 0050 含息（t 收盤→出場前一日收盤，0050 只有收盤）',
   't': '事件先按進場月份平均，再對月平均做 t（btlib.tstat）；t_date 僅參考',
 },
 'portfolio_stats': '日曆時間組合：每日持有所有在場事件、等權（每日再平衡），進場日報酬＝收/開，出場日＝出場價/前收，進出各按 1/n 權重扣成本；無持股日現金 0 報酬。報年化、日頻最大回撤、對 0050 與同池等權（收對收）的月超額 t、逐年、曝險天數比例',
 'windows': {'select': '2005-01-01~2019-04-30（以訊號日歸屬）', 'valid': '2019-05-01~2026-10-22', 'valid_2024_2025': '2024-01-01~2025-12-31',
             'valid_ex_src3': '驗證期剔除持有期碰到 src 3（2024-02-15~2025-09-23）的事件（僅事件層）'},
 'verdict_rule': 'PRIMARY 在挑選期與驗證期都要：事件淨異常報酬（對同池等權）> 0 且 t_month >= 2，且日曆組合淨年化 >= 0050 同期年化 → 可用；只有一段過 → 待觀察；否則不可用；某段事件 < 30 筆或事件月份 < 24 → 無法測試。',
 'known_limits': ['2019–2023 面板只有存活者（驗證期偏樂觀）', '2024-01-02~2024-02-14 無資料（rolling 跨缺口）',
                  'src 3 期間收盤為 13:20 近似、成交值亦為近似（成交量＝value/close 也是近似）', '面板無最低價，停損以觸發日開盤代替紅 K 低點，並以收盤判斷',
                  '成交量用 value/close_raw 推算，減資／分割日前後股數口徑會跳動'],
}
if __name__ == '__main__':
    f = HERE / 'results.json'
    d = json.loads(f.read_text(encoding='utf-8')) if f.exists() else {}
    assert 'PREREG' not in d, 'PREREG 已存在，不覆寫'
    d = {'PREREG': PREREG, **d}
    f.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding='utf-8')
    print('PREREG written')
