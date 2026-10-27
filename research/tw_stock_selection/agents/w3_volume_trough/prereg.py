#!/usr/bin/env python3
"""事前設定（看任何策略結果前寫入 results.json 的 PREREG；之後不改）。"""
import json, datetime
from pathlib import Path
HERE = Path(__file__).resolve().parent
PREREG = {
 'written_at': datetime.datetime.now().isoformat(timespec='seconds'),
 'rule_name': '地量見地價（N 日地量＋低位；版本 A 直接買、版本 B 放量收紅確認）',
 'sources': ['https://licai.cofool.com/ask/qa_6520065.html', 'https://cs.com.cn/gppd/jp/201808/t20180831_5866541.html',
             'https://www.thepaper.cn/newsDetail_forward_28393949', 'https://www.cnfin.com/stock-xh08/a/20111209/868115.shtml'],
 'type': '事件型：主報告＝事件報酬（扣成本、對同池等權開盤對開盤異常報酬、t 按進場月份分群）；另報同時最多 20 檔的日曆組合（年化、日頻最大回撤、對 0050／同池等權月超額 t、逐年）',
 'data': {
   'panel': 'scratchpad/panel 上市日線（含已下市；2019–2023 只有存活者）；先剔除「有值檔數 < 300」的日子（2018-01-13）再做任何 rolling；flag 5 價格視為缺值',
   'volume': 'V＝value/close_raw；value<=0、NaN、close_raw 缺值或 flag ∉ {0,1,2} 的日子 V 設 NaN（停牌／缺漏不算地量）',
   'known_truncated_days': '2025-01-21（多數股票資料停在中午，成交值約 0.68 倍）、2025-04-10（缺部分 5 分 K，約 0.82 倍）：這兩天 V 設 NaN，不當地量日、也不進 LLV 窗（value_fix 已記錄）',
   'price': '均線、收紅、停損、報酬一律用 adj（含息還原，避免除息假訊號）；進出場用還原開盤＝open×adj/close_raw',
 },
 'universe': '上市普通股（4 碼數字、排除 91xx）；以訊號日（地量日）前一交易日判斷：60 日均成交值（上市後停牌日記 0）>= NT$1,000 萬、close_raw >= 10、flag ∈ {0,1,2}；訊號日本身需 flag ∈ {0,1,2} 且有收盤',
 'signal_trough_day_t': {
   'LLV': 'V_t 有值且 V_t <= min(V_{t-L+1..t})（通達信 V=LLV(V,L)，含當日；以交易日曆滾動，窗內有值天數 >= 0.8L，NaN 不參與）',
   'low_position': 'adj_t < MA60_t（adj 60 日簡單均線，含當日，min_periods 60）',
   'L_primary': 120, 'L_grid': [60, 120, 250],
 },
 'version_A': {
   'entry': 't+1 開盤（還原開盤）；t+1 無開盤價或開盤漲停（開盤/前收−1 >= 漲跌幅−0.6%，2015-06-01 前 7%、之後 10%）→ 放棄、不遞補',
   'exit': '進場日 e 起持有 H 個交易日，e+H 日開盤賣（無停損）',
 },
 'version_B': {
   'confirm': '地量日 t 之後的 t+1..t+10 中第一個 c 滿足：V_c >= 2 × MA5(V)_c（含 c 當日的 5 日均量，通達信 MA(V,5) 口徑，min_periods 5）且 adj_c > adj_{c-1}（收紅＝收盤高於前收）；c 需 flag ∈ {0,1,2}',
   'entry': 'c+1 開盤；無開盤價或開盤漲停 → 放棄',
   'stop': '自 e 起任一收盤 adj_s < 0.93 × adj_t（地量日收盤）→ s+1 開盤賣',
   'exit': '未停損則 e+H 日開盤賣',
   'trough_pick': '同一次確認對應多個地量日時，取最早的地量日（依序處理）',
 },
 'common_execution': {
   'dedupe': '同一檔持有中（出場開盤日之前）的新進場忽略',
   'exit_price': '出場日有開盤價用開盤；停牌則用之後第一個開盤；已下市用最後收盤',
   'cost': 'exlib.cost_rates：買＝0.1425%×0.6＋滑價 0.1%；賣＝0.1425%×0.6＋稅 0.3%＋滑價 0.1%（來回約 0.67%）；壓力情境 買 0.5%＋賣 0.5%（來回 1%）',
 },
 'portfolio': {
   'cap': '同時最多 20 檔；每檔權重 1/20（空格放現金、報酬 0），每日依 1/20 再平衡近似',
   'priority': '同日可進場數 > 空格時，依訊號日前一日 60 日均成交值由大到小',
   'secondary': '另報「在場事件全額等權」（不設上限、全部資金分給在場部位）供參考，不用於判定',
   'benchmarks': '0050 含息日報酬；同池等權（前一日符合股池、收盤對收盤）',
 },
 'variants': {
   'PRIMARY': 'B_L120_H20（教學建議的放量確認版）',
   'CO_REPORT': 'A_L120_H20（直接版）',
   'grid': '版本 A／B × L ∈ {60,120,250} × H ∈ {10,20,60}，共 18 組全部報告；判定只看 PRIMARY；不做事後調參',
 },
 'event_stats': {
   'raw': '出場價/進場價 − 1（還原）；net＝(1+raw)×(1−賣)/(1+買) − 1',
   'abnormal_EW': 'net − 同期間（進場開盤→出場開盤）同池等權報酬（前一日符合股池的上市普通股，開盤對開盤每日等權）',
   'vs_0050': 'net − 0050 含息（進場前一日收盤→出場前一日收盤）',
   't': '事件先按進場月份平均，再對月平均做 t（btlib.tstat）；t_date 僅參考',
   'sample': '主統計＝全部可成交事件（不受 20 檔上限）；另報 20 檔上限實際成交的子集合',
 },
 'windows': {'select': '2005-01-01~2019-04-30（以地量日歸屬）', 'valid': '2019-05-01~2026-10-22', 'valid_2024_2025': '2024-01-01~2025-12-31',
             'valid_ex_src3': '驗證期剔除持有期碰到 src 3（2024-02-15~2025-09-23）的事件'},
 'verdict_rule': 'PRIMARY 在挑選期與驗證期都要：事件淨異常報酬（對同池等權）> 0 且 t_month >= 2，且 20 檔上限組合淨年化 >= 0050 同期 → 可用；只有一段過 → 待觀察；兩段都不過 → 不可用；某段事件 < 30 筆或事件月份 < 24 → 無法測試',
 'src3_check': '比較 src 3 期間與官方期間的「池內地量日比例」（每日地量檔數／池內檔數），並報 src 切換日（2024-02-15、2025-09-24）前後的地量日聚集；驗證期另報剔除 src 3 的事件統計',
 'known_limits': ['2019–2023 面板只有存活者', '2024-01-02~2024-02-14 無資料（滾動窗直接跨過）', 'src 3 期間收盤為 13:20 近似、量為 5 分 K 加總近似',
                  '量＝value/close_raw，減資／分割前後口徑跳動', '無最低價：停損以收盤判斷'],
}
if __name__ == '__main__':
    f = HERE / 'results.json'
    d = json.loads(f.read_text(encoding='utf-8')) if f.exists() else {}
    assert 'PREREG' not in d, 'PREREG 已存在，不覆寫'
    d = {'PREREG': PREREG, **d}
    f.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding='utf-8')
    print('PREREG written')
