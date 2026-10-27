#!/usr/bin/env python3
"""事前設定（在看任何結果前寫入 results.json 的 PREREG；之後不改）。執行：python3 -I prereg.py"""
import json, datetime
from pathlib import Path
HERE = Path(__file__).resolve().parent
PREREG = {
 'written_at': datetime.datetime.now().isoformat(timespec='seconds'),
 'rule_name': 'Connors R3（RSI(2) 連三降）',
 'sources': ['https://www.whselfinvest.de/en/trading-platform/free-trading-strategies/tradingsystem/81-r3-larry-connors',
             'https://tw.tradingview.com/script/QJSXeJnv',
             'https://quantifiedstrategies.substack.com/p/larry-connors-r3-strategy-it-still-795'],
 'rule_text_as_given': '股池：同 BRIEF 流動性條件。訊號（全部成立）：(1) 收盤 > SMA(200)；(2) RSI(2) 連續 3 天下降，RSI_t < RSI_t-1 < RSI_t-2 < RSI_t-3；'
                       '(3) 下降第一天的 RSI(2)（RSI_t-2）< 60；(4) 今日 RSI(2) < 10。進場：隔日開盤買進。出場：收盤時 RSI(2) > 70，隔日開盤賣出。'
                       '原版不設停損、不設停利；時間停損預設 10 個交易日。每檔 10%，最多 10 檔，同日依 RSI(2) 由低到高排序。',
 'web_note': '本環境無法連外（DNS 失敗），無法再讀原文；條件 (3) 的兩種讀法依任務說明：WHS「第一天 RSI < 60」、TradingView「從 60 以上開始下降」。',
 'type': '事件觸發＋固定槽位組合。兩種都報：(1) 事件報酬（所有訊號、不受槽位限制；扣成本、對同池等權異常、t 按進場月份分群）；'
         '(2) 規則字面的 10 槽組合（每檔 10%、最多 10 檔、RSI(2) 低者優先）的年化、日頻最大回撤、對 0050 與同池等權月超額 t、逐年。',
 'data': {
   'panel': 'scratchpad/panel 上市日線（含已下市；2019–2023 只有存活者）。有值檔數 < 300 的日子（2018-01-13）整列剔除後才做任何 rolling；flag 5 的價格視為缺值。',
   'price_series': '只用收盤：SMA200、RSI(2) 全部用 adj（含息還原收盤）。adj 先 ffill（最多 10 日）以便跨停牌計算；訊號要求 t、t-1、t-2、t-3 四天都有真實收盤。',
   'universe': '上市普通股（排除 91xx）；t 日 60 日均成交值（上市後停牌日記 0、min_periods=60）>= NT$1,000 萬、原始收盤 >= 10 元、t 日 flag ∈ {0,1,2}。',
   'flag_exclusion': '訊號股在 [t-4, t] 任何一天 flag ∈ {3,4,5} → 剔除該訊號；「全市場同日 flag 4 檔數 >= 50」的面板缺口日不算。',
   'disposal_exclusion': 't 或 t+1 在處置期間 → 剔除（名單 2021-09-24 起才有）。',
 },
 'indicators': {
   'RSI2': 'Wilder RSI(2)，ewm(alpha=0.5, adjust=False)；AL=0 時 RSI=100（AG=AL=0 時 50）。與 w3_connors_rsi2_sma5 完全相同的計算。',
   'SMA200': 'adj 的 200 日簡單均線（min_periods=200）',
 },
 'signal_t_close': {
   'common': 'adj_t > SMA200_t 且 RSI_t < RSI_t-1 < RSI_t-2 < RSI_t-3（嚴格小於）且 RSI_t < 10',
   'C3_WHS': '加上 RSI_t-2 < 60（下降第一天的 RSI < 60）— PRIMARY 用此讀法（即任務規則原文）',
   'C3_TV': '加上 RSI_t-3 > 60（下降起點、即連三降開始前一天的 RSI > 60；「從 60 以上開始下降」）',
   'C3_none': '不加條件 (3)（只作參考，看條件 (3) 本身的作用）',
 },
 'execution': {
   'entry': 't+1 開盤（還原開盤＝open×adj/close_raw）。t+1 無開盤價或開盤漲停（2015-06-01 前 7%、之後 10%，各減 0.6%）→ 不買（事件層：放棄；組合層：換下一個候選）。',
   'exit_signal': '從進場日 e 收盤開始，每日收盤檢查 RSI_s > 70；成立 → s+1 開盤賣。',
   'time_stop': 'T10（PRIMARY）：到 e+9 收盤仍未觸發，e+10 開盤賣。NT：不設時間停損（原版）。',
   'sell_frictions': '預定賣出日鎖跌停（還原最高價/前收−1 <= −(漲跌幅−0.6%)）或無開盤價 → 順延；下市 → 最後收盤賣。',
   'dedupe': '持有中同檔新訊號忽略；賣出日當天收盤起可再觸發。',
   'cost': 'exlib.cost_rates：買＝0.1425%×0.6＋滑價 0.1%；賣＝0.1425%×0.6＋稅 0.3%＋滑價 0.1%（來回約 0.67%）；另報毛報酬與壓力情境（買 0.5%＋賣 0.5%）。',
 },
 'portfolio': {
   'slots': '10 槽，每筆＝當日開盤 NAV×10%（現金不足用剩餘現金，不足 2.5% NAV 不買）；最多 10 檔；不加碼、不再平衡。',
   'order': '每日開盤先賣後買；候選依訊號日 RSI(2) 由低到高（同值依 60 日均成交值大者優先）；當天剛賣出的不買回。',
   'cash': '現金 0 報酬（訊號少時大部分是現金，這是規則字面的結果）。',
   'start': '2004-07 起模擬，統計從 2005-01。',
 },
 'variants': {
   'PRIMARY': 'C3_WHS × T10',
   'all_reported': '3 種條件 (3)（WHS、TV、none）× 2 種時間停損（T10、NT）＝ 6 組全部報告；TV 讀法與 PRIMARY 並列報告，但判定只用 PRIMARY。',
   'reference_for_overlap': 'w3_connors_rsi2_sma5 的 PRIMARY（RSI2<10、收盤>SMA5 出場、T10），在同一程式重算，只用來算重疊度與相關，不參與判定。',
 },
 'overlap_stats': ['訊號重疊：R3 訊號（股票×日）中屬於 RSI2<10 訊號的比例（定義上應為 100%），與 RSI2<10 訊號中屬於 R3 的比例',
                   '事件重疊：R3 事件的進場（股票、進場日）同時是 rsi2_sma5 事件進場的比例；以及持有期間重疊天數比例',
                   '組合重疊：兩個 10 槽組合每日持股的 Jaccard 平均、日報酬相關係數'],
 'event_stats': {
   'raw': '賣出價/進場價−1（還原）；net＝(1+raw)×(1−賣)/(1+買)−1',
   'abnormal_EW': 'net − 同期間（進場開盤→賣出開盤）同池等權報酬（前一日符合股池條件的上市普通股，每日等權、開盤對開盤）',
   'vs_0050': 'net − 0050 含息（t 收盤→賣出前一日收盤）',
   't': '事件按進場月份平均後對月平均做 t（btlib.tstat）；t_date 只供參考',
 },
 'portfolio_stats': '10 槽組合日頻 NAV；年化、日頻最大回撤、對 0050 含息與同池等權（每月第一個 >= 11 日交易日起算的月窗；同池＝流動性股池、隔日開盤等權、月換股、無成本）的月超額 t、逐年、平均持股數。',
 'windows': {'select': '2005-01-01~2019-04-30（事件以訊號日歸屬）', 'valid': '2019-05-01~2026-10-22', 'y2024_2025': '2024-01-01~2025-12-31（另報）',
             'valid_survivor_2019-05_2023-12': '只有存活者的區段（另報）', 'valid_officialPx_only': '驗證期剔除 src 3（2024-02-15~2025-09-23）'},
 'verdict_rule': 'PRIMARY 在挑選期與驗證期各自：事件淨異常報酬（對同池等權，滑價 0.1% 成本）> 0 且 t_month >= 2，且 10 槽組合淨年化 >= 0050 同期年化 → 該段通過；'
                 '兩段都過 → 可用；一段 → 待觀察；都沒過 → 不可用。組合對同池等權月超額 t 一併報告；事件層過但組合層 t < 2 時在結論明示。'
                 '若 PRIMARY 某段事件數 < 100 → 該段視為資料不足（無法測試）。',
 'diagnostics_preplanned': ['t 收盤進場（不可成交）vs t+1 開盤的隔夜跳空', '壓力成本與毛報酬', '挑選期分 2005-2011 / 2012-2019-04', '逐年事件異常報酬',
                            '極端報酬（>100%、<−50%）與 flag 3/4/5 檢查', '驗證期剔除 src 3', '平均持股數／現金比例'],
 'known_limits': ['2019–2023 面板只有存活者（抄底策略偏樂觀）', '2024-01-02~2024-02-14 無資料', 'src 3 期間收盤為 13:20 近似',
                  '處置名單 2021-09 起才有', '面板無最低價：鎖跌停用最高價判斷', '原文無法連線複核'],
}
if __name__ == '__main__':
    f = HERE / 'results.json'
    d = json.loads(f.read_text(encoding='utf-8')) if f.exists() else {}
    assert 'PREREG' not in d, 'PREREG 已存在，不覆寫'
    d = {'PREREG': PREREG, **d}
    f.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding='utf-8')
    print('PREREG written')
