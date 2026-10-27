#!/usr/bin/env python3
"""事前設定（在看任何結果前寫入 results.json 的 PREREG；之後不改）。執行：python3 -I prereg.py"""
import json, datetime
from pathlib import Path
HERE = Path(__file__).resolve().parent
PREREG = {
 'written_at': datetime.datetime.now().isoformat(timespec='seconds'),
 'rule_name': 'Larry Connors 雙重 7 日（Double 7s）：SMA200 之上收盤創 7 日新低買、收盤創 7 日新高賣',
 'sources': ['https://strategyquant.com/blog/larry-connors-double-7-strategy-tested-on-spy-and-8-other-markets/',
             'https://strategyquant.com/shared/larry-connors-double-7-for-spy/',
             'https://quantifiedstrategies.substack.com/p/larry-connors-double-seven-trading-aa3'],
 'rule_text_as_given': '原版（ETF/SPY）：收盤 > SMA(200)，且收盤為最近 7 日最低收盤 → 收盤買；收盤為最近 7 日最高收盤 → 收盤賣；不設停損。'
                       '台股改編：股池同 BRIEF 流動性條件；訊號日收盤後才知道，改成隔日開盤買、出場訊號隔日開盤賣。每檔 10%、最多 10 檔。',
 'type': '事件觸發＋固定槽位組合。兩種都報：(1) 事件報酬（所有訊號、不受槽位限制；扣成本、對同池等權異常、t 按進場月份分群）；'
         '(2) 規則字面的 10 槽組合（每檔 10%、最多 10 檔）的年化、日頻最大回撤、對 0050 與同池等權月超額 t、逐年。',
 'data': {
   'panel': 'scratchpad/panel 上市日線（含已下市；2019–2023 只有存活者）。有值檔數 < 300 的日子（2018-01-13）整列剔除後才做任何 rolling；flag 5 的價格視為缺值。',
   'price_series': '只用收盤：SMA200、N 日最低／最高收盤全部用 adj（含息還原收盤），避免除權息造成假的 7 日新低。adj 先 ffill（最多 10 日）以便跨停牌計算；但訊號日與出場判斷日必須有當日真實收盤。',
   'universe': '上市普通股（排除 91xx）；t 日 60 日均成交值（停牌日記 0、min_periods=60）>= NT$1,000 萬、原始收盤 >= 10 元、t 日 flag ∈ {0,1,2}。',
   'flag_exclusion': '訊號股在 [t-N+1, t]（N 日通道用到的區間）任何一天 flag ∈ {3,4,5} → 剔除該訊號；全市場同日 flag 4 檔數 >= 50 的面板缺口日不算。',
   'disposal_exclusion': 't 或 t+1 在處置期間 → 剔除（名單 2021-09-24 起才有）。',
 },
 'indicators': {
   'SMA200': 'adj 的 200 日簡單均線（min_periods=200）',
   'LOW_N': 'adj 的 N 日 rolling min（含今日，min_periods=N）',
   'HIGH_N': 'adj 的 N 日 rolling max（含今日，min_periods=N）',
   'N': '主設定 N=7（原版）；敏感度 N=5、10 只在挑選期報告，不得用來改判定。',
 },
 'signal_t_close': 'adj_t > SMA200_t 且 adj_t <= LOW_N_t（收盤創 N 日新低，含平手）',
 'execution': {
   'entry': 't+1 開盤（還原開盤＝open×adj/close_raw）。t+1 無開盤價或開盤漲停（開盤/前收−1 >= 漲跌幅−0.6%；2015-06-01 前 7%、之後 10%）→ 不買（事件層：放棄；組合層：換下一個候選）。',
   'exit_signal': '從進場日 e 收盤開始，每日收盤檢查 adj_s >= HIGH_N_s（收盤創 N 日新高，含平手）；成立 → s+1 開盤賣。',
   'time_stop': 'T20（預設，教學沒給）：到 e+19 收盤（持有第 20 個交易日收盤）仍未觸發，e+20 開盤賣。NT：不設時間停損（原版）。',
   'stop_loss': '不設價格停損（原版）。也不因跌破 SMA200 出場（原版沒有）。',
   'sell_frictions': '預定賣出日整天鎖跌停（還原最高價/前收−1 <= −(漲跌幅−0.6%)）或無開盤價 → 順延到之後第一個可賣開盤；下市（之後再無價格）→ 最後收盤賣。',
   'dedupe': '持有中（到賣出開盤日前）同檔的新訊號忽略；賣出日當天收盤起可再觸發。',
   'cost': 'exlib.cost_rates：買＝0.1425%×0.6＋滑價 0.1%；賣＝0.1425%×0.6＋稅 0.3%＋滑價 0.1%（來回約 0.67%）；另報毛報酬與壓力情境（買 0.5%＋賣 0.5%）。',
 },
 'portfolio': {
   'slots': '10 槽，每筆進場金額＝當日開盤 NAV×10%（現金不足用剩餘現金，不足 2.5% NAV 不買）；最多 10 檔；不加碼、不再平衡。',
   'order': '每日開盤先處理賣出，再依前一日收盤訊號買進；同日訊號依「距 200 日均線乖離 adj/SMA200−1」由大到小排序（教學沒給排序規則，此為任務指定預設）；同值依 60 日均成交值大者優先；當天剛賣出的股票當天不再買回。',
   'cash': '現金 0 報酬。',
   'start': '2004-07 起模擬，統計從 2005-01。',
 },
 'variants': {
   'PRIMARY': 'N=7 × T20（20 日時間停損）',
   'main_reported_all_windows': ['N7_T20（PRIMARY）', 'N7_NT（原版不設時間停損）'],
   'sensitivity_select_only': ['N5_T20', 'N5_NT', 'N10_T20', 'N10_NT'],
   'note': '其他變體只作穩健性參考，不得改判。',
 },
 'event_stats': {
   'raw': '賣出價/進場價−1（還原）；net＝(1+raw)×(1−賣)/(1+買)−1',
   'abnormal_EW': 'net − 同期間（進場開盤→賣出開盤）同池等權報酬（同池＝前一日符合股池條件的上市普通股，每日等權、開盤對開盤）',
   'vs_0050': 'net − 0050 含息（t 收盤→賣出前一日收盤）',
   't': '事件按進場月份平均後對月平均做 t（btlib.tstat）；t_date 只供參考',
 },
 'portfolio_stats': '10 槽組合日頻 NAV；年化、日頻最大回撤、對 0050 含息與同池等權（每月第一個 >= 11 日交易日起算的月窗，同池＝流動性股池、隔日開盤等權、月換股，無成本）的月超額 t、逐年、平均持股數。',
 'windows': {'select': '2005-01-01~2019-04-30（事件以訊號日歸屬）', 'valid': '2019-05-01~2026-10-22', 'y2024_2025': '2024-01-01~2025-12-31（另報）',
             'valid_survivor_2019-05_2023-12': '只有存活者的區段（另報）', 'valid_officialPx_only': '驗證期剔除 src 3（2024-02-15~2025-09-23）'},
 'verdict_rule': 'PRIMARY 在挑選期與驗證期各自：事件淨異常報酬（對同池等權，滑價 0.1% 成本）> 0 且 t_month >= 2，且 10 槽組合淨年化 >= 0050 同期年化 → 該段通過；兩段都過 → 可用；只有一段 → 待觀察；都沒過 → 不可用。'
                 '組合對同池等權月超額 t 一併報告；若事件層通過但組合層對等權 t < 2，在結論中明示（不改判定規則）。（與已測 RSI(2) 規則同一判定口徑。）',
 'diagnostics_preplanned': ['t 收盤進場（不可成交）vs t+1 開盤的隔夜跳空', '壓力成本與毛報酬', '挑選期分 2005-2011 / 2012-2019-04', '逐年事件異常報酬',
                            '組合候選隨機排序 10 種子（檢查乖離排序是否關鍵）', '極端報酬（>100%、<−50%）與 flag 3/4/5 檢查', '驗證期剔除 src 3', '每筆損益兩平的來回成本'],
 'known_limits': ['2019–2023 面板只有存活者（抄底策略偏樂觀）', '2024-01-02~2024-02-14 無資料', 'src 3 期間收盤為 13:20 近似',
                  '處置名單 2021-09 起才有', '面板無最低價：鎖跌停用最高價判斷',
                  '和已測「乖離率反轉」與「RSI(2)」不同：本規則看 N 日收盤通道新低／新高，加 SMA200 濾網；但同屬短線超賣反彈家族'],
}
if __name__ == '__main__':
    f = HERE / 'results.json'
    d = json.loads(f.read_text(encoding='utf-8')) if f.exists() else {}
    assert 'PREREG' not in d, 'PREREG 已存在，不覆寫'
    d = {'PREREG': PREREG, **d}
    f.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding='utf-8')
    print('PREREG written')
