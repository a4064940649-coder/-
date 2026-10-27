#!/usr/bin/env python3
"""事前設定（在看任何結果前寫入 results.json 的 PREREG；之後不改）。執行：python3 -I prereg.py"""
import json, datetime
from pathlib import Path
HERE = Path(__file__).resolve().parent
PREREG = {
 'written_at': datetime.datetime.now().isoformat(timespec='seconds'),
 'rule_name': 'Larry Connors 多日下跌（MDD, Multiple Days Down）：SMA200 之上、收盤 < SMA5、最近 5 日至少 4 日收黑 → 買；收盤 > SMA5 → 賣',
 'sources': ['https://www.quantifiedstrategies.com/multiple-days-up-and-multiple-days-down-trading-strategy-larry-connors/',
             'https://quantifiedstrategies.substack.com/p/multiple-days-up-mdu-and-multiple',
             'https://quantifiedstrategies.substack.com/p/larry-connors-multiple-days-up-and-ba3',
             'https://quantifiedstrategies.substack.com/p/larry-connors-multiple-days-up-and-d41'],
 'rule_text_as_given': '原版（Connors《High Probability ETF Trading》，ETF）：(1) 收盤 > SMA200；(2) 收盤 < SMA5；(3) 最近 5 個交易日至少 4 天收盤低於前一日收盤 → 收盤買。'
                       '積極版：持有中若收盤低於買進價，再加碼一單位。出場：收盤 > SMA5 → 收盤賣；不設停損。'
                       '台股改編：訊號日收盤後才知道 → 隔日開盤買／加碼／賣；開盤漲停買不到。',
 'type': '事件觸發＋固定槽位組合。兩種都報：(1) 事件報酬（全部訊號、不受槽位限制；扣成本、對同池等權異常、t 按進場月份分群）；'
         '(2) 20 槽組合（每單位開盤 NAV 的 1/20、最多 20 檔）年化、日頻最大回撤、對 0050 含息與同池等權月超額 t、逐年。另測 0050 單一標的。',
 'data': {
   'panel': 'scratchpad/panel 上市日線（含已下市；2019–2023 只有存活者）。有值檔數 < 300 的日子（2018-01-13）整列剔除後才做任何 rolling；flag 5 的價格視為缺值。',
   'price_series': '只用收盤：SMA200、SMA5、收黑天數全部用 adj（含息還原收盤），避免除權息造成假的收黑。adj 先 ffill（最多 10 日）以便跨停牌計算（停牌日 ffill 後不算收黑）；訊號日與出場判斷日必須有當日真實收盤。',
   'universe': '上市普通股（排除 91xx）；t 日 60 日均成交值（停牌日記 0、min_periods=60）>= NT$1,000 萬、原始收盤 >= 10 元、t 日 flag ∈ {0,1,2}。',
   'flag_exclusion': '[t-5, t]（收黑計數用到的區間）任何一天 flag ∈ {3,4,5} → 剔除該訊號；全市場同日 flag 4 檔數 >= 50 的面板缺口日不算。',
   'disposal_exclusion': 't 或 t+1 在處置期間 → 剔除（名單 2021-09-24 起才有）。',
 },
 'indicators': {
   'SMA200': 'adj 的 200 日簡單均線（min_periods=200）',
   'SMA5': 'adj 的 5 日簡單均線（含今日，min_periods=5）',
   'DOWN5': 'sum_{k=0..4} [adj_{t-k} < adj_{t-k-1}]（嚴格小於；平盤不算收黑）',
 },
 'signal_t_close': 'adj_t > SMA200_t 且 adj_t < SMA5_t 且 DOWN5_t >= 4',
 'execution': {
   'entry': 't+1 開盤（還原開盤＝open×adj/close_raw）。t+1 無開盤價或開盤漲停（開盤/前收−1 >= 漲跌幅−0.6%；2015-06-01 前 7%、之後 10%）→ 不買（事件層：放棄；組合層：換下一個候選）。',
   'add_on_AGG': '積極版：進場日 e 起每日收盤（尚未觸發出場時），若 adj_s < 首筆進場價（還原開盤）且尚未加碼 → s+1 開盤加碼一單位（同等金額）；最多 2 單位。s+1 開盤漲停或無開盤 → 該次不加，之後收盤仍低於進場價可再觸發。同日出場訊號優先於加碼。',
   'exit_signal': '從進場日 e 收盤開始，每日收盤檢查 adj_s > SMA5_s；成立 → s+1 開盤全部賣出。',
   'time_stop': 'NT：不設時間停損（原版）。T10：到 e+9 收盤（持有第 10 個交易日收盤，從首筆進場算）仍未觸發，e+10 開盤全部賣出。',
   'stop_loss': '不設價格停損（原版）。也不因跌破 SMA200 出場。',
   'sell_frictions': '預定賣出日整天鎖跌停（還原最高價/前收−1 <= −(漲跌幅−0.6%)）或無開盤價 → 順延到之後第一個可賣開盤；下市（之後再無價格）→ 最後收盤賣。',
   'dedupe': '持有中（到賣出開盤日前）同檔的新訊號忽略；賣出日當天收盤起可再觸發。',
 },
 'costs': {
   'main': 'exlib.cost_rates(0.001)：買＝0.1425%×0.6＋滑價 0.1%；賣＝0.1425%×0.6＋稅 0.3%＋滑價 0.1%（來回約 0.67%）。判定用這個。',
   'rt0.5': '任務口徑來回 0.5%（買 0.25%＋賣 0.25%），報告用。',
   'stress': 'exlib 壓力（買 0.5%＋賣 0.5%，來回 1%）。',
   'gross': '不扣成本（扣成本前），報告用。',
   'etf_0050': '0050 單一標的：買＝手續費 0.0855%＋滑價 0.1%；賣＝手續費＋ETF 證交稅 0.1%＋滑價 0.1%；另報 rt0.5 與壓力。',
 },
 'portfolio': {
   'slots': '最多 20 檔；每單位＝當日開盤 NAV×1/20（現金不足用剩餘現金，不足 0.25 單位不買）；不再平衡、不用槓桿。',
   'AGG_cash': '積極版加碼同樣是 1/20 NAV 一單位，只用現金（滿倉時加碼不到就跳過，記錄次數）。每日開盤順序：先賣、再加碼、再新進場。',
   'AGG40_sensitivity': '敏感度：積極版每單位 1/40 NAV（20 檔各 2 單位剛好 100%）。只報告，不改判定。',
   'order': '同日新訊號依「距 5 日均線跌幅」adj/SMA5−1 由小到大（跌最深者優先，任務指定）；同值依 60 日均成交值大者優先；當天剛賣出的股票當天不再買回。',
   'cash': '現金 0 報酬。',
   'start': '2004-07 起模擬，統計從 2005-01。',
 },
 'variants': {
   'PRIMARY': 'AGG_NT（書中積極版：加碼至多 2 單位、收盤 > SMA5 出場、無停損）',
   'main_reported_all_windows': ['AGG_NT（PRIMARY）', 'STD_NT（標準版，不加碼）', 'AGG_T10', 'STD_T10', 'AGG40_NT（敏感度，組合層）'],
   'note': '判定只用 PRIMARY；其他變體只作穩健性參考，不得改判。',
 },
 'single_0050': {
   'data': 'etf_0050.csv 只有含息收盤（無開盤價），所以訊號日 t 收盤判斷後，以 t+1 收盤成交（比隔日開盤再晚半天，可成交）；加碼與出場同樣 s+1 收盤。',
   'sizing': 'STD：進場 100%；AGG：每單位 50%，加碼後 100%。現金 0 報酬。',
   'bench': '0050 買進持有（含息）；報年化、日頻 MDD、月窗對 0050 超額 t、在場比率、交易筆數、事件淨報酬對同期 0050（t 按進場月份分群）。',
   'role': '補充參考，不進入主判定。',
 },
 'event_stats': {
   'raw': 'STD：賣出價/進場價−1（還原）。AGG：兩單位等金額，raw＝(px/p0 + px/pa)/2 − 1（未加碼則同 STD）。',
   'net': 'STD：(1+raw)×(1−賣)/(1+買)−1；AGG 同（各單位成本比例相同）。',
   'abnormal_EW': 'net − 同期間同池等權報酬（同池＝前一日符合股池條件的上市普通股，每日等權、開盤對開盤）；AGG 的對照＝兩單位各自持有窗口的等權報酬平均。',
   'vs_0050': 'net − 0050 含息（t 收盤→賣出前一日收盤，參考）',
   't': '事件按進場月份平均後對月平均做 t（btlib.tstat）；t_date 只供參考',
 },
 'portfolio_stats': '20 槽組合日頻 NAV；年化、日頻最大回撤、對 0050 含息與同池等權（每月第一個 >= 11 日交易日起算的月窗；同池等權＝流動性股池、隔日開盤等權、月換股、無成本）的月超額 t、逐年、平均持股數。',
 'windows': {'select': '2005-01-01~2019-04-30（事件以訊號日歸屬）', 'valid': '2019-05-01~2026-10-22', 'y2024_2025': '2024-01-01~2025-12-31（另報）',
             'valid_survivor_2019-05_2023-12': '只有存活者的區段（另報）', 'valid_officialPx_only': '驗證期剔除 src 3（2024-02-15~2025-09-23）',
             'r21': '2021–2023 / 2024–2026-10（另報，價格類規則以 2005 切分為準）'},
 'verdict_rule': '依任務：挑選期與驗證期各自，PRIMARY 扣成本（滑價 0.1%）後須「勝同池等權 t >= 2」且「不輸 0050」。'
                 '操作化（與已測 Double 7 / RSI(2) 同口徑）：事件淨異常（對同池等權）> 0 且月分群 t >= 2，且 20 槽組合淨年化 >= 0050 同期年化 → 該段通過。'
                 '另報組合對同池等權月超額 t；若事件層通過但組合對等權 t < 2，結論中明示（不改判定規則）。兩段都過 → 可用；只有一段 → 待觀察；都沒過 → 不可用。',
 'diagnostics_preplanned': ['t 收盤進場（不可成交）vs t+1 開盤的隔夜跳空', '毛報酬、rt0.5、壓力成本', '挑選期分 2005-2011 / 2012-2019-04', '逐年事件異常報酬',
                            '加碼筆數比例與加碼單位的貢獻', '與 Double 7（7 日最低收盤）、RSI(2)<10 訊號的同日重疊比例',
                            '組合候選隨機排序 10 種子（檢查跌幅排序是否關鍵）', '極端報酬（>100%、<−50%）與 flag 3/4/5 檢查', '驗證期剔除 src 3', '每筆損益兩平的來回成本'],
 'known_limits': ['2019–2023 面板只有存活者（抄底策略偏樂觀）', '2024-01-02~2024-02-14 無資料', 'src 3 期間收盤為 13:20 近似',
                  '處置名單 2021-09 起才有', '面板無最低價：鎖跌停用最高價判斷', '0050 無開盤價：單一標的改用隔日收盤成交',
                  '與已測 Double 7、RSI(2)+SMA5 同屬短線超賣反彈家族；差別在於用 5 日中 4 日收黑的天數計數 + SMA5 出場 + 加碼'],
}
if __name__ == '__main__':
    f = HERE / 'results.json'
    d = json.loads(f.read_text(encoding='utf-8')) if f.exists() else {}
    assert 'PREREG' not in d, 'PREREG 已存在，不覆寫'
    d = {'PREREG': PREREG, **d}
    f.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding='utf-8')
    print('PREREG written')
