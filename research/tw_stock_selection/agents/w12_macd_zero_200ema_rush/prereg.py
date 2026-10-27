#!/usr/bin/env python3
"""Trading Rush MACD（200 EMA 之上、零軸下黃金交叉、1.5R 停利）事前設定：看任何結果前寫入 results.json 的 PREREG；之後不改。"""
import json, datetime
from pathlib import Path
HERE = Path(__file__).resolve().parent
PREREG = {
 'written_at': datetime.datetime.now().isoformat(timespec='seconds'),
 'rule_name': 'Trading Rush MACD：收盤 > 200 日 EMA、MACD(12,26,9) DIF 在零軸下上穿訊號線 → 隔日開盤買；停損＝進場前 10 日最低收盤、停利＝1.5R、60 日時間出場',
 'sources': ['https://www.tradingview.com/script/4gXXEI5Y-Trading-Rush-Signals-Alerts',
             'https://www.energyexch.com/thread/post/13166450', 'https://br.tradingview.com/scripts/rush'],
 'type': '選股型（組合，最多 20 檔等權）為主報告與判定；另報事件型（全部訊號、不受 20 檔限制）供參考',
 'data': {
   'panel': 'scratchpad/panel 上市日線（含已下市，2019–2023 只有存活者）；有值檔數 < 300 的日子（2018-01-13）整列剔除後再做任何 rolling/EMA，模擬只在其餘交易日進行',
   'price': 'EMA、MACD、停損、停利一律用含息還原收盤（adj）；成交用還原開盤（exlib.AO）。adj 是起點=1 的指數，原始 DIF 跨股不可比',
   'universe': 'BRIEF 流動性池：上市普通股（4 碼、排除 91xx）；訊號日 t：60 日均成交值 >= NT$1,000 萬（mlib.V60，剔除缺漏日後滾動）、原始收盤 >= 10 元；t+1 有開盤價',
 },
 'signal_t_close': {
   'EMA200': 'adj 的 EMA(200)，adjust=False、min_periods=200（至少 200 個有效收盤）',
   'MACD': 'DIF = EMA12(adj) − EMA26(adj)（adjust=False，min_periods 12/26）；DEA = EMA9(DIF)（min_periods 9）',
   'entry': '(1) adj_t > EMA200_t；(2) DIF_t > DEA_t 且 DIF_{t-1} <= DEA_{t-1}；(3) DIF_t < 0 且 DEA_t < 0；且 t 在流動性池、DIF/EMA200 有值',
   'stop_price': '停損價 = min(adj_{t-9..t})（進場日 e=t+1 之前 10 個交易日的最低還原收盤；原教學為「最近擺動低點」，面板無最低價，以此近似）',
   'target_price': '停利價 = p0 + 1.5 ×（p0 − 停損價），p0 = e 日還原開盤成交價',
   'invalid_risk': '若 p0 <= 停損價（開盤即在停損下方、風險 <= 0）→ 該筆不進場（事前決定，報告筆數）',
 },
 'exits': {
   'A_PRIMARY_1.5R': '自進場日 e 收盤起：adj_t < 停損價 → 停損；adj_t >= 停利價 → 停利；兩者擇一觸發 → t+1 開盤賣；若持有滿 60 交易日（t = e+59 收盤仍未觸發）→ e+60 開盤賣（時間出場）',
   'B_macd_cross_down': '自 e 收盤起：DIF_t < DEA_t 且 DIF_{t-1} >= DEA_{t-1}（下穿）→ t+1 開盤賣；保留同一停損價（教學的風控）、不設 1.5R 停利；60 日時間出場同 A',
   'diag_B0_pure_cross': '診斷：只有 MACD 下穿＋60 日時間出場（無停損無停利），只報告',
   'diag_no_filters_plain_cross': '診斷：去掉 200 EMA 與零軸條件（一般 DIF 上穿 DEA），出場同 A，用來看濾網是否改變結論，只報告',
   'execution': '出場訊號在收盤判斷 → 隔日開盤賣；隔日無開盤價延到下一個有開盤價的日子；已下市以最後還原收盤賣出；跌停賣不掉不模擬',
 },
 'execution': {
   'entry': 't 收盤訊號 → t+1 開盤買（還原開盤）；t+1 開盤漲停（開盤/前收−1 >= 漲跌幅−0.6%，2015-06-01 前 7%、之後 10%，mlib.GAP_UP）→ 買不到、該訊號放棄',
   'max_positions': '最多同時 20 檔；每檔 = 當日開盤 NAV/20，不足用剩餘現金（現金 < 0.25 份不買）；同日候選依訊號日 DIF/adj 由低到高（零軸下最深者優先；用 DIF/價格是因 adj 為指數、原始 DIF 跨股不可比）；另以 10 個隨機順序種子做診斷',
   'reentry': '持有中同檔新訊號忽略；出場日收盤起的新交叉可再進場',
   'order_at_open': '先出場 → 再新進場',
   'cash': '現金零利率',
   'cost': 'exlib.cost_rates：滑價 0.1%（買 0.0855%+0.1%、賣 0.0855%+0.3%+0.1%，來回約 0.67%）為主；另報滑價 0（來回約 0.47%，對應 0.5%）、壓力（買 0.5%+賣 0.5%=1%）、毛報酬',
   'sim_start': '2004-07-01 起模擬（暖身，EMA200 需約 200 日），統計自各期間起點',
 },
 'benchmarks': {
   '0050': '0050 含息（日頻 ffill 到面板日曆）',
   'EW_pool': '同池等權：每月第一個 >= 11 日交易日 d 收盤的同一流動性股池全部等權，d+1 開盤換股，不扣成本（exlib.sim_portfolio）',
 },
 'portfolio_stats': '日頻 NAV：年化、日頻最大回撤、逐年；t＝月窗（每月第一個 >= 11 日交易日收盤 → 下一個）超額報酬 t 對 0050 與對同池等權',
 'event_stats': '全部訊號（不設 20 檔上限；同檔持有中不重複）：net=(1+raw)(1−賣)/(1+買)−1；異常 = net − 同池同窗（進場開盤→出場開盤）等權報酬；另報對「全上市（所有 e 日有開盤者）」等權的異常；先按進場月份平均再做 t',
 'windows': {'select': '2005-01-01~2019-04-30', 'valid': '2019-05-01~2026-10-22', 'y2024_2025': '2024-01-01~2025-12-31',
             'extra': ['驗證期存活者段 2019-05~2023-12', '驗證期官方價格段（剔除 src3 2024-02-15~2025-09-23）', '全期']},
 'verdict_rule': '以滑價 0.1% 的 PRIMARY（A，1.5R）組合判定。某期間「通過」= 對同池等權月窗 t >= 2 且 年化 >= 0050 同期年化。兩段都通過 → 可用；一段 → 待觀察；都不通過 → 不可用。B 只作穩健性：A 不可用但 B 兩段都通過 → 最多升為待觀察並註明多重檢定。',
 'comparison_prior': '技術組 T02 一般 MACD 交叉（無濾網、交叉反向出場）：挑選期組合 −13.8%、對等權 t −7.86',
 'known_limits': ['無最低價：擺動低點以 10 日最低收盤近似，停損/停利以收盤判斷、隔日開盤成交（原教學為盤中觸價）',
                  '原教學為外匯 15/30 分 K 日內，套到台股日線屬延伸測試', '2019–2023 面板只有存活者（驗證期偏樂觀）',
                  '2024-01-02~2024-02-14 無資料（EMA 跨過缺口）', 'src 3 期間收盤為 13:20 近似', '跌停賣不掉未模擬'],
}
if __name__ == '__main__':
    f = HERE / 'results.json'
    d = json.loads(f.read_text(encoding='utf-8')) if f.exists() else {}
    assert 'PREREG' not in d, 'PREREG 已存在，不覆寫'
    d = {'PREREG': PREREG, **d}
    f.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding='utf-8')
    print('PREREG written')
