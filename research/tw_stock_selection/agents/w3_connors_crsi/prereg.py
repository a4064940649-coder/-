#!/usr/bin/env python3
"""事前設定（在看任何結果前寫入 results.json 的 PREREG；之後不改）。執行：python3 -I prereg.py"""
import json, datetime
from pathlib import Path
HERE = Path(__file__).resolve().parent
PREREG = {
 'written_at': datetime.datetime.now().isoformat(timespec='seconds'),
 'rule_name': 'ConnorsRSI（CRSI）超賣＋限價低接',
 'sources': ['https://www.luxalgo.com/blog/connors-rsi-guide-enhanced-rsi-for-short-term-signals/',
             'https://therighttrader.com/learn/indicators/connors-rsi',
             'https://doc.stocksharp.com/en/topics/api/indicators/list_of_indicators/connors_rsi',
             'https://therighttrader.com/learn/comparisons/rsi-vs-connors-rsi'],
 'rule_text_as_given': 'CRSI = [RSI(close,3) + RSI(streak,2) + PercentRank(1 日報酬,100)] / 3。股池：BRIEF 流動性條件＋100 日歷史波動率（年化）> 30%。'
                       '訊號：收盤 > SMA200（濾網版；原版不加濾網，兩版都報）且 CRSI < 10（另測 < 5）。進場：隔日限價 收盤×(1−4%)。'
                       '面板無最低價 → (A) 隔日開盤 <= 限價才以開盤價成交；(B) 隔日開盤無條件買。出場：收盤 CRSI > 70（另測 > 50）或收盤 > SMA5，隔日開盤賣；'
                       '時間停損 10 個交易日。每檔 10%，最多 10 檔，同日訊號依 CRSI 由低到高。',
 'type': '事件觸發＋固定槽位組合。兩種都報：(1) 事件報酬（所有成交訊號、不受槽位限制；扣成本、對同池等權異常、t 按進場月份分群）；(2) 10 槽組合（年化、日頻最大回撤、對 0050 與同池等權月超額 t、逐年）。',
 'data': {
   'panel': 'scratchpad/panel 上市日線（含已下市；2019–2023 只有存活者）。有值檔數 < 300 的日子（2018-01-13）整列剔除後才做任何 rolling；flag 5 的價格視為缺值。排除 91xx。',
   'price_series': '指標全部用 adj（含息還原收盤），adj 先 ffill（最多 10 日）以便跨停牌；訊號日與出場判斷日必須有當日真實收盤。',
   'liquidity_pool': 't 日 60 日均成交值（上市後停牌日記 0、min_periods=60）>= NT$1,000 萬、原始收盤 >= 10、flag ∈ {0,1,2}（BRIEF 股池＝LIQ）。',
   'rule_pool': 'LIQ 且 HV100_t > 30%；HV100＝adj 日對數報酬最近 100 日標準差（ddof=1、min_periods=100）× sqrt(252)（Connors 原文未指明年化天數，用 252 交易日常見口徑）。',
   'flag_exclusion': '訊號股在 [t-4, t] 任何一天 flag ∈ {3,4,5} → 剔除（全市場同日 flag 4 >= 50 檔的缺口日不算）。',
   'disposal_exclusion': 't 或 t+1 在處置期間 → 剔除（名單 2021-09-24 起才有）。',
 },
 'indicators': {
   'RSI3': 'Wilder RSI(3) on adj：AG/AL 用 ewm(alpha=1/3, adjust=False)；AL=0 且 AG>0 → 100；皆 0 → 50。',
   'streak': 'adj_t > adj_{t-1} → 前值>0 則 +1 否則 +1 起算（連漲天數，正）；adj_t < adj_{t-1} → 連跌天數（負）；相等 → 0。',
   'RSI_streak_2': 'Wilder RSI(2) 套在 streak 序列上（d=streak_t − streak_{t-1}），alpha=1/2。',
   'PercentRank100': '1 日報酬 r_t=adj_t/adj_{t-1}−1；前 100 個交易日（t-100..t-1）中 r 嚴格小於 r_t 的比例 ×100；需 100 個值都有效。',
   'CRSI': '(RSI3 + RSI_streak_2 + PercentRank100)/3',
   'SMA200': 'adj 200 日簡單均線（min_periods=200）', 'SMA5': 'adj 5 日簡單均線（min_periods=5）',
 },
 'signal_t_close': {'F200': 'rule_pool 且 adj_t > SMA200_t（濾網版）', 'NOF': 'rule_pool（原版，不加濾網）',
                    'c10': 'CRSI_t < 10', 'c5': 'CRSI_t < 5'},
 'execution': {
   'limit_price': 'L = adj_t × (1 − 4%)（以還原價比較，等同對除權息參考價下 4%；不做 tick 取整）。',
   'fill_A': 't+1 還原開盤 <= L → 以開盤價成交；否則當天不成交、訊號作廢（保守：漏掉盤中觸價的成交，成交率被低估）。',
   'fill_B': 't+1 開盤無條件買（開盤漲停或無開盤價則不買）。此為「沒有限價」的不同規則，只作對照。',
   'fill_A2_diag': '（只在事件層作診斷）A 之外，若 t+1 還原收盤 <= L（盤中必定觸及 L）→ 以 L 成交。仍是下界（收盤回到 L 以上但盤中觸及者無法得知）。',
   'exit_signals': 'E70＝CRSI_s > 70；E50＝CRSI_s > 50；SMA5＝adj_s > SMA5_s；E70orSMA5＝兩者任一。從進場日 e 收盤開始逐日檢查，成立 → s+1 開盤賣。'
                   '教學原文「CRSI > 70（另測 > 50），或收盤 > SMA5」有歧義，四種都報；PRIMARY 用 E70（教學預設）。',
   'time_stop': '到 e+9 收盤仍未觸發 → e+10 開盤賣（持有最多 10 個交易日）。',
   'sell_frictions': '預定賣出日鎖跌停（還原最高/前收−1 <= −(漲跌幅−0.6%)）或無開盤 → 順延；下市 → 最後收盤賣。',
   'dedupe': '持有中同檔新訊號忽略。',
   'cost': 'exlib.cost_rates：買 0.0855%＋滑價 0.1%；賣 0.0855%＋稅 0.3%＋滑價 0.1%；另報毛與壓力（買 0.5%＋賣 0.5%）。',
 },
 'portfolio': {
   'slots': '10 槽，每筆＝當日開盤 NAV×10%（現金不足用剩餘，不足 2.5% NAV 不買）；不加碼、不再平衡；現金 0 報酬。',
   'order': '每日開盤先賣後買；候選＝前一日收盤訊號，依 CRSI 由低到高（同值成交值大者優先）。fill A：依序檢查開盤 <= L 者成交，直到槽滿（等同對所有候選掛單、超額成交取消；另報嚴格版：只對前「空槽數」名候選掛單）。fill B：依序買，開盤漲停跳過遞補。',
   'start': '2004-07 起模擬，統計從 2005-01。',
 },
 'variants': {
   'PRIMARY': 'F200 × c10 × E70 × fill A × 時間停損 10 日',
   'all_reported': '{F200, NOF} × {c10, c5} × {E70, E50, SMA5, E70orSMA5} × {A, B} ＝ 32 組，全部報告；其他組只作穩健性參考，不得改判。',
   'secondary_judgement': '同時報 NOF_c10_E70_A（原版無濾網）與 F200_c10_E70_B（無限價）的判定，僅供參考。',
 },
 'benchmarks': {
   'EW_rule_pool': '同池＝rule_pool（LIQ 且 HV100>30%）。事件層：每日等權開盤對開盤（前一日在池者）；組合層：每月第一個 >= 11 日交易日以收盤池、隔日開盤等權月換股、無成本。',
   'EW_liq': 'BRIEF 流動性股池同法計算（另報）。', '0050': '含息 0050。',
 },
 'event_stats': {'net': '(1+raw)×(1−賣)/(1+買)−1', 'abnormal': 'net − 同期間（進場開盤→賣出開盤）EW_rule_pool；另報對 EW_liq、對 0050',
                 't': '按進場月份平均後 t（btlib.tstat）；t_date 只供參考'},
 'windows': {'select': '2005-01-01~2019-04-30（事件以訊號日歸屬）', 'valid': '2019-05-01~2026-10-22', 'y2024_2025': '2024-01-01~2025-12-31（另報）',
             'others': '挑選期子段 2005-2011 / 2012-2019-04、驗證期存活者區段 2019-05~2023-12、驗證期剔除 src3、2021-2023 / 2024-2026（參考）'},
 'verdict_rule': 'PRIMARY 在挑選期與驗證期各自：事件淨異常（對 EW_rule_pool，滑價 0.1%）> 0 且 t_month >= 2，且 10 槽組合淨年化 >= 0050 同期 → 該段通過；兩段都過 → 可用；一段 → 待觀察；都沒過 → 不可用。'
                 '若 PRIMARY 在任一段成交事件 < 100 筆或有事件月份 < 24 → 該段資料不足；兩段都不足 → 無法測試。組合對同池等權月超額 t 一併報告。',
 'diagnostics_preplanned': ['fill A 成交率、fill A2 成交下界的事件報酬', 'PRIMARY 設定下限價 2/4/6/8/10% 的事件層報酬（fill A，只作參考）',
                            'fill A 組合嚴格掛單版', 't 收盤進場（不可成交）vs t+1 開盤', '壓力成本與毛報酬', '挑選期分 2005-2011 / 2012-2019-04', '逐年事件異常',
                            '極端報酬與 flag 3/4/5 檢查', '驗證期剔除 src 3', '依 CRSI 分層'],
 'known_limits': ['面板無最低價：限價單成交率被低估（A）或限價被忽略（B）', '2019–2023 面板只有存活者（抄底偏樂觀）', '2024-01-02~2024-02-14 無資料',
                  'src 3 收盤為 13:20 近似', '處置名單 2021-09 起才有', '已測 RSI(2)+SMA5（w3_connors_rsi2_sma5）：本規則換成 CRSI、加 HV 股池與限價低接'],
}
if __name__ == '__main__':
    f = HERE / 'results.json'
    d = json.loads(f.read_text(encoding='utf-8')) if f.exists() else {}
    assert 'PREREG' not in d, 'PREREG 已存在，不覆寫'
    d = {'PREREG': PREREG, **d}
    f.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding='utf-8')
    print('PREREG written')
