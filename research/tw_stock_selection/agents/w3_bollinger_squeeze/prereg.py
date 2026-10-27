#!/usr/bin/env python3
"""布林通道壓縮（Bollinger Squeeze）事前設定：看任何結果前寫入 results.json 的 PREREG；之後不改。"""
import json, datetime
from pathlib import Path
HERE = Path(__file__).resolve().parent
PREREG = {
 'written_at': datetime.datetime.now().isoformat(timespec='seconds'),
 'rule_name': '布林通道壓縮：BandWidth 為 125 日最低（壓縮）→ 10 個交易日內第一次收盤 > 上軌且量 > 20 日均量×1.5 → 隔日開盤買；收盤跌破 MB 隔日開盤賣，時間停損 40 日',
 'sources': ['https://www.luxalgo.com/library/concept/bollinger-squeeze/', 'https://www.luxalgo.com/library/concept/bandwidth/',
             'https://sentimentrader.com/education/technical-indicators/bollinger-bandwidth', 'https://daytradingtoolkit.com/strategies/bollinger-band-squeeze-strategy',
             'https://questdb.com/docs/cookbook/sql/finance/bollinger-bandwidth/', 'https://www.luxalgo.com/library/indicator/HcbazIig-bollinger-bands-w-squeeze-alert/'],
 'type': '事件型（主判定：全部訊號的逐筆事件報酬，每檔同時只持有一筆，不受部位上限限制）；另報組合（最多 20 檔、每檔開盤 NAV 5%，候選依當日量比排序）供參考',
 'data': {
   'panel': 'scratchpad/panel 上市日線（含已下市；2019–2023 只有存活者）。有值檔數 < 300 的日子（2018-01-13）整列剔除後再做所有 rolling；模擬只在其餘交易日進行',
   'price_basis': '布林、Keltner、出場判斷一律用 adj（含息還原收盤）。布林的 20 日均線與標準差用未 ffill 的 adj（窗內 20 日都要有值，避免停牌平盤造成假壓縮）。股價門檻用 close_raw',
   'volume': '成交量（股）= value / close_raw（面板無成交量欄）',
 },
 'universe_t': '上市普通股（排除 91xx）；訊號日 t：mlib.V60（剔除缺漏日的 60 日均成交值）>= NT$1,000 萬、close_raw >= 10 元、adj 有值',
 'bollinger': 'MB = SMA20(adj)；σ = 20 日母體標準差（ddof=0，Bollinger 原定義／TradingView ta.stdev）；UB = MB + 2σ、LB = MB − 2σ；BandWidth BW = (UB − LB)/MB = 4σ/MB',
 'squeeze_MAIN': 'SQ_t：BW_t <= 過去 125 個交易日（含 t）BW 最小值（rolling min，窗內至少 120 個有效 BW）',
 'entry_MAIN': {
   'episode': '每檔維護「最近一次壓縮日」s。若 t − s <= 10（含 s = t）且這次壓縮後尚未出現過收盤 > UB，則 t 為候選日；新的壓縮日會重設 s 與「已用過」旗標',
   'first_breakout': 'adj_t > UB_t 的第一個日子（壓縮後第一次）。這一天就把該次壓縮「用掉」：不論是否符合量能或股池，該次壓縮之後不再找第二次突破（逐字：第一次出現收盤 > UB，且當日量 > 均量×1.5）',
   'volume': '量_t > 1.5 × 20 日均量（20 日均量含 t，TradingView ta.sma(volume,20) 預設；窗內 20 日皆需有值）',
   'universe': 't 須在股池內',
   'execution': 't+1（下一個有效交易日）開盤買；t+1 無開盤價或開盤漲停（相對前收跳空 >= 漲跌幅 − 0.6%，2015-06-01 前 7%、之後 10%）視為買不到、該訊號放棄。t 收盤鎖漲停本身不排除（隔日開盤仍可能成交），另報診斷',
 },
 'exit': '從進場當日收盤起，第一個 adj < MB 的收盤日 x → x+1 開盤賣；若進場後 40 個交易日內（e … e+39 收盤）未觸發，e+40 開盤賣（時間停損）。停牌：順延至下一個有開盤價的日子；下市：以最後收盤計。每檔持有中不重複進場（事件與組合都是）',
 'variants_preregistered': {
   'a_NOVOL': '主版本去掉量能條件（第一次收盤 > UB 即買）',
   'b_PCT5': '壓縮改為 BW_t <= 過去 250 日（含 t）BW 的 5% 分位（rolling quantile 0.05，窗內至少 240 個有效 BW）；其餘同主版本',
   'c_TTM': 'TTM Squeeze 近似：壓縮 = UB < KCU 且 LB > KCL，KC = EMA20(adj) ± 1.5 × ATR20；ATR20 = 20 日簡單平均的近似真實區間 TR = max(|c_t − c_{t−1}|, h_t − c_{t−1})（h 為還原最高價 = high × adj/close_raw；面板無最低價，TR 偏小 → Keltner 偏窄 → 壓縮較難成立，標註為近似）；其餘（10 日內第一次收盤 > UB、量能、出場）同主版本',
   'd_HEADFAKE': 'Bollinger 假突破：主版本壓縮後 10 個交易日內，第一次收盤跑出通道是往下（adj < LB；若第一次跑出是往上則該次壓縮作廢）；之後 10 個交易日內第一次 adj > MB → 隔日開盤買。不要求量能（量能是教學對「向上突破」的確認）；另報 d_HEADFAKE_VOL（收回 MB 當日量 > 20 日均量×1.5）作診斷',
 },
 'control_not_judged': 'BB_BREAK：無壓縮條件，adj_t > UB_t 且 adj_{t−1} <= UB_{t−1}（當日首次站上），量能同主版本，出場同主版本。用來看「壓縮」本身有沒有增量',
 'costs': 'exec_rebuild/exlib.cost_rates：主情境滑價 0.1%（買 0.0855%+0.1%、賣 0.0855%+0.3%+0.1%，來回約 0.67%）；另報 gross、0 滑價（約 0.47%）、壓力（來回 1%）',
 'benchmarks': {
   'event_poolEW': '同一持有窗（e 開盤 → x 開盤）訊號日 t 股池內、t+1 有開盤價的所有股票等權報酬（不扣成本）',
   'event_allEW': '同窗全上市普通股等權（診斷）',
   'event_0050': '0050 含息：E50[x−1]/E50[e−1] − 1（收盤對收盤近似，0050 無開盤價）',
   'portfolio': '0050 含息與同池等權（每月第一個 >= 11 日交易日 d 取股池，d+1 開盤換股，不扣成本）；組合 t 為月窗超額',
 },
 'stats': '事件異常報酬 = 扣成本淨報酬 − 同窗同池等權；t 按進場月份分群（各月平均異常 → 月序列 t）；挑選期 2005-01～2019-04、驗證期 2019-05～2026-10 以訊號日分段；另報 2024-01～2025-12、2021–2023／2024–2026、驗證期只用官方價格（剔除 src3 2024-02-15～2025-09-23）、驗證期存活者段 2019-05～2023-12',
 'judgement': {
   'basis': 'MAIN 版本、滑價 0.1% 的事件報酬',
   'period_pass': '月份分群 t（對同池等權）>= 2 且平均異常 > 0，且對 0050 月份分群平均超額 >= 0（不輸 0050）',
   'verdict': '兩段都過 → 可用；只一段 → 待觀察；都不過 → 不可用；任一段可成交事件 < 30 筆或進場月份 < 12 → 無法測試',
   'variants': '變體與對照全部報告，不參與判定、不據以換主版本',
 },
}
if __name__ == '__main__':
    p = HERE / 'results.json'
    assert not p.exists(), 'PREREG 已存在，不覆寫'
    p.write_text(json.dumps({'PREREG': PREREG}, ensure_ascii=False, indent=1), encoding='utf-8')
    print('PREREG written', PREREG['written_at'])
