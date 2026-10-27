#!/usr/bin/env python3
"""VCP（波動收縮型態）突破 程式化近似版 事前設定：看任何結果前寫入 results.json 的 PREREG；之後不改。"""
import json, datetime
from pathlib import Path
HERE = Path(__file__).resolve().parent
PREREG = {
 'written_at': datetime.datetime.now().isoformat(timespec='seconds'),
 'rule_name': 'VCP 突破（程式化近似）：趨勢模板股池 + 收盤 zigzag(5%) 近 60～150 日 2～4 段遞減回檔（後段 ≤ 前段×0.7、最後一段 ≤ 10%、最後一段量縮）→ 收盤突破樞紐且量 > 50 日均×1.5，隔日開盤買；停損最後低點／跌破 SMA50／持有 60 日；每檔 5%、最多 20 檔',
 'source': 'https://www.luxalgo.com/library/concept/volatility-contraction-pattern.md',
 'type': '選股型（組合：最多 20 檔、每檔 NAV 5%）為主報告與判定；另報事件型（全部訊號、不受 20 檔限制）',
 'data': {
   'panel': 'scratchpad/panel 上市日線（含已下市；2019–2023 只有存活者）。有值檔數 < 300 的日子（2018-01-13）整列剔除後再做任何 rolling／zigzag，模擬也只在其餘交易日進行',
   'price_basis': '型態、均線、52 週高、停損一律用 adj（含息還原收盤；以收盤近似，面板無 low 欄），adj 先 ffill(limit=10) 補短暫停牌再算 zigzag／均線；訊號日本身須有 adj 值。股價門檻用原始收盤。成交值用 value（NT$）',
 },
 'universe_t': '上市普通股（排除 91xx）；訊號日 t：mlib.V60（剔除缺漏日後的 60 日均成交值）>= NT$1,000 萬、close_raw >= 10 元、adj 有值',
 'trend_filter_t': 'adj_t > SMA50_t > SMA150_t > SMA200_t（簡單均線，min_periods = n）且 adj_t >= 0.75 × 252 日最高 adj（min_periods 240；「距 52 週最高收盤 < 25%」）',
 'zigzag': {
   'definition': '因果式收盤 zigzag，反轉門檻 5%：上升段追蹤最高收盤 Hx，當 adj <= Hx×0.95 時確認 Hx 為高點樞紐（確認日＝該日）；下降段追蹤最低收盤 Lx，當 adj >= Lx×1.05 時確認 Lx 為低點樞紐。起始方向由先觸發者決定。訊號日 t 只用確認日 <= t 的樞紐',
   'contraction': '一段回檔＝已確認的高點樞紐 H_i 與其後緊接的已確認低點樞紐 L_i；深度 d_i = 1 − L_i/H_i',
   'base': 't 時最後一個已確認樞紐須為低點。取最近 k 段回檔（k ∈ {2,3,4}，存在任一 k 滿足即可）：k 段的所有 H_i 位於 [t−150, t] 個有效交易日內、且 t − idx(最早那段 H) >= 60；深度遞減 d_{i+1} <= 0.7 × d_i（k 段內每一對相鄰回檔）；最後一段 d_k <= 10%',
   'volume_dryup (PRIMARY)': '最後一段回檔的低點日 idx(L_k)：10 日均成交值 < 50 日均成交值（均以剔除缺漏日的 rolling，min_periods = n）',
   'pivot': '樞紐價 = H_k（最後一段的高點收盤，adj）；停損價 = L_k（最後一段低點收盤，adj）',
 },
 'entry_signal_t_close': {
   'breakout': 'adj_t > 樞紐 且 adj_{t−1} <= 樞紐（當日首次站上；同一底部只觸發一次）',
   'volume': 'value_t > 1.5 × 50 日均成交值（截至 t−1）',
   'exclude_lockup': 't 收盤鎖漲停（ret_t >= 漲跌幅−0.6% 且 close_raw >= high×0.999；漲跌幅 2015-06-01 前 7%、之後 10%）→ 視為買不到、不進場',
   'repeat': '持有中的股票不重複；出場後可由新的突破再觸發',
   'disposal': '教學未提，不另外過濾處置股（官方名單只有 2021-09 起）',
 },
 'execution': {
   'entry': 't 收盤訊號 → t+1 開盤買（還原開盤）；t+1 無開盤價或開盤漲停（mlib.GAP_UP，依年代 7%/10%）→ 買不到，由同日下一候選遞補',
   'priority': '同日候選依突破量比 value_t / V50_{t−1} 由大到小；另以 10 個隨機順序種子做診斷',
   'sizing': '最多同時 20 檔；每檔 = 當日開盤 NAV/20（5%），不足用剩餘現金（現金 < 0.25 份則不買）；現金零利率',
   'exit_rules_at_close (含進場當日收盤起判斷，任一成立 → 隔日開盤賣)': {
     'stop_last_low': 'adj_t < 停損價 L_k',
     'below_sma50': 'adj_t < SMA50_t',
     'max_hold': '持有滿 60 個有效交易日（第 e+60 日開盤賣）',
   },
   'exit_fill': '隔日無開盤價（停牌）延到下一個有開盤價的日子；已下市以最後還原收盤賣出；跌停賣不掉未模擬',
   'order_at_open': '先出場 → 再新進場',
   'cost': 'exlib.cost_rates：主情境滑價 0.1%（買 0.0855%+0.1%、賣 0.0855%+0.3%+0.1%，來回約 0.67%）；另報 0 滑價（來回約 0.47%）、壓力（買賣各 0.5%，來回 1%）與毛報酬',
   'sim_start': '2005-01-03（SMA200 與 252 日高需約 1 年歷史，第一個可能訊號約 2005-01）',
 },
 'variants': {
   'PRIMARY': 'VCP（上述）',
   'robustness_only (不參與判定)': {
     'DRY_T1': '量縮改在突破前一日 t−1 衡量：MA10(value)_{t−1} < MA50(value)_{t−1}',
     'NO_DRY': '拿掉量縮條件（看量縮條件的增量）',
     'TREND_BRK (對照)': '只有趨勢股池 + 收盤突破前 20 日最高收盤（首次）+ 量 > 50 日均×1.5，無收縮條件；停損改為前 20 日最低收盤、其餘出場相同 → 衡量「收縮型態」的增量',
   },
 },
 'minervini_overlap': '訊號日 t 是否同時符合 Minervini 8 條件趨勢模板（w1_minervini_trend_template BASE：C1 收盤>MA150 且>MA200、C2 MA150>MA200、C3 MA200(t)>MA200(t−21)、C4 MA50>MA150 且>MA200、C5 收盤>MA50、C6 收盤 >= 1.30×252 日最低收盤、C7 收盤 >= 0.75×252 日最高收盤、C8 RS 百分位 >= 70，RS=0.4r63+0.2r126+0.2r189+0.2r252 在流動性股池內排名）；報 VCP 訊號中符合模板的比例、組合持股日中符合比例，以及符合／不符合兩組的事件異常報酬',
 'benchmarks': {
   '0050': '0050 含息（ffill 到面板日曆）',
   'EW_pool': '同池等權：每月第一個 >= 11 日交易日 d 收盤的流動性股池（普通股、V60>=1,000 萬、收盤>=10、d+1 可交易）全部等權，d+1 開盤換股，不扣成本',
 },
 'portfolio_stats': '日頻 NAV：年化、日頻最大回撤、逐年；t＝月窗（每月第一個 >= 11 日交易日收盤 → 下一個）超額報酬 t（每月一個觀察值＝月份分群）對 0050 與對同池等權',
 'event_stats': '全部訊號（不設 20 檔上限；同檔持有中不重複）：net＝(1+raw)(1−賣)/(1+買)−1；異常＝net − 訊號日同池等權同窗（進場開盤→出場開盤）報酬；另報對全上市等權；先按進場月份平均再做 t',
 'windows': {'select': '2005-01-01~2019-04-30', 'valid': '2019-05-01~2026-10-22', 'y2024_2025': '2024-01-01~2025-12-31',
             'extra': ['2021-01~2023-12 / 2024-01~2026-10（參考）', '驗證期存活者段 2019-05~2023-12', '驗證期官方價格段（剔除 src3 2024-02-15~2025-09-23）']},
 'verdict_rule': '以滑價 0.1% 的 PRIMARY 組合判定。某期間「通過」＝ 對同池等權月窗 t >= 2 且 年化 >= 0050 同期年化。挑選期與驗證期都通過 → 可用；只一段 → 待觀察；都不通過 → 不可用；挑選期或驗證期平均持股 < 2 檔或交易 < 30 筆 → 無法測試。變體不參與判定。',
 'known_limits': ['無最低價：zigzag、停損以收盤判斷，停損隔日開盤出場（教學為盤中停損單）', '無成交量欄，量縮／放量以成交值代替',
                  '2019–2023 面板只有存活者（對突破策略偏樂觀）', '2024-01-02~2024-02-14 無資料', 'src 3 期間收盤為 13:20 近似',
                  'VCP 原為主觀型態，程式近似的參數（zigzag 5%、0.7 遞減、60～150 日）照任務給定'],
}
if __name__ == '__main__':
    f = HERE / 'results.json'
    d = json.loads(f.read_text(encoding='utf-8')) if f.exists() else {}
    assert 'PREREG' not in d, 'PREREG 已存在，不覆寫'
    d = {'PREREG': PREREG, **d}
    f.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding='utf-8')
    print('PREREG written')
