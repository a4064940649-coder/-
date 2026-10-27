#!/usr/bin/env python3
"""DUKE 新高値ブレイク（上場來新高＋營收成長）事前設定：看任何結果前寫入 results.json 的 PREREG；之後不改。"""
import json, datetime
from pathlib import Path
HERE = Path(__file__).resolve().parent
PREREG = {
 'written_at': datetime.datetime.now().isoformat(timespec='seconds'),
 'rule_name': 'DUKE 新高値ブレイク：上市以來（面板 2004 起）收盤新高＋月營收成長（近 3 月合計年增 >10% 且單月年增 >0），5 檔、停損 10%／相對弱勢 5%／獲利後跌破 50 日線／最長 250 日',
 'sources': ['https://media.rakuten-sec.net/articles/-/20223', 'https://media.rakuten-sec.net/articles/print/20225',
             'https://media.rakuten-sec.net/articles/-/41682', 'https://toyokeizai.net/articles/-/645082?display=b',
             'https://online.nikkei-cnbc.co.jp/guide/news/20160713tsubo/'],
 'type': '選股型（組合，最多 5 檔、資金 5 等份）為主報告與判定；另報事件型（全部訊號、不受 5 檔限制）',
 'data': {
   'panel': 'scratchpad/panel 上市日線（含已下市；2019–2023 只有存活者）。有值檔數 < 300 的日子（2018-01-13）整列剔除後再做任何 rolling／expanding，模擬也只在其餘交易日進行',
   'revenue': '長期月營收 niklavsd/bottleneck-signals tw_revenue.csv（千元；sii 優先去重；2013 起只剩存活公司）',
   'taiex': 'jonatw/finlab-txd taiex_twii.csv close（ffill 到面板日曆）',
   'disposal': '2021-09-24 起 disposal_5y 官方名單；之前用 disposal_bear_longhistory 的近似規則（6 日還原漲幅 >=32% 且比加權多 >=20pp、連 3 日 → 隔日起 10 日）',
 },
 'universe_t': '上市普通股（排除 91xx）；訊號日 t：mlib.V60（剔除缺漏日後的 60 日均成交值）>= NT$1,000 萬、close_raw >= 10 元、adj 有值；上市滿 3 年＝面板（剔除缺漏日後）累計有效 adj 天數 >= 750',
 'signal_t_close': {
   'ATH_raw (PRIMARY)': 'close_raw_t > max(close_raw 於面板起點 2004-01-02 至 t-1 的所有收盤)（2004 前上市者實為「2004 以來高點」）',
   'ATH_adj': 'adj_t > max(adj 於 2004-01-02 至 t-1)（避免除權息影響；含息還原）',
   'H52_raw (對照)': 'close_raw_t > max(close_raw_{t-252..t-1})（252 個有效交易日、至少 240 個有值）',
   'H52_adj (對照)': 'adj_t > max(adj_{t-252..t-1})（同上）',
   'revenue': '最近一期已公布月份 m*＝t 的日期 >= 11 日則上月，否則上上月；條件：(Σrev m*-2..m*)/(Σrev m*-14..m*-12) − 1 > 10% 且 rev_m*/rev_{m*-12} − 1 > 0（6+1 個月都須有值且 > 0）',
   'ifrs_break': 'm* 落在 2013-01～2014-02（年增跨 IFRS 口徑斷點）不產生訊號',
   'exclude_lockup': 't 收盤鎖漲停（ret_t >= 漲跌幅−0.6% 且 close_raw >= high×0.999；漲跌幅 2015-06-01 前 7%、之後 10%）不進場',
   'exclude_disposal': 't 或 t+1 在處置期間（官方名單 start～end，或近似規則）不進場',
   'repeat_signal': '只要當日收盤創新高且條件成立即為訊號（不要求第一次突破）；持有中的股票不重複；出場日收盤起可再觸發',
 },
 'execution': {
   'entry': 't 收盤訊號 → t+1 開盤買（還原開盤）；t+1 無開盤價或開盤漲停（mlib.GAP_UP，依年代 7%/10%）→ 買不到、由同日下一候選遞補',
   'priority': '同日候選依 t 的 60 日均成交值由小到大（DUKE 偏好中小型）；另以 10 個隨機順序種子做診斷',
   'sizing': '最多同時 5 檔；每檔 = 當日開盤 NAV/5，不足用剩餘現金（現金 < 0.25 份則不買）；現金零利率',
   'exit_rules_at_close (含進場當日收盤起判斷，任一成立 → 隔日開盤賣)': {
     'stop10': 'adj_t <= 0.90 × 進場價',
     'rel_weak5': '加權指數 20 日報酬（TX_t/TX_{t-20}−1，有效交易日）> 0 且 adj_t <= 0.95 × 進場價',
     'ma50_after_profit': '進場後曾有收盤 > 進場價（獲利後）且 adj_t < adj 50 日簡單均線（有效交易日、50 個都需有值）',
     'max_hold': '持有滿 250 個有效交易日（第 e+250 日開盤賣）',
   },
   'exit_fill': '隔日無開盤價（停牌）延到下一個有開盤價的日子；已下市以最後還原收盤賣出；跌停賣不掉未模擬',
   'order_at_open': '先出場 → 再新進場',
   'cost': 'exlib.cost_rates：主情境滑價 0.1%（買 0.0855%+0.1%、賣 0.0855%+0.3%+0.1%，來回約 0.67%）；另報 0 滑價（來回約 0.47%≈0.5%）、壓力（買賣各 0.5%，來回 1%）與毛報酬',
   'sim_start': '2006-07-01 起模擬（上市滿 3 年＋營收 15 個月使第一個可能訊號約在 2007-01）',
 },
 'variants': {
   'PRIMARY': 'ATH_raw（5 檔組合）',
   'reported': ['ATH_raw', 'ATH_adj', 'H52_raw（對照）', 'H52_adj（對照）'],
   'incremental': '「上市以來新高比 52 週高多出來的部分」：(a) 事件：同口徑 H52 訊號拆成「同時是 ATH」與「52 週高但非 ATH」，月份分群異常報酬差與 t；(b) 組合：ATH − H52 月窗報酬差 t',
 },
 'benchmarks': {
   '0050': '0050 含息（ffill 到面板日曆）',
   'EW_pool': '同池等權：每月第一個 >= 11 日交易日 d 收盤的股池（流動性＋普通股＋上市滿 750 日＋d+1 可交易）全部等權，d+1 開盤換股，不扣成本',
 },
 'portfolio_stats': '日頻 NAV：年化、日頻最大回撤、逐年；t＝月窗（每月第一個 >= 11 日交易日收盤 → 下一個）超額報酬 t（每月一個觀察值＝月份分群）對 0050 與對同池等權',
 'event_stats': '全部訊號（不設 5 檔上限；同檔持有中不重複）：net＝(1+raw)(1−賣)/(1+買)−1；異常＝net − 訊號日同池等權同窗（進場開盤→出場開盤）報酬；另報對「全上市等權」（所有 t+1 可交易的上市普通股）；先按進場月份平均再做 t',
 'windows': {'select': '2007-01-01~2019-04-30（價格類挑選期 2005-01 起，但首個可能訊號約 2007-01）', 'valid': '2019-05-01~2026-10-22', 'y2024_2025': '2024-01-01~2025-12-31',
             'extra': ['2021-01~2023-12 / 2024-01~2026-10（2021 起資料口徑的挑選／驗證，參考）', '驗證期存活者段 2019-05~2023-12', '驗證期官方價格段（剔除 src3 2024-02-15~2025-09-23）']},
 'verdict_rule': '以滑價 0.1% 的 PRIMARY 組合判定。某期間「通過」＝ 對同池等權月窗 t >= 2 且 年化 >= 0050 同期年化。挑選期與驗證期都通過 → 可用；只一段 → 待觀察；都不通過 → 不可用。ATH_adj 只作穩健性：若 PRIMARY 不可用但 ATH_adj 兩段都通過，整體最多升為「待觀察」。H52 為對照，不參與判定。',
 'known_limits': ['無最低價：停損以收盤判斷、隔日開盤出場（教學為盤中停損單）', '2004 前上市股票的真實上市以來高點看不到，只能用 2004 以來', '原始價口徑受減資／股票股利影響（減資會抬高原始價造成假突破）→ 另報 ATH_adj 與「訊號前 20 日還原因子下降 >5%」剔除診斷',
                  '2019–2023 面板只有存活者（對突破策略偏樂觀）；2013 起營收只有存活公司', '2024-01-02~2024-02-14 無資料', 'src 3 期間收盤為 13:20 近似', '2021-09 前處置名單為近似規則（召回約 56%）', '「增收增益」以月營收替代（無即時 EPS）'],
}
if __name__ == '__main__':
    f = HERE / 'results.json'
    d = json.loads(f.read_text(encoding='utf-8')) if f.exists() else {}
    assert 'PREREG' not in d, 'PREREG 已存在，不覆寫'
    d = {'PREREG': PREREG, **d}
    f.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding='utf-8')
    print('PREREG written')
