#!/usr/bin/env python3
"""Weinstein 階段分析：第二階段突破。事前設定：在看任何結果前寫入 results.json 的 PREREG；之後不改。"""
import json, datetime
from pathlib import Path
HERE = Path(__file__).resolve().parent
PREREG = {
 'written_at': datetime.datetime.now().isoformat(timespec='seconds'),
 'rule_name': 'Weinstein Stage 2 突破：週線 MA30w 走平基底 → 週收盤突破基底上緣且站上上彎 MA30w、Mansfield RS > 0、週成交值放大；週收盤跌破 MA30w 出場',
 'sources': ['https://www.thepatternsite.com/Stages.html', 'https://www.mql5.com/en/articles/22746'],
 'sources_note': '本環境無法連外（DNS 失敗），規則以任務給定文字為準；教學未給的參數用最常見預設（下列「預設」標註）',
 'type': '選股型（組合，最多 20 檔等權、先到先得）為主報告與判定；另報事件型（全部訊號、不設 20 檔上限）供參考',
 'data': {
   'panel': 'scratchpad/panel 上市日線（含已下市，2019–2023 只有存活者）；有值檔數 < 300 的日子（2018-01-13）整列剔除後才建週線與任何 rolling；模擬只在其餘交易日進行',
   'price': '週收盤＝該週（週一～週五）最後一個交易日的含息還原收盤 adj（避免除息造成假跌破）；訊號週要求該股在該週最後交易日有收盤',
   'index': '加權指數 taiex_twii.csv close（價格指數，ffill 到面板日曆），週收盤同上',
   'weekly_value': '週成交值＝該週日成交值 value 加總（NT$；教學用成交量，此處依規則用成交值）',
   'universe': '上市普通股（4 碼、排除 91xx）；訊號週最後交易日：60 日均成交值 >= NT$1,000 萬（mlib.V60，已剔除缺漏日）、原始收盤 >= 10 元；下週第一個交易日有開盤價',
 },
 'signal_week_k': {
   'MA30w': 'MA30_k＝最近 30 個週收盤平均（30 週都要有值）',
   'MRS': 'RS_k＝週收盤 adj / 加權指數週收盤；MRS_k＝(RS_k / SMA52(RS)_k − 1)×100（52 週都要有值）；「MRS > 0 或由負轉正」＝ MRS_k > 0（由負轉正是其子集）',
   'stage1_base': '|MA30_{k-1} / MA30_{k-11} − 1| < 2%（突破週前 10 週 MA30w 走平；只在 k-1 檢查一次，預設）',
   'base_upper': 'max(週收盤_{k-10..k-1})（過去 10 週最高週收盤，10 週都要有值）',
   'entry': '週收盤_k > 基底上緣 且 週收盤_k > MA30_k 且 MA30_k >= MA30_{k-1} 且 MRS_k > 0 且 基底條件 且 量能條件',
   'volume_V1': '週成交值_k >= 2 × mean(週成交值_{k-4..k-1})（主）',
   'volume_V2': '週成交值_k >= 1.5 × median(週成交值_{k-20..k-1})（另測）',
   'exit_X1': '週收盤_k < MA30_k → 下週第一個交易日開盤賣（主）',
   'exit_X2': '最低週收盤移動停損（另測，取代 X1）：stop_k = max(L10_s, …, L10_k)，L10_j＝min(週收盤_{j-10..j-1})；初始＝基底 10 週最低週收盤；週收盤_k < stop_k → 下週第一個交易日開盤賣（10 週為預設，與基底長度相同）',
   'first_exit_check': '進場後第一個週收盤（進場週）開始檢查',
 },
 'execution': {
   'entry': '訊號週最後交易日收盤後 → 下週第一個交易日（通常週一）開盤買（還原開盤）；該開盤漲停（開盤/前收−1 >= 漲跌幅−0.6%，2015-06-01 前 7%、之後 10%，mlib.GAP_UP）→ 買不到、該訊號放棄（下週若仍符合可再進）',
   'exit': '出場訊號週的下週第一個交易日開盤賣；無開盤價（停牌）延到下一個有開盤價的日子；已下市以最後還原收盤賣出',
   'max_positions': '最多同時 20 檔；同一執行日的候選依訊號週 MRS 由大到小（Weinstein 偏好強 RS，預設）；另以 10 個隨機順序種子做診斷',
   'sizing': '每檔 = 當日開盤 NAV/20，不足用剩餘現金（現金 < 0.25 份則不買）；持有期間不再平衡；同檔持有中忽略新訊號',
   'order_at_open': '先出場 → 再新進場',
   'cash': '現金零利率',
   'cost': 'exlib.cost_rates：買＝0.1425%×0.6＋滑價 0.1%；賣＝0.1425%×0.6＋稅 0.3%＋滑價 0.1%（來回約 0.67%）；壓力情境 買 0.5%＋賣 0.5%；另報毛報酬',
   'sim_start': '2004-07-01 起模擬（實際第一個訊號需約 2 年週線暖身：MA30w + RS SMA52），統計自各期間起點',
 },
 'variants': {
   'PRIMARY': 'V1X1：量能 2×前 4 週平均、週收盤 < MA30w 出場',
   'reported': ['V1X1（主）', 'V2X1（1.5×20 週中位數量能）', 'V1X2（移動停損出場）', 'V2X2'],
 },
 'benchmarks': {
   '0050': '0050 含息（日頻 ffill 到面板日曆）',
   'EW_pool': '同池等權：每月第一個 >= 11 日交易日 d 收盤的同一股池（流動性 + 普通股 + d+1 可交易）全部等權，d+1 開盤換股，不扣成本（exlib.sim_portfolio）',
 },
 'portfolio_stats': '日頻 NAV：年化、日頻最大回撤、逐年；t＝月窗（每月第一個 >= 11 日交易日收盤 → 下一個）超額報酬 t（每月一觀察值）對 0050 與對同池等權',
 'event_stats': '全部訊號（不設上限；同檔持有中不重複）：net＝(1+raw)(1−賣)/(1+買)−1；異常＝net − 訊號週股池同窗（進場開盤→出場開盤）等權報酬；先按進場月份平均再做 t',
 'windows': {'select': '2005-01-01~2019-04-30', 'valid': '2019-05-01~2026-10-22', 'y2024_2025': '2024-01-01~2025-12-31',
             'extra': ['2021-01~2023-12 / 2024-01~2026-10（參考）', '驗證期存活者段 2019-05~2023-12', '驗證期官方價格段（剔除 src3 2024-02-15~2025-09-23）']},
 'verdict_rule': '以滑價 0.1% 的組合判定。某期間「通過」＝ 對同池等權月窗 t >= 2 且 年化 >= 0050 同期年化。PRIMARY 挑選期與驗證期都通過 → 可用；只一段通過 → 待觀察；都不通過 → 不可用。其他 3 個變體只作穩健性：若 PRIMARY 不可用但某變體兩段都通過，整體最多升為「待觀察」並註明多重檢定。',
 'known_limits': ['無最低價：停損以週收盤判斷（原版為盤中停損單）', '週成交值以成交值代替成交量；連假短週的週成交值偏低',
                  'RS 用含息還原價 / 價格指數（RS 每年約有股息率的上漂）', '2019–2023 面板只有存活者（驗證期偏樂觀）',
                  '2024-01-02~2024-02-14 無資料（週線 rolling 直接跨過，約 6 週缺口）', 'src 3 期間收盤為 13:20 近似', '跌停賣不掉未模擬（開盤即成交）'],
}
if __name__ == '__main__':
    f = HERE / 'results.json'
    d = json.loads(f.read_text(encoding='utf-8')) if f.exists() else {}
    assert 'PREREG' not in d, 'PREREG 已存在，不覆寫'
    d = {'PREREG': PREREG, **d}
    f.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding='utf-8')
    print('PREREG written')
