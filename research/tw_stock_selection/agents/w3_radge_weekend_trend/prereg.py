#!/usr/bin/env python3
"""Nick Radge 週末趨勢交易者（Weekend Trend Trader）事前設定：在看任何結果前寫入 results.json 的 PREREG；之後不改。"""
import json, datetime
from pathlib import Path
HERE = Path(__file__).resolve().parent
PREREG = {
 'written_at': datetime.datetime.now().isoformat(timespec='seconds'),
 'rule_name': 'Nick Radge Weekend Trend Trader：週收盤 = 20 週最高收盤 且 ROC(20 週) >= 30% 且 加權指數週收盤 > 10 週 SMA → 下週一開盤買；移動停損 40%（大盤濾網成立）/10%（大盤跌破 10 週均線），只上移；每檔 5%、最多 20 檔、同週依 ROC 排序',
 'sources': ['https://usethinkscript.com/goto/post?id=5029', 'https://il.tradingview.com/script/QytXbleE', 'https://thechartist.com.au/?p=101421'],
 'type': '選股型（組合，每檔 5%、最多 20 檔）為主報告與判定；另報事件型（不受 20 檔限制的全部訊號）供參考',
 'data': {
   'panel': 'scratchpad/panel 上市日線（含已下市，2019–2023 只有存活者）；有值檔數 < 300 的日子（只有 2018-01-13）整列剔除後才建週線與做任何 rolling，模擬也只在其餘交易日進行',
   'price': '個股一律用含息還原收盤（adj，ffill 最多 4 個交易日以處理短暫停牌）；只用收盤價（不用最高價）',
   'index': '加權指數 taiex_twii.csv 收盤（價格指數）；在指數自己的完整日曆上取每週最後一個交易日收盤、算 10 週 SMA，再對到面板的週末日（面板 2024-01-02~02-14 缺資料，不影響指數濾網）',
   'universe': '上市普通股（排除 91xx 存託憑證）；週末日 w：60 日均成交值 >= NT$1,000 萬（mlib.V60：剔除缺漏日後滾動）、原始收盤 >= 10 元、w 當天有收盤；原版另有成交值門檻，以 BRIEF 流動性條件代替',
 },
 'weekly_bars': '週 = 週一~週日的日曆週；週收盤 = 該週最後一個（保留的）交易日收盤（週五休市則取週四，週六補班日則取週六）。面板缺資料的週直接不存在（20 週視窗跨過缺口）',
 'signal_week_close': {
   'C1_20w_high': '本週還原收盤 >= 前 20 根週收盤（含本週）的最大值（rolling 20、min_periods 20）',
   'C2_ROC20': '本週週收盤 / 20 週前週收盤 − 1 >= 30%',
   'C3_index_filter': '加權指數本週收盤 > 加權指數 10 週收盤 SMA（含本週）',
   'not_held': '同檔已持有中不重複進場',
 },
 'execution': {
   'entry': '週末日 w 收盤訊號 → 下一個交易日（通常週一）開盤買（還原開盤）；開盤漲停（開盤/前收−1 >= 漲跌幅−0.6%，2015-06-01 前 7%、之後 10%，mlib.GAP_UP）或無開盤價 → 不買、由下一名遞補；隔週若仍符合可再進',
   'ranking': '同週訊號依 ROC(20) 由高到低排序，依序填入空位',
   'sizing': '每檔 = 當日開盤 NAV × 5%（不足用剩餘現金；現金 < 0.25 份則不買）；最多 20 檔；現金零利率；持股不再平衡',
   'stop': '持有期間每個週末日：HH = 進場後（含進場當週）最高週收盤；p = 40%（加權指數週收盤 > 10 週 SMA）否則 10%；stop = max(前一週 stop, HH × (1 − p))（只上移）；週收盤 < stop → 下一交易日開盤賣',
   'exit_fill': '下一交易日無開盤價（停牌）延到下一個有開盤價的日子；已下市以最後還原收盤賣出；跌停賣不掉未模擬',
   'order_at_open': '先賣出 → 再新進場（同一個週一）',
   'holding_period': '無固定持有期，只靠移動停損出場',
   'cost': 'exlib.cost_rates：買＝0.1425%×0.6＋滑價 0.1%；賣＝0.1425%×0.6＋稅 0.3%＋滑價 0.1%；壓力情境 買 0.5%＋賣 0.5%；另報毛報酬',
   'sim_start': '2004-07-01 起模擬（暖身），統計自各期間起點',
 },
 'variants': {
   'PRIMARY': '如上（5% × 20 檔、ROC 排序）— 唯一判定依據',
   'diag_only': ['D1：同週候選改隨機順序（10 個種子）', 'D2：Radge 原版 10% × 10 檔', 'D3：不分大盤、停損固定 40%（看濾網收緊停損的貢獻）'],
 },
 'benchmarks': {
   '0050': '0050 含息（日頻 ffill 到面板日曆）',
   'EW_pool': '同池等權：每月第一個 >= 11 日交易日 d 收盤的同一流動性股池（普通股 + d+1 可交易）全部等權，d+1 開盤換股，不扣成本（exlib.sim_portfolio）',
 },
 'portfolio_stats': '日頻 NAV：年化、日頻最大回撤、逐年；t＝月窗（每月第一個 >= 11 日交易日收盤 → 下一個）超額報酬 t（每月一觀察值），對 0050 與對同池等權',
 'event_stats': '全部訊號（不設 20 檔上限；同檔事件進行中不重複）：net＝(1+raw)(1−賣)/(1+買)−1；異常＝net − 訊號週股池同窗（進場開盤→出場開盤）等權報酬；先按進場月份平均再做 t',
 'windows': {'select': '2005-01-01~2019-04-30', 'valid': '2019-05-01~2026-10-22', 'y2024_2025': '2024-01-01~2025-12-31',
             'extra': ['2021-01~2023-12 / 2024-01~2026-10（參考）', '驗證期存活者段 2019-05~2023-12', '驗證期官方價格段（剔除 src3 2024-02-15~2025-09-23）']},
 'verdict_rule': '以滑價 0.1% 的 PRIMARY 組合判定。某期間「通過」＝ 對同池等權月窗 t >= 2 且 年化 >= 0050 同期年化。挑選期與驗證期都通過 → 可用；只一段通過 → 待觀察；都不通過 → 不可用。診斷變體不影響判定。',
 'known_limits': ['TradingView 版停損用進場後最高價；本研究用最高週收盤', '停損以週收盤判斷、週一開盤成交（跳空可能遠低於停損）', '2019–2023 面板只有存活者（驗證期偏樂觀）',
                  '2024-01-02~2024-02-14 無資料（週線缺 6 週）', 'src 3 期間收盤為 13:20 近似', '處置股名單只有 2021 起，不排除處置股（與長歷史研究一致）'],
}
if __name__ == '__main__':
    f = HERE / 'results.json'
    d = json.loads(f.read_text(encoding='utf-8')) if f.exists() else {}
    assert 'PREREG' not in d, 'PREREG 已存在，不覆寫'
    d = {'PREREG': PREREG, **d}
    f.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding='utf-8')
    print('PREREG written')
