#!/usr/bin/env python3
"""事前設定（看任何策略結果前寫入 results.json 的 PREREG；之後不改）。"""
import json, datetime
from pathlib import Path
HERE = Path(__file__).resolve().parent
PREREG = {
 'written_at': datetime.datetime.now().isoformat(timespec='seconds'),
 'rule_name': '一進二／二板接力（A 股連板戰法、日本ストップ高翌日）',
 'sources': ['https://www.toutiao.com/zixun/7500086229745305634/', 'https://sina.cn/news/detail/5231826177361317.html',
             'https://www.thepaper.cn/newsDetail_forward_4058183', 'https://jioinc.jp/investleaders/?p=11374'],
 'rule_verbatim': '股池：上市流動性池（60 日均成交值 ≥ NT$1,000 萬、股價 ≥ 10 元），排除處置與全額交割。Day0：連續第 2 天收盤漲停（二板），且 Day−1 是首板（t−11～t−2 沒有收盤漲停）。'
                  'Day1：開盤 < 漲停價（不是一字板），且開盤 ≥ Day0 收盤 × 0.97，開盤買進。出場 A Day1 收盤（當沖）、B Day2 開盤、C 連板持有（收盤漲停續抱，第一個收盤沒漲停的隔日開盤賣，最多 10 日；開盤跌停延後）。'
                  '敏感度：Day0 量 ≤ Day−1 量（縮量）／ > Day−1 量（放量）；三板版本；對照組 Day0 首板。',
 'type': '事件型：主報告＝事件報酬（扣成本、對同池等權與上市等權的異常報酬、t 按進場月份分群）；另報日曆時間組合（年化、日頻最大回撤、對 0050／同池等權月超額 t、逐年）',
 'data': {
   'panel': 'scratchpad/panel 上市日線（含已下市；2019–2023 只有存活者）。先剔除「有值檔數 < 300」的日子（2018-01-13）再做任何 rolling；flag 5 價格視為缺值（沿用 w3_limitup_pullback/lim.py）',
   'stocks': '4 碼數字代碼、排除 91xx（存託憑證）',
   'limit_price': '直接 import agents/w3_limitup_pullback/lim.py（不改原檔、不寫 bytecode）：參考價 = close_raw/(1+ret)，漲停價 = 參考價×(1+漲幅) 依升降單位無條件捨去，漲幅 2015-06-01 前 7%、之後 10%（分段），2005-03-01 前舊制升降單位；'
                  '收盤漲停 = close_raw ≥ 漲停價 − 半檔 且 close_raw ≥ high×0.999 且 flag ∈ {0,1,2}；開盤漲停（一字板）= open ≥ 漲停價 − 半檔',
   'limit_down': '跌停價 = 參考價×(1−漲幅) 依升降單位無條件進位；價格 ≤ 跌停價 + 半檔 視為跌停（同 w12_first_board_gap_down）',
   'volume': '成交量 = value / close_raw（面板沒有量欄）',
   'adjusted_prices': '跨日比較一律用還原價：還原收盤 adj、還原開盤 = open×adj/close_raw',
 },
 'universe': {
   'liquidity': 'Day0：60 日均成交值（上市後停牌日記 0，滿 60 筆）≥ NT$1,000 萬且 Day0 原始收盤 ≥ 10 元，flag ∈ {0,1,2}',
   'disposal': 'disposal_5y（2021-09 起）每筆 start→end 對到面板交易日；Day0 或 Day1 在處置期間 → 排除。2021-09 前無名單（限制）',
   'full_delivery': '全額交割股：無名單資料，無法排除（限制；多半已被股價 ≥ 10、流動性條件擋掉）',
 },
 'signal': {
   'BASE (二板)': 'Day0 收盤漲停、Day−1 收盤漲停、Day−11..Day−2 都沒有收盤漲停',
   'day1_entry': 'Day1 = Day0 下一個面板交易日；需有開盤價、flag ∈ {0,1,2}、不在處置；開盤 < 漲停價 − 半檔（非一字板，否則買不到）；還原開盤 / Day0 還原收盤 ≥ 0.97 → 以 Day1 開盤價買進',
   'timing_note': '以開盤價成交、條件也用開盤價（需開盤撮合後立即下單；差異由滑價 0.1% 吸收，限制）',
   'dedupe': '同一檔若前一事件尚未出場，新事件跳過（實際上首板條件使重疊幾乎不可能）',
 },
 'exits': {
   'A': 'Day1 收盤賣（現股當沖）；只計 2014-01-06（現股當沖開放）起的事件。Day1 收盤跌停 → 賣不掉，改 Day2 開盤以一般賣出成本賣',
   'B': 'Day2 開盤賣',
   'C': '連板持有：k = Day1, Day2, …；Day k 收盤漲停 → 續抱；第一個收盤沒漲停（或停牌無收盤）的 Day k → Day k+1 開盤賣；Day1～Day10 全部收盤漲停 → Day11 開盤賣',
   'halt_or_limit_down': '出場時點無價或處在跌停 → 順延到下一個價格時點（下一個收盤、再下一個開盤…），最多再 10 個交易日；已下市 → 最後收盤賣',
 },
 'costs': {
   'B_C': 'exlib.cost_rates(0.001)：買 = 0.1425%×0.6 + 滑價 0.1%；賣 = 0.1425%×0.6 + 證交稅 0.3% + 滑價 0.1%（來回約 0.67%）；壓力：買 0.5% + 賣 0.5%',
   'A': '當沖 0.23% 來回（稅 0.15%＋手續費）＋ 滑價每邊 0.1% ＝ 0.43%；2017-04-28 前當沖稅 0.3%（再加 0.15%）；壓力：來回 1%',
 },
 'variants': {
   'BASE': '二板（主規格）',
   'VOL_SHRINK': 'BASE 且 Day0 量 ≤ Day−1 量',
   'VOL_EXPAND': 'BASE 且 Day0 量 > Day−1 量',
   'THREE_BOARD': 'Day0、Day−1、Day−2 收盤漲停，Day−12..Day−3 無收盤漲停（三板），Day1 規則相同',
   'CONTROL_FIRST': '對照組：Day0 首板（Day−10..Day−1 無收盤漲停），Day1 規則相同',
   'all': '每個變體都報 A／B／C；判定只看 BASE',
 },
 'event_stats': {
   'raw': '出場價 / 進場價 − 1（還原）；net = (1+raw)×(1−賣)/(1+買) − 1（A：net = raw − 來回成本）',
   'abnormal_pool': 'net − 同區間同池等權報酬（同池 = 前一日符合流動性條件的上市普通股；區間拆成開→收、收→開段，每段等權後連乘；判定用）',
   'abnormal_listed': 'net − 同區間上市等權（所有有價上市普通股，同樣分段；必報）',
   'vs_0050': 'net − 0050 含息（Day0 收盤→出場日收盤，近似）',
   't': '事件先按進場月份平均，再對月平均做 t（btlib.tstat）；t_date 僅參考',
 },
 'portfolio_stats': '日曆時間組合（每個出場版各一條）：每日持有在場事件、等權；進出成本以事件總成本計在進場日；無持股日現金 0 報酬。報年化、日頻最大回撤、對 0050 與同池等權（收對收）月超額 t、逐年、曝險天數',
 'windows': {'select': '2005-01-01~2019-04-30（以 Day0 歸屬）', 'select_sub_7pct': '2005-01~2015-05', 'select_sub_10pct': '2015-06~2019-04',
             'valid': '2019-05-01~2026-10-22', 'valid_2024_2025': '2024-01-01~2025-12-31',
             'src3': 'Day0～出場碰到 src 3（2024-02-15~2025-09-23）的事件另報；驗證期剔除 src 3 另報；另報 src 3 期間收盤漲停偵測率',
             'data2021': '處置名單 2021-09 起：另報 2021-01~2023-12 與 2024-01~2026-10',
             'A_select': 'A 版挑選期實際只有 2014-01-06~2019-04-30'},
 'report_first': '每年 Day0 事件數與實際進場數（各變體）',
 'verdict_rule': '判定只看 BASE 的 C 版（連板持有，接力戰法本意且全期可測）：挑選期與驗證期都要「事件淨異常（對同池等權）> 0 且 t_month ≥ 2」且「日曆組合淨年化 ≥ 0050 同期年化」→ 可用；只有一段過 → 待觀察；都沒過 → 不可用；某段事件 < 30 筆或事件月份 < 24 → 無法測試。'
                 'A、B 版與敏感度只作參考；若 BASE 的 A 或 B 兩段都過而 C 沒過，最多判待觀察（多重檢定）。',
 'known_limits': ['2019–2023 面板只有存活者', '2024-01-02~2024-02-14 無資料', 'src 3 收盤為 13:20 近似，收盤漲停判斷會漏抓（二板要連兩天抓到，漏抓加倍）',
                  '無最低價：跌停以收盤／開盤判斷', '2021-09 前無處置名單、全程無全額交割名單', '漲停價以 ref 近似', '以開盤價成交且條件用開盤價'],
}
if __name__ == '__main__':
    f = HERE / 'results.json'
    d = json.loads(f.read_text(encoding='utf-8')) if f.exists() else {}
    assert 'PREREG' not in d, 'PREREG 已存在，不覆寫'
    d = {'PREREG': PREREG, **d}
    f.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding='utf-8')
    print('PREREG written')
