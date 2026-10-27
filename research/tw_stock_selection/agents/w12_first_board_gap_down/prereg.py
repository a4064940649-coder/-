#!/usr/bin/env python3
"""事前設定（看任何策略結果前寫入 results.json 的 PREREG；之後不改）。"""
import json, datetime
from pathlib import Path
HERE = Path(__file__).resolve().parent
PREREG = {
 'written_at': datetime.datetime.now().isoformat(timespec='seconds'),
 'rule_name': '首板低開（聚寬／BigQuant 打板社群：昨日首板、今日低開買）',
 'sources': ['https://developer.volcengine.com/articles/7423250312833531913',
             'https://quant.10jqka.com.cn/view/article/F3ETVJ6UEZ1554442IQSKQRMOS',
             'https://bigquant.com/wiki/doc/OCkaNT91EC'],
 'rule_verbatim': '股池：上市流動性池（60 日均成交值 ≥ NT$1,000 萬、股價 ≥ 10 元），排除當日處置股與全額交割股。Day0 收盤漲停、t−10～t−1 無收盤漲停（首板），'
                  '相對位置 rp=(Day0 收盤 − 60 日最低收盤)/(60 日最高 high − 60 日最低收盤) ≤ 0.5。Day1 開盤／Day0 收盤 ∈ [0.96, 0.99] 開盤買，開盤跌停不買。'
                  '出場 A Day1 收盤（現股當沖）、B Day2 開盤、C Day2 收盤。',
 'type': '事件型：主報告＝事件報酬（扣成本、對同池等權與上市等權的異常報酬、t 按進場月份分群）；另報日曆時間組合（年化、日頻最大回撤、對 0050／同池等權月超額 t、逐年）',
 'data': {
   'panel': 'scratchpad/panel 上市日線（含已下市；2019–2023 只有存活者）。先剔除「有值檔數 < 300」的日子（2018-01-13）再做任何 rolling；flag 5 價格視為缺值',
   'stocks': '4 碼數字代碼、排除 91xx（存託憑證）',
   'limit_price': '直接 import agents/w3_limitup_pullback/lim.py（不改原檔、不寫 bytecode）：參考價 = close_raw/(1+ret)，漲停價 = 參考價×(1+漲幅) 依升降單位無條件捨去，漲幅 2015-06-01 前 7%、之後 10%，2005-03-01 前用舊制升降單位；'
                  '收盤漲停 = close_raw ≥ 漲停價 − 半檔 且 close_raw ≥ high×0.999 且 flag ∈ {0,1,2}',
   'limit_down': '跌停價 = 參考價×(1−漲幅) 依升降單位無條件進位（同 lim.py 的升降單位表）；價格 ≤ 跌停價 + 半檔 視為跌停',
   'adjusted_prices': '跨日比較一律用還原價：還原收盤 adj、還原開盤 = open×adj/close_raw、還原最高 = high×adj/close_raw',
 },
 'universe': {
   'liquidity': 'Day0：60 日均成交值（上市後停牌日記 0，滿 60 筆）≥ NT$1,000 萬且 Day0 原始收盤 ≥ 10 元（Day0 收盤後即已知）',
   'disposal': '處置股：disposal_reconcile 正式版用的標準事件表（rlib.records：disposal_5y 每筆記錄，start → 第一個 ≥ start 的面板交易日、end → 最後一個 ≤ end 的交易日）；Day0 或 Day1 落在處置期間 → 排除。名單 2021-09 起才有，之前無法排除（限制）',
   'full_delivery': '全額交割股：沒有名單資料，無法排除（限制；這類股票多半股價 < 10 或流動性差，大多已被流動性池擋掉）',
 },
 'signal': {
   'day0': 'Day0 收盤漲停（上面定義），且 Day-10..Day-1 都沒有收盤漲停（首板）',
   'rp': 'rp = (adj_Day0 − min(adj, Day-59..Day0)) / (max(還原 high, Day-59..Day0) − min(adj, Day-59..Day0))，窗口含 Day0、至少 40 筆有值；主規格 rp ≤ 0.5',
   'gap': 'g = Day1 還原開盤 / Day0 還原收盤（= Day1 開盤 / Day1 參考價，避開除權息）；主規格 0.96 ≤ g ≤ 0.99',
   'day1_ok': 'Day1 = Day0 的下一個面板交易日；需有開盤價、flag ∈ {0,1,2}；開盤跌停（open ≤ 跌停價 + 半檔）→ 不買；Day1 在處置期間 → 不買',
   'timing_note': '以開盤價成交、條件也用開盤價：實務上需在開盤撮合後立即下單，成交價與開盤價的差異由滑價 0.1% 吸收（限制）',
 },
 'exits': {
   'A': 'Day1 收盤賣（現股當沖）。只從 2014-01-06（現股當沖開放）起的事件計算。Day1 收盤跌停 → 視為賣不掉，改 Day2 開盤以一般賣出成本賣',
   'B': 'Day2 開盤賣',
   'C': 'Day2 收盤賣',
   'halt_or_limit_down': '出場時點（開盤或收盤）無價或價格處在跌停 → 順延到下一個價格時點（下一個開盤、再下一個收盤…），最多 10 個交易日；已下市 → 以最後收盤賣',
   'dedupe': '持有最多 2 日，首板定義使同檔事件不可能重疊（新 Day0 前 10 日不能有漲停），不需去重',
 },
 'costs': {
   'B_C': 'exlib.cost_rates(0.001)：買 = 0.1425%×0.6 + 滑價 0.1%；賣 = 0.1425%×0.6 + 證交稅 0.3% + 滑價 0.1%（來回約 0.67%）；壓力：買 0.5% + 賣 0.5%',
   'A': '當沖：BRIEF 的 0.23% 來回（稅 0.15%＋手續費）＋ exlib 滑價每邊 0.1% ＝ 0.43%；2017-04-28 前當沖稅為 0.3%，再加 0.15%（＝0.58%）；壓力：來回 1%；另報「exlib 費率口徑」2×0.0855%＋當沖稅＋2×0.1%（約 0.52%）',
 },
 'variants': {
   'PRIMARY': 'rp ≤ 0.5、g ∈ [0.96, 0.99]',
   'GAP_SHALLOW': 'g ∈ [0.98, 1.00)，其他同主規格',
   'GAP_DEEP': 'g ∈ [0.94, 0.97]，其他同主規格',
   'NO_RP': '拿掉 rp 濾網，g ∈ [0.96, 0.99]',
   'all': '每個變體都報 A／B／C 三種出場；判定只看 PRIMARY',
 },
 'event_stats': {
   'raw': '出場價 / 進場價 − 1（還原）；net = (1+raw)×(1−賣)/(1+買) − 1（A：net = raw − 來回成本）',
   'abnormal_pool': 'net − 同區間同池等權報酬（同池 = 前一日符合流動性條件的上市普通股；區間拆成「開→收」「收→開」段，每段等權平均後連乘；判定用）',
   'abnormal_listed': 'net − 同區間上市等權（所有有價的上市普通股，同樣分段）',
   'vs_0050': 'net − 0050 含息（Day0 收盤→出場日收盤；0050 沒有開盤價，為近似）',
   't': '事件先按進場月份平均，再對月平均做 t（btlib.tstat）；另報 t_date 僅供參考',
 },
 'portfolio_stats': '日曆時間組合（每個出場版各一條）：每日持有當日在場事件、等權；進場日報酬 = 收/開，B 版出場日 = 出場開/前收，C 版 Day2 = 收/前收；進出各按 1/n 權重扣成本；無持股日現金 0 報酬。報年化、日頻最大回撤、對 0050 與同池等權（收對收）月超額 t、逐年、曝險天數比例',
 'windows': {'select': '2005-01-01~2019-04-30（以 Day0 歸屬）', 'select_sub_7pct': '2005-01~2015-05（7% 漲跌幅）', 'select_sub_10pct': '2015-06~2019-04',
             'valid': '2019-05-01~2026-10-22', 'valid_2024_2025': '2024-01-01~2025-12-31',
             'src3_only': 'Day0～出場碰到 src 3（2024-02-15~2025-09-23）的事件另報（收盤是 13:20 近似，漲停會漏抓）', 'valid_ex_src3': '驗證期剔除碰到 src 3 的事件',
             'data2021': '處置名單 2021 起才有：另報 2021-01~2023-12 與 2024-01~2026-10',
             'A_select': 'A 版挑選期實際只有 2014-01-06~2019-04-30'},
 'verdict_rule': '判定只看 PRIMARY 的 C 版（Day2 收盤賣，最接近社群「次日賣出」預設，且全期可測）：挑選期與驗證期都要「事件淨異常（對同池等權）> 0 且 t_month ≥ 2」且「日曆組合淨年化 ≥ 0050 同期年化」→ 可用；只有一段過 → 待觀察；都沒過 → 不可用；某段事件 < 30 筆或事件月份 < 24 → 無法測試。'
                 'A、B 版與敏感度只作參考；若 A 或 B 兩段都過而 C 沒過，最多判待觀察（多重檢定）。',
 'known_limits': ['2019–2023 面板只有存活者（驗證期偏樂觀）', '2024-01-02~2024-02-14 無資料', 'src 3 收盤為 13:20 近似，收盤漲停判斷會漏抓',
                  '無最低價：無法判斷盤中跌停鎖死；跌停一律以收盤／開盤價判斷', '2021-09 前無處置名單、全程無全額交割名單',
                  '漲停價以 ref 近似（除權息日參考價四捨五入差異）', '以開盤價成交且條件用開盤價（需開盤後立即下單）'],
}
if __name__ == '__main__':
    f = HERE / 'results.json'
    d = json.loads(f.read_text(encoding='utf-8')) if f.exists() else {}
    assert 'PREREG' not in d, 'PREREG 已存在，不覆寫'
    d = {'PREREG': PREREG, **d}
    f.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding='utf-8')
    print('PREREG written')
