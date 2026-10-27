#!/usr/bin/env python3
"""事前設定（看任何策略結果前寫入 results.json 的 PREREG；之後不改）。"""
import json, datetime
from pathlib import Path
HERE = Path(__file__).resolve().parent
PREREG = {
 'written_at': datetime.datetime.now().isoformat(timespec='seconds'),
 'rule_name': '跌停翌日開盤買反彈（日本ストップ安翌日寄り買い）',
 'sources': ['https://www.invest-concierge.com/qa/post-limit-price-moves',
             'https://gentosha-go.com/articles/-/38476',
             'https://finance.matsui.co.jp/news/619374/index'],
 'source_access_note': '本環境連不到三個來源網站（DNS 失敗），規則以主研究員轉述為準',
 'rule_verbatim': '股池同上（上市流動性池），排除處置、全額交割、注意股當日。Day0：收盤 ≤ 跌停價 + 半檔（收盤跌停），且 t−10～t−1 沒有收盤跌停（首次跌停）。'
                  'Day1：開盤 > Day1 跌停價（可成交），且開盤 ≥ Day0 收盤 × 0.97（「翌日に気配を切り上げて寄る」），開盤買進。'
                  '出場（事前三版）：A. Day1 收盤賣（當沖 0.23%）；B. Day3 開盤賣；C. Day5 開盤賣。開盤跌停就延後賣出。'
                  '敏感度：不設開盤門檻；Day0 開盤 > 前收 與 ≤ 前收 兩組；大盤 Day0 跌幅 ≤ −2% 與否分開報。',
 'type': '事件型：主報告＝事件報酬（扣成本、對同池等權與上市等權的異常報酬、t 按進場月份分群）；另報日曆時間組合（年化、日頻最大回撤、對 0050／同池等權月超額 t、逐年）',
 'data': {
   'panel': 'scratchpad/panel 上市日線（含已下市；2019–2023 只有存活者）。先剔除「有值檔數 < 300」的日子（2018-01-13）再做任何 rolling；flag 5 價格視為缺值（沿用 w3 lim.py）',
   'stocks': '4 碼數字代碼、排除 91xx（存託憑證）',
   'limit_price': '直接 import agents/w3_limitup_pullback/lim.py（不改原檔、不寫 bytecode）的參考價與升降單位表：參考價 = close_raw/(1+ret)；'
                  '跌停價 = 參考價×(1−幅度) 依升降單位（以跌停價水準查表）無條件進位；漲停價 = lim.py 的 LIMUP。幅度 2015-06-01 前 7%、之後 10%；2005-03-01 前舊制升降單位',
   'close_limit_down': '收盤跌停 = close_raw ≤ 跌停價 + 半檔 且 flag ∈ {0,1,2}（無最低價，無法確認全天鎖死；以收盤跌停近似）',
   'open_limit_down': '開盤跌停 = open ≤ 跌停價 + 半檔；開盤漲停 = open ≥ 漲停價 − 半檔（lim.py OPEN_LU）',
   'adjusted_prices': '跨日比較一律用還原價：還原收盤 adj、還原開盤 = open×adj/close_raw',
   'taiex': 'jonatw/finlab-txd taiex_twii.csv 收盤（價格指數），Day0 大盤報酬 = Day0 收盤 / 前一交易日收盤 − 1',
 },
 'universe': {
   'liquidity': 'Day0：60 日均成交值（上市後停牌日記 0、滿 60 筆）≥ NT$1,000 萬且 Day0 原始收盤 ≥ 10 元（收盤後已知）。注意：收盤跌停本身會使股價跌破 10 元的事件被排除',
   'disposal': '處置股：disposal_5y（disposal_reconcile 標準：start → 第一個 ≥ start 的交易日、end → 最後一個 ≤ end 的交易日）；Day0 或 Day1 在處置期間 → 排除。名單 2021-09 起才有（限制）',
   'attention': '注意股：attention_5y 公告日（盤後已知）＝ Day0 → 排除（「注意股當日」）。名單 2021-10-06 起才有，之前無法排除（限制）。另報不排除注意股的 2021 起結果供參考',
   'full_delivery': '全額交割股：沒有名單資料，無法排除（限制；這類多半股價 < 10 或流動性差，大多已被股池擋掉）',
 },
 'signal': {
   'day0': 'Day0 收盤跌停、在股池內、非處置、非注意股公告日；Day-10..Day-1 都沒有收盤跌停（首次跌停；跌停判斷不受股池限制）',
   'day1': 'Day1 = Day0 的下一個面板交易日；需有開盤價、flag ∈ {0,1,2}、非處置；開盤跌停 → 不買（買得到但規則要求「開盤 > 跌停價」）；開盤漲停 → 視為買不到（鎖漲停買不到）',
   'gate': 'g = Day1 還原開盤 / Day0 還原收盤（= Day1 開盤 / Day1 參考價，避開除權息）；主規格 g ≥ 0.97',
   'timing_note': '以開盤價成交、條件也用開盤價：實務上需在開盤撮合（或試撮氣配）後立即下單，差異由滑價 0.1% 吸收（限制）',
 },
 'exits': {
   'A': 'Day1 收盤賣（現股當沖）。只從 2014-01-06（現股當沖開放）起的事件計算。Day1 收盤跌停 → 視為賣不掉，改 Day2 開盤以一般賣出成本賣',
   'B': 'Day3 開盤賣（Day1 進場後第 2 個交易日的開盤）',
   'C': 'Day5 開盤賣',
   'halt_or_limit_down': '出場時點無價或價格處在跌停 → 順延到下一個價格時點（同日收盤、下一日開盤…），最多 10 個交易日；已下市 → 以最後收盤賣',
   'overlap': '同檔事件持有最長 5 日、首次跌停定義要求前 10 日無跌停，同檔事件不會重疊',
 },
 'costs': {
   'B_C': 'exlib.cost_rates(0.001)：買 = 0.1425%×0.6 + 滑價 0.1%；賣 = 0.1425%×0.6 + 證交稅 0.3% + 滑價 0.1%（來回約 0.67%）；壓力：買 0.5% + 賣 0.5%',
   'A': '當沖：BRIEF 的 0.23% 來回（稅 0.15%＋手續費）＋ exlib 滑價每邊 0.1% ＝ 0.43%；2017-04-28 前當沖稅為 0.3%，再加 0.15%；壓力：來回 1%',
 },
 'variants': {
   'PRIMARY': '開盤門檻 g ≥ 0.97',
   'NO_GATE': '不設開盤門檻：所有可成交（開盤 > 跌停價、非開盤漲停）的 Day1 開盤都買',
   'splits': '對 PRIMARY 與 NO_GATE 各自分組：Day0 開盤 > Day0 參考價（開高殺跌停）vs ≤；大盤 Day0 報酬 ≤ −2%（系統性）vs > −2%',
   'all': '每個變體都報 A／B／C；判定只看 PRIMARY 的 B 版',
 },
 'event_stats': {
   'raw': '出場價 / 進場價 − 1（還原）；net = (1+raw)×(1−賣)/(1+買) − 1（A：net = raw − 來回成本）',
   'abnormal_pool': 'net − 同區間同池等權報酬（同池 = 前一日符合流動性條件的上市普通股；區間拆成「開→收」「收→開」段，各段等權平均後連乘；判定用）',
   'abnormal_listed': 'net − 同區間上市等權（所有有價的上市普通股，同樣分段）',
   'vs_0050': 'net − 0050 含息（Day0 收盤→出場日收盤，開盤出場用前一日收盤；0050 沒有開盤價，為近似）',
   't': '事件先按進場月份平均，再對月平均做 t（btlib.tstat）；另報 t_date 僅供參考',
 },
 'portfolio_stats': '日曆時間組合（每個出場版各一條）：每日持有當日在場事件、等權；進場日報酬 = 收/開，開盤出場日 = 出場開/前收；成本以事件總成本分到進場日（÷ 當日持股數）；無持股日現金 0 報酬。'
                    '報年化、日頻最大回撤、對 0050 與同池等權（收對收）月超額 t、逐年、曝險天數比例',
 'windows': {'select': '2005-01-01~2019-04-30（以 Day0 歸屬）', 'select_sub_7pct': '2005-01~2015-05', 'select_sub_10pct': '2015-06~2019-04',
             'valid': '2019-05-01~2026-10-22', 'valid_2024_2025': '2024-01-01~2025-12-31',
             'src3_only': 'Day0～出場碰到 src 3（2024-02-15~2025-09-23）的事件另報（收盤是 13:20 近似，跌停判斷會漏抓或誤抓）', 'valid_ex_src3': '驗證期剔除碰到 src 3 的事件',
             'data2021': '處置／注意名單 2021 起才有：另報 2021-01~2023-12 與 2024-01~2026-10',
             'A_select': 'A 版挑選期實際只有 2014-01-06~2019-04-30'},
 'verdict_rule': '判定只看 PRIMARY 的 B 版（Day3 開盤賣；三版中間的持有期、全期可測）：挑選期與驗證期都要「事件淨異常（對同池等權）> 0 且 t_month ≥ 2」且「日曆組合淨年化 ≥ 0050 同期年化」→ 可用；'
                 '只有一段過 → 待觀察；都沒過 → 不可用；某段事件 < 30 筆或事件月份 < 24 → 無法測試。A、C 版與敏感度只作參考；若 A 或 C 兩段都過而 B 沒過，最多判待觀察（多重檢定）。',
 'known_limits': ['2019–2023 面板只有存活者（驗證期偏樂觀；對「跌停後反彈」尤其有利，因為跌停後下市的股票不在樣本內）', '2024-01-02~2024-02-14 無資料',
                  'src 3 收盤為 13:20 近似，收盤跌停判斷不準', '無最低價：無法判斷 Day0 是否全天鎖死', '2021-09 前無處置名單、2021-10 前無注意股名單、全程無全額交割名單',
                  '跌停價以 ref 近似（除權息日參考價四捨五入差異）', '以開盤價成交且條件用開盤價'],
}
if __name__ == '__main__':
    f = HERE / 'results.json'
    d = json.loads(f.read_text(encoding='utf-8')) if f.exists() else {}
    assert 'PREREG' not in d, 'PREREG 已存在，不覆寫'
    d = {'PREREG': PREREG, **d}
    f.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding='utf-8')
    print('PREREG written')
