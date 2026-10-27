#!/usr/bin/env python3
"""事前設定（看任何策略結果前寫入 results.json 的 PREREG；之後不改）。"""
import json, datetime
from pathlib import Path
HERE = Path(__file__).resolve().parent
PREREG = {
 'written_at': datetime.datetime.now().isoformat(timespec='seconds'),
 'rule_name': '炸板次日反包（A 股「曾漲停未封」反包戰法）',
 'sources': ['https://quant.10jqka.com.cn/view/article/8CA6FSR30N125810KFE7V8HBQ8',
             'https://clawhub.ai/gracexiaoo/skills/akshare-backtest',
             'https://hub.docker.com/r/birdxs/tickflow-stock-panel'],
 'rule_verbatim': '股池同上（上市流動性池），排除處置與全額交割。Day0（炸板）：high ≥ 漲停價（盤中觸及漲停），收盤 < 漲停價 − 1 檔（沒封住），而且收盤 ≥ 前收 × 1.03（仍收紅 3% 以上）；'
                  't−10～t−1 沒有觸及或收盤漲停（首次）。主版（確認型反包）：Day1 收盤 > Day0 high（反包）就在 Day2 開盤買（開盤漲停不買）。'
                  '出場：收盤 < Day0 收盤就隔日開盤停損；否則持滿 5 個交易日開盤賣。激進版（教學另一種做法）：所有炸板股 Day1 開盤買、Day1 收盤賣（當沖）或 Day2 開盤賣。'
                  '敏感度：持有 3／10 日；反包門檻改成 Day1 收盤 > Day0 收盤 × 1.05。',
 'type': '事件型：主報告＝事件報酬（扣成本、對同池等權與上市等權的異常報酬、t 按進場月份分群）；另報日曆時間組合（年化、日頻最大回撤、對 0050／同池等權月超額 t、逐年）',
 'data': {
   'panel': 'scratchpad/panel 上市日線（含已下市；2019–2023 只有存活者）。先剔除「有值檔數 < 300」的日子（2018-01-13）再做任何 rolling／t−10 視窗；flag 5 價格視為缺值（沿用 w3_limitup_pullback/lim.py）',
   'stocks': '4 碼數字代碼、排除 91xx（存託憑證）',
   'limit_price': '直接 import agents/w3_limitup_pullback/lim.py（不改原檔、不寫 bytecode）：參考價 REF = close_raw/(1+ret)，漲停價 LIMUP = REF×(1+漲幅) 依升降單位無條件捨去（漲幅 2015-06-01 前 7%、之後 10%；2005-03-01 前舊制升降單位），tk = 漲停價所在的升降單位',
   'touch_limit': '觸及漲停 = 原始 high ≥ LIMUP − 半檔（flag ∈ {0,1,2}）',
   'close_limit': '收盤漲停 = lim.CLOSE_LU（close_raw ≥ LIMUP − 半檔 且 close_raw ≥ high×0.999 且 flag ∈ {0,1,2}）',
   'not_sealed': '沒封住 = close_raw < LIMUP − 1 檔（逐字：嚴格小於；實作 close_raw < LIMUP − tk − tk/2 的浮點容忍，即收盤至少比漲停價低 2 檔）',
   'red3': '收盤 ≥ 前收 × 1.03：前收用參考價 REF（避開除權息），close_raw ≥ REF × 1.03 − 1e-9',
   'limit_down': '跌停價 = REF×(1−漲幅) 依升降單位無條件進位；價格 ≤ 跌停價 + 半檔 視為跌停（同 w12_first_board_gap_down）',
   'adjusted_prices': '跨日比較一律用還原價：還原收盤 adj、還原開盤 = open×adj/close_raw、還原 high = high×adj/close_raw（Day1 收盤 > Day0 high、停損 收盤 < Day0 收盤 都用還原價比）',
   'src3': 'src 3（2024-02-15～2025-09-23）的 high 與收盤為 5 分 K 近似，炸板／封板會互相誤判 → 這段期間的事件（Day0～出場任一天落在此區間）一律從判定剔除，另外分開報',
 },
 'universe': {
   'liquidity': 'Day0：60 日均成交值（上市後停牌日記 0，滿 60 筆）≥ NT$1,000 萬且 Day0 原始收盤 ≥ 10 元，Day0 flag ∈ {0,1,2}',
   'disposal': 'disposal_5y（2021-09 起）每筆 start→end 對到面板交易日；Day0 或進場日在處置期間 → 排除（主版另加 Day1）。2021-09 前無名單（限制）',
   'full_delivery': '全額交割股：無名單資料，無法排除（限制；多半已被股價 ≥ 10、流動性條件擋掉）',
 },
 'signal': {
   'day0_broken_board': 'touch_limit & not_sealed & red3 & 流動性池 & 非處置',
   'first_time': 'Day−10..Day−1（面板交易日）都沒有觸及漲停、也沒有收盤漲停',
   'MAIN_fanbao': 'Day1 = Day0 下一個面板交易日；Day1 還原收盤 > Day0 還原 high（Day1 收盤後才知道）→ Day2 開盤買；Day2 需有開盤價、flag ∈ {0,1,2}、Day1/Day2 不在處置；Day2 開盤 ≥ 漲停價 − 半檔（開盤漲停）→ 不買',
   'AGGR_entry': '所有 Day0 炸板股 Day1 開盤買；Day1 需有開盤價、flag ∈ {0,1,2}、不在處置；Day1 開盤漲停 → 買不到（不買）。教學沒給開盤條件，不加任何開盤濾網',
   'timing_note': '激進版以 Day1 開盤價成交，訊號為 Day0 盤後已知，可掛開盤單（合規）',
 },
 'exits': {
   'MAIN': '進場日 te = Day2；檢查 te, te+1, …, te+H−1 的收盤：第一個還原收盤 < Day0 還原收盤的日子 s → s+1 開盤賣（停損）；都沒觸發 → te+H 開盤賣（持滿 H=5 個交易日）。停牌無收盤的日子不檢查',
   'AGGR_A': 'Day1 收盤賣（現股當沖）；只計 2014-01-06（現股當沖開放）起的事件。Day1 收盤跌停 → 賣不掉，改 Day2 開盤以一般賣出成本賣',
   'AGGR_B': 'Day2 開盤賣',
   'halt_or_limit_down': '出場時點無價或處在跌停 → 順延到下一個價格時點（下一個收盤、再下一個開盤…），最多再 10 個交易日；已下市 → 最後收盤賣',
   'no_intraday_stop': '面板沒有最低價 → 無法做盤中停損，只能收盤檢查、隔日開盤成交',
   'dedupe': '同一檔前一事件尚未出場時新事件跳過（首次條件使重疊幾乎不可能）',
 },
 'costs': {
   'normal': 'exlib.cost_rates(0.001)：買 = 0.1425%×0.6 + 滑價 0.1%；賣 = 0.1425%×0.6 + 證交稅 0.3% + 滑價 0.1%（來回約 0.67%）；壓力：exlib cost_rates("stress") 買 0.5% + 賣 0.5%',
   'daytrade_A': '當沖：BRIEF 的 0.23% 來回（稅 0.15%＋手續費）＋滑價每邊 0.1% ＝ 0.43%；2017-04-28 前當沖稅為 0.3%，再加 0.15%；壓力：來回 1%',
 },
 'variants': {
   'MAIN_H5': '主版：反包門檻 Day1 收 > Day0 high、持有 5 日、收盤停損（判定用）',
   'MAIN_H3': '敏感度：持有 3 日',
   'MAIN_H10': '敏感度：持有 10 日',
   'THR105_H5': '敏感度：反包門檻改成 Day1 還原收盤 > Day0 還原收盤 × 1.05，其他同主版',
   'AGGR_A': '激進版：Day1 開盤買、Day1 收盤賣（當沖）',
   'AGGR_B': '激進版：Day1 開盤買、Day2 開盤賣',
 },
 'event_stats': {
   'raw': '出場價 / 進場價 − 1（還原）；net = (1+raw)×(1−賣)/(1+買) − 1（當沖：net = raw − 來回成本）',
   'abnormal_pool': 'net − 同區間同池等權報酬（同池 = 前一日符合流動性條件的上市普通股；區間拆成「開→收」「收→開」段，每段等權平均後連乘；判定用）',
   'abnormal_listed': 'net − 同區間上市等權（所有有價的上市普通股，同樣分段）',
   'vs_0050': 'net − 0050 含息（進場前一日收盤→出場日（開盤出場取前一日）收盤；0050 沒有開盤價，為近似）',
   't': '事件先按進場月份平均，再對月平均做 t（btlib.tstat）；另報 t_date 僅供參考',
 },
 'portfolio_stats': '日曆時間組合（每個變體一條）：每日持有當日在場事件、等權；進場日報酬 = 收/開，開盤出場日 = 出場開/前收；成本於進場日按 1/n 扣；無持股日現金 0 報酬。報年化、日頻最大回撤、對 0050 與同池等權（收對收）月超額 t、逐年、曝險天數比例',
 'windows': {'select': '2005-01-01~2019-04-30（以 Day0 歸屬）', 'select_sub_7pct': '2005-01~2015-05（7% 漲跌幅）', 'select_sub_10pct': '2015-06~2019-04',
             'valid_judge': '2019-05-01~2026-10-22 剔除 src 3：事件 = Day0～出場都不碰 2024-02-15~2025-09-23；組合 = 只用 2019-05-01~2024-02-14 與 2025-09-24~2026-10-22 兩段日報酬（0050、等權同日）',
             'valid_all': '2019-05-01~2026-10-22 含 src 3（只供參考）', 'src3_only': 'Day0～出場碰到 src 3 的事件另報',
             'valid_2024_2025': '2024-01-01~2025-12-31（含 src 3，絕大部分是 src 3，僅供參考；另報其中剔除 src 3 的部分＝2025-09-24~2025-12-31）',
             'data2021': '處置名單 2021 起才有：另報 2021-01~2023-12 與 2024-01~2026-10（後者分含／剔除 src 3）',
             'A_select': 'AGGR_A 挑選期實際只有 2014-01-06~2019-04-30'},
 'verdict_rule': '判定只看 MAIN_H5：挑選期（2005-01~2019-04）與驗證期（valid_judge，剔除 src 3）各自要「事件淨異常（對同池等權）> 0 且 t_month ≥ 2」且「日曆組合淨年化 ≥ 0050 同期年化」→ 兩段都過 可用；只有一段過 待觀察；都沒過 不可用；某段事件 < 30 筆或事件月份 < 24 → 無法測試。'
                 '敏感度與激進版只作參考；若某個非主版兩段都過而主版沒過，最多判待觀察（多重檢定）。另報上市等權異常與壓力成本。',
 'known_limits': ['2019–2023 面板只有存活者（驗證期偏樂觀）', '2024-01-02~2024-02-14 無資料', 'src 3 期間 high／收盤為 5 分 K 近似 → 剔除',
                  '無最低價：無盤中停損，跌停一律以收盤／開盤價判斷', '2021-09 前無處置名單、全程無全額交割名單',
                  '漲停價以 REF 近似（除權息日參考價四捨五入差異）', '盤中觸及漲停用日 high，無法區分「瞬間觸及」與「封板後打開」'],
}
if __name__ == '__main__':
    f = HERE / 'results.json'
    d = json.loads(f.read_text(encoding='utf-8')) if f.exists() else {}
    assert 'PREREG' not in d, 'PREREG 已存在，不覆寫'
    d = {'PREREG': PREREG, **d}
    f.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding='utf-8')
    print('PREREG written')
