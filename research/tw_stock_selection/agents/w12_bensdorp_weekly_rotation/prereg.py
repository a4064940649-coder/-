#!/usr/bin/env python3
"""Bensdorp 週輪動（Weekly Rotation）事前設定：看任何結果前寫入 results.json 的 PREREG；之後不改。"""
import json, datetime
from pathlib import Path
HERE = Path(__file__).resolve().parent
PREREG = {
 'written_at': datetime.datetime.now().isoformat(timespec='seconds'),
 'rule_name': 'Laurens Bensdorp《The 30-Minute Stock Trader》Weekly Rotation：大盤 > 200 日線（2% 緩衝）、RSI(3) < 50、ROC(200) 前 10 名、10 檔等權、週換股',
 'sources': ['https://www.turingtrader.com/portfolios/bensdorp-weekly-rotation/',
             'https://quantopian-archive.netlify.app/forum/threads/weekly-rotation-s-and-p-500.html',
             'https://www.quantopian.com/posts/weekly-rotation-s-and-p-500',
             'https://www.turingtrader.com/2020/01/30-minute-stock-trader/'],
 'source_rule_verbatim': '股池 S&P 500 成分股、股價 > 1 美元、日均量 > 100 萬股；每週最後一個交易日收盤算訊號、下週第一個交易日開盤成交；'
                         'S&P 500 收盤 > 200 日 SMA × 0.98 才允許新買進；依 ROC(200) 由高到低排序；RSI(3) < 50 才能新買進（不追高）；'
                         '10 檔、每檔 10%；持股跌出 ROC 前 10 名就在下週開盤賣出，換入排名最高且 RSI(3) < 50 的新股。',
 'note_on_access': '本環境無法連到上列網址（DNS 失敗），兩種解讀依任務說明（Quantopian 討論串 = A、TuringTrader = B）事前寫定，未看原始碼。',
 'type': '選股型（週換股、10 檔）→ 組合報告為主（年化、日頻最大回撤、對 0050 t、對同池等權 t、逐年）',
 'data': {
   'panel': 'scratchpad/panel 上市日線（含已下市；2019–2023 只有存活者）。有值檔數 < 300 的日子（2018-01-13）整列剔除後才建週線、算任何 rolling，模擬也只在其餘交易日進行。',
   'price': '個股一律用含息還原收盤 adj（只用收盤價）；成交用還原開盤（exlib.AO）。',
   'codes': '排除 91xx 存託憑證。',
   'index': 'taiex_twii.csv 收盤（價格指數）。剔除週日列（2026-09-20 疑似錯誤列；週六補班日保留）。在指數自己的日曆上算 200 日 SMA（min_periods 200），再取面板週末日當天的值。',
 },
 'weekly_calendar': '週 = W-SUN 日曆週；訊號日 d = 該週最後一個（保留的）交易日；成交日 = d 之後下一個保留交易日（開盤）。面板 2024-01-02~02-14 缺資料，該段沒有訊號，持股跨過缺口。',
 'universe': '訊號日 d：mlib.V60（剔除稀疏日後 60 日均成交值）>= NT$1,000 萬、原始收盤 >= 10 元、d 當天有 adj、非 91xx；在此之中取 V60 前 300 名（對應 S&P 500）。不排除處置股（名單只有 2021 起）。',
 'indicators': {
   'ROC200': 'adj(d) / adj(d 之前第 200 個保留交易日) − 1；過去端允許 ffill 最多 5 日（停牌），d 端必須有值。無值者不參與排名。',
   'RSI3': 'Wilder RSI(3)：日變動 Δ = adj 收盤差（保留交易日；停牌 NaN 不計），平均漲跌幅用 Wilder 平滑 α = 1/3（ewm adjust=False, ignore_na=True），至少 20 個有效觀察值；RSI = 100 − 100/(1+AG/AL)。d 當天須有 adj。',
   'REGIME': 'TAIEX(d) > SMA200_TAIEX(d) × 0.98。',
 },
 'version_A': {
   'name': 'A（Quantopian 解讀）',
   'rank': '全池（前 300 且 ROC 有值）依 ROC200 由高到低，取前 10 名 = TOP10。',
   'sell': '持股不在 TOP10（含已不在股池、無 ROC）→ 下週開盤賣出。大盤濾網失效時仍照此規則賣，不強制出場。',
   'buy': '濾網成立時：TOP10 中未持有且 RSI3 < 50 者依 ROC 排序填空位（空位 = 10 − 賣後持股數）；TOP10 中 RSI3 >= 50 者不買，空位留現金。濾網失效：不新買。',
 },
 'version_B': {
   'name': 'B（Turing Trader 解讀）',
   'rank': '先排除 RSI3 >= 50 的股票，再依 ROC200 取前 10 名 = TOP10B。已持有股票不受 RSI 排除（RSI 只限制新買進），即 TOP10B = 依 ROC 排序的 {池內且 RSI3 < 50} ∪ {池內已持有} 前 10 名。',
   'sell': '持股不在 TOP10B → 下週開盤賣出。大盤濾網失效 → 全部持股下週開盤出場、不新買。',
   'buy': '濾網成立時：TOP10B 中未持有者依序填入空位。',
   'diag_B_literal': '診斷（不判定）：已持有股票也套 RSI 排除（持股 RSI3 >= 50 就跌出 TOP10B → 賣出）。',
 },
 'execution': {
   'order': '成交日開盤：先賣後買。',
   'sizing': '每檔新買 = 成交日開盤 NAV × 10%（不足用剩餘現金；現金 < 0.25 份不買）；持股不再平衡（原書規則：10% 部位、不調整）；現金零利率。',
   'limit_up': '成交日開盤漲停（開盤/前收−1 >= 漲跌幅−0.6%，2015-06-01 前 7%、之後 10%，mlib.GAP_UP）或無開盤價 → 不買，該空位由同一候選清單的下一名遞補（A：TOP10 內 RSI<50 的下一名；B：TOP10B 之後依序第 11、12…名的 RSI<50 未持有股），清單用完則留現金。',
   'sell_fill': '成交日無開盤價（停牌）→ 延到下一個有開盤價的日子；已下市以最後還原收盤賣出；跌停賣不掉未模擬（偏樂觀）。',
   'holding_period': '無固定持有期；只靠週排名出場。',
   'cost': 'exlib.cost_rates：判定用滑價 0.1%（買 0.0855%+0.1%、賣 0.0855%+0.3% 稅+0.1%，來回約 0.67%）；另報滑價 0（來回約 0.47%，即「0.5% 來回」）、壓力（買賣各 0.5%，來回 1%）、毛報酬。',
   'sim_start': '第一個 ROC200 可算的週（約 2004-11）起模擬；統計自各期間起點。',
 },
 'benchmarks': {
   '0050': '0050 含息（ffill 到面板日曆）。',
   'EW_pool': '同池等權（判定用）：每個週訊號日取同一前 300 池（ROC 有值者），下週開盤換股等權，exlib.sim_portfolio，不扣成本（毛）。另報月頻（每月第一個 >= 11 日交易日）同池等權供參考。',
   'MAIN_52w': '研究中心 52 週高主策略：revenue_longhistory/lhlib.py Setup(yoy_long()).selection(n)（52 週高百分位 0.5 + 近 3 月營收年增 0.5、前 300 流動性、月換股、隔日開盤），n30 與 n10，滑價 0.1%；另報 momentum_family 長歷史 P1（只用 52 週高）n10/n30 滑價 0.1%。報年化與月配對差 t（本策略 − 主策略）。',
 },
 'stats': '日頻 NAV 算年化（實際天數）、日頻最大回撤、波動、逐年；t = 日曆月（保留交易日的月底 NAV）超額報酬 btlib.tstat，對 0050、對同池等權分開；另報周轉、平均持股數、平均投入比、濾網成立比例、每年換股次數。',
 'windows': {'select': '2005-01-01~2019-04-30', 'valid': '2019-05-01~2026-10-22', 'y2024_2025': '2024-01-01~2025-12-31',
             'extra': ['r21_select 2021-01~2023-12', 'r21_valid 2024-01~2026-10', 'valid_survivor 2019-05~2023-12', 'valid_officialPx_only（剔除 src3 段）', 'full 2005-01~2026-10']},
 'verdict_rule': '每個版本（A、B）以滑價 0.1% 判定：某段通過 = 淨年化 >= 0050 同期年化 且 對同池等權（毛）月超額 t >= 2。兩段都過 → 可用；一段 → 待觀察；否則不可用。'
                 '整條規則判定取 A、B 中較好者，但因兩個主版本屬多重檢定，若只有一個版本通過且其 t < 2.24（Bonferroni 2 檢定），降一級。',
 'prior_expectation': '動能家族 12-1 挑選期 t 約 −1～0.5；本題週頻、周轉高，預期不高。',
 'known_limits': ['2019–2023 面板只有存活者（對動能偏樂觀）', 'src 3 期間（2024-02-15~2025-09-23）收盤為 13:20 近似價，影響 RSI3 與週訊號',
                  '2024-01-02~02-14 無資料', '只有上市股', '跌停賣不掉未模擬', '原書股池為 S&P 500 成分股，此處用流動性前 300 代替'],
}
if __name__ == '__main__':
    f = HERE / 'results.json'
    d = json.loads(f.read_text(encoding='utf-8')) if f.exists() else {}
    assert 'PREREG' not in d, 'PREREG 已存在，不覆寫'
    d = {'PREREG': PREREG, **d}
    f.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding='utf-8')
    print('PREREG written')
