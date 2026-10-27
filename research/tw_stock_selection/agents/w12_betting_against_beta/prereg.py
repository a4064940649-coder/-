#!/usr/bin/env python3
"""事前設定（看任何結果前寫入 results.json 的 PREREG；之後不改）。"""
import json, datetime
from pathlib import Path
HERE = Path(__file__).resolve().parent
PREREG = {
 'written_at': datetime.datetime.now().isoformat(timespec='seconds'),
 'rule_name': '低 Beta／押注反 Beta（Frazzini–Pedersen 2014 BAB；只做多版）月換股',
 'sources': ['https://quantpedia.com/strategies/betting-against-beta-factor-in-stocks',
             'https://alphaarchitect.com/betting-against-beta-new-insights/',
             'https://www.johnhcochrane.com/s/BettingAgainstBeta.pdf',
             'https://robeco.com/en-za/glossary/quantitative-investing/low-volatility-anomaly',
             'https://irtest.lib.nccu.edu.tw/handle/140.119/158287',
             'https://financialforum.be/en/bfw-digitaal/the-low-risk-anomaly-in-large-emerging-markets'],
 'source_rule_verbatim': '股池同上（流動性 ≥ NT$1,000 萬、價格 ≥ 10 元）。市場＝加權指數 taiex_twii（另做 0050 含息版）。'
     'Beta 依原論文：相關係數 ρ 用過去 5 年「3 日重疊對數報酬」（最少 750 個交易日）；波動 σ 用過去 1 年日對數報酬（最少 120 日）；'
     'β_TS = ρ × σ_i/σ_m；縮減 β = 0.6×β_TS + 0.4×1。每月底依縮減 β 排序成 5 組。做多版：買 β 最低 20%（另報最低 30 檔），等權'
     '（另報論文的排名加權：權重 ∝ 排名離中位數的距離）；月底收盤算訊號、隔日開盤成交，每月換股、持有 1 個月。'
     'BAB 多空版只當作參考計算：低 β 組乘上 1/β_L、高 β 組乘上 1/β_H，做成 beta 中性（只報理論值）。'
     '報告低 β 組對 0050 的 CAPM α、t 值、逐年，以及「低 β 組 ÷ 平均 β」槓桿到 β=1 的理論績效。'
     '挑選期 2009-01（5 年暖機）～2019-04，驗證 2019-05～2026-10。',
 'type': '選股型（月換股）：組合報告（年化、日頻最大回撤、對 0050 日曆月 t、對同池等權日曆月 t、逐年、周轉）',
 'data': {
   'panel': 'scratchpad/panel 上市日線（含已下市；2019–2023 只有存活者）。有值檔數 < 300 的日子（2018-01-13）整列剔除後才算任何報酬與 rolling 窗。只用 4 碼普通股、排除 91xx（同 w3_idio_vol）。',
   'stock_logret': '在剔除後的日曆上 r = ln(adj_t / adj_{t-1})（不 ffill，任一邊缺值就缺）；t 日 flag ∈ {3,4,5} 或 |adj_t/adj_{t-1}−1| > 25% 視為缺值。',
   'market_logret': 'PRIMARY：taiex_twii 收盤（價格指數）reindex 到同一日曆，ln 比值。另一版：0050 含息 adj（不 ffill；2025-06-11～06-17 0050 停止交易缺值 → 該段報酬缺值）。',
   'rf': '無風險利率用固定 1%／年（台灣一年期定存約 1%～1.7% 的保守近似），日 = 1%/252。只用在 CAPM α、槓桿到 β=1、BAB 理論值；β 估計不扣 rf（日頻影響可忽略）。',
 },
 'beta': {
   'rho': '3 日重疊對數報酬 r3_t = r_t + r_{t-1} + r_{t-2}（三日都有值才算，股票與市場各自）。d 往回 1260 個交易日（含 d）內，股票與市場 r3 同時有值的配對算 Pearson 相關，配對數 ≥ 750 才有值。',
   'sigma': 'd 往回 252 個交易日（含 d）的日對數報酬標準差（ddof=1）。股票需 ≥ 120 個有效日；市場用同窗全部有效日。',
   'beta_TS': 'ρ × σ_i / σ_m',
   'beta_shrunk': '0.6 × β_TS + 0.4 × 1（排序與 β_L、β_H 都用縮減 β）',
 },
 'universe': '換股日 d（每月最後一個交易日）：60 日均成交值 ≥ NT$1,000 萬（停牌日記 0、需 60 日窗）、原始收盤 ≥ 10 元、d 的 flag ∈ {0,1,2}、adj 有值；d+1 有開盤價與 adj；縮減 β 有值。不限前 300 名（教學只寫 1,000 萬／10 元，與 w3_idio_vol 同）。',
 'timing': {
   'signal': '每月最後一個交易日 d 收盤。d 從 2008-12 最後交易日到 2026-09 最後交易日。',
   'trade': 'd+1 開盤成交（exlib M2 價格陣列 AO），持有到下個月最後交易日的隔日開盤；最後一期持有到資料最後一日 2026-10-22。',
   'blocked': 'd+1 開盤漲停（2015-06-01 前 7%、之後 10%，容差 0.6%）且未持有 → 視為買不到。等權組合由下一個更低 β 遞補；排名加權組合直接剔除、其他權重等比例放大；Q2～Q5 診斷組不遞補。',
   'hold': '1 個月，每月換股，換股時調回目標權重（漂移後權重計周轉，同 exlib.sim_portfolio）。',
 },
 'portfolios': {
   'Q1_EW_TAIEX': 'PRIMARY（判定用）：β 排序 qcut 5 組（rank method=first），買最低 β 組 Q1，等權。',
   'LOW30_EW_TAIEX': '最低 β 30 檔等權。',
   'RW_LOW_TAIEX': '論文排名加權低 β 腿：z = β 的排名，z̄ = 平均排名，w_i ∝ max(z̄ − z_i, 0)，總和 1（約低於中位數的一半股池）。',
   'Q2..Q5_EW_TAIEX': '診斷組（看 β 單調性與報酬是否平坦），等權不遞補。',
   'Q1_EW_0050': '市場改 0050 含息算 β 的 Q1 等權（Q5 也做，用於 BAB）。',
   'TVOL60_Q1': '對照：同股池（β 有值者）依 60 日日報酬標準差（≥ 45 有效日）分 5 組的 Q1，等權；用來比較與算重疊率。',
   'LEV_Q1_EW': '理論槓桿：日報酬 = rf + (r_Q1淨 − rf) / β̄_L，β̄_L = 換股時目標持股的縮減 β 等權平均（事前 β），整月固定；借款利率 = rf，不加融資利差。',
   'LEV_RW_LOW': '同上，排名加權腿，β̄_L = 加權平均縮減 β。',
   'BAB_RW': '論文版理論多空：r = (r_L − rf)/β_L − (r_H − rf)/β_H，L/H 為排名加權兩腿（H：w ∝ max(z − z̄, 0)），兩腿用同引擎；報毛（不扣成本、不扣借券費）與兩腿各扣基本成本兩版。',
   'BAB_Q': '五分組版：L = Q1_EW、H = Q5_EW，其餘同上。',
 },
 'costs': 'exlib.cost_rates：基本（判定用）滑價每邊 0.1%（來回約 0.67%）；c05 滑價 0（約 0.47%）；壓力 來回 1%；另報毛報酬。',
 'benchmarks': '同池等權（同股池＝β 有值者、同引擎、不扣成本）；0050 含息（exlib.E50）。t 用日曆月超額報酬（每月一個觀察值，即按月份分群）。',
 'windows': {'select_2009-01_2019-04': ['2009-01-01', '2019-04-30'], 'valid_2019-05_2026-10': ['2019-05-01', '2026-10-31'],
             'valid_2024-01_2025-12': ['2024-01-01', '2025-12-31'], 'valid_2024-01_2026-10': ['2024-01-01', '2026-10-31'],
             'valid_survivor_2019-05_2023-12': ['2019-05-01', '2023-12-31']},
 'capm': '日曆月 (r_p − rf) 對 (r_0050 − rf) OLS，報 α（%/月）與其 OLS t、實現 β、R²；每個窗分開。',
 'overlap': '每月 Q1(β) 與 Q1(TVOL60) 成員重疊率 = |交集| / |Q1(β)|（訊號名單、未處理漲停）；另報 β 與 TVOL60、與 1 年 σ_i、與 ρ 的截面 Spearman 相關平均值。',
 'verdict_rule': 'PRIMARY Q1_EW_TAIEX、基本成本。某段「通過」= 扣成本後對同池等權月超額 > 0 且 t ≥ 2，且淨年化 ≥ 0050 年化。挑選期與驗證期都過 → 可用；只一段過 → 待觀察；都不過 → 不可用；β 有值股池 < 50 檔的月份過多（> 1/3）→ 無法測試。其他組合只做參考，不換主規則。',
 'no_tuning': '所有參數（5 年／1 年窗、750／120 最少觀察值、0.6 縮減、5 組、30 檔、rf 1%）照論文與教學，看結果後不改。',
}
if __name__ == '__main__':
    f = HERE / 'results.json'
    d = json.loads(f.read_text(encoding='utf-8')) if f.exists() else {}
    if 'PREREG' in d:
        raise SystemExit('PREREG 已存在，不覆寫')
    d['PREREG'] = PREREG
    f.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding='utf-8')
    print('PREREG written', PREREG['written_at'])
