#!/usr/bin/env python3
"""事前設定（看任何結果前寫入 results.json 的 PREREG；之後不改）。"""
import json, datetime
from pathlib import Path
HERE = Path(__file__).resolve().parent
PREREG = {
 'written_at': datetime.datetime.now().isoformat(timespec='seconds'),
 'rule_name': '多均線趨勢因子（Han–Zhou–Zhu 2016 JFE；中國 A 股版 Liu–Zhou–Zhu）月換股',
 'sources': ['https://profiles.wustl.edu/en/publications/a-trend-factor-any-economic-gains-from-using-information-over-inv/',
             'https://acfr.aut.ac.nz/__data/assets/pdf_file/0014/324113/Y-Liu-New-TrendChina_12_1_WithAppendix.pdf',
             'https://wrds-www.wharton.upenn.edu/documents/1119/WARSP_TrendChina.pdf',
             'https://www.r-bloggers.com/2014/03/performance-of-trend-factor-in-chinese-market/',
             'https://aeaweb.org/conference/2021/preliminary/paper/K8eNGFSz'],
 'source_rule_verbatim': '股池同上。每月最後交易日 t，對每檔股票算正規化均線 A_L = MA_L(adj)/adj_t，L ∈ {3,5,10,20,50,100,200,400,600,800,1000} 日（論文預設 11 條）。'
                         '(1) 每月做橫斷面回歸 r_{i,t} = β0 + Σ_L β_L·A_{L,i,t-1} + ε（r 為當月報酬，A 為上月底值）；'
                         '(2) 預期報酬 E[r_{i,t+1}] = Σ_L β̄_L·A_{L,i,t}，β̄ 為過去 12 個月係數平均（只用已實現月份）；'
                         '(3) 依預期報酬排序，買進前 20%（另報前 30 檔），等權；月底收盤算訊號、隔日開盤成交，持有 1 個月。'
                         '歷史不足 1000 日的股票，缺的均線用該股可得最長均線替代（論文作法），或另做只用 L≤200 的版本。'
                         '中國版變體：加上成交量均線 MA_L(volume)/volume_t，volume＝value/close_raw，同一組 L 最長到 400。'
                         '挑選期 2009-01～2019-04（需 1000 日＋12 個月暖機），驗證期 2019-05～2026-10。',
 'deviation_from_paper': ['原論文（美股）用全體股票、十分位多空；這裡只做多前 20%（另報前 30 檔），等權，股池用中心流動性門檻。',
                          '預期報酬公式依教學：E = Σ β̄_L·A_L（不加 β̄0；截距對排序沒有影響）。',
                          '論文用月報酬與月底均線，這裡相同；實際成交是隔日開盤（M2），回歸用的仍是月底收盤對收盤報酬（訊號只用已實現資料）。',
                          '中國版的量均線用 L ∈ {3,5,10,20,50,100,200,400}（價、量各 8 條，共 16 個回歸變數），長度上限 400 依教學。'],
 'type': '選股型（月換股）：組合報告（年化、日頻最大回撤、對 0050 日曆月 t、對同池等權日曆月 t、逐年、周轉）',
 'data': {
   'panel': 'scratchpad/panel 上市日線（含已下市；2019–2023 只有存活者）。有值檔數 < 300 的日子（2018-01-13）整列剔除後，才算月底日曆、均線、60 日均量與任何回看。',
   'codes': '4 碼數字代碼、排除 91xx（存託憑證）。',
   'month_end': '剔除稀疏日後，每個日曆月最後一個交易日 = 月底 ME（2024-01 面板無資料 → 無 2024-01 月底，2023-12→2024-02 視為一期）。',
   'monthly_return': 'r[m] = adj(ME[m]) / adj(ME[m-1]) − 1；任一端缺值 → 缺。|r| ≥ +200% 或 ≤ −80% 視為資料異常設缺（報告筆數）。',
   'MA_price': 'MA_L(adj) 用剔除稀疏日後的交易日序列：adj 在上市（首個有值日）後向前填補（停牌日沿用最後價），取最近 L 個交易日的平均；需該股已有 ≥ L 個交易日歷史才算「真值」。A_L = MA_L / adj_t，且月底當天 adj 必須有值（未停牌）。',
   'substitution': 'PRIMARY（TF11）：歷史 ≥ 200 日的股票才有訊號；L > 歷史長度的均線，用該股已有真值的最長均線（≥200 日中最長者）替代（論文作法）。歷史 < 200 日 → 無訊號。',
   'MA_volume': '量 = value / close_raw，上市後停牌日記 0；MA_L(vol) 同樣最近 L 個交易日平均。A^V_L = MA_L(vol) / vol_t，月底當天 vol_t 必須 > 0；L = 400 歷史不足時用 200 替代（同價格規則）。',
   'winsorize': '每個月、每個回歸變數在當月樣本內做 1%/99% 截尾（回歸與預測都一樣；量比在個股單日量極小時會爆大，事前固定此常見作法）。',
   'V60': '60 日均成交值：上市後停牌日記 0、需滿 60 個交易日（剔除稀疏日後）。',
   'base_pool': '月底 ME 的基礎股池：V60 ≥ NT$1,000 萬、原始收盤 ≥ 10 元、當日 adj 有值、flag ∈ {0,1,2}。',
 },
 'signal': {
   'regression': '月 m（ME[m-1]→ME[m]）：樣本 = ME[m-1] 基礎股池 ∩ r[m] 有值 ∩ 該版本所有 A 有值；OLS r = β0 + Σβ_L A_L(ME[m-1]) + ε，樣本 ≥ 50 檔才記錄 β[m]。',
   'regression_start': '第一個回歸月 = ME[m-1] 是面板上第一個「已有 ≥ 1000 個交易日」的月底之後（約 2007-12 月底 → 2008-01 月）。所有版本同一起點（可比較）。',
   'forecast': '訊號日 d = ME[m]：β̄ = β[m-11..m] 共 12 個已實現月份的平均（月 m 報酬在 ME[m] 收盤已實現），需 12 個都存在；E_i = Σ β̄_L·A_L,i(ME[m])。第一個訊號約 2008-12 月底，持有從 2009-01 起。',
   'TF11': 'PRIMARY：11 條價格均線 L ∈ {3,5,10,20,50,100,200,400,600,800,1000}，長均線缺值用可得最長均線替代。',
   'S200': '只用 L ≤ 200 的 7 條價格均線（不需要替代）。',
   'CN': '中國版：價格 + 量均線，L ∈ {3,5,10,20,50,100,200,400} 各 8 條，共 16 變數；400 缺用 200 替代。',
 },
 'universe': '排序股池（所有組合共用、比的是同一群股票）：d 的基礎股池 ∩ d+1（隔月第一個交易日）有開盤價與 adj ∩ TF11、S200、CN 三個預期報酬都有值。若排序股池 < 50 檔，該月不換股。',
 'portfolios': {
   'TF11_Q20': 'PRIMARY（判定用）：TF11 預期報酬由高到低，買前 floor(20%×排序股池檔數) 檔，等權。',
   'TF11_N30': 'TF11 前 30 檔。',
   'S200_Q20': 'S200 前 20%。', 'S200_N30': 'S200 前 30 檔。',
   'CN_Q20': '中國版前 20%。', 'CN_N30': '中國版前 30 檔。',
   'note': '非 PRIMARY 只作穩健性與機制參考，不得用來改判。',
 },
 'execution': {
   'calendar': '每月底 d 收盤算訊號；d+1 開盤成交（exec_rebuild/exlib 價格陣列 AO/AC 與 cost_rates，M2 規則）；持有到下一個月底的隔日開盤換股；最後一期（訊號 2026-09-30）持有到資料最後一日。',
   'tradable': 'd+1 開盤漲停（開盤/前收−1 ≥ 漲跌幅−0.6%；2015-06-01 前 7%、之後 10%）且未持有 → 買不到，由排序下一名遞補，維持目標檔數；已持有者照常保留。',
   'cost': 'exlib.cost_rates：基本（判定用）＝買 0.1425%×0.6＋滑價 0.1%；賣 0.1425%×0.6＋稅 0.3%＋滑價 0.1%（來回約 0.67%）；另報「0.5%」情境＝滑價 0（來回約 0.47%）、壓力＝買 0.5%＋賣 0.5%（來回 1%）、毛報酬。周轉以漂移後市值計。',
 },
 'benchmarks': {'EW': '同池等權：同一換股日、同一排序股池全部等權，同一引擎、不扣成本（毛）——比扣成本的策略嚴格；另報 EW 扣成本年化。',
                '0050': '0050 含息（finlab-txd etf_0050.csv）。'},
 'stats': '年化（實際天數）、日頻最大回撤、波動、對 0050 與同池等權的日曆月超額報酬 t（btlib.tstat，月份分群）、逐年、周轉；挑選期與驗證期分開，另報 2024–2025。'
          '診斷：(i) 截面 IC（預期報酬 vs d+1 開盤→下一 d+1 開盤報酬，Spearman）月平均與 t；(ii) 五分位 Q5−Q1 毛價差；(iii) β̄ 各係數平均；'
          '(iv) 預期報酬與上月報酬（短期反轉）、12-1 動能的截面等級相關；(v) 排序股池中真有 1000 日歷史的比例（替代比例）；(vi) TF11 − S200、CN − TF11 配對月差 t。',
 'windows': {'select': '2009-01-01~2019-04-30', 'valid': '2019-05-01~2026-10-22', 'valid_2024_2025': '2024-01-01~2025-12-31',
             'extra_reported': ['valid_survivor_2019-05_2023-12', 'valid_2024-01_2026-10', 'full_2009-01_2026-10']},
 'verdict_rule': 'PRIMARY TF11_Q20（基本成本）在挑選期與驗證期都要：扣成本後對同池等權（毛）月超額平均 > 0 且 t ≥ 2，且淨年化 ≥ 0050 同期年化 → 可用；只有一段過 → 待觀察；否則不可用。',
 'known_limits': ['只做多、等權，非論文的多空十分位', '2019–2023 面板只有存活者（長均線樣本偏向存活者、驗證期偏樂觀）',
                  '2024-01-02~2024-02-14 無資料（無 2024-01 月底，持有跨兩個月）',
                  'src 3 期間（2024-02-15~2025-09-23）收盤為 Yahoo 5 分 K 近似；成交量 = value/close_raw 亦為近似',
                  '只有上市股，無上櫃', '處置股名單只有 2021-09 起，本題不排除處置股（月頻、全期一致）'],
}
if __name__ == '__main__':
    f = HERE / 'results.json'
    d = json.loads(f.read_text(encoding='utf-8')) if f.exists() else {}
    assert 'PREREG' not in d, 'PREREG 已存在，不覆寫'
    d = {'PREREG': PREREG, **d}
    f.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding='utf-8')
    print('PREREG written')
