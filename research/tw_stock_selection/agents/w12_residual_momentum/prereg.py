#!/usr/bin/env python3
"""事前設定（看任何結果前寫入 results.json 的 PREREG；之後不改）。"""
import json, datetime
from pathlib import Path
HERE = Path(__file__).resolve().parent
PREREG = {
 'written_at': datetime.datetime.now().isoformat(timespec='seconds'),
 'rule_name': '殘差動能（Blitz–Huij–Martens 2011；Robeco、Quantpedia #136）月換股',
 'sources': ['https://quantpedia.com/strategies/residual-momentum-factor', 'https://repub.eur.nl/pub/22252',
             'https://www.robeco.com/en-ch/insights/2023/02/quant-chart-taming-momentum-crashes',
             'https://thesis.eur.nl/pub/66795/Emmanouil_Giakoumakis_641404_FinalThesis_FE_Masters.pdf',
             'https://www.quantconnect.com/research/15304/residual-momentum/'],
 'source_rule_verbatim': '股池：每月最後交易日，上市股中 60 日均成交值 ≥ NT$1,000 萬、收盤價 ≥ 10 元、至少有 36 個月月報酬（最少 24 個月）。'
                         '市場因子：同池等權月報酬。(1) 對每檔股票用過去 36 個月（t-36～t-1）月報酬對市場因子做 OLS，取殘差 e；'
                         '(2) 訊號＝t-12～t-2 共 11 個月殘差之和 ÷ 同 11 個月殘差的標準差（跳過最近 1 個月）；(3) 依訊號排序，買進前 10%（另報前 30 檔），等權；'
                         '(4) 每月換股，月底收盤算訊號、隔日開盤成交，開盤漲停遞補；(5) 持有 1 個月。對照：同池原始 12-1 動能前 10%、同池等權、0050 含息。'
                         '挑選期 2007-01～2019-04，驗證期 2019-05～2026-10；成本來回 0.5%／壓力 1%。',
 'deviation_from_paper': ['原論文（Blitz, Huij & Martens 2011, JEF）用 Fama-French 三因子回歸殘差、美股、十分位多空。台股面板沒有長期股本與淨值（shares_hist 只有 2025-09 起），'
                          'SMB/HML 無法照原法建構 → PRIMARY 退一步用單因子（同池等權市場）殘差；另做一版「市場＋成交值大小分組 SMB 代理」雙因子殘差；HML 完全缺。',
                          '只做多前 10%（不放空、不做多空價差），組合等權（非市值加權）。',
                          '原文訊號同樣是 t-12～t-2 殘差和 ÷ 殘差標準差、跳過最近一個月、36 個月回歸窗，這部分照原文。'],
 'type': '選股型（月換股）：組合報告（年化、日頻最大回撤、對 0050 日曆月 t、對同池等權日曆月 t、逐年、周轉）',
 'data': {
   'panel': 'scratchpad/panel 上市日線（含已下市；2019–2023 只有存活者）。有值檔數 < 300 的日子（2018-01-13）整列剔除後才算月底日曆、60 日均量與任何回看。',
   'codes': '4 碼數字代碼、排除 91xx（存託憑證）。',
   'month_end': '剔除稀疏日後，每個日曆月最後一個交易日 = 月底 ME。',
   'monthly_return': '月報酬 R[m] = adj(ME[m]) / adj(ME[m-1]) − 1；任一端 adj 缺值（月底停牌）→ 該月缺值。資料異常保護：R ≥ +200% 或 ≤ −80% 視為缺值（事前固定，報告筆數）。2004-01 為第一個月底，第一筆月報酬是 2004-02。',
   'V60': '60 日均成交值：上市後停牌日記 0、需滿 60 個交易日（剔除稀疏日後）。',
   'base_pool_m': '月底 ME[m] 的基礎股池 BP[m]：V60 ≥ NT$1,000 萬、原始收盤 ≥ 10 元、當日 adj 有值、flag ∈ {0,1,2}。',
   'MKT_factor': 'MKT[m] = 前一月底基礎股池 BP[m-1] 成員在月 m 的月報酬等權平均（只平均有值者，至少 50 檔，否則缺值）。成員在月初就已知，無前視。',
   'SMB_proxy': 'SMB[m]：BP[m-1] 成員依 ME[m-1] 的 V60 以中位數切兩組（成交值當作市值代理；FF 的規模斷點也用中位數），SMB = 小半等權月報酬 − 大半等權月報酬（各組至少 25 檔）。',
 },
 'signal': {
   'window': '訊號日 d = ME[m]。回歸窗 = 最近 36 個月報酬 R[m-35..m]（即持有月 t 的 t-36～t-1）；該股有效月數 ≥ 24 才回歸（含截距 OLS）。',
   'RESID_CAPM': 'PRIMARY：R_i = a + b·MKT + e。',
   'RESID_FF2P': '另一版：R_i = a + b·MKT + s·SMB_proxy + e（有效月數 ≥ 24）。',
   'score': '訊號 = Σ e[m-11..m-1]（t-12～t-2，跳過最近一個月 R[m]）÷ std(e[m-11..m-1], ddof=1)。11 個月殘差都要有值，否則該股無訊號。',
   'RAW_12_1': '對照：原始 12-1 動能 = Π(1+R[m-11..m-1]) − 1（同樣 11 個月都要有值）。',
 },
 'universe': '排序股池（所有組合共用、保證比的是同一群股票）：d 的基礎股池 ∩ d+1（隔月第一個交易日）有開盤價與 adj ∩ RESID_CAPM、RESID_FF2P、RAW_12_1 三個訊號都有值。若排序股池 < 50 檔，該月不換股。',
 'portfolios': {
   'RC_D10': 'PRIMARY（判定用）：RESID_CAPM 由高到低，買前 floor(10%×排序股池檔數) 檔，等權。',
   'RC_N30': 'RESID_CAPM 前 30 檔。',
   'RF_D10': 'RESID_FF2P 前 10%。', 'RF_N30': 'RESID_FF2P 前 30 檔。',
   'RAW_D10': '對照：RAW_12_1 前 10%。', 'RAW_N30': '對照：RAW_12_1 前 30 檔。',
   'note': '非 PRIMARY 只作穩健性與機制參考，不得用來改判。',
 },
 'execution': {
   'calendar': '每月底 d 收盤算訊號；d+1 開盤成交（exec_rebuild/exlib 價格陣列 AO/AC 與 cost_rates，M2 規則）；持有到下一個月底的隔日開盤換股；最後一期（訊號 2026-09-30）持有到資料最後一日 2026-10-22。訊號最早在有 24 個月報酬時出現（約 2006-01），挑選期統計從 2007-01 起算。',
   'tradable': 'd+1 開盤漲停（開盤/前收−1 ≥ 漲跌幅−0.6%；2015-06-01 前 7%、之後 10%）且未持有 → 買不到，由排序下一名遞補，維持目標檔數；已持有者照常保留。',
   'cost': 'exlib.cost_rates：基本（判定用）＝買 0.1425%×0.6＋滑價 0.1%；賣 0.1425%×0.6＋稅 0.3%＋滑價 0.1%（來回約 0.67%）；另報「0.5%」情境＝滑價 0（來回約 0.47%）、壓力＝買 0.5%＋賣 0.5%（來回 1%）、毛報酬。周轉以漂移後市值計。',
 },
 'benchmarks': {'EW': '同池等權：同一換股日、同一排序股池全部等權，同一引擎、不扣成本（毛）——比扣成本的策略嚴格；另報 EW 扣成本年化。',
                '0050': '0050 含息（finlab-txd etf_0050.csv）。'},
 'stats': '年化（實際天數）、日頻最大回撤、波動、對 0050 與同池等權的日曆月超額報酬 t（btlib.tstat，月份分群）、逐年、周轉；挑選期與驗證期分開，另報 2024–2025。'
          '診斷：(i) RC_D10 − RAW_D10 扣成本月差配對 t；(ii) 截面 IC（訊號 vs d+1 開盤→下一 d+1 開盤報酬，Spearman）月平均與 t；(iii) 十分位 D10−D1 毛價差；(iv) 殘差訊號與原始 12-1 的截面等級相關；(v) 最差月份（動能崩盤：Robeco 宣稱殘差動能崩得較輕）。',
 'windows': {'select': '2007-01-01~2019-04-30', 'valid': '2019-05-01~2026-10-22', 'valid_2024_2025': '2024-01-01~2025-12-31',
             'extra_reported': ['select_ext_2006-02_2019-04（最少 24 個月暖機即開始）', 'valid_survivor_2019-05_2023-12', 'valid_2024-01_2026-10', 'full_2007-01_2026-10']},
 'verdict_rule': 'PRIMARY RC_D10（基本成本）在挑選期與驗證期都要：扣成本後對同池等權（毛）月超額平均 > 0 且 t ≥ 2，且淨年化 ≥ 0050 同期年化 → 可用；只有一段過 → 待觀察；否則不可用。',
 'prior_center_result': 'momentum_family P4（每月 11 日後換股、前 300 流動性、21 日區塊、對加權指數殘差、n30）挑選期對等權 t −2.14；本題日曆、股池、市場因子、檔數都不同，獨立重測，不以 P4 改判。',
 'known_limits': ['無 FF3：PRIMARY 為 CAPM（等權市場）殘差，SMB 只能用成交值代理，HML 缺',
                  '等權組合而非市值加權', '2019–2023 面板只有存活者（對動能類驗證期偏樂觀）',
                  '2024-01-02~2024-02-14 無資料（2024-01 月報酬缺，該月底無法換股 → 持有跨兩個月）',
                  'src 3 期間（2024-02-15~2025-09-23）收盤為 Yahoo 5 分 K 近似', '只有上市股，無上櫃', '處置股名單只有 2021-09 起，本題不排除處置股（月頻、全期一致）'],
}
if __name__ == '__main__':
    f = HERE / 'results.json'
    d = json.loads(f.read_text(encoding='utf-8')) if f.exists() else {}
    assert 'PREREG' not in d, 'PREREG 已存在，不覆寫'
    d = {'PREREG': PREREG, **d}
    f.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding='utf-8')
    print('PREREG written')
