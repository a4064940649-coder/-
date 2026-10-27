#!/usr/bin/env python3
"""事前設定（在看任何結果前寫入 results.json 的 PREREG；之後不改）。"""
import json, datetime
from pathlib import Path
HERE = Path(__file__).resolve().parent
PREREG = {
 'written_at': datetime.datetime.now().isoformat(timespec='seconds'),
 'rule_name': '價格延遲溢酬（Hou-Moskowitz 2005 D1）：買延遲最大十分位，月換股',
 'sources': ['https://ideas.repec.org/a/eee/finmar/v17y2014icp150-173.html',
             'https://researchonline.stthomas.edu/esploro/outputs/journalArticle/Price-Delay-Premium-and-Liquidity-Risk/991015132254303691',
             'https://ideas.repec.org/a/eee/pacfin/v72y2022ics0927538x22000270.html', 'https://dx.doi.org/10.1007/s11156-013-0410-8',
             'https://arxiv.org/pdf/2505.01921', 'https://uva.academia.edu/TseChunLin'],
 'source_access_note': '本環境無法連線（arxiv.org、ideas.repec.org DNS 失敗），規則依任務敘述逐條轉成程式；教學未寫明的細節用 Hou-Moskowitz (2005) 最常見寫法，於下方固定，看結果後不改。',
 'type': '選股型（月換股）：組合報告（年化、日頻最大回撤、對 0050 日曆月 t、對同池等權日曆月 t、逐年、周轉）',
 'data': {
   'panel': 'scratchpad/panel 上市日線（含已下市；2019–2023 只有存活者）。有值檔數 < 300 的日子（2018-01-13）整列剔除後才做任何 rolling。',
   'weekly_returns': '週＝週四～週三（pandas W-WED）；每週取該週最後一個交易日（通常為週三，假日則前一個交易日）的 adj（ffill 至多 4 個交易日，避免短暫停牌造成整週缺值；更久停牌為缺值）。週報酬 r_i,t = adj(週末日)/adj(前一週末日) − 1；|r| > 200% 視為資料錯誤設缺。',
   'Rm_EW': '同池等權週報酬：週 t 的 Rm = 在前一週末日屬於「流動基本池」（60 日均成交值 >= 1,000 萬、原始收盤 >= 10 元、4 碼普通股非 91xx）的股票，其週 t 報酬的等權平均（至少 50 檔）。',
   'Rm_TAIEX': '穩健性：加權指數 taiex_twii.csv close（價格指數）同一週末日的週報酬。',
   'D1': '每月最後一個交易日 d：取週末日 <= d 的最近 52 週。兩個回歸用同一組樣本週（個股週報酬有值，且 Rm_t..Rm_{t-4} 都有值），有效週 >= 40。受限式 r = a + b0·Rm_t；不受限式 r = a + b0·Rm_t + Σ_{k=1..4} b_k·Rm_{t−k}；R² = 1 − SSR/SST（SST 以個股樣本均值）。D1 = 1 − R²_r / R²_u；R²_u <= 0 或非有限 → 缺值。',
   'D1_daily': '變體 (c)：日報酬（ret，|ret|>50% 設缺）＋同池等權日 Rm（前一日流動基本池、至少 50 檔）＋ 4 個落後期；回看 d 往前 250 個（剔除稀疏日後的）交易日，有效日 >= 200。',
   'V250': '變體 (b) 用：過去 250 日平均成交值（value 有值且 > 0 的日子，>= 200 日）。',
 },
 'universe': '與 w3_amihud_illiquidity 相同（方便比較重疊）：4 碼數字代碼、排除 91xx；判斷日 d：60 日均成交值（上市後停牌日記 0、需滿 60 日）>= NT$1,000 萬、原始收盤 >= 10 元、d 的 flag ∈ {0,1,2}、d 有 adj、d+1 有開盤價與 adj、D1 有值。分組在這個股池內做。',
 'sort': '依 D1 在股池內 pd.qcut(rank(method=first), 10) 分 10 組，D10 = 延遲最大。PRIMARY 買 D10 全部等權。',
 'execution': {
   'calendar': '每月最後一個交易日 d（剔除稀疏日後的日曆，2004-12～2026-09）收盤算訊號；d+1（隔月第一個交易日）開盤成交（exec_rebuild/exlib 價格陣列與成本，同 M2）；持有到下一個月底的隔日開盤換股；最後一期持有到 2026-10-22 收盤。',
   'tradable': 'd+1 開盤漲停（開盤/前收−1 >= 漲跌幅−0.6%，2015-06-01 前 7%、之後 10%）且未持有 → 買不到，由排序下一名（D9 頂端）遞補，維持 D10 原檔數；已持有者照常保留。',
   'cost': 'exlib.cost_rates：買 0.1425%×0.6＋滑價 0.1%；賣 0.1425%×0.6＋稅 0.3%＋滑價 0.1%（來回約 0.67%，即「0.5%」情境加滑價）；壓力：買 0.5%＋賣 0.5%（來回 1%）；另報毛報酬。周轉以漂移後市值計。',
   'empty': '股池 < 50 檔該月不換股（沿用持股）。',
 },
 'variants': {
   'PRIMARY': 'W_EW_D10（週報酬、Rm=同池等權、月換股、D10 等權）',
   'all_reported': [
     'W_TX_D10：Rm 改加權指數（穩健性）',
     'W_EW_D10_ANNUAL (a)：只在每年 6 月最後交易日算訊號、隔日開盤買、持有 12 個月（期間不換股），2005-06 起',
     'W_EW_D10_TERC (b)：股池內先依 V250 分三組，各組內依 D1 分 10 組，三個 D10 合併等權',
     'DAILY_EW_D10 (c)：日報酬＋4 落後期估 D1，其餘同 PRIMARY',
     '診斷：W_EW D1～D9 毛報酬單調性、D10−D1 毛價差',
   ],
   'note': '非 PRIMARY 只作穩健性與機制參考，不得用來改判。',
 },
 'diagnostics': {
   'amihud_overlap': '每個換股日：D10 名單與 Amihud ILLIQ_250 Q5（照 w3_amihud_illiquidity 定義在同股池重算）重疊比例（隨機期望 20%）；D1 與 ILLIQ_250 的截面 Spearman；D10 與 Amihud A250_Q5 組合（讀取該資料夾 nav.parquet）月超額報酬（對 EW）相關係數。',
   'limit_hits': '每檔過去 250 日「漲跌停日」數：ret >= 漲停−0.6% 且收盤 >= 最高價×0.999（鎖漲停）或 ret <= −(跌停−0.6%)（無 low 欄，跌停只能用報酬判斷）；比較 D10 與股池其他股的平均；D1 與漲跌停日數的截面 Spearman。',
   'fama_macbeth': '每換股日在股池內：下月報酬（d+1 開盤→下一期 d+1 開盤，clip −50%～+100%）對 D1 百分位−0.5 的單變量回歸，以及加入 ILLIQ_250 百分位、漲跌停日數百分位的多變量回歸；係數月平均與 t（月份分群），挑選期／驗證期分開。',
   'ex_limit': '診斷組合 W_EW_D10_EXLIM：先剔除過去 250 日漲跌停日 >= 5 的股票再分 10 組取 D10（只作機制參考，不改判）。',
 },
 'benchmarks': {
   'EW': '同池等權：同一換股日、同一股池（PRIMARY 的股池），同一引擎、不扣成本（毛）——比扣成本的策略嚴格；另報 EW 扣成本年化。(a) 年換股版另報同股池年換股等權。',
   '0050': '0050 含息（finlab-txd etf_0050.csv）',
 },
 'stats': '年化（實際天數）、日頻最大回撤、波動、對 0050 與同池等權的日曆月超額報酬 t（btlib.tstat）、逐年報酬、周轉；挑選期與驗證期分開，另報 2024–2025。',
 'windows': {'select': '2005-01-01~2019-04-30', 'valid': '2019-05-01~2026-10-22', 'valid_2024_2025': '2024-01-01~2025-12-31',
             'extra_reported': ['valid_survivor_2019-05_2023-12（存活者面板）', 'valid_2024-01_2026-10']},
 'verdict_rule': 'PRIMARY（W_EW_D10，滑價 0.1%）在挑選期與驗證期都要：扣成本後對同池等權（毛）月超額平均 > 0 且 t >= 2，且淨年化 >= 0050 同期年化 → 可用；只有一段過 → 待觀察；否則不可用。',
 'known_limits': ['只有上市股、無上櫃', '2019–2023 面板只有存活者', '2024-01-02~2024-02-14 無資料（週報酬跨缺口那週會是多週報酬，照常使用）',
                  'src 3 期間（2024-02-15~2025-09-23）收盤為 Yahoo 5 分 K 13:20 近似，週三收盤不準會影響 D1',
                  '無市值資料，(b) 用平均成交值三分組近似規模／流動性', '無 low 欄，跌停只能用日報酬近似判斷'],
}
if __name__ == '__main__':
    f = HERE / 'results.json'
    d = json.loads(f.read_text(encoding='utf-8')) if f.exists() else {}
    assert 'PREREG' not in d, 'PREREG 已存在，不覆寫'
    d = {'PREREG': PREREG, **d}
    f.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding='utf-8')
    print('PREREG written')
