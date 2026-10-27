#!/usr/bin/env python3
"""事前設定（在看任何結果前寫入 results.json 的 PREREG；之後不改）。"""
import json, datetime
from pathlib import Path
HERE = Path(__file__).resolve().parent
PREREG = {
 'written_at': datetime.datetime.now().isoformat(timespec='seconds'),
 'rule_name': 'Amihud 非流動性溢酬（流動股池內，含 Lou-Shu 成交值分量）月換股',
 'sources': ['https://ideas.repec.org/a/asi/aeafrj/v9y2019i7p778-788id1837.html',
             'https://ideas.repec.org/a/eee/pacfin/v79y2023ics0927538x23000501.html',
             'https://pure.kfupm.edu.sa/en/publications/do-emerging-stock-markets-offer-an-illiquidity-premium-for-local-/',
             'https://osuva.uwasa.fi/handle/10024/633', 'https://tdr.lib.ntu.edu.tw/jspui/handle/123456789/57586',
             'https://etds.lib.tku.edu.tw/ETDS/Home/Detail/U0002-2506201921044300'],
 'source_access_note': '規則依任務敘述逐條轉成程式；教學未寫明的細節用最常見寫法，於下方固定，看結果後不改。',
 'type': '選股型（月換股）：組合報告（年化、日頻最大回撤、對 0050 日曆月 t、對同池等權日曆月 t、逐年、周轉）',
 'data': {
   'panel': 'scratchpad/panel 上市日線（含已下市；2019–2023 只有存活者）。有值檔數 < 300 的日子（2018-01-13）整列剔除後才做任何 rolling。只用 ret、value（另用 close_raw 做 10 元門檻、flag 剔除公司行動日、open/adj 做成交）。',
   'ILLIQ_250': '每月最後一個交易日 d 收盤：過去 250 個（剔除稀疏日後的）交易日內，有效日 = value 有值且 > 0、ret 有值、flag 不屬於 {3,4,5}；ILLIQ = mean(|ret_t| / (value_t / 1e6))，有效日 >= 200 才有值。',
   'ILLIQ_21': '變體 (a)：同上但回看 21 日、有效日 >= 15（AHKZ／Amihud 月頻常用的 15 日門檻）。',
   'value_component': '變體 (b1)：VC = 1 / mean(value_t/1e6)（同一 250 日有效日），越大越不流動。',
   'residual_component': '變體 (b2)：每月在股池內做截面 OLS：log(ILLIQ_250) = a + b·log(mean value_250) + e，用殘差 e 排序（越大越不流動）。取 log 是因 ILLIQ 極度右偏（常見作法）。',
 },
 'universe': '上市普通股：4 碼數字代碼、排除 91xx；判斷日 d：60 日均成交值（上市後停牌日記 0、需滿 60 日）>= NT$1,000 萬（變體 (d)：>= NT$3,000 萬）、原始收盤 >= 10 元、d 的 flag ∈ {0,1,2}、d 有 adj、d+1 有開盤價與 adj、排序變數有值。分組在這個股池內做。',
 'sort': '依排序變數在股池內 pd.qcut(rank, 5) 分 5 組，Q5 = 最不流動（ILLIQ 最大）。PRIMARY 買 Q5 全部等權。',
 'variant_c': '變體 (c)：先依 250 日平均成交值（同 VC 的分母）在股池內分三組，各組內依 ILLIQ_250 分 5 組，三個 Q5 合併等權（近似控制規模；面板無股本／市值）。',
 'execution': {
   'calendar': '每月最後一個交易日 d（剔除稀疏日後的日曆，2004-12～2026-09）收盤算訊號；d+1（隔月第一個交易日）開盤成交（M2 引擎：exec_rebuild/exlib 價格陣列與成本）；持有到下一個月底的隔日開盤換股；最後一期持有到資料最後一日 2026-10-22 收盤。',
   'tradable': 'd+1 開盤漲停（開盤/前收−1 >= 漲跌幅−0.6%，2015-06-01 前 7%、之後 10%）且未持有 → 買不到，由排序下一名（Q4 頂端）遞補，維持 Q5 原檔數；已持有者照常保留。',
   'cost': 'exlib.cost_rates：買 0.1425%×0.6＋滑價 0.1%；賣 0.1425%×0.6＋稅 0.3%＋滑價 0.1%（來回約 0.67%）；壓力：買 0.5%＋賣 0.5%（來回 1%）；另報毛報酬。周轉以漂移後市值計（每次換股賣出金額／總值）。',
   'empty': '若股池 < 50 檔該月不換股（沿用持股）。',
 },
 'variants': {
   'PRIMARY': 'A250_Q5（ILLIQ_250、門檻 1,000 萬、Q5 等權）',
   'all_reported': ['A21_Q5 (a)', 'VC_Q5 (b1)', 'RES_Q5 (b2)', 'A250_TERC_Q5 (c)', 'A250_Q5_T30M (d)', '診斷：A250 Q1～Q4、Q5−Q1 毛報酬單調性'],
   'note': '非 PRIMARY 只作穩健性與機制參考，不得用來改判。',
 },
 'lou_shu_test': '每個換股日在股池內做 Fama–MacBeth：下月報酬（d+1 開盤→下一期 d+1 開盤，clip −50%～+100%）對 [VC 百分位−0.5, 殘差百分位−0.5] 回歸，另做只放 ILLIQ_250 百分位的單變量回歸；報係數月平均與 t（月份分群），挑選期／驗證期分開。',
 'capacity_and_blocking': 'Q5 持股：換股時 60 日均成交值的中位數與 10 分位；容量估算 = 檔數 × 10 分位 V60 × 10%（每檔單日參與率 10% 以內一天建倉的組合規模）；開盤漲停買不到的檔數占 Q5 原名單比例；另報持有期內碰到 flag 3/4/5、src 3 期間的占比。',
 'benchmarks': {
   'EW': '同池等權：同一換股日、同一股池（PRIMARY 用 1,000 萬股池；(d) 用 3,000 萬股池），同一引擎、不扣成本（毛）——比扣成本的策略嚴格；另報 EW 扣成本年化',
   '0050': '0050 含息（finlab-txd etf_0050.csv）',
 },
 'stats': '年化（實際天數）、日頻最大回撤、波動、對 0050 與同池等權的日曆月超額報酬 t（btlib.tstat）、逐年報酬、周轉；挑選期與驗證期分開，另報 2024–2025。',
 'windows': {'select': '2005-01-01~2019-04-30', 'valid': '2019-05-01~2026-10-22', 'valid_2024_2025': '2024-01-01~2025-12-31',
             'extra_reported': ['valid_survivor_2019-05_2023-12（存活者面板）', 'valid_2024-01_2026-10']},
 'verdict_rule': 'PRIMARY（A250_Q5，滑價 0.1%）在挑選期與驗證期都要：扣成本後對同池等權（毛）月超額平均 > 0 且 t >= 2，且淨年化 >= 0050 同期年化 → 可用；只有一段過 → 待觀察；否則不可用。',
 'known_limits': ['面板無股本／市值，無法真正控制規模，只能用平均成交值分組近似', '2019–2023 面板只有存活者（對最不流動組驗證期偏樂觀）',
                  '2024-01-02~2024-02-14 無資料（rolling 跨過缺口）', 'src 3 期間（2024-02-15~2025-09-23）收盤與成交值為 Yahoo 5 分 K 近似',
                  '只有上市股，無上櫃（台灣最不流動的股票多在上櫃）'],
}
if __name__ == '__main__':
    f = HERE / 'results.json'
    d = json.loads(f.read_text(encoding='utf-8')) if f.exists() else {}
    assert 'PREREG' not in d, 'PREREG 已存在，不覆寫'
    d = {'PREREG': PREREG, **d}
    f.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding='utf-8')
    print('PREREG written')
