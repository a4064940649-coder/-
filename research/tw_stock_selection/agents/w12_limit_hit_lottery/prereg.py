#!/usr/bin/env python3
"""事前設定（看任何結果前寫入 results.json 的 PREREG；之後不改）。"""
import json, datetime
from pathlib import Path
HERE = Path(__file__).resolve().parent
PREREG = {
 'written_at': datetime.datetime.now().isoformat(timespec='seconds'),
 'rule_name': '漲停次數樂透股（價格限制下的 MAX：Hung–Yang 2018 JFM；Wang–Lien 2022 RQFA 台股淨觸板率）',
 'sources': ['https://business.oregonstate.edu/faculty-and-research/research/publications-list/max-effect-lottery-stocks-price-limits-and-limits',
             'https://ideas.repec.org/a/eee/finmar/v41y2018icp77-91.html',
             'https://ideas.repec.org/a/kap/rqfnac/v59y2022i2d10.1007_s11156-022-01051-1.html',
             'https://thesis.lib.nccu.edu.tw/detail/795b691088dcfeb024f8bfe38c10c423/',
             'https://www.quantconnect.com/research/20886/the-max-effect-with-vix-based-leverage-scaling/'],
 'source_rule_verbatim': '觸板：當日 close_raw 相對前一日 close_raw 的漲幅 ≥ 漲停幅度 − 0.3%（2015-06-01 前 7%、之後 10%，用 6.7%／9.7% 近似，'
                         '跳過除權息與 flag 3/4/5 的日子）；跌停同理。(A) Wang–Lien 淨觸板率＝過去 20 個交易日（另報 60 日）漲停天數減跌停天數，除以交易天數；'
                         '(B) Hung–Yang 修正 MAX＝過去 1 個月最大日報酬（上限為漲停幅度），同為觸板時用觸板天數排序。每月底依訊號分 5 組，等權，隔日開盤成交，持有 1 個月。'
                         '用法：(1) 最低組做多（排除樂透股）；(2) 最高減最低價差（只作診斷）；(3) 主策略排除條件：持股或候選股過去 20 日漲停 ≥ 2 次就剔除，報配對 t。'
                         '依 Wang–Lien 另依「上月營收已公告後」與「營收公告前」分開統計。挑選期 2005-01～2019-04（2015-06 漲跌幅制度改變，兩段分開報），驗證期 2019-05～2026-10。',
 'type': '選股型（月換股）＋主策略排除濾網：組合報告（年化、日頻最大回撤、對 0050 日曆月 t、對同池等權日曆月 t、逐年、周轉）',
 'data': {
   'panel': 'scratchpad/panel 上市日線（含已下市；2019–2023 只有存活者）。有值檔數 < 300 的日子（2018-01-13）整列剔除後才做任何 rolling／回看窗與「前一日收盤」。',
   'stocks': '4 碼數字代碼、排除 91xx（同 w3_idio_vol）。',
   'day_valid': '觸板可判定日：當日與前一（非稀疏）交易日 close_raw 都有值、當日 flag == 0（flag 1/2 = 除權息日、3 = 參考價重設、4 = 資料缺口、5 = 資料錯誤，一律跳過）、'
                '且 |(1+ret) − close/prev_close| ≤ 0.2%（額外防止未標記的除權息），且 raw 漲跌幅 |r_raw| ≤ 漲跌停幅度 + 0.5%（超過者視為無漲跌幅限制日（新上市前 5 日）或資料錯誤，跳過）。',
   'limit': '漲跌停幅度 L：日期 < 2015-06-01 為 7%，之後 10%。',
   'UP': '漲停（收盤觸板）：day_valid 且 r_raw ≥ L − 0.3%（6.7%／9.7%）。',
   'DN': '跌停：day_valid 且 r_raw ≤ −(L − 0.3%)。',
   'src3_note': 'src 3（2024-02-15～2025-09-23）close_raw 是 Yahoo 13:20 近似，觸板判定會失真；history.csv.gz 官方收盤只從 2025-09-24 起，無法回補 src 3 段，'
                '故不改資料，另報「驗證期_官方訊號段」（訊號回看窗不含 src 3 日：2019-05-01～2024-02-29 與 2025-11-01～2026-10-22 的持有期），並報 src 3 段與官方段的觸板頻率對照。',
 },
 'signals': {
   'NET20': 'PRIMARY。Wang–Lien 淨觸板率：d 往回 20 個（非稀疏）交易日含 d，(ΣUP − ΣDN) / Σday_valid；需 Σday_valid ≥ 15。',
   'NET60': '另報：回看 60 日，需 Σday_valid ≥ 45。',
   'MAXC': 'Hung–Yang 修正 MAX：d 往回 21 個交易日含 d 的清理日報酬（ret 含息、flag ∉ {3,4,5}、|ret| ≤ 25%）最大值，上限截在 L（當日的 L）；需 ≥ 15 個有效日。'
           '若該窗 UP 天數 ≥ 1，分數 = 1 + UP 天數（即所有觸板股排在未觸板股之上，觸板股之間以觸板天數排序）；否則分數 = min(max ret, L)。',
   'UP20': '排除條件用：d 往回 20 個交易日 ΣUP。',
 },
 'universe': '同 w3_idio_vol 基礎股池：判斷日 d 的 60 日均成交值（上市後停牌日記 0、需滿 60 日）≥ NT$1,000 萬、原始收盤 ≥ 10 元、d 的 flag ∈ {0,1,2}、d 有 adj、d+1 有開盤價與 adj。排序股池 = 基礎股池且訊號有值。',
 'sort': '排序股池內依訊號由低到高排序，同值者以固定亂數（numpy seed 20260427，每個換股日重抽）打破，然後 qcut 成 5 組（各 20%）。Q1 = 最低（最不像樂透）、Q5 = 最高。'
         '診斷：「同值同組」版本（pct rank average，floor(pct×5) 分組，所以 0 值大團塊整塊落在同組）只報 Q1、Q5，不判定。',
 'usages': {
   'U1_long_low': 'PRIMARY（判定用）：NET20 的 Q1 等權做多。另報 NET60 Q1、MAXC Q1（非判定）。',
   'U2_spread': 'Q5 − Q1（兩腿各自扣成本的日曆月報酬差，以及毛報酬差），只作診斷；另報 Q2～Q4 毛報酬看單調性。',
   'U3_main_exclusion': '主策略 = revenue_longhistory/lhlib.py 的 Setup(yoy_long(), start=2005-01-01).selection(30)（M2：每月第一個 ≥ 11 日交易日 d 收盤算分數、d+1 開盤成交、開盤漲停未持有者遞補，成本同 exlib）。'
                        '排除版：d 的 UP20 ≥ 2 的股票（不論是否持有）從合格股池剔除後再排名選 30 檔。報 排除版 − 原版 的淨日曆月報酬配對差與 t（月份分群），挑選期／驗證期分開。'
                        '敏感度（不判定）：UP20 ≥ 1、UP20 ≥ 3、NET20 位於當日排序股池最高 20%。',
   'revenue_split': 'Wang–Lien「月營收公告前／後」：(a) 持有期拆段——每個持有月以「當月 10 日（含）前最後一個交易日收盤」為界：公告前段 = d+1 開盤 → 該日收盤；公告後段 = 該日收盤 → 下一換股日開盤。'
                    '各段報 Q1、Q5 個股等權平均報酬減同排序股池平均（毛），及 Q5−Q1，t 按月份分群。'
                    '(b) 形成日改為「營收已公告後」：每月第一個 ≥ 11 日交易日收盤算訊號（同主策略日曆）、隔日開盤成交、持有到下一個換股日，報 NET20 Q1 組合（同成本）與 Q5−Q1（毛）。',
 },
 'execution': {
   'calendar': '每月最後一個交易日 d（剔除稀疏日後的日曆，2004-12～2026-09）收盤算訊號；d+1 開盤成交（exlib 價格陣列 AO/AC 與 cost_rates）；持有到下一個月底的隔日開盤；最後一期持有到資料最後一日。',
   'tradable': 'd+1 開盤漲停（開盤/前收−1 ≥ L − 0.6%）且未持有 → 買不到，由同方向排序下一名遞補（Q1 由低往高、Q5 由高往低）；已持有者照常保留。Q2～Q4 診斷組不遞補（直接剔除）。',
   'cost': '基本（判定用）：exlib.cost_rates(0.001)＝買 0.0855%＋0.1%、賣 0.0855%＋0.3%＋0.1%（來回約 0.67%）；另報 滑價 0（約 0.47%）、壓力（買 0.5%＋賣 0.5%）、毛報酬。周轉以漂移後市值計。',
   'empty': '排序股池 < 50 檔該月不換股（沿用持股）。',
 },
 'benchmarks': {'EW': '同池等權：同一換股日的排序股池（該訊號有值）、同一引擎、不扣成本（毛）；另報 EW 扣成本年化。',
                '0050': '0050 含息（finlab-txd etf_0050.csv）'},
 'windows': {'select': '2005-01-01～2019-04-30', 'select_7pct': '2005-01-01～2015-05-31', 'select_10pct': '2015-06-01～2019-04-30',
             'valid': '2019-05-01～2026-10-22', 'valid_2024_2025': '2024-01-01～2025-12-31',
             'extra_reported': ['valid_survivor_2019-05_2023-12', 'valid_official_signal（2019-05-01～2024-02-29 ＋ 2025-11-01～2026-10-22）', 'valid_2024-01_2026-10']},
 'stats': '年化（實際天數）、日頻最大回撤、波動、對 0050 與同池等權（毛）的日曆月超額報酬 t（btlib.tstat）、逐年、周轉。',
 'verdict_rule': {
   'U1': 'NET20 Q1（基本成本）在挑選期與驗證期都要：對同池等權（毛）月超額平均 > 0 且 t ≥ 2，且淨年化 ≥ 0050 同期年化 → 該段通過。',
   'U3': '排除版（UP20 ≥ 2，基本成本）在挑選期與驗證期都要：排除版 − 原版 淨月報酬配對差平均 > 0 且 t ≥ 2，且排除版淨年化 ≥ 0050 同期年化 → 該段通過。',
   'overall': '兩段都過 → 可用；只有一段過 → 待觀察；否則不可用。U1 與 U3 各自判定並列；總判定取 U1 與 U3 中較好者（兩個都是事前登記的用法，只有兩個檢定，於 FINDINGS 註明）。資料不足 → 無法測試。',
 },
 'known_limits': ['用收盤觸板（無最低價；漲停也只用收盤判定，不用盤中最高價），「打開的漲停板」不算觸板',
                  '漲停價有 tick 捨去，6.7%／9.7% 門檻對極少數跨 tick 級距的價格（如前收 9.3 元）會漏判',
                  '2019–2023 面板只有存活者（對高觸板／高 MAX 組驗證期偏樂觀）',
                  'src 3 期間收盤為 13:20 近似：觸板與漲跌幅判定失真（另報官方訊號段）',
                  '2024-01-02～2024-02-14 無資料（回看窗跨過缺口）',
                  '主策略長期營收 2013 起只剩存活公司；2013 IFRS 斷點', '只有上市股，無上櫃；組合等權，非論文的市值加權'],
}
if __name__ == '__main__':
    f = HERE / 'results.json'
    d = json.loads(f.read_text(encoding='utf-8')) if f.exists() else {}
    assert 'PREREG' not in d, 'PREREG 已存在，不覆寫'
    d = {'PREREG': PREREG, **d}
    f.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding='utf-8')
    print('PREREG written')
