#!/usr/bin/env python3
"""S54 事前登記：寫 SPEC.json、PREREG.json 與 sha256（在任何非現行格報酬計算之前執行；已存在就拒絕覆寫）。
用法：python3 -I -B prereg_write.py"""
import json, hashlib, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
for f in ('SPEC.json', 'PREREG.json'):
    assert not (HERE / f).exists(), f'{f} 已存在，拒絕覆寫'
NOW = datetime.datetime.now().isoformat(timespec='minutes')

SPEC = {
 "id": "s54_market_filter_family",
 "title": "S54 大盤濾網家族：0050 趨勢轉弱時暫停衛星新買進（已持有照 E0 出場），回撤能否變淺而報酬不掉太多",
 "written": f"{NOW}，寫於本題任何非現行格報酬計算之前（只讀過程式碼與既有 FINDINGS；現行格數字已知＝deadline_engine_fix ALL）",
 "literature": "Faber (2007) A Quantitative Approach to Tactical Asset Allocation, Journal of Wealth Management（10 個月均線趨勢濾網）；Daniel & Moskowitz (2016) Momentum crashes, JFE（空頭後動能崩跌）；課程講義（類股輪動、Beta）。",
 "prior_work_checked": {
  "S29": "BACKLOG 的 S29 是課程 QMV（品質／動能／估值排序），FINDINGS 內沒有大盤濾網或法人 regime；與本題不重疊。",
  "S22": "法人籌碼確認：資料不足（三大法人逐股只到 2021-10 起），沒做。",
  "S23（真正做過的 regime 題）": "現行衛星、只在換股日判斷：BRD_STOP_Q20／Q33（SURGE 廣度）、LMA_STOP_Q20／Q33、MA_HALF（0050 < 200 日線新買減半）、MA_STOP（0050 含息收盤 < 200 日均線 → 不新買、錢進 0050）、MA_EXIT（同上並清出衛星）。舊引擎（三個正式開關關閉，48.13%）。FINDINGS 明寫『放現金版沒做』。",
  "其他": "timing_switch（加權指數 < 200 日線 → 整個策略換現金／0050，舊主策略、全切換）、breadth_derisk（新高家數減碼到現金，舊 n10）、o_concentration_regime（0050 與主策略輪動）、S30／S38（固定降曝險，不是濾網）、S47（空位常態留現金 −8.15、回撤淺 8.6，沒有濾網）。",
  "dedup_decision": "8 格中『0050 > 200 日均線 × 暫停新買進（空位買 0050）』就是 S23 MA_STOP（同定義：0050 含息還原收盤 vs 含當日 200 日簡單均線、只在換股訊號日判斷、沒買的錢由引擎掃進 0050）→ 依指示不重做，只引用 S23 數字（舊引擎，與本題不能逐位比較），不計入本題新增 N。本題實際新跑 7 格：60／120 日均線、60 日報酬 × 空位買 0050（3 格），以及 4 個指標 × 空位留現金（4 格，S23 沒做過現金版）。"
 },
 "engine": {
  "chain": "s20lib → s17lib → s10lib（全部 import，不改任何既有檔案）；載入與訊號逐行照 agents/s48_exit_rule_family/run.py（其現行格已逐位重現 deadline_engine_fix ALL）。",
  "flags": "cslib.EXRIGHT_FIX＝True（load 前）；SURGE（STD）在 DEADLINE_FIX／INS15_MASK 關閉下建，之後兩者設 True，用 cslib.rev_avail_days(P, months, True)；s17lib.e50_fix_suspension。三個正式開關全開。",
  "universe_entry_exit": "股池 twse & ~創新板；F1＝make_signal_s10(depth=10, mom_lo=0.10, nsel=10)；出場 E0（XV['E0']）完全不動；simulate W10_70（sat_w=0.7, sizing='W10', lock=False）、嚴成本。",
  "filter_signal": "每條路徑的每個訊號列 k（訊號日收盤，次一交易日開盤執行），只用 ≤ k 的資料：E＝P.E50（0050 含息還原收盤，已做 e50_fix_suspension；用還原價是因為 2025-06 0050 1 拆 4，原始收盤不連續）。MA_n＝E[k−n+1..k] 簡單平均（需滿 n 筆，不足視為『不弱』）；弱＝E[k] ≤ MA_n（即『收盤 > 均線』不成立）。R60＝E[k] ÷ E[k−60] − 1；弱＝R60 ≤ 0。",
  "action_stop_0050": "弱的訊號列：當月買進名單 sig[k,:]＝False、rank[k,:]＝NaN（照 S23 apply_regime 'stop'），既有持股照 E0；沒買的錢與賣出的錢照原引擎掃進 0050。直接用 s10lib.simulate_s10。",
  "action_stop_cash": "同上不新買；另外從該訊號列的執行日（k+1）起到下一個訊號列執行日的前一天，衛星空位留現金（利率 0）：0050 部位只維持在底倉簿記 B（差 ≥ 2 萬才補買或賣回，照 S47 idle='cash' 已驗證的寫法），衛星賣股的錢留現金；下一個『不弱』訊號列的執行日起恢復原行為（現金 ≥ 2 萬整筆掃進 0050）。底倉 30%（B）不動。實作：inspect 取 s10lib.simulate_s10 原始碼，只做 3 處字串替換後 exec（原檔不動），新參數 cashmode（長度 T 的布林陣列）；全 False 時必須與 simulate_s10 逐位相同（驗收 A3）。",
  "paths": "20 條：資金 {60, 80, 100, 150, 200} 萬 × 換股日位移 {0,1,2,3}；主跑＝100 萬、位移 0。",
  "current_cell": "不設濾網＝現行；驗收 A2：主跑 2019 起 48.76%（0.4875617247721724）、已出場 381 筆、全期、全期回撤與 deadline_engine_fix/out/results.json ALL 逐位相同，20 條逐條相同（中位數 46.40%）。不符就停止、不報結果。"
 },
 "family": {
  "indicators": ["MA60：0050 > 60 日均線", "MA120：> 120 日均線", "MA200：> 200 日均線", "R60：0050 過去 60 日報酬 > 0"],
  "actions": ["E50：暫停新買進（空位買 0050）", "CASH：暫停新買進（空位留現金）"],
  "cells_total": 8,
  "cells_run": ["MA60_E50", "MA120_E50", "R60_E50", "MA60_CASH", "MA120_CASH", "MA200_CASH", "R60_CASH"],
  "cell_cited_not_run": {"MA200_E50": "＝S23 MA_STOP（舊引擎），引用其全期連續數字：2019 起年化差 −5.82 個百分點（0／20）、全期回撤淺 0.40（15／20）、2018 段淺 1.77、2020-03 段深 1.02、2022 段深 1.47、2024-07～2025-04 段淺 1.19（20／20）。"},
  "plus": "現行（不設濾網）"
 },
 "metrics_per_cell": [
  "2019 起年化（2018 最後交易日 → 2026-10-22）：主跑、20 條中位數、最小、最大",
  "最大回撤：全期（2015-07 起）與 2019 起；主跑與 20 條中位數",
  "Calmar＝2019 起年化 ÷ |全期最大回撤|（同 S47 定義）；主跑與 20 條中位數",
  "逐條配對（對現行）：Calmar 差、2019 起年化差、全期回撤差（正＝變淺）的正數條數與中位數",
  "四段空頭（s17lib.SEGMENTS：2018、2020-01～06、2022、2024-07～2025-04）段內最大跌幅：20 條中位數與逐條差（正＝變淺）",
  "觸發（弱）訊號列數：位移 0，2016 起與 2019 起；觸發月份清單；已出場筆數、平均持股、衛星現金平均占比"
 ],
 "outputs": "results.json（不存淨值）、run.py、prereg_write.py、SPEC.json／PREREG.json 與 sha256、FINDINGS.md（若可寫）。暫存 scratchpad/s54/（結束刪除）。",
 "resources": "單行程；峰值記憶體 ≤ 3GB；python3 -I -B。"
}

PREREG = {
 "id": "s54_market_filter_family",
 "written": f"{NOW}，寫於本題任何非現行格報酬計算之前；規格見 SPEC.json（同時寫、一起算 sha256）。",
 "question": "大盤（0050）趨勢轉弱時暫停衛星新買進，回撤（特別是 2024-07～2025-04）能否變淺而報酬不掉太多？以 Calmar 為主判定。",
 "not_blind": "不是盲測：S23 已知 0050 < 200 日線停買（錢進 0050）2019 起少約 5.8 個百分點、回撤只淺 0.4；S47 已知空位常態留現金少 8.15、回撤淺 8.6。其餘 7 格在正式開關下沒算過。",
 "primary（事前寫死）": {
  "statistic": "每條路徑 Calmar＝2019 起年化 ÷ |全期最大回撤|；與現行同一路徑配對差 dC（20 個）。",
  "較好": "dC 中位數 ≥ +0.10 且 dC > 0 的路徑 ≥ 15／20。",
  "較差": "dC 中位數 ≤ −0.10 且 dC > 0 的路徑 ≤ 5／20。",
  "差不多": "其他。",
  "明顯較好（列後續候選的條件）": "『較好』且 2019 起年化差中位數 ≥ −3 個百分點 且 2024-07～2025-04 段跌幅變淺 ≥ 15／20。只列為後續訓練／驗證候選，不改預設。"
 },
 "secondary（描述，不改主判定）": {
  "回撤": "全期最大回撤逐條差（正＝變淺）中位數與條數；四段空頭各段逐條差中位數與條數。",
  "報酬（S17 規則）": "家族＝7 個新跑格＋現行共 8 格；P3＝現行 20 條中位數 − 8 個格子中位數的中位數；P3 ≥ 3 個百分點 →『現行偏樂觀（被挑過）』，否則『沒被挑過』。每格逐條配對 2019 起年化：≥ 15／20 正且中位數 ≥ +2 →『比現行好』；≤ 5／20 且 ≤ −2 →『比現行差』；其他『差不多』。（濾網家族裡現行是『不設濾網』，P3 主要用來確認沒有任何濾網格在報酬上勝過現行。）",
  "動作比較": "同一指標下 CASH 減 E50 的 Calmar、年化、回撤差（描述）。"
 },
 "default_change": "不論結果都不改現行預設；明顯較好的格只列為後續訓練／驗證候選（需另立題、用新資料或事前切訓練／驗證）。",
 "acceptance": "A2：現行格逐位重現 deadline_engine_fix ALL（48.76%、381 筆、20 條逐條）；A3：cashmode 全 False 的副本引擎在 100 萬位移 0 與 simulate_s10 淨值逐位相同；A4：濾網全不觸發的訊號（sig 不改）與現行相同（由 A2 涵蓋）。任一不符 → 停止、不報結果。",
 "predictions（事前寫下，可被否證）": [
  "E50 版 3 格 Calmar 都『差不多』；年化少 1～6 個百分點，全期回撤變化 < 2 個百分點（S23：底倉 30% 與空位本來就在 0050，0050 自己 2022 段 −34%、2024-07 段 −27.5%）。",
  "CASH 版回撤明顯較淺（3～9 個百分點），年化少 3～8 個百分點；Calmar 可能較好，最可能是 MA200_CASH 或 MA120_CASH。",
  "MA60 觸發最多、來回最多，報酬代價最大或與 R60 相近；2020-03 段所有格都沒有變淺（V 型反彈錯過）。",
  "2024-07～2025-04 段：CASH 版變淺 ≥ 15／20；E50 版變淺但幅度 < 2 個百分點。",
  "S17 報酬口徑：P3 ≥ 3（現行在報酬上排第 1），但這只代表濾網都在付報酬，不是現行被挑過。"
 ],
 "data_snooping": {"added": 7, "note": "新跑 7 格；MA200_E50 已在 S23 計入。N 以研究中心彙總為準。"},
 "resources": "單行程、峰值記憶體 ≤ 3GB；python3 -I -B；暫存 scratchpad/s54/，結束刪除。"
}

for name, obj in (('SPEC', SPEC), ('PREREG', PREREG)):
    p = HERE / f'{name}.json'
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=1))
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    (HERE / f'{name}.sha256').write_text(f'{h}  {name}.json  {NOW}\n')
    print(name, h)
