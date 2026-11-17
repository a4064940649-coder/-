#!/usr/bin/env python3
"""S48 事前登記：寫 SPEC.json、PREREG.json 與 sha256（在任何非現行格報酬計算之前執行；已存在就拒絕覆寫）。
用法：python3 -I -B prereg_write.py"""
import json, hashlib, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
for f in ('SPEC.json', 'PREREG.json'):
    assert not (HERE / f).exists(), f'{f} 已存在，拒絕覆寫'
NOW = datetime.datetime.now().isoformat(timespec='minutes')

SPEC = {
 "id": "s48_exit_rule_family",
 "title": "S48 出場規則 E0（跌出 L_MA 前 20 名、或換股日收盤 ≤ 60 日均線就賣）的兩個數字有沒有被挑過——家族檢查（同 S45／S46）",
 "written": f"{NOW}，寫於本題任何非現行格報酬計算之前（只讀過程式碼與既有 FINDINGS；現行格的數字已知＝deadline_engine_fix ALL）",
 "engine": {
  "chain": "s20lib → s17lib → s10lib（全部 import，不改任何既有檔案）；做法逐行照 agents/s46_momentum_gate_family/run.py（其現行格已逐位重現 deadline_engine_fix ALL）。",
  "flags": "cslib.EXRIGHT_FIX＝True（load 前）；SURGE（s17lib.rev_family STD）在 DEADLINE_FIX／INS15_MASK 關閉下建，之後 cslib.DEADLINE_FIX＝True、cslib.INS15_MASK＝True，用 cslib.rev_avail_days(P, months, True) 算可得日；s17lib.e50_fix_suspension。三個正式開關全開。",
  "universe": "stlib.universe(P, 'twse') & ~創新板（s4lib.innov_panel）。",
  "entry": "F1 不動：s10lib.make_signal_s10(P, U, RT, shift, depth=10, mom_lo=0.10, nsel=10)（L_MA＝股池內、SURGE 有值、還原收盤 > 60 日均線，依 SURGE 排；前 10 名中 RET20 > 10%，不遞補）。進場的 60 日均線（P.F_MA60）不動。",
  "exit_code_location": "現行 E0 在 s10lib.make_signal_s10：每個訊號列 k，o_ma＝L_MA 依 SURGE 排序；XV['E0'][k,:]＝True、XV['E0'][k, o_ma[:20]]＝False（『20』寫死在 for key, n in (('E0', 20), …)）；60 日均線是 P.F_MA60＝P.G > P.MA60（s2lib.load，P.MA60＝stlib.rmean(P.G, 60)），與進場 L_MA 用的是同一個矩陣、同一個 o_ma 名單 → 在共用程式裡進場與出場的 60MA 是同一個參數，沒有獨立開關。s17lib 只負責 SURGE 與 0050 修正，不設定出場。",
  "decoupling": "可以分開（不改共用程式）：照 S10 已驗證的拆法（S10 驗收 A2：『SURGE ≥ 當月 L_MA 第 20 名分數 且 還原收盤 > 60 日均線』與『L_MA 前 20 名』不同格數 0），本題在 run.py 自己重建每個訊號列的出場矩陣 X，再交給 s10lib.simulate_s10（S['X'] 換掉，進場 sig／rank 不動）。所以本題只改出場那個 60MA，進場 L_MA 的 60MA 不動，不需要聯動版。",
  "exit_rule_cell": "出場格 (N, M)：每個訊號列 k（只在換股訊號日檢查，同 E0；月中不檢查）——fin＝股池內且當月 SURGE 有值（INS15 遮掉＝缺值，同 E0）；thr_N＝當月 L_MA（進場名單，60MA 篩過）第 N 名的 SURGE 分數，L_MA 不足 N 檔時 −∞；續抱＝fin 且 SURGE ≥ thr_N（N＝不用排名時省略此條件）且 還原收盤 > M 日簡單均線（M＝不用均線時省略此條件）；不續抱 → 標記待賣，次一交易日開盤賣（同 E0 引擎）。(20, 60) 必須與 XV['E0'] 逐格相同（驗收 A1，四個位移）。(N, 不用均線) ＝ S10 的 MANO 寫法（名次門檻仍取自 L_MA 名單），(N, 20) ＝ S10 的 MA20 寫法。",
  "ma_def": "M 日均線＝stlib.rmean(P.G, M)（min_periods＝M，P.G＝還原收盤，EXRIGHT_FIX 版）；F_MA_M＝P.G > MA_M（NaN 比較為 False）。(60) 必須與 P.F_MA60 逐格相同（驗收 A0）。",
  "simulate": "s10lib.simulate_s10(P, S', U, cost=COSTS['strict'], capital=資金, sat_w=0.7, sizing='W10', lock=False)＝W10_70，嚴成本，min_frac 0.5；不開 trail／requal。",
  "paths": "20 條：資金 {60, 80, 100, 150, 200} 萬 × 換股日位移 {0, 1, 2, 3}。主跑＝100 萬、位移 0。",
  "current_cell": "(N＝20, M＝60)；驗收：主跑 2019 起 48.76%（0.4875617247721724）、已出場 381 筆、全期與全期回撤，20 條逐條與 deadline_engine_fix/out/results.json ALL 逐位相同（中位數 46.40%）。不符就停止、不報結果。"
 },
 "family": {
  "exit_rank_N": [15, 20, 25, 30, "不用排名"],
  "exit_ma_M": [20, 40, 60, 120, "不用均線"],
  "excluded": "(不用排名, 不用均線)——沒有出場規則（只剩離開股池／營收缺值），不列入。",
  "n_cells": 24,
  "fixed": "F1 進場、W10_70、嚴成本、20 條路徑、只在換股訊號日檢查出場，全部不動。"
 },
 "metrics_per_cell": [
  "2019 起年化（2018 最後交易日 → 2026-10-22）：主跑、20 條中位數、最小、最大、第 10 百分位",
  "最大回撤：全期（2015-07 起）與 2019 起，主跑與 20 條中位數",
  "全期年化（主跑）",
  "交易數：主跑已出場筆數（why ≠ open_end）、買進筆數；20 條已出場筆數中位數",
  "平均持有天數：已出場交易 exit_t − ent_t（交易日）平均與中位數，主跑與 20 條中位數",
  "平均持股檔數（avg_npos）",
  "分段年化：2019-01～2023-12、2024-01 起（逐條配對）；2016-03～2018-12（2016-02 最後交易日 → 2018 最後交易日，事後描述，不影響判定）"
 ],
 "outputs": "results.json（全部格子與路徑，不存淨值）、run.py、SPEC.json／PREREG.json 與 sha256、FINDINGS.md（若可寫）。暫存 /tmp/claude-0/-home-user/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s48/（結束刪除）。",
 "resources": "單行程；峰值記憶體 ≤ 3GB；python3 -I -B；輸出 ≤ 50MB。"
}

PREREG = {
 "id": "s48_exit_rule_family",
 "written": f"{NOW}，寫於本題任何非現行格報酬計算之前。規格見 SPEC.json（同一時間寫、一起算 sha256）。",
 "question": "現行出場 E0 的兩個數字（L_MA 排名門檻前 20 名、出場均線 60 日）是不是看結果挑出來的？在事前列出的 24 格出場家族裡排第幾、比家族中位數高多少個百分點？",
 "history_checked_before_prereg（只讀文件與程式，沒有重算任何報酬）": {
  "how_current_was_chosen": [
   "60 日均線：S1／12 號腳本的季線慣例；S2 D1 事前登記『還原收盤 > 60 日均線才列入排名、不符合的持股換股日視同不在名單』→ 進場與出場一起用同一條 60MA。S2 事後跑 40／50／80／100／120 日（進出場聯動、舊 D1 基底），60 日反而是六個裡最低（41.56% 對 43.21%～45.83%），S2 寫明不拿來挑參數。",
   "前 20 名：S3 事前登記 E1『掉出前 20 名才賣』（= 2 × 持股數 10 的整數），E1 單獨『沒差』，與 E2 合併成 E3 後成為現行。之後沒有換過別的名次。",
   "因此 20 與 60 都是慣例整數，不是從鄰域挑最高；但 S10 已看過一維鄰域（見下）。"
  ],
  "neighbourhoods_already_seen": [
   "S10（舊引擎：無除權息、期限修正，W10_70，20 條）：E0 46.23%；R10 43.15%、R15 46.21%、R30 44.80%（排名線，60MA 固定）；MA20 47.45%、MANO（不看均線、名次門檻 20）48.87%（均線線，N＝20 固定）；E0 在 11 個出場排 6／11、判沒被挑過。MANO 依規則『比標準好』但 2019～2023 0／20，列可選不改預設。",
   "S10 沒有做過 N × M 的交叉，沒有 N＝25、M＝40／120、『不用排名』的版本；也沒有在正式開關（EXRIGHT／DEADLINE／INS15）下重跑。",
   "所以本題不是盲測：一維線上 6 格（R15、R30、MA20、MANO，含現行）的舊引擎數字已知；其餘 18 格盲。"
  ]
 },
 "family（事前列全，不挑最好）": "排名門檻 N ∈ {15, 20, 25, 30, 不用排名} × 出場均線 M ∈ {20, 40, 60, 120 日, 不用均線}，去掉（不用排名, 不用均線），共 24 格。現行＝(20, 60)。F1 進場、W10_70、嚴成本、20 條路徑不動。",
 "primary_statistics（事前指定）": {
  "P1 家族分布": "每格 20 條路徑 2019 起年化中位數（路徑中位數）；24 個路徑中位數的中位數、最小、最大；24 × 20＝480 條路徑的中位數與第 10 百分位。另報主跑、最大回撤（20 條中位數）、交易數、平均持有天數、平均持股。",
  "P2 現行名次": "現行在 24 格中依路徑中位數的名次（1＝最高）；依 20 條全期最大回撤中位數的名次（1＝最淺）。",
  "P3 現行減家族中位數": "現行路徑中位數 − 24 個路徑中位數的中位數（百分點）。",
  "P4 逐條配對": "每格對現行：同一條路徑 2019 起年化差（20 個）為正的條數與中位數；兩段（2019-01～2023-12、2024-01 起）逐條差中位數；全期最大回撤差（正＝變淺）。"
 },
 "decision_rules（事前寫死）": {
  "偏樂觀（主判定，24 格）": "P3 ≥ 3 個百分點 →「現行出場偏樂觀（被挑過）」，報酬預期改用家族 480 條路徑中位數（附第 10 百分位）；P3 < 3 →「沒被挑過（不必因出場打折）」。",
  "現行是不是剛好最好": "名次 1 →「是」；2～6（前四分之一）→「前段」；其他 →「不是」。",
  "次要子家族（只描述，不改主判定）": "A＝兩者都用的 4 × 4＝16 格（N ∈ {15,20,25,30} × M ∈ {20,40,60,120}）；排名線 5 格（M＝60，N ∈ {15,20,25,30,不用排名}）；均線線 5 格（N＝20，M ∈ {20,40,60,120,不用均線}）。各報現行名次與 P3，套同一條 3 個百分點規則，標「次要」。若主判定與 A 不同，兩個都寫，以主判定為準。",
  "比現行好／差（每格，描述）": "逐條配對：≥ 15／20 條為正且中位數 ≥ +2 個百分點 →「比現行好」；≤ 5／20 且中位數 ≤ −2 →「比現行差」；其他 →「差不多」。「比現行好」又兩段逐條差中位數都 > 0 → 標「比現行好且兩段都正」。",
  "改不改預設": "不論結果都不改現行預設（題目要求）。任何「比現行好」的格子是 23 格裡事後挑出的，只列為後續題目候選（要另立訓練／驗證題）；報酬預期不得直接用最好那格。",
  "鄰域平滑（描述）": "現行的 4 個一步鄰居（N 15／25、M 40／120）路徑中位數與現行的差；現行比 4 個鄰居都高 ≥ 2 個百分點 → 標「孤峰」。",
  "2016-03～2018-12 分段": "事後描述（參數決定前的期間），只報各格中位數與現行名次，不影響判定。"
 },
 "acceptance": "A0：rmean(P.G, 60) 重建的 F_MA60 與 P.F_MA60 逐格相同。A1：(20, 60) 自建出場矩陣與 make_signal_s10 的 XV['E0'] 在位移 0～3 逐格相同。A2：現行格主跑 2019 起年化、全期、全期回撤、已出場筆數與 deadline_engine_fix/out/results.json ALL 逐位相同（48.76%、381 筆），20 條逐條相同（中位數 46.40%）。任一不符 → 停止、不報結果。",
 "predictions（事前寫下，可被否證）": [
  "現行不是第 1；名次在中段（6～16／24）。依據：S10 舊引擎 MANO、MA20 都比 E0 高。",
  "P3 落在 −2～+2 個百分點；主判定『沒被挑過』。",
  "『不用排名』那一列（只看均線）是各列裡最差的，持有天數最長、交易數最少。",
  "『不用均線』那一欄在 2024 起比現行好、2019～2023 比現行差（S10 MANO 的型態）。",
  "N＝15 與 N＝20 差 < 1 個百分點；N＝30 較差、回撤較深。",
  "均線越長，平均持有天數越長。"
 ],
 "data_snooping": {
  "added": 23,
  "note": "24 格扣掉現行。N 以研究中心彙總為準（S46 後約 2,948 → 約 2,971，若 S47 另計則順延）。不做 SPA／DSR。"
 },
 "resources": "單行程、峰值記憶體 ≤ 3GB；python3 -I -B；暫存 scratchpad/s48/，結束刪除；輸出只存 results.json（不存淨值）。"
}

for name, obj in (('SPEC.json', SPEC), ('PREREG.json', PREREG)):
    p = HERE / name
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=1))
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    (HERE / name.replace('.json', '.sha256')).write_text(f'{h}  {name}\n')
    print(name, h)
