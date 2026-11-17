#!/usr/bin/env python3
"""S57 事前登記：寫 SPEC.json、PREREG.json 與 sha256（在任何非現行格報酬計算之前執行；已存在就拒絕覆寫）。
用法：python3 -I -B prereg_write.py"""
import json, hashlib, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
for f in ('SPEC.json', 'PREREG.json'):
    assert not (HERE / f).exists(), f'{f} 已存在，拒絕覆寫'
NOW = datetime.datetime.now().isoformat(timespec='minutes')

SPEC = {
 "id": "s57_intramonth_exit",
 "title": "S57 月中出場檢查家族：檢查頻率 {每月換股日（現行）, 每週最後交易日, 每日} × 月中觸發 {收盤 ≤ 60 日均線, 連 2 日 ≤ 60 日均線, 低於買進價 −15%}，回撤能否變淺",
 "written": f"{NOW}，寫於本題任何非現行格報酬計算之前（只讀過程式碼與既有 FINDINGS；現行格數字已知＝deadline_engine_fix ALL 48.76%／381 筆，S56 現行 20 條中位數 46.40%、Calmar 中位數 1.379）",
 "prior_work_checked": {
  "S10（s10_exit_details）": "已做『每日 × 收盤 ≤ 60 日均線』＝DMA60（XV['E0'] | ~F_MA60，月中標記 'ma_daily'，次日開盤賣，月中不補買）。舊引擎（只開 EXRIGHT_FIX 前的版本，E0 主跑 48.13%）W10_70：20 條中位數 −1.42 個百分點（4／20 正）、回撤淺 +1.53（20／20），WN_70 回撤淺 7.27 但只在 2024-07～2025-04 段，2022 段較深、2024-06 前回撤較深。S10 也做了移動停損（自高點回落 15%／25%，每日或換股日看），不是『自買進價 −15%』。",
  "exit_rules（舊題）": "舊 SURGE 前 10% 策略、一般成本下測過固定停損 15%：變差，被停損賣掉的股票到下一換股日平均再漲。進場、引擎、成本都不同，不能直接比，但方向已知 → 本題 −15% 格不是盲測。",
  "S48／S49": "S48 是換股日的名次 N × 均線 M；S49 是趨勢濾網；都沒有月中檢查。",
  "dedup_decision": "『每日 × 收盤 ≤ 60 日均線』S10 已做 → 不列入本題新家族（不計入 N、不做判定），只在現行引擎重跑一次當『參考格 D_MA』（與 S10 定義差別：本題只在 G、MA60 都有值時觸發，見 trigger_defs）。新家族 5 格：W_MA、W_MA2、W_SL15、D_MA2、D_SL15。『每月 × 任何觸發』＝現行（換股日照 E0，月中不檢查），去重。"
 },
 "engine": {
  "chain": "s20lib → s17lib → s10lib（全部 import，不改任何既有檔案）；載入與訊號逐行照 agents/s56_industry_cap_family/run.py（＝s48／s54，其現行格已逐位重現 deadline_engine_fix ALL）。",
  "flags": "cslib.EXRIGHT_FIX＝True（load 前）；SURGE（STD）在 DEADLINE_FIX／INS15_MASK 關閉下建，之後兩者設 True、cslib.rev_avail_days(P, months, True)；s17lib.e50_fix_suspension。三個正式開關全開。",
  "universe_entry": "股池 twse & ~創新板；F1＝make_signal_s10(depth=10, mom_lo=0.10, nsel=10)；換股日出場 E0（跌出 L_MA 前 20 名，L_MA 已含還原收盤 > 60 日均線）完全不動。",
  "cell": "simulate W10_70（sat_w=0.7, sizing='W10', lock=False）、嚴成本 COSTS['strict']；20 條路徑＝資金 {60, 80, 100, 150, 200} 萬 × 位移 {0,1,2,3}；主跑 100 萬位移 0。",
  "mechanism": "均線類觸發：出場矩陣 X＝E0 的 X（只在訊號列有值）OR（月中檢查日 × 觸發條件），直接傳給 s10lib.simulate_s10（S10 DMA60 同一做法；非訊號列標記原因 'ma_daily'）。−15% 觸發：inspect 取 s10lib.simulate_s10 原始碼，只做字串替換後 exec（原檔不動），新參數 stop（0.15）、chk（長度 T 的月中檢查日布林）；在原有標記段 requal 之前加一條：chk[t] 且非訊號列 且 p['g'] < 1 − stop → 標記 'stop'。p['g']＝原引擎的持有期含息報酬指數（進場日收盤 g＝收盤 ÷ 買進開盤價，之後 ×(1＋R)），即『收盤（含息還原）相對買進開盤價』。",
  "sell_and_refill": "月中被標記 → 次一交易日開盤賣（原引擎規則：開盤 ≤ 跌停價或無開盤價順延；超過當日成交值 5% 分批；嚴成本）。月中不補買：賣出所得照原引擎在當天掃進 0050；下一個換股日照常依 F1 買滿空位（被賣的股票若再進 F1 可以買回）。換股日當天只用 E0 規則（月中觸發在訊號列不另外檢查）。",
  "check_days": "每週＝每個日曆週（ISO 年-週）在面板中的最後一個交易日；每日＝每個交易日。都排除訊號列（訊號列由 E0 處理）。",
  "trigger_defs": {
   "MA": "還原收盤 G ≤ 60 日簡單均線 rmean(G,60)（同 F_MA60 的反面），且 G 與均線都有值（停牌／無值不觸發）。",
   "MA2": "MA 在檢查日 t 與前一交易日 t−1 都成立。",
   "SL15": "收盤（含息還原）低於買進價 15%：p['g'] < 0.85。"
  },
  "cells": {"CUR": "現行（每月換股日、E0）", "W_MA": "每週 × MA", "W_MA2": "每週 × MA2", "W_SL15": "每週 × SL15", "D_MA2": "每日 × MA2", "D_SL15": "每日 × SL15", "D_MA（參考，S10 已做）": "每日 × MA"},
  "acceptance": "A1：現行格主跑 2019 起 48.76%（0.4875617247721724）、已出場 381 筆、全期年化、全期回撤與 deadline_engine_fix/out/results.json ALL 逐位相同，20 條逐條相同。A2：副本引擎 stop＝None 時與 simulate_s10 淨值逐位相同（100 萬，每個位移）。A3：副本引擎 stop＝1.0、chk 全 True（永遠不會觸發）淨值逐位相同。A4：每週檢查日每週恰一天、位在週內最後交易日。任一不符 → 停止、不報結果。另記錄（不擋）：D_MA 觸發矩陣與 S10 DMA60 定義（~F_MA60）不同的格數（只差在 G 或均線無值）。"
 },
 "metrics_per_cell": [
  "2019 起年化（2018 最後交易日 → 2026-10-22）：主跑、20 條中位數、最小、最大",
  "最大回撤：全期（2015-07 起）與 2019 起；主跑與 20 條中位數",
  "Calmar＝2019 起年化 ÷ |全期最大回撤|（同 S47／S54／S56）；主跑與 20 條中位數",
  "逐條配對（對現行）：Calmar 差、2019 起年化差、全期回撤差（正＝變淺）的正數條數與中位數",
  "四段空頭（s17lib.SEGMENTS：2018、2020-01～06、2022、2024-07～2025-04）段內最大跌幅：20 條中位數與逐條差（正＝變淺）",
  "交易：已出場筆數、買進筆數、月中觸發賣出筆數、年化單邊換手（(買進額＋賣出額)／2／平均淨值／年數，同 S10）、成本占淨值／年；20 條中位數與主跑",
  "事後描述（主跑）：月中被賣股票從賣出日收盤到下一個訊號列收盤的還原報酬，對同期 0050 的超額平均／中位數"
 ],
 "outputs": "results.json（不存淨值）、run.py、prereg_write.py、SPEC.json／PREREG.json 與 sha256、FINDINGS.md（若可寫）。暫存 scratchpad/s57/（結束刪除）。",
 "resources": "單行程；峰值記憶體 ≤ 3GB；python3 -I -B。"
}

PREREG = {
 "id": "s57_intramonth_exit",
 "written": f"{NOW}，寫於本題任何非現行格報酬計算之前；規格見 SPEC.json（同時寫、一起算 sha256）。",
 "question": "月中也檢查出場（每週或每日；均線、連 2 日均線、買進價 −15%）會不會讓回撤變淺？以 Calmar 為主判定。",
 "not_blind": "不是盲測：S10 已知每日季線（D_MA）在舊引擎報酬 −1.42、回撤淺 1.53 個百分點、好處只在 2025-04 一段；exit_rules 已知固定停損 15% 在舊策略變差。",
 "family": {"judged": ["W_MA", "W_MA2", "W_SL15", "D_MA2", "D_SL15"], "current": "CUR", "reference_not_judged": ["D_MA（S10 已做 DMA60；現行引擎重跑，只報數字，不計入 N、不做判定、不進 S17 家族中位數主算）"]},
 "primary（事前寫死）": {
  "statistic": "每條路徑 Calmar＝2019 起年化 ÷ |全期最大回撤|；與現行同一路徑配對差 dC（20 個）。",
  "較好": "dC 中位數 ≥ +0.10 且 dC > 0 的路徑 ≥ 15／20。",
  "較差": "dC 中位數 ≤ −0.10 且 dC > 0 的路徑 ≤ 5／20。",
  "差不多": "其他。",
  "明顯較好（列後續候選的條件）": "『較好』且 2019 起年化差中位數 ≥ −3 個百分點。只列為後續訓練／驗證候選，不改預設。"
 },
 "secondary（描述，不改主判定）": {
  "回撤": "全期最大回撤逐條差（正＝變淺）中位數與條數；四段空頭各段逐條差中位數與條數（S10 複製員建議：回撤看各段空頭逐段比）。",
  "交易": "筆數、月中觸發筆數、換手、成本，對現行。",
  "報酬（S17 規則）": "家族＝5 個新格＋現行共 6 格；P3＝現行 20 條中位數 − 6 格 20 條中位數的中位數；P3 ≥ 3 個百分點 →『現行偏樂觀（被挑過）』，否則『沒被挑過』（另報含 D_MA 的 7 格作敏感度）。每格逐條配對 2019 起年化：≥ 15／20 正且中位數 ≥ +2 →『比現行好』；≤ 5／20 且 ≤ −2 →『比現行差』；其他『差不多』。"
 },
 "default_change": "不論結果都不改現行預設；明顯較好的格只列為後續訓練／驗證候選（需另立題）。",
 "acceptance": "見 SPEC.engine.acceptance（A1～A4）；任一不符 → 停止、不報結果。",
 "predictions（事前寫下，可被否證）": [
  "D_MA（參考）在現行引擎與 S10 方向相同：年化少 0.5～2.5 個百分點、全期回撤淺 0.5～3 個百分點、Calmar 差不多。",
  "W_MA 介於現行與 D_MA 之間（觸發較少、較晚），Calmar 差不多。",
  "連 2 日（W_MA2、D_MA2）比單日少觸發，結果更接近現行；Calmar 差不多。",
  "−15% 停損（W_SL15、D_SL15）年化少 1～4 個百分點（賣在短期低點、切斷大贏家），全期回撤變化 < 3 個百分點，Calmar 差不多或較差；D_SL15 比 W_SL15 差。",
  "回撤變淺若有，集中在 2024-07～2025-04 段；2022 段變深或不變。",
  "5 格中 0 格『明顯較好』；S17 報酬口徑：現行排前段、P3 < 3（沒被挑過）。"
 ],
 "data_snooping": {"added": 5, "note": "新跑 5 格（D_MA 為 S10 已做的重跑，不加）；N 以研究中心彙總為準。"},
 "resources": "單行程、峰值記憶體 ≤ 3GB；python3 -I -B；暫存 scratchpad/s57/，結束刪除。"
}

for name, obj in (('SPEC', SPEC), ('PREREG', PREREG)):
    p = HERE / f'{name}.json'
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=1))
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    (HERE / f'{name}.sha256').write_text(f'{h}  {name}.json  {NOW}\n')
    print(name, h)
