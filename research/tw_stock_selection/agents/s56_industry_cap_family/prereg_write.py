#!/usr/bin/env python3
"""S56 事前登記：寫 SPEC.json、PREREG.json 與 sha256（在任何非現行格報酬計算之前執行；已存在就拒絕覆寫）。
用法：python3 -I -B prereg_write.py"""
import json, hashlib, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
for f in ('SPEC.json', 'PREREG.json'):
    assert not (HERE / f).exists(), f'{f} 已存在，拒絕覆寫'
NOW = datetime.datetime.now().isoformat(timespec='minutes')

SPEC = {
 "id": "s56_industry_cap_family",
 "title": "S56 同產業持股上限家族：同一 PIT 產業最多持有 2／3／4 檔（遞補／不遞補），回撤能否分散而報酬不掉太多",
 "written": f"{NOW}，寫於本題任何非現行格報酬計算之前（只讀過程式碼與既有 FINDINGS；現行格數字已知＝deadline_engine_fix ALL 48.76%／381 筆）",
 "literature": "Moskowitz & Grinblatt (1999) Do Industries Explain Momentum?, Journal of Finance（個股動能有很大一部分是產業動能）；課程講義（類股輪動）。",
 "prior_work_checked": {
  "grep": "grep『產業上限｜industry cap｜集中』：original_o12／o_lc_sector_rotation_pit 的『單一產業上限 3×B』是大型股產業超配權重上限，與本策略無關。",
  "S21": "產業共振加分（同業一起爆發才買／加分），不通過；本題不重做。",
  "S31（最接近）": "S31 CAP2：只看『當月買進名單』同一 PIT 產業最多 2 檔，剔除者從 L_MA 第 11～20 名遞補；已持有股票不計入；舊引擎（只開 EXRIGHT_FIX）；以年化報酬做訓練／驗證選擇，訓練期 −0.79 個百分點、未凍結。S31 描述：持股同產業 ≥ 3 檔天數 27.0%、單一產業最大占比 42.9%。",
  "dedup_decision": "本題與 S31 CAP2 不同：(1) 已持有股票計入上限；(2) 上限 2／3／4 三層 × 遞補／不遞補；(3) 正式開關全開的新引擎；(4) 以 Calmar 為主的家族檢查，不是訓練／驗證選規則。最接近的 C2_R 與 S31 CAP2 仍不相同（持股計入），所以 6 格全部新跑、全部計入 N。"
 },
 "industry_pit": {
  "source": "agents/o_lc_sector_rotation_pit/raw/industry_pit.csv（只讀、不複製；同 S21／S31 所用）",
  "layer": "證交所上市『產業別』單層（約 33 類，電子業已細分為半導體業、電腦及週邊設備業、光電業、通信網路業、電子零組件業、電子通路業、資訊服務業、其他電子業；2023-07-03 起新增綠能環保、數位雲端、運動休閒、居家生活）。沒有更細的子產業（例如 AI 伺服器、散熱分屬電腦及週邊／電子零組件等），所以本題的上限是『證交所產業別』上限。",
  "pit_rule": "每一列 code、start、end（空白＝至今）：日期 d 的產業＝start ≤ d < end 的那一段（多份快照逐段生效）。訊號列 k 用 P.IDX[k] 當天的產業；判定已持有股票的產業也用同一個訊號日的產業。",
  "missing": "『未分類』、空白或代碼不在檔內 → 缺值，不受上限限制，也不計入任何產業的檔數。"
 },
 "engine": {
  "chain": "s20lib → s17lib → s10lib（全部 import，不改任何既有檔案）；載入與訊號逐行照 agents/s54_market_filter_family/run.py／s48（其現行格已逐位重現 deadline_engine_fix ALL）。",
  "flags": "cslib.EXRIGHT_FIX＝True（load 前）；SURGE（STD）在 DEADLINE_FIX／INS15_MASK 關閉下建，之後兩者設 True，用 cslib.rev_avail_days(P, months, True)；s17lib.e50_fix_suspension。三個正式開關全開。",
  "universe_entry_exit": "股池 twse & ~創新板；F1＝make_signal_s10(depth=10, mom_lo=0.10, nsel=10)；出場 E0（跌出 L_MA 前 20 名）完全不動；simulate W10_70（sat_w=0.7, sizing='W10', lock=False）、嚴成本；20 條路徑＝資金 {60, 80, 100, 150, 200} 萬 × 位移 {0,1,2,3}；主跑 100 萬位移 0。",
  "cap_mechanism": "上限要看『當下持股』，所以放在引擎買進步驟：inspect 取 s10lib.simulate_s10 原始碼，只做字串替換後 exec（原檔不動），新參數 cap_n（None＝不限）、capind（T×N 產業代碼）、reserve（訊號列 → 遞補候選）、refill（布林）。執行日 t（訊號列 k＝t−1）買進時，依原引擎的 SURGE 順序逐檔看 F1 候選：若該檔產業（訊號日分類）目前持股檔數（pos：含已持有、含被標記待賣但當天沒賣掉者；不含當天已賣出者；含本日先買進者）≥ cap_n → 擋下（不買）。已持有股票不強制賣。",
  "refill": "遞補版：每擋下 1 檔 F1，就有 1 個遞補名額；遞補候選＝同一訊號列 L_MA（依 SURGE 排序）第 11～20 名中 RET20 > 10%、不在 F1、不在持股、當天沒賣出者，依 SURGE 由高到低；候選產業已達上限 → 跳過；送進原引擎買進流程（漲停、無開盤、資金、5% 成交值、零股、最小一半等規則照舊），實際買到才用掉名額。遞補只到第 20 名，所以新買的股票照同一條 E0 出場。不遞補版：擋下的名額空著，錢照原引擎掃進 0050。",
  "acceptance": "A2：現行格（cap_n＝None）主跑 2019 起 48.76%（0.4875617247721724）、已出場 381 筆、全期年化、全期回撤與 deadline_engine_fix/out/results.json ALL 逐位相同，20 條逐條相同。A3：副本引擎 cap_n＝None 時與 simulate_s10 淨值逐位相同（100 萬位移 0）。A4：副本引擎 cap_n＝99、refill＝True（上限永遠不會綁）淨值與現行逐位相同。A5：每個訊號列重建的 F1 與 make_signal_s10 的 targets 相同。任一不符 → 停止、不報結果。"
 },
 "family": {
  "caps": [2, 3, 4, "不限（現行）"],
  "refill": ["R：遞補", "N：不遞補"],
  "cells_run": ["C2_R", "C2_N", "C3_R", "C3_N", "C4_R", "C4_N"],
  "plus": "現行 NONE（不限）"
 },
 "metrics_per_cell": [
  "2019 起年化（2018 最後交易日 → 2026-10-22）：主跑、20 條中位數、最小、最大",
  "最大回撤：全期（2015-07 起）與 2019 起；主跑與 20 條中位數",
  "Calmar＝2019 起年化 ÷ |全期最大回撤|（同 S47／S54）；主跑與 20 條中位數",
  "逐條配對（對現行）：Calmar 差、2019 起年化差、全期回撤差（正＝變淺）的正數條數與中位數",
  "四段空頭（s17lib.SEGMENTS：2018、2020-01～06、2022、2024-07～2025-04）段內最大跌幅：20 條中位數與逐條差（正＝變淺）",
  "平均同產業最大持股數：每個交易日收盤持股依當天 PIT 產業分組取最大檔數，2019 起（及全期）平均；另報 ≥ 3 檔天數占比；20 條中位數",
  "擋下檔數、遞補買到檔數、已出場筆數、平均持股檔數"
 ],
 "outputs": "results.json（不存淨值）、run.py、prereg_write.py、SPEC.json／PREREG.json 與 sha256、FINDINGS.md（若可寫）。暫存 scratchpad/s56/（結束刪除）。",
 "resources": "單行程；峰值記憶體 ≤ 3GB；python3 -I -B。"
}

PREREG = {
 "id": "s56_industry_cap_family",
 "written": f"{NOW}，寫於本題任何非現行格報酬計算之前；規格見 SPEC.json（同時寫、一起算 sha256）。",
 "question": "同一 PIT 產業持股上限能否讓回撤分散（變淺）而報酬不掉太多？以 Calmar 為主判定。",
 "not_blind": "不是盲測：S31 已知『只看當月名單、上限 2、遞補』訓練期年化 −0.79 個百分點，持股同產業 ≥ 3 檔天數約 27%。本題 6 格在正式開關、持股計入下沒算過。",
 "primary（事前寫死）": {
  "statistic": "每條路徑 Calmar＝2019 起年化 ÷ |全期最大回撤|；與現行同一路徑配對差 dC（20 個）。",
  "較好": "dC 中位數 ≥ +0.10 且 dC > 0 的路徑 ≥ 15／20。",
  "較差": "dC 中位數 ≤ −0.10 且 dC > 0 的路徑 ≤ 5／20。",
  "差不多": "其他。",
  "明顯較好（列後續候選的條件）": "『較好』且 2019 起年化差中位數 ≥ −3 個百分點。只列為後續訓練／驗證候選，不改預設。"
 },
 "secondary（描述，不改主判定）": {
  "回撤": "全期最大回撤逐條差（正＝變淺）中位數與條數；四段空頭各段逐條差中位數與條數。",
  "集中度": "平均同產業最大持股數（2019 起）與 ≥ 3 檔天數占比，對現行。",
  "報酬（S17 規則）": "家族＝6 個新跑格＋現行共 7 格；P3＝現行 20 條中位數 − 7 格中位數的中位數；P3 ≥ 3 個百分點 →『現行偏樂觀（被挑過）』，否則『沒被挑過』。每格逐條配對 2019 起年化：≥ 15／20 正且中位數 ≥ +2 →『比現行好』；≤ 5／20 且 ≤ −2 →『比現行差』；其他『差不多』。",
  "遞補比較": "同一上限下 R 減 N 的 Calmar、年化、回撤差（描述）。"
 },
 "default_change": "不論結果都不改現行預設；明顯較好的格只列為後續訓練／驗證候選（需另立題）。",
 "acceptance": "見 SPEC.engine.acceptance（A2～A5）；任一不符 → 停止、不報結果。",
 "predictions（事前寫下，可被否證）": [
  "上限 4 幾乎不綁（擋下 < 15 檔），兩格都與現行差不多（年化差 |·| < 1 個百分點）。",
  "上限 2 綁得最多；年化少 1～5 個百分點（S31 CAP2 訓練期 −0.79，且產業動能是動能的一部分，擋掉同產業第 3 檔常是擋掉贏家）；全期回撤變化 < 3 個百分點；Calmar 『差不多』。",
  "遞補版比不遞補版年化高、回撤略深或相近（S21／S4：L_MA 第 11～20 名平均只有約 +1～2%，遞補本身不太賺）。",
  "6 格中 0 格『明顯較好』；2024-07～2025-04 段變淺不超過 2 個百分點（那段是全市場一起跌，不只是單一產業）。",
  "S17 報酬口徑：現行排前段，P3 < 3（沒被挑過）。"
 ],
 "data_snooping": {"added": 6, "note": "新跑 6 格；N 以研究中心彙總為準。"},
 "resources": "單行程、峰值記憶體 ≤ 3GB；python3 -I -B；暫存 scratchpad/s56/，結束刪除。"
}

for name, obj in (('SPEC', SPEC), ('PREREG', PREREG)):
    p = HERE / f'{name}.json'
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=1))
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    (HERE / f'{name}.sha256').write_text(f'{h}  {name}.json  {NOW}\n')
    print(name, h)
