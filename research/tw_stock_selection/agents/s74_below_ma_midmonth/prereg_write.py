#!/usr/bin/env python3
"""S74 事前登記：寫 SPEC.json、PREREG.json 與 sha256（寫於任何事件報酬計算之前）。用法：python3 -I -B prereg_write.py"""
import json, hashlib
from pathlib import Path
HERE = Path(__file__).resolve().parent
SPEC = {
 "id": "s74_below_ma_midmonth",
 "title": "S74（描述性，不改規則）：月中跌破 60 日均線的持股，撐到換股日是漲回來還是繼續跌？依跌破深度、距換股日天數、買進以來賺賠、大盤位置分組的事件描述",
 "written": "2026-11-19，寫於本題任何事件報酬計算之前（只讀過 S57／S10 的 FINDINGS、results.json 與程式碼）",
 "prior_work_checked": {
  "S57（s57_intramonth_exit）": "已做月中出場規則家族（每週／每日 × 收盤 ≤ 60 日均線、連 2 日、−15% 停損）；結論不改預設。事件面只報過一個總數：月中被賣股票賣出日收盤 → 下個訊號列收盤對 0050 的超額（D_MA 135 筆平均 −0.36%、中位數 −0.37%、勝率 44%）。results.json／results_posthoc.json 沒有任何依深度、天數、賺賠、大盤分組的數字。",
  "S10（s10_exit_details）": "DMA60（每日看季線）規則回測；『只因跌破季線在換股日被賣』41 筆之後一個換股日超額 +3.56%（換股日事件，不是月中事件）；DMA60 月中賣出 134 筆之後 −0.27%。沒有分組。",
  "dedup_decision": "不重做任何規則回測（不跑 D_MA、D_MA2 等）。只在現行引擎主跑的實際持股上做事件描述與分組；規則對照只引用 S57 既有數字。"
 },
 "engine": {
  "load": "載入與訊號逐行照 agents/s57_intramonth_exit/run.py 的現行格（＝s56／s48）：cslib.EXRIGHT_FIX＝True（load 前）；SURGE（STD）在 DEADLINE_FIX／INS15_MASK 關閉下建，之後兩者設 True、rev_avail_days(P, months, True)；s17lib.e50_fix_suspension。全部 import，不改任何既有檔案。",
  "path": "標準路徑＝主跑：資金 100 萬、位移 0、W10_70（sat_w=0.7, sizing='W10', lock=False）、嚴成本、出場 E0。只跑這一條。",
  "acceptance": "A1：2019 起年化、已出場筆數、全期年化、全期最大回撤與 deadline_engine_fix/out/results.json standard.ALL 逐位相同（48.76%／381 筆）。不符 → 停止、不報結果。"
 },
 "sample": "主跑交易紀錄的全部持有期間（含期末未出場），面板起點（2016-02）～2026-10-22。持有期間的『持有收盤日』＝進場日 entry_t 到出場日前一日 exit_t−1（期末未出場到最後一日）。",
 "event_def": {
  "below": "還原收盤 G ≤ 60 日簡單均線 stlib.rmean(G,60)，兩者都有值（同 S57 的 MA 觸發）。",
  "first_breach": "每個持有期間內第一個 below 的持有收盤日 t*。",
  "event": "t* 不是訊號列 → 月中事件（本題樣本）。t* 是訊號列 → 歸『首次跌破就在訊號日』（規則當天就會標記賣出，不是月中事件，只報筆數）。整段都沒跌破 → 『從未跌破』。每個持有期間最多一個事件。",
  "next_rebalance": "k＝t* 之後第一個訊號列；換股執行日 e＝k+1。沒有 k 或 e 超出資料 → 排除（報筆數）。e 日該股無開盤價 → 改用其後第一個有開盤價的交易日（報筆數），0050 同日對齊。"
 },
 "returns": {
  "主指標 hold_excess": "股票：G[e−1]/G[t*] × O[e]/REF[e] − 1（事件日還原收盤 → 換股執行日開盤，含息）；0050：E50O[e]/E50[t*] − 1；超額＝兩者相減。未扣成本（兩邊都遲早要賣，賣出成本相同）。正＝撐到換股日比『事件日收盤就換成 0050』好。",
  "副指標 hold_excess_nextopen": "起點改成事件次日開盤（散戶實際能賣的價位）：股票 G[e−1]/G[t*] × O[e]/REF[e] ÷ (O[t*+1]/REF[t*+1]) − 1；0050：E50O[e]/E50O[t*+1] − 1。t*+1 無開盤價者排除。",
  "也報": "股票自身報酬（不減 0050）的平均、中位數、上漲比例。"
 },
 "stats_per_group": "樣本數 n、平均、中位數、勝率（超額 > 0 的比例）、P10、P90（numpy percentile 線性內插）、涵蓋的換股月份數、以換股月份（k）分群的群聚 t 值（平均 ÷ 群聚標準誤）。n < 20 標示『不可靠』。",
 "groups": {
  "深度 depth＝G/MA60−1（事件日）": ["0～−2%（depth > −0.02）", "−2～−5%（−0.05 < depth ≤ −0.02）", "−5～−10%（−0.10 < depth ≤ −0.05）", "< −10%（depth ≤ −0.10）"],
  "距換股執行日交易日數 D＝e−t*": ["≤5", "6～10", ">10"],
  "買進以來賺賠": "事件日還原收盤 ÷ 買進日還原開盤（G[entry_t−1]×O/REF）≥ 1＝賺，< 1＝賠",
  "大盤": "事件日 0050 還原收盤 E50 ≤ 自己的 60 日均線＝大盤也在季線下；否則在季線上",
  "時期（次要）": "事件日在 2023 年底以前／2024 年起（S57、S10 都發現月中出場的好壞前後期相反）"
 },
 "rebalance_outcome": {
  "sold": "訊號列 k 上 E0 出場矩陣 X[k,j]＝True（會在 e 開盤賣）。再拆：G[k] ≤ 均線（仍在季線下）／G[k] > 均線（已站回但名次掉出等）。",
  "kept": "X[k,j]＝False（已站回季線且仍在前 20 名）→ 續抱。之後報酬：e 開盤 → 實際出場日開盤（交易紀錄 exit_t；期末未出場算到最後收盤），含息，對同期 0050 超額。",
  "也報": "k 日已站回季線的比例（不論賣不賣）。"
 },
 "rule_reference": "只引用 agents/s57_intramonth_exit/results.json 的 D_MA、D_MA2（20 條中位數對現行的年化、回撤、Calmar 差與判定），不新跑。",
 "outputs": "run.py、results.json（含逐事件明細）、SPEC.json／PREREG.json 與 sha256、code.sha256；FINDINGS.md 若不能寫則全文放最後回覆。暫存 scratchpad/s74/（結束刪除）。",
 "resources": "單行程；峰值記憶體 ≤ 3GB；python3 -I -B；不連外網。"
}
PREREG = {
 "id": "s74_below_ma_midmonth",
 "nature": "描述性事件研究，不產生規則建議、不改預設、不計入規則家族的 N。無論結果如何，結論都不能寫成『應該月中先賣』或『應該續抱』的規則（規則層級的答案以 S57 為準）。",
 "primary": "全部月中事件的 hold_excess（事件日收盤 → 換股執行日開盤，對 0050 超額）。",
 "wording_rule_overall": {
  "幫忙": "平均 > 0 且群聚 t ≥ 2 且中位數 > 0",
  "幫倒忙": "平均 < 0 且群聚 t ≤ −2 且中位數 < 0",
  "看不出差別": "其他情況（包括平均與中位數方向不同）"
 },
 "wording_rule_groups": "某一組『明顯不同』須同時：該組 n ≥ 20；該組平均與『同一分組維度其餘事件』平均的差，以換股月份分群的群聚 t 絕對值 ≥ 2.5（4 個維度共 11 組＋時期 2 組，事後多重比較，所以門檻高於 2）；且中位數差同方向。否則寫『看不出明顯不同』。n < 20 的組一律標『不可靠』，不下結論。",
 "depth_question": "『季線下越深越該先賣』的回答方式：看四個深度組平均超額是否單調遞減，以及最深兩組（≤ −5%）合併對較淺兩組的差是否達上述門檻；兩者都成立才說『有跡象』，否則說『資料不支持』。",
 "limits_to_state": ["事後分組多重比較", "事件在時間上扎堆（同月多檔一起跌破），有效樣本少於事件數", "單一路徑（100 萬、位移 0）", "未扣成本與賣出後再買回的摩擦", "描述不是規則：賣掉以後錢放哪、換股日買不買回，要看 S57 的規則回測"],
 "expected_before_run": "依 S57 總數（−0.4%～+0.1%）預期整體落在『看不出差別』；深度分組沒有先驗方向。"
}
for n, o in (('SPEC', SPEC), ('PREREG', PREREG)):
    p = HERE / f'{n}.json'
    p.write_text(json.dumps(o, ensure_ascii=False, indent=1))
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    (HERE / f'{n}.sha256').write_text(f'{h}  {n}.json\n')
    print(n, h)
