#!/usr/bin/env python3
"""S72 事前登記：寫 SPEC.json、PREREG.json 與 sha256（在任何安慰劑報酬計算之前執行；已存在就拒絕覆寫）。
用法：python3 -I -B prereg_write.py"""
import json, hashlib, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
for f in ('SPEC.json', 'PREREG.json'):
    assert not (HERE / f).exists(), f'{f} 已存在，拒絕覆寫'
NOW = datetime.datetime.now().isoformat(timespec='minutes')

SPEC = {
 "id": "s72_surge_placebo",
 "title": "S72 安慰劑檢定：現行策略的超額有多少來自月營收驚奇 SURGE 排名、有多少來自「站上季線＋近 20 日漲 >10%＋集中持股」這個殼",
 "written": f"{NOW}，寫於本題任何安慰劑報酬計算之前。此時只跑過 run.py --quick（現行格驗收：訊號矩陣四個位移逐格相同、主跑逐位重現 48.76%／381 筆、單次計時），沒有算過任何安慰劑的報酬。",
 "type": "描述性研究，不改策略、不新增可選方案、不計入嘗試次數 N。",
 "dedup（只讀文件與程式）": {
  "grep": "BACKLOG.md 與 agents/ 搜尋 安慰劑｜placebo｜隨機排名｜亂數排名｜shuffle｜permut。BACKLOG.md 無命中。",
  "found": {
   "o_canslim_wide_exit／o_ext_vcp_xs": "同日期、同筆數改買『通過趨勢樣板的隨機股票』的安慰劑——對象是 VCP／CANSLIM 型態，不是現行 SURGE 策略。",
   "o_largecap_leg／o_lc_sector_*／o_capmatched_*／o_0050_tilt_pit": "大型股腿：池等權 30% 安慰劑、產業內／市值組內重排隨機分數（t 值）——對象是大型股腿，不是現行主策略。",
   "o_msci_inclusion_avoid／o_disposal_rejudge／o_cb_board_runup／o_concentration_regime": "事件研究的安慰劑事件（臺灣50 納入、注意股未處置、隨機排列狀態區段）——與本題無關。",
   "s20_persistent_growth": "『隨機篩掉同樣比例』對照（各 20 次）：在 SURGE 名單裡隨機刪股，不是把排名換成隨機。",
   "s64_retail_behavior_tax": "『隨機只買一半』（200 次 × 20 條）：SURGE 名單內隨機少買，不是換排名。",
   "s28／s70": "禁買規則對『隨機剔除同檔數』：同上，名單內隨機刪。",
   "s9（複製員）": "隨機挑 49 個月不換股：時間上的隨機，不是排名。",
   "s60_factor_attribution": "迴歸歸因：四因子＋營收動能後 α +5.16（t 1.18）不顯著——本題的動機；做法是因子迴歸，不是安慰劑。",
   "s46_momentum_gate_family／s49": "動能門檻、進場均線的參數家族（殼的參數），SURGE 排名不動。"
  },
  "conclusion": "現行 SURGE 策略沒有做過『把 SURGE 排名整個換成隨機／純動能排名、其餘不動』的安慰劑。本題新做，正式開關全開。"
 },
 "engine": {
  "chain": "s20lib → s17lib → s10lib（全部 import，不改任何既有檔案）；載入與開關逐行照 agents/s48_exit_rule_family/run.py（＝deadline_engine_fix ALL＝official_nav 的 E0｜W10_70）。",
  "flags": "cslib.EXRIGHT_FIX＝True（load 前）；SURGE（s17lib.rev_family STD）在 DEADLINE_FIX／INS15_MASK 關閉下建，之後兩者＝True，cslib.rev_avail_days(P, months, True)；s17lib.e50_fix_suspension。",
  "universe": "stlib.universe(P, 'twse') & ~創新板。",
  "signal_builder": "run.py 的 build(shift, fn)＝s10lib.make_signal_s10(depth=10, mom_lo=0.10, nsel=10) 的逐行複製（只留 E0），唯一差別：排序分數由 SURGE（s）換成 q。每個訊號月：合格＝股池內 且 SURGE 有值（INS15 遮罩後）且 q 有值；L_MA＝合格 且 還原收盤 > 60 日均線，依 q 由大到小（stable）；進場＝L_MA 前 10 名中 RET20 > 10%（不遞補）；出場 E0＝不在 L_MA 前 20 名（含跌破 60 日均線、離開股池、SURGE 缺值）；買進順序 rank＝q。月份是否有訊號（可得日、股池內 SURGE 有值 ≥ 50 檔）的判斷照舊用 SURGE。",
  "simulate": "s10lib.simulate_s10(P, S, U, cost=COSTS['strict'], capital, sat_w=0.7, sizing='W10', lock=False)＝W10_70、嚴成本、A10 換股日。",
  "paths": "標準路徑＝100 萬、位移 0；20 條＝{60, 80, 100, 150, 200} 萬 × 位移 {0, 1, 2, 3}。",
  "acceptance": "A1：q＝SURGE 時 build() 的 sig／rank／X／每月名單與 s10lib.make_signal_s10 四個位移逐格相同。A2：主跑 2019 起 0.4875617247721724（48.76%）、已出場 381 筆、全期年化與全期回撤與 deadline_engine_fix/out/results.json ALL 逐位相同；20 條 2019 起逐條相同。不符就停止。"
 },
 "placebos": {
  "P1 隨機排名": "q＝均勻亂數，每個訊號月對全部股票重抽（np.random.default_rng([72, 1, 次序]).random((月數, 股票數))；同一次抽樣的 20 條路徑共用同一組分數）。300 次跑標準路徑；其中前 30 次（次序 0～29）另跑 20 條路徑。",
  "P2 隨機但保留延續性": "每檔股票的隨機分數持續 3 個月才重抽：每檔股票抽一個相位 φ∈{0,1,2}（rng([72, 2, 次序])），月序 mi 滿足 (mi＋φ) mod 3＝0 的月份才換新亂數，否則沿用上月（各股票錯開，不是全體同月重抽）。300 次、標準路徑。另報實際換手（已出場筆數、平均持有天數、年化單邊換手＝引擎 buy_val ÷ 全期平均淨值 ÷ 年數）與現行對照。",
  "P3 純動能排名（各 20 條路徑，確定性）": {
   "P3_RET20": "q＝P.RET20[訊號列]（引擎的近 20 日報酬，與進場門檻同一個量）",
   "P3_RET60": "q＝還原收盤 G[k] ÷ G[k−60] − 1",
   "P3_MOM12_1": "q＝G[k−21] ÷ G[k−252] − 1。面板自 2015-01-05 起，k < 252 的訊號月（約 2016-01 以前）分數缺值 → 當月沒有名單（資金在 0050）；所以這一格的『全期』數字不可比，只看 2016 起各段。"
  },
  "P4 反向（20 條）": "q＝−SURGE（SURGE 最低的排前面）。",
  "P5 營收年增率原始值（20 條）": "q＝當月營收 ÷ 去年同月 − 1（營收 ≤ 0 缺值、2013 年各月缺值，同 s17lib 的讀法；不做 μ24／σ24 標準化）。合格名單仍要求 SURGE 有值，所以股票集合與現行相同，只有排序不同。",
  "unchanged": "RET20 > 10%、E0（前 20 名＋60 日均線）、W10_70（衛星 70%、每檔 10%、空位放 0050）、嚴成本、A10 換股日、INS15 遮罩、股池。"
 },
 "metrics": [
  "2019 起年化（2018 最後交易日 → 2026-10-22）、全期年化（2015-07-01 起）、最大回撤（全期與 2019 起）",
  "對 0050 超額＝策略年化 − 0050 年化（同期間；0050 為引擎含息序列 P.E50，e50_fix_suspension 後），單位百分點",
  "三段（描述）：2016～2018（2015 最後交易日 → 2018 最後交易日）、2019～2023、2024 起",
  "隨機安慰劑：分布的中位數、P5、P95、最小、最大；現行的百分位＝安慰劑 < 現行的比例；單尾 p＝(1＋安慰劑 ≥ 現行的次數) ÷ (次數＋1)",
  "換手：已出場筆數、平均持有交易日、平均持股檔數、年化單邊換手、每月名單平均檔數",
  "確定性安慰劑（P3～P5）：標準路徑與 20 條中位數、對現行逐條配對差、在 P1 分布中的百分位"
 ],
 "outputs": "results.json、run.py、prereg_write.py、SPEC.json／PREREG.json 與 sha256、code.sha256、FINDINGS.md（若可寫）。暫存 /tmp/claude-0/-home-user/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s72/（結束刪除）。",
 "resources": "預設單行程；峰值記憶體 ≤ 3GB；python3 -I -B；只用本地資料、不連外網。"
}

PREREG = {
 "id": "s72_surge_placebo",
 "written": f"{NOW}，寫於任何安慰劑報酬計算之前。規格見 SPEC.json（同一時間寫、一起算 sha256）。",
 "question": "把 SURGE 排名換成隨機排名（其餘規則完全不動），現行 2019 起 48.76% 落在安慰劑分布的哪裡？殼本身賺多少、SURGE 多賺多少？",
 "primary": {
  "statistic": "標準路徑（100 萬、位移 0）2019 起年化；P1 隨機排名 300 次。",
  "p": "單尾 p＝(1＋#{P1 ≥ 現行}) ÷ 301。",
  "gap": "現行 − P1 中位數（百分點）。"
 },
 "verdict_rule（事前寫死）": {
  "SURGE 有明確貢獻": "p < 0.05 且 現行 − P1 中位數 ≥ 5 個百分點",
  "有貢獻但小": "p < 0.05 但 差 < 5 個百分點",
  "看不出 SURGE 的貢獻": "其他（p ≥ 0.05）"
 },
 "decomposition": "殼＝P1 中位數 − 0050（2019 起年化差，百分點）；SURGE＝現行 − P1 中位數。全期另報一次（描述）。",
 "descriptive_only（不影響判定）": [
  "全期年化、最大回撤、對 0050 超額的 P1 分布與現行位置",
  "30 次 × 20 條：每次的 20 條中位數分布，對現行 20 條中位數 46.40%",
  "P2（延續 3 個月的隨機分數）：同一套統計＋換手對照；若 P2 與 P1 判定不同，兩者都報，判定仍以 P1 為準，並在結論註明",
  "P3（RET20／RET60／12-1 月動能）、P4（反向）、P5（YoY 原始值）：對現行的差與在 P1 分布中的百分位",
  "三段（2016～2018、2019～2023、2024 起）各報 P1 分布與現行位置"
 ],
 "expectations（事前猜測，寫下來供事後對照，不是判定）": "S60 說 58% 的超額可由因子（主要是動能＋營收動能）解釋，所以猜：殼（P1 中位數）明顯贏 0050，SURGE 再多幾個百分點；P4 反向應低於 P1 中位數；P5 YoY 應接近現行。",
 "multiple_testing": "判定只看一個事前指定的統計量（P1、標準路徑、2019 起）。其餘全是描述，不各自下顯著結論。",
 "caveats_known_in_advance": [
  "現行 48.76% 本身是多輪研究後留下來的規格（S52：資料窺探校正後不顯著；S46：動能門檻偏樂觀）；本題的 p 只回答『在同一個殼裡，SURGE 排名是否比隨機排名好』，不是策略整體是否顯著。",
  "隨機排名每月重抽 → 換手比現行高、成本較重，對 P1 不利；所以另做 P2（延續 3 個月）與換手對照。",
  "合格名單仍要求 SURGE 有值（有 24 個月以上營收歷史的公司），殼包含這個條件。",
  "樣本只有 2015-07～2026-10 一段歷史，集中持股 10 檔，個股運氣成分大；P1 分布寬度就是這個運氣的量。"
 ],
 "no_change": "不論結果，不改策略、不改網站數字；不動共用引擎、既有研究、raw/。"
}

for name, obj in (('SPEC', SPEC), ('PREREG', PREREG)):
    p = HERE / f'{name}.json'
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=1))
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    (HERE / f'{name}.sha256').write_text(f'{h}  {name}.json\n')
    print(name, h)
h = hashlib.sha256((HERE / 'run.py').read_bytes()).hexdigest()
(HERE / 'code.sha256').write_text(f'{h}  run.py\n')
print('run.py', h)
