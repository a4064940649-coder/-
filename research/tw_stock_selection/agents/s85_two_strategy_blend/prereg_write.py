#!/usr/bin/env python3
"""S85 事前登記：寫 PREREG.json 與 PREREG.sha256（已存在就拒寫）。在算任何合併結果之前執行。"""
import json, hashlib, datetime
from pathlib import Path
H = Path(__file__).resolve().parent
assert not (H / 'PREREG.json').exists(), '已登記，拒絕覆寫'
P = {
 "id": "S85", "topic": "兩套策略合併／閒置資金放哪：能不能降低對大贏家與冷環境的依賴（不改 SURGE 選股規則）",
 "written_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
 "seen_before_prereg": [
  "各候選原研究的 FINDINGS（含 2019 起、2024 起成績與『每條路徑都輸現行』等結論）；S11／S14／S15 的『各半合併、不再平衡』已做過（都不加）；S30 防禦底倉、S47／S38 留現金、S66 ETF、strategy_ensemble 已做過。驗證段不是盲的，照實揭露。",
  "第 0 步（surge_sleeve.py、build_candidates.py）印出的全期期末淨值：E0 2,832 萬、E0 禁買前 5% 交易 893 萬、DISP_D 1,870 萬、PULLBACK_P 384 萬、PEAD_Q 657 萬；溫度計冷月數：訓練 17、驗證 7。",
  "尚未計算：任何候選在訓練／驗證段的分段統計、任何相關係數、任何合併結果。"],
 "data": {
  "surge": "現行 E0｜W10_70 標準路徑（100 萬、位移 0），surge_daily.csv.gz（與 xq_scripts/site/navs_official.csv.gz 逐日差 < 0.01 元）；含逐日個股占比 stockfrac、換股執行日。",
  "bench": "navs_official.csv.gz 的 0050 欄（含息）。",
  "thermometer": "s83_environment_thermometer/thermometer_monthly.csv 的 state（冷／溫／熱；當月同步讀數，門檻為 S83 事後訂）。",
  "comparable_candidates（同引擎 s10lib、嚴成本、正式三開關、100 萬、W10_70 帳戶、空位放 0050；可判定）": {
   "PEAD_Q": "S11 主格 Q（季報驚喜 SUE⊥ 前 10 名，季換股）；cand_navs_official_switch.csv.gz",
   "DISP_D": "S14 主格 D（上市第一次處置，起日開盤買、處置結束隔日賣，再處置延長）；同上",
   "PULLBACK_P": "S15 主格 P（強勢股拉回、固定持有 20 日）；同上",
   "OTC_SURGE": "S68 OTC（同一套 SURGE 規則只做上櫃）；s68_otc_universe_oos/navs.csv.gz 的 OTC 欄"},
  "reference_candidates（成本或引擎口徑不同；只列參考，不得直接判『值得獨立複製』，最多寫『值得先用同口徑重做』）": {
   "MAIN_N10": "主策略 V7X 可執行引擎、100 萬、官方換股日 j＝0、一般成本：o_g6_stage3/sealed/v7x_navs_jmean_100w.csv.gz 欄 base|REAL_opt|0|normal|100萬|std（敏感度：同欄 stress）",
   "CANSLIM_W": "o_canslim_wide_exit/navs.csv.gz 欄 T15_MA|E50（主規格、空位放 0050、一般成本）",
   "MA200_0050": "w4_lev2_half_rebalance/nav.parquet 欄 BENCH_MA200_0050（0050 站上 200 日線才持有，否則現金；擇時非選股）"},
  "describe_only（只進相關性與分環境表，不做合併；合併已由前題做過或屬同一因子）": {
   "LOWB／LBQ／HDLV／MIX": "s30_defensive_base/basket_navs.csv.gz（S30 已做『底倉＋空位換防禦組合』：不通過）",
   "H52_N10": "momentum_family/long_nav.parquet 欄 P1_52wHigh_n10_stress（到 2026-10-12）",
   "N30／TURN_C／MINV": "strategy_ensemble/strategy_nav.parquet（到 2026-10-12；strategy_ensemble 已做 11 條組合：不可用）"},
  "calendar": "一律對齊正式淨值的交易日；候選缺日用前一日淨值補（報酬 0）。日曆月報酬＝月底淨值相除；2026-10 算到 10-22（describe_only 中到 10-12 的三條，2026-10 不計）。"},
 "windows": {"train": ["2016-02", "2021-12"], "valid": ["2022-01", "2026-10"], "nav_base": "各段第一個月的前一個月最後交易日收盤"},
 "structures": {
  "A_x（分資金，x＝給 SURGE 帳戶的比例）": "總資金 x 給現行 W10_70 帳戶、(1−x) 給候選帳戶；每個日曆月最後交易日收盤拉回目標比例；拉回成本＝2×κ×被移動金額（賣方與買方各 κ）。x ∈ {0.5, 0.7}。",
  "B_k（閒置資金與底倉改放候選，k＝改放比例）": "SURGE 個股部位完全不動；帳戶裡原本放 0050 的錢（底倉＋空位，＝1−stockfrac）有 k 改放候選帳戶。日報酬＝s_t＋k×(1−w_{t−1})×(c_t−b_t)−k×κ×|w_t−w_{t−1}|×[t 是換股執行日]；s＝現行帳戶日報酬、c＝候選日報酬、b＝0050 日報酬、w＝前一日收盤個股占比。k ∈ {0.5, 1.0}。原帳戶已付的 0050 買賣成本不退（對合併略不利）。",
  "kappa": "κ＝0.5%（嚴成本個股買 0.34%／賣 0.64% 的平均，取整）；敏感度 κ＝0 另列，不影響判定。",
  "n_cells": "可判定 4 候選 × 4 結構＝16 格；參考 3 候選 × 4 結構＝12 格。資料窺探次數 +28。"},
 "metrics（每格、每段，並與同段現行比較）": {
  "cagr": "(期末÷期初)^(12÷月數)−1",
  "maxdd": "段內逐日最大回撤",
  "excess": "月超額（−0050）平均×12；t＝Newey-West（Bartlett、落後 6 個月）",
  "IR": "月超額平均÷月超額標準差×√12",
  "cold_excess": "溫度計為『冷』的月份之月超額平均×12（個百分點年化）；另列月數",
  "ex_top5pct": "SURGE 帳戶改用『禁買報酬率前 5% 交易（全期 389 筆中 20 筆，引擎 block、名額空著放 0050）』的淨值與 stockfrac 後，同格的 cagr；候選那一邊不動。",
  "corr": "候選與現行的月超額相關係數（全期重疊、訓練、驗證；另分冷／溫／熱）"},
 "decision（事前寫死）": {
  "pass_rule": "同段內：合併 cagr ≥ 現行 cagr − 2 個百分點，且（最大回撤淺 ≥ 3 個百分點 或 cold_excess 改善 ≥ 5 個百分點年化 或 IR 提高 ≥ 0.1）。",
  "train": "train.py 只算訓練段。可判定 16 格中符合 pass_rule（用訓練段數字）者為『訓練合格』；凍結＝訓練合格格中訓練段 IR 最高的一格（同分取 cagr 高者）。沒有訓練合格格 → 不凍結任何格，結論＝維持現行；另把『訓練段 IR 最高的可判定格』當紀錄格帶進驗證（只紀錄、不能翻案）。參考 12 格同法選一格『參考凍結格』（可無）。寫入 FROZEN.json＋sha256。",
  "valid": "validate.py 只跑一次。凍結格在驗證段符合 pass_rule →『值得獨立複製』；否則『維持現行』。參考凍結格通過只寫『值得先用同口徑（s10lib 嚴成本、正式開關）重做』。其餘格的驗證段數字全部列出但只作描述。",
  "no_change": "不論結果，不改正式策略、不改任何既有檔。"},
 "predictions（事前寫下，可被否證）": [
  "可判定四個候選與現行的月超額相關在 0.2～0.6（都是中小型強勢股行情），冷月多半一起輸 0050。",
  "A 結構（分資金）在訓練與驗證段的 cagr 都比現行低超過 2 個百分點 → 不合格。",
  "B 結構只有 DISP_D 有機會在訓練段合格（S14：2019～2023 有補充價值），但驗證段（2024 起處置單獨輸 0050）不過。",
  "禁買前 5% 交易後，任何合併格的 cagr 仍明顯低於不禁買；合併不能實質降低對大贏家的依賴。"]
}
b = json.dumps(P, ensure_ascii=False, indent=1).encode('utf-8')
(H / 'PREREG.json').write_bytes(b)
(H / 'PREREG.sha256').write_text(hashlib.sha256(b).hexdigest() + '  PREREG.json\n')
print(hashlib.sha256(b).hexdigest())
