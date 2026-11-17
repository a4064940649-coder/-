#!/usr/bin/env python3
"""S59 事前登記：寫 SPEC.json、PREREG.json 與 sha256。
寫於任何「環境變數 × 月超額」計算之前（已知事項：S55 FINDINGS 的逐年超額與三段年化超額，見 PREREG.known_before）。
已存在就拒絕覆寫。用法：python3 -I -B prereg_write.py"""
import json, hashlib, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
NOW = datetime.datetime.now(datetime.timezone.utc).isoformat()
for f in ('SPEC.json', 'PREREG.json'):
    assert not (HERE / f).exists(), f'{f} 已存在，拒絕覆寫'

SPEC = {
 "topic": "S59 環境歸因（描述性、事後、不改規則）：凍結現行規則（E0|W10_70）的月超額（策略−0050）在什麼市場環境下為正",
 "written_at": NOW,
 "strategy_series": {
  "oos": "S55 凍結規則 2008-01～2015-06：本題用 S55 同一套引擎（s47lib.sim，正式開關全開）在本題暫存重建，月報酬須與 agents/s55_pre2015_oos/oos_monthly.csv 逐月差 < 1e-9、交易與 oos_trades.csv.gz 逐筆相同，否則停止",
  "is": "現行正式 2015-07-01～2026-10-22：用官方面板快取重跑同引擎，日淨值須與 xq_scripts/site/navs_official.csv.gz 的 E0|W10_70 差 < 0.01 元（該檔四捨五入到分），否則停止",
  "pre_panel": "由 raw/pre2015/asoul_twse_2007_2015H1.csv.gz（S55 衍生長表）＋ data_panel 補格，逐行照 S55 build_pre_panel.py 重建到本題暫存（不寫 raw/、不寫 S55 目錄）",
  "monthly": "同 S55：日曆月最後交易日；月超額＝策略月報酬−0050 含息月報酬（算術差）"
 },
 "segments": {
  "OOS": "2008-02～2013-01（60 個月；有訊號的樣本外。2008-01 為起始月、尚無訊號，不列入）",
  "STUCK": "2013-02～2015-06（29 個月；IFRS 斷點，凍結規則沒有訊號、持股停在 2013-01 名單）",
  "IS": "2015-07～2026-10（136 個月；已知樣本內）"
 },
 "env_variables": {
  "timing": "除另註明，月 t 的環境變數一律用月 t−1 月底（含）以前可得的資料（事前、可監控）",
  "MKT_MOM12": "0050 含息還原收盤（cslib.E50_CSV）t−1 月底對 t−13 月底的報酬",
  "MKT_VOL": "0050 含息日報酬 t−1 月底往回 63 個交易日的標準差 ×√252",
  "SMB_MOM12": "小型減大型（以流動性代理規模）12 個月累積：每月 t−1 月底在 data_panel 上市普通股中，取收盤 ≥ 10 且 60 日均成交值 ≥ 1,000 萬者依 60 日均成交值排序，前 50 名＝大型、第 151 名以後＝小型，月 t 等權報酬（data_panel ret.parquet 日報酬連乘，缺值記 0）；SMB_RET＝小型−大型；SMB_MOM12＝t−12…t−1 的 SMB_RET 連乘−1。說明：沒有時點正確的市值（o_pit_shares_industry_2006_2013 閘門未過），以成交值排序代理規模；0051 中型 100 本地只有 2025-09 起，不用",
  "TURN_CHG": "市場成交值變化：data_panel 上市普通股每日成交值加總；t−1 月底往回 63 個交易日平均 ÷ 往回 250 個交易日平均，取 log",
  "REV_DISP": "截面營收成長離散度：營收月 t−2（t−1 月 10 日前公布，t−1 月底可得）的 log(營收／12 個月前營收) 截面標準差；股票＝data_panel 當時上市（該營收月最後交易日收盤有值）且兩月營收 > 0；先依截面 1%／99% 分位縮尾。營收月 2013-01～2013-12 一律缺值（IFRS 斷點，同凍結規則）",
  "ELEC_W": "電子股權重（成交值代理）：t−1 月底往回 60 個交易日，電子類（半導體、電腦及週邊設備、光電、通信網路、電子零組件、電子通路、資訊服務、其他電子）成交值 ÷ 有分類的上市普通股成交值。產業別＝agents/o_lc_sector_rotation_pit/raw/industry_pit.csv（2011-07 以前多為回溯沿用最早觀測，非嚴格時點正確）。沒有時點正確的市值，故不用市值權重",
  "SURGE_TOP": "月營收驚奇名單平均 SURGE：月 t 的訊號列（該月訊號日，約 11 日）股池內 SURGE 有值且收盤 > 60 日均線者，SURGE 前 10 名的平均（凍結規則的 STD SURGE）。只在有訊號的月份有值（STUCK 期與 2015-07～2016-01 缺值）。此變數在月 t 中旬才知道，不是嚴格事前",
  "secondary_contemporaneous": "MKT_RET（0050 月 t 報酬）、SMB_RET（月 t）：同期共變描述，非事前",
  "trade_level": "NTGT（月 t 訊號實際選出檔數）、W_ACT（月平均個股部位占淨值比例：每日 Σ 股數×原始收盤 ÷ 淨值，再取月平均）"
 },
 "analyses": {
  "A1_univariate": "月超額（年化百分點＝月超額×12×100）對每個標準化環境變數（以主樣本平均、標準差標準化）的單變量 OLS，Newey-West（Bartlett 落後 6）t；主樣本＝OOS＋IS（196 個月，有缺值的月份逐變數剔除）；敏感度：全 225 個月（含 STUCK）、只 OOS、只 IS",
  "A1_terciles": "主樣本內依各變數三分位，各組年化平均超額",
  "A2_multivariate": "主樣本同時放 MKT_MOM12、MKT_VOL、SMB_MOM12、TURN_CHG、REV_DISP、ELEC_W（6 個事前變數，不含 SURGE_TOP 以便 2013～2014 也能預測）＋另一版加 SURGE_TOP；NW6",
  "A2_gap": "月超額＝a＋d·IS 虛擬變數（＋6 個環境變數）：比較加入環境變數前後的 d，看樣本內較高的超額有多少能被環境解釋",
  "A3_segments": "三段各變數的平均、中位數、四分位、標準化平均差（相對主樣本標準差）",
  "A4_trades": "交易依進場日分段（OOS／IS；STUCK 沒有新進場）：筆數、勝率、單筆持有報酬（proceeds÷cost−1，已扣成本）平均與中位數、持有交易日、贏家集中度（報酬前 10% 交易占正報酬總和比例；前 5 筆占全部報酬總和比例）、每月選出檔數分布、W_ACT 平均；單位個股部位超額＝月超額 ÷ W_ACT（W_ACT ≥ 5% 的月份）；月超額對 NTGT 的 NW 迴歸",
  "A5_scenario": "用 A2 六變數模型（主樣本估計），代入 2011-01～2014-12 的環境平均，算預期年化超額與 NW 標準誤；另報該期實際超額（2011-01～2013-01 有訊號、之後卡住）與 OOS 中 2011-01～2013-01 的單位部位超額",
  "A6_light": "環境燈號（事前定義、描述用）：5 個條件 MKT_MOM12>0、SMB_MOM12>0、TURN_CHG>0、REV_DISP>主樣本中位數、MKT_VOL<主樣本中位數；符合 ≥4 綠、2～3 黃、≤1 紅；報各燈號在主樣本與各段的月數與年化平均超額（中位數門檻用到全樣本，屬事後）"
 },
 "stats": "NW：Bartlett 落後 6（同 S52／S55）；年化百分點＝月平均×12×100；多重比較：A1 主樣本 7 個事前變數用 Holm 校正（家族 α 0.05），其餘一律標示為探索",
 "not_changed": "引擎、既有研究、raw/、xq_scripts 一律只讀；不改任何策略規則；不呼叫 git",
 "resources": "python3 -I -B；峰值記憶體 ≤ 3GB；暫存 scratchpad/s59/，結束刪除"
}

PREREG = {
 "written_at": NOW,
 "known_before": [
  "S55 FINDINGS：三段年化超額 OOS 主窗 +1.27、STUCK −6.09、IS +11.68 個百分點；逐年 2008 +5.44、2009 +10.12、2010 +4.94、2011 −2.13、2012 −11.35、2013 −2.93、2014 −8.72、2015H1 −3.83；OOS 每月平均 3.3 檔、閒置 74%",
  "本題為事後描述性分析：分段與部分變數的選擇受上述已知結果影響，不能當成假說檢定的獨立證據"
 ],
 "expected_direction": {
  "MKT_MOM12": "+（多頭延續時動能型選股較好）",
  "MKT_VOL": "−（高波動常伴隨動能崩跌）",
  "SMB_MOM12": "+（策略持股偏中小型，小型股順風時較好）",
  "TURN_CHG": "+（量能放大、題材股活躍）",
  "REV_DISP": "+（營收成長離散度大，營收驚奇較能區分贏家）",
  "ELEC_W": "+（弱；樣本內電子／AI 行情占比高）",
  "SURGE_TOP": "+（驚奇強度越大越好）",
  "NTGT": "+（選出檔數多，個股部位比重高、超額放大）",
  "segments": "預期 IS 的 REV_DISP、SMB_MOM12、TURN_CHG、ELEC_W 高於 OOS；2011～2014 環境偏不利"
 },
 "decision_language": {
  "A1": "Holm 校正後 p<0.05 才說「與超額有統計關聯」；未校正 p<0.05 只說「探索性關聯」；其餘說「看不出關聯」",
  "A2_gap": "加入環境變數後 IS 虛擬係數縮小 ≥ 50% 才說「樣本內超額大半可由環境解釋」；縮小 < 25% 說「環境解釋不了樣本內較高的超額」",
  "A5": "預期超額以點估計±1 個 NW 標準誤表示；不得據以改規則或宣稱可擇時"
 },
 "multiple_comparisons": "A1 主檢定 7 個（Holm）；敏感度、三分位、多變數、燈號、交易層皆屬探索，未計入校正；研究中心累計試驗數 N 不因本題增加（不產生新策略格），但燈號若日後要採用須另題事前登記並計入 N",
 "no_rule_change": "本題結論只描述，不得據此改任何規則或參數"
}

for name, obj in (('SPEC', SPEC), ('PREREG', PREREG)):
    p = HERE / f'{name}.json'
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=1))
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    (HERE / f'{name}.sha256').write_text(f'{h}  {name}.json  {NOW}\n')
    print(name, h)
