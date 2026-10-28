# FINDINGS：換股緩衝帶與延長持有：只續抱訊號仍新鮮的舊持股（o_hold_buffer_fresh）

子代理不能寫 .md，請主研究員把本文存成 `agents/o_hold_buffer_fresh/FINDINGS.md`。
資料夾：`/home/user/-/research/tw_stock_selection/agents/o_hold_buffer_fresh/`

## 結論

- **判定：不可用。機制：不支持。**
  - 依事前規則選出的版本是 A20，也就是純排名緩衝：舊持股排名 ≤ 20 就續抱。
  - 周轉從 59.4% 降到 43.1%，但五個判定門檻只過了一個（損益兩平滑價）。
  - 挑選期配對差（pp／月，括號為 t）：一般成本 −0.154（−0.88）、壓力成本 −0.055（−0.32）、扣成本前 −0.264（−1.50）。
- **「慢漂移」假說被否定。** 依事前登記的關閉條件，緩衝題永久關閉。
  - 排名 11～30、52 週比 ≥ 0.90 的舊股，下一窗報酬比新進前 10 名少 −1.64pp（t −2.05，122 個月），已經跨過事前寫好的 −2 門檻。
  - 連「仍在前 10 名」的續抱股都比新進股少 −0.93pp（t −1.64）。
  - 主策略的超額報酬集中在入選後第一個月：持有第 2 個月 −1.48pp（t −2.05），第 4 個月以後才回到和新進股差不多。
- **周轉的來源和預測相反。** 調回等權只占約 1.6pp，不是預測的 10～15pp。漂移帶 ±40% 只能把周轉從 59.4% 降到 56.8%。59% 的周轉幾乎全部來自換名額。
- **和主策略 V3 比（一般成本）**
  - 跨循環：A20 26.56%、對 0050 t 2.546、回撤 −58.0%；V3 26.95%、t 2.549、回撤 −65.0%。
  - 2024～2026：A20 67.1%，V3 60.1%，配對 t 0.54。這段只報告，而且只有 32 個窗。
  - AB 跨循環：A20 25.55%、t 2.37；V3 26.47%、t 2.46。
- **SOP 不改**：主策略維持每月全換、全部調回等權。
- 唯一的副作用好處是回撤淺約 7pp，主要來自 2008 年。但它不在判定門檻內，2019 年以後也沒有一致改善，不構成改版理由。

## 1. 研究問題

主策略 V3 的月單邊周轉約 62%。一般成本一年吃掉約 5pp，壓力成本下對 0050 的 t 從 2.55 掉到 1.54。

本題要回答：能不能只續抱「訊號仍新鮮」的舊持股，把周轉降到 35～40%，同時保住扣成本前的優勢？

## 2. 機制假說（事前）

- **對應的贏家**
  - 新高突破的順勢買方，以及持有 1～6 個月的投信。
  - 處分效應造成反應不足，按理會在數個月內逐步消化。
- **對手**：0.3% 證交稅加滑價。
- **假說**
  - 報酬流失只來自價格腿已經衰退的舊股，也就是 52 週比下滑的股票。
  - 只因營收百分位被新公告洗牌、掉到 11～30 名，但仍貼近 52 週高點的舊股，下月報酬應和新進股差不多。
  - 另一個周轉來源是續抱股每月被調回等權，可以用權重漂移帶省下來。
- **和舊研究的差別**
  - 舊研究：turnover_buffer 只有 2021 年以後的資料；main_refine R4 K=20 用的是 ORIG 營收和 v1 處置定義。
  - 本題用 V3 資料、2006～2018 長歷史挑選，並且做條件式續抱、權重漂移帶、門檻差、兩期重疊。

## 3. 資料與引擎

- **資料：全部是 V3**
  - o_infra_v3_rebuild 的 P（下市股價格，已補 15,375 格）與 X（併購出清規則 166 條）。
  - R_DEL 營收。
  - o_disposal_rejudge 官方處置名單 E1。
  - 都用只讀 import，sha256 由 o3_core 斷言。
  - 沒有新資料，raw/ 是空的。
- **引擎**
  - `hb_core.sim` 是 o3_core.simulate 的推廣，多了漂移帶、逐檔成本向量和換股拆解紀錄。
  - `hb_core.planner` 是 V2.plan 的推廣，多了 K、h、δ 三個參數與兩期重疊。
- **驗證**（validate.py）
  - planner 和 V2.plan 的每期名單相同。
  - sim 和 o3_core.simulate 逐日相同（最大相對差 < 1e-12）。
  - 重現 v3_results R_DEL|E1：跨循環 26.9459%、t 2.5486，逐位相同。
- **成本**
  - 一般：買 0.1855%、賣 0.4855%。
  - 壓力：每邊再加 0.3%。
  - 另報：flat 0.5%／0.5%、1,000 萬容量（每邊加 2.5%×sqrt(成交額／60 日均成交值)）、全額手續費。

## 4. 探索發現（只用挑選期 2006-04～2018-12，扣成本前；explore.json）

### 4.1 周轉拆解（預測 1 不成立）

- 賣出 59.4%：出場名額 56.6%，續抱股減碼 2.7%。
- 買進 59.0%：新進名額 58.5%，續抱股加碼 0.5%。
- 調回等權的部分＝（減碼＋加碼）／2，約 **1.6pp**。
- 續抱股完全不調回時，周轉是 56.3%。
- 平均每月換 5.9 檔，續抱 4.1 檔。

### 4.2 舊持股本月的排名

| 本月排名 | 比例 |
|---|---|
| 仍在前 10 | 41.2% |
| 11～15 | 11.6% |
| 16～20 | 8.0% |
| 21～30 | 10.3% |
| 31～50 | 12.9% |
| 51+ | 12.8% |
| 不合格 | 3.2% |

### 4.3 下一窗報酬：減去當月新進前 10 名，按月分群

- 新進股本身是每窗 +2.40%。
- **續抱股（仍在前 10）**：−0.93pp（t −1.64）。
- **依排名桶**
  - 11～15：−0.30（t −0.30）
  - 16～20：−2.35（t −1.96）
  - 21～30：−0.68（t −0.68）
  - 31 名以後：−2.58（t −3.64）
- **依 52 週比**（合格舊股）
  - ≥ 0.95：−1.05（t −1.19）
  - 0.90～0.95：−3.34（t −3.79）
  - 0.85～0.90：−1.20（t −1.02）
  - < 0.85：−3.19（t −2.99）
- **事前檢定 P2a**：排名 11～30 且 52 週比 ≥ 0.90，**−1.64pp（t −2.05）**，共 325 筆、122 個月。
  - 敏感度：winsor 5% 後 t −2.33；中位數 −0.84pp；拿掉最差 3 個月 t −1.44；剔除單一年份後 t 在 −1.61～−2.43。
  - 分段：2006～2010 −2.60（t −1.60）；2011～2014 −2.18（t −1.96）；2015～2018 −0.08（t −0.06）。
- **事前檢定 P2b**：合格舊股且 52 週比 < 0.85，−3.19（t −2.99），通過。但只看排名 11～30 的話是 −2.20（t −1.23，只有 31 個月）。
- **事後觀察**：排名 11～30 且 52 週比 ≥ 0.95，+0.04pp（t 0.05，87 個月）；0.90～0.95 是 −4.28（t −4.15）。
  - 52 週比和報酬不是單調關係，這個門檻是看了資料才找到的。
  - 它對應的 B30h95 在 2019～2023 的配對差是 −0.237（t −1.27），不支持把它當成新規則。
- **營收前三分之一**：排名 11～30、≥ 0.90、營收在前 1/3 是 −1.61（t −1.90）；不在前 1/3 是 −2.93（t −1.73）。營收新鮮與否，救不了價格腿。
- **依連續持有月數**

| 持有第幾個月 | 和新進股的差 pp（t） |
|---|---|
| 第 2 個月 | −1.48（−2.05） |
| 第 3 個月 | −0.78（−0.96） |
| 第 4 個月以上 | −0.08（−0.09） |

- **解讀**：主策略的超額報酬集中在入選後第一個窗。這是「新高突破後立即反應」，不是「數月慢漂移」。
- **事前關閉條件已經觸發**：P2a 的 t < −2，所以「慢漂移」機制在本策略不成立。模型族仍照事前登記全部跑完，只作紀錄。

### 4.4 容量（1,000 萬，每檔 100 萬）

- 新買名額部位超過 60 日均成交值 1% 的比例：挑選期 40.6%、2019 後 11.8%、跨循環 29.0%。
- A20 把每月新買檔數從 6.15 降到 4.40，但超過 1% 的比例不變（29.2%）。

## 5. 模型迭代紀錄（iterations.json，5 輪共 20 版，全部列入多重檢定帳）

- 下表是挑選期數字，括號為 t。
- 周轉是一般成本下的月單邊。
- 損益兩平滑價是「挑選期對 0050 月均超額＝0」的單邊滑價；原版是 0.895%。

| 版本 | 周轉 | 扣成本前 | 一般成本 | 壓力成本 | 損益兩平 |
|---|---|---|---|---|---|
| A15 | 49.5% | −0.012（−0.10） | +0.054（0.44） | +0.114（0.93） | 1.106% |
| **A20（選中）** | **43.1%** | **−0.264（−1.50）** | **−0.154（−0.88）** | **−0.055（−0.32）** | **1.017%** |
| A25 | 37.8% | −0.265（−1.34） | −0.118（−0.60） | +0.013（0.07） | 1.193% |
| A30 | 34.4% | −0.461（−2.26） | −0.290（−1.44） | −0.138（−0.69） | 1.055% |
| B20h85 | 45.1% | −0.292（−1.74） | −0.195（−1.18） | −0.109（−0.66） | 0.931% |
| B20h90 | 47.6% | −0.356（−2.38） | −0.275（−1.86） | −0.203（−1.39） | 0.806% |
| B20h95 | 51.8% | −0.105（−0.91） | −0.054（−0.47） | −0.008（−0.07） | 0.959% |
| B30h85 | 36.8% | −0.392（−2.04） | −0.238（−1.25） | −0.101（−0.53） | 1.062% |
| B30h90 | 40.6% | −0.461（−2.60） | −0.332（−1.90） | −0.218（−1.25） | 0.857% |
| B30h95 | 50.2% | −0.054（−0.44） | +0.009（0.07） | +0.064（0.52） | 1.047% |
| C03 | 51.4% | +0.041（0.32） | +0.095（0.74） | +0.143（1.12） | 1.108% |
| C05 | 47.2% | −0.137（−0.96） | −0.054（−0.39） | +0.019（0.14） | 1.041% |
| C10 | 38.2% | −0.298（−1.60） | −0.154（−0.83） | −0.027（−0.14） | 1.132% |
| D20 | 57.7% | −0.018（−1.09） | −0.016（−0.98） | −0.002（−0.13） | 0.903% |
| D40 | 56.8% | −0.024（−0.93） | −0.007（−0.27） | +0.012（0.45） | 0.929% |
| E2 | 35.7% | −0.444（−2.66） | −0.282（−1.69） | −0.138（−0.83） | 1.030% |
| X1＝B30h95＋D40 | 47.5% | −0.105（−0.80） | −0.026（−0.20） | +0.045（0.35） | 1.068% |
| X2＝C03＋D40 | 48.7% | +0.022（0.16） | +0.094（0.68） | +0.157（1.14） | 1.159% |
| X3＝B30h95＋重疊 | 31.6% | −0.465（−2.49） | −0.275（−1.49） | −0.106（−0.58） | 1.162% |
| X4＝B30h95＋C03 | 46.3% | +0.027（0.20） | +0.115（0.85） | +0.193（1.42） | 1.238% |

- **迭代輪次**
  - it1：A 族對照。
  - it2：B 族。
  - it3：C 族。
  - it4：D 族與 E2。
  - it5：跨族組合 X1～X4，依探索 4.3 和 it1～it4 的結果決定（理由寫在 iterations.json 的 X_rationale）。
- **規律**
  - 周轉進入 35～45% 帶的版本（A20、A25、B30h85、B30h90、C10、E2），扣成本前全部是 −0.26～−0.46pp／月，一般成本全部為負。
  - 一般成本配對差為正的版本（A15、C03、X2、X4），周轉都在 46～51%。
  - 也就是說，只要把周轉壓到 45% 以下，扣成本前的流失就大於省下的成本。
- **選版**（事前規則：35～45% 帶內壓力配對 t 最高，鄰居不得為負）
  - 帶內最高是 A25（t 0.07）。
  - 但它的鄰居 A20（−0.055）和 A30（−0.138）為負，所以退回較保守的 A20。
  - A20 的鄰居 A15（+0.114）、A25（+0.013）都不為負，選定 A20。
  - 選出對照族的版本，表示新鮮度條件（B 族）在周轉帶內沒有勝過純排名緩衝。
- **揭露**
  - it4 和 it5 之間，planner 增加「K 與 δ 並用」的分支，給 X4 使用。我在凍結後用 scratchpad 重算 A20、B30h95、C05、D40、E2，挑選期配對差和 iterations.json 逐位相同。
  - X 組合的佔位規格在 it5 前改寫過；佔位版本從未執行。

## 6. 凍結規格

- 規格全文見 frozen_spec 欄位。
- PREREG sha256 `cfd4b29f…ec2f`，時間 2026-10-27T22:59:59Z。
- oos.py 會斷言 PREREG 存在，並核對 PREREG、程式與 iterations.json 的 sha256 都沒被改過，才計算樣本外。

## 7. 樣本外結果（oos.json，凍結後只跑一次）

| 期間 | 原版 年化／t／回撤 | A20 年化／t／回撤 | 配對差（一般） | 配對差（壓力） | 0050 | 同池等權 |
|---|---|---|---|---|---|---|
| 挑選期 06-04～18 | 18.02／2.37／−65.0 | 16.28／2.13／−58.0 | −0.154（−0.88） | −0.055（−0.32） | 6.47 | 4.67 |
| 驗證期 2019～23 | 34.26／1.40／−36.8 | 34.47／1.39／−33.2 | +0.006（0.02） | +0.115（0.39） | 16.43 | 21.16 |
| 保留期 24～26-10-12 | 60.09／0.38／−43.7 | 67.05／0.68／−36.3 | +0.330（0.54） | +0.451（0.75） | 59.23 | 25.67 |
| 2019 後合計 | 42.99／1.29 | 45.33／1.51 | +0.119（0.42） | +0.232（0.82） | 30.23 | 22.76 |
| 跨循環 | 26.95／2.549／−65.0 | 26.56／2.546／−58.0 | −0.051（−0.34） | +0.053（0.35） | 14.94 | 11.20 |

- **壓力成本，跨循環**：原版 21.42%（t 1.54），A20 22.53%（t 1.77）。
- **其他成本，跨循環**
  - flat 0.5%：23.89% 對 24.34%。
  - cap10m：23.56% 對 24.16%。
  - 全額手續費：25.88%（t 2.36）對 25.79%（t 2.40）。
- **v2 舊正式**（main_v2 navs_gapfix EXFIN|n10|normal）：跨循環 28.15%、t 2.64，只作對照。
- **AB 實盤基準**（只用已公告營收，Monte Carlo 20 次平均淨值）
  - 原版：跨循環 26.47%、t 2.46、回撤 −65.1%；2024～2026 58.13%（t 0.32）。
  - A20：跨循環 25.55%、t 2.37、回撤 −59.5%；2024～2026 67.66%（t 0.70）。
  - AB 下的配對差：跨循環 −0.086（t −0.61），挑選期 −0.203（t −1.26）。
- **成本損益兩平**
  - 挑選期「對 0050 超額＝0」的單邊滑價：0.895% → 1.017%（+0.12%）。
  - 跨循環「t 降到 2.3」的單邊滑價：0.174% → 0.195%（只多 +0.02%）。
- **判定門檻**：T1、T2、T4、T5 未過，只有 T3 通過（定義見 frozen_spec 第 7 點），**判定不可用**。
- **2019 年後配對差略正**（一般成本 +0.12，t 0.42），但沒有統計意義，而且不參與判定。
- **其餘 19 版的 2019 後結果**（只報告，不改選版）
  - X4：驗證期 −0.147（t −0.65）、保留期 +0.537（t 2.04）。
  - C03：驗證期 −0.025、保留期 +0.091。
  - B30h95：驗證期 −0.237（t −1.27）、保留期 +0.442（t 1.74）。
  - 方向在驗證期和保留期之間不一致，挑選期的排序也沒有延續。

## 8. 穩健性

### 8.1 逐年（A20 減原版，pp，一般成本）

| 年 | 差 | 年 | 差 | 年 | 差 |
|---|---|---|---|---|---|
| 2006 | −22.8 | 2013 | −2.1 | 2020 | −15.2 |
| 2007 | −21.8 | 2014 | +7.1 | 2021 | +4.2 |
| 2008 | +10.1 | 2015 | −10.8 | 2022 | +8.7 |
| 2009 | +6.7 | 2016 | −2.4 | 2023 | −1.7 |
| 2010 | +4.3 | 2017 | +8.5 | 2024 | +11.0 |
| 2011 | −0.2 | 2018 | −3.9 | 2025 | +13.2 |
| 2012 | −8.7 | 2019 | −0.7 | 2026 | −11.2 |

- 空頭和盤整年（2008、2022）較好；強勢輪動年（2006、2007、2015、2020）明顯較差。

### 8.2 子樣本（預測 4）

- **高成本情境改善較大：成立。**
  - 挑選期配對差：一般 −0.154、壓力 −0.055、cap10m −0.077、flat −0.100。
  - 但在挑選期，即使是壓力成本也是負的。
- **V 型反彈與快速輪動期改善較小：只對了一半。**
  - 2020-04～06 是 −2.17pp／月（只有 3 個窗），符合預測。
  - 2009 是 +0.33（t 0.59），和預測相反。
- **「S1 改善最大」：不成立。**
  - S1 2006～2013：−0.219（t −0.94）。
  - 2014～2018：−0.053（t −0.20）。

### 8.3 分產業（扣成本前每窗等權貢獻，A20 減原版，挑選期合計 −40pp）

- 損失分散在半導體（−16）、通信網路（−14）、電腦及週邊設備（−11）、紡織（−8）、光電（−6）。
- 只有電機機械為正（+11）。

### 8.4 去掉最大貢獻股

- 挑選期配對差最大的正貢獻股是 6269、3504、3515；最大的負貢獻股是 4426、3016、2498，各 ±5～7pp 累計。拿掉最大正貢獻股後，差距更負（−46pp）。
- 單股最大貢獻：原版和 A20 在挑選期都是 5534 長虹。原版在保留期是 2059 川湖，剔除後年化少 11.4pp；A20 剔除後少 8.5pp。

### 8.5 端點與多重檢定

- **端點敏感度**（跨循環對 0050 t，A20／原版）
  - 拿掉最好 3 個月：2.02／2.04。
  - 拿掉最好 5 個月：1.74／1.73。
  - 拿掉最好年份（2021）：2.14／2.15。
  - 日曆月 t：2.47／2.46。
- **Bonferroni**
  - m＝10（主策略累計改版）＋20（本題）＝30：A20 的 p＝0.163，原版 0.162，都不顯著。
  - 20 版中挑選期壓力配對 t 最高的 X4 只有 1.42，Bonferroni p > 1。

## 9. 機制是否被支持：不支持

- **預測 1 不成立。** 調回等權只占周轉約 1.6pp，漂移帶頂多降 3pp。扣成本前配對差 |Δ| < 0.03pp 這一點成立（D20 −0.018、D40 −0.024），但沒有實際價值。
- **預測 2 依事前關閉條件不成立。** 仍貼近高點（≥ 0.90）的舊股，比新進股少 1.64pp（t −2.05）。主策略的超額在入選後第一個月就大部分兌現，續抱只是持有「已經被市場消化的訊號」。
- **預測 3 不成立。** 條件續抱加漂移帶（X1）周轉 47.5%，沒有進入 35～40%。周轉帶內的 B 族版本，壓力成本配對差全部為負。
- **預測 4 只對一部分**，見 8.2。
- **白話**
  - 新高突破加營收年增，是「剛發生」的訊號，市場在 1 個月內就把大部分反應做完。
  - 慢漂移在主策略的股池裡看不到。
  - 0.3% 證交稅的對手很硬，但為了省稅去續抱舊股，代價比稅還高。

## 10. 散戶怎麼執行

- 照現行 SOP：每月全換、全部調回等權，不加緩衝帶、漂移帶或延長持有。
- 想省成本的話：用限價單，避免追開盤市價；不要靠續抱舊股省成本。
- 調回等權的成本本來就只有約 1.6pp 的周轉，每年約 0.1～0.2pp。嫌麻煩的人可以容許 ±40% 的權重漂移，結果幾乎相同（扣成本前 −0.02pp／月）。這只是操作便利，不是改版。

## 11. 限制

- **2019 年以後的資料已經看過**：主策略 V3 的數字已公開，turnover_buffer 也看過 2021 年以後的緩衝結果。所以判定上限本來就只有待觀察；本題在挑選期就沒過門檻。
- **選版規則把周轉帶（35～45%）設為硬條件**，結果選出對照族的 A20。帶外表現最好的 X4（46.3%）沒有被選；它的樣本外結果不一致，也不構成反證。
- **探索的下一窗報酬是近似值**：停牌股以最後價值持有，沒有套用 X 出清。這只影響描述統計，不影響回測。
- **1,000 萬容量的成本**是平方根衝擊的假設模型（2.5%×sqrt(成交額／60 日均成交值)），沒有實測滑價。
- **常態近似 p 值**；B 族的 h 門檻在探索中呈非單調，0.95 的門檻是看了資料才找到的。

## 檔案

- `/home/user/-/research/tw_stock_selection/agents/o_hold_buffer_fresh/`
  - 程式：hb_core.py、validate.py、plan.py、explore.py、iterate.py、prereg.py、oos.py、finalize.py
  - 結果：validate.json、explore.json、explore_rows.csv、iterations.json、oos.json、navs_oos.parquet、results.json（PLAN、PREREG、SUMMARY）
- 執行順序：
  1. `python3 -I -B validate.py`
  2. `plan.py`
  3. `explore.py`
  4. `iterate.py it1…it5 <版本>`，再 `iterate.py select`
  5. `prereg.py`
  6. `oos.py`
  7. `finalize.py`
- 沒有修改其他資料夾，沒有 git commit。

## 凍結規格

凍結規格（PREREG sha256 cfd4b29fdfa0bbbeeacdff5fd179c7ef35e87dde560b17e1bd6c6d347465ec2f，時間 2026-10-27T22:59:59Z；PLAN sha256 4608388930bca2c081608df43b7c694d2061bdc5ca63428da10dc5fe7c0f18da，時間 2026-10-27T22:53:00Z，早於任何探索與迭代報酬）

選中版本：A20（純排名緩衝 K=20）。依事前選版規則選出：周轉 35～45% 帶內壓力配對 t 最高的是 A25，但它的鄰居 A20、A30 壓力配對差為負，所以退回較保守的 A20；A20 的鄰居 A15、A25 都不為負。

1. 基礎策略＝主策略 V3（總監 3.5 節正式口徑）
- 程式：main_v2/v2_core.V2('gapfix')。
- 營收：R_DEL 長期營收（o_rev_survivorship 凍結版，由 o3_core.load_rdel 斷言 sha256）；2021-03 起改用 rev_5y yoy。
- 排除金融保險：stock_list 的金融保險業，另加 28xx、5876、5880。
- 股池：60 日均成交值前 300 名，條件是 60 日均成交值 ≥ 1,000 萬、收盤 ≥ 10、隔日有開盤成交。處置股照 o_disposal_rejudge 官方名單 E1 全部排除（公告日 ≤ 訊號日 d，而且 start ≤ d ≤ end）。
- 分數＝0.5×52 週比百分位＋0.5×近 3 月營收年增率平均百分位。滾動窗用 gapfix。
- 換股：每月第一個 ≥ 11 日的交易日收盤算訊號，隔日開盤成交，等權 n10。
- 同分時先比營收 3 月年增率（高者優先），再比代碼。
- 開盤 ≥ 跳動點漲停價、而且還沒持有的股票不買，由下一名遞補。
- 停牌部位凍結；P 補下市股價格；X 併購出清用 o3_core.exit_rules('main')。

2. 緩衝規則（A20）
- 每個訊號日，上月實際持股若同時符合三個條件就續抱：本月仍合格、隔日可交易、排名 ≤ 20（rank 1 起算）。
- 其餘名額依排名由高到低補滿 10 檔，漲停遞補規則不變。
- 續抱股與新進股全部調回等權，沒有漂移帶。

3. 引擎
- hb_core.sim：band=None、純量成本時，與 o3_core.simulate 逐日相同（validate 斷言，最大相對差 < 1e-12）。
- hb_core.planner：K=None 時，每期名單集合與 V2.plan 相同（斷言）。
- 原版重現 o_disposal_rejudge v3_results R_DEL|E1：跨循環 26.9459%、t 2.5486，逐位相同。

4. 成本
- 一般：買 0.1855%、賣 0.4855%（手續費 0.0855%＋稅 0.3%＋滑價 0.1%）。
- 壓力（選版與判定用）：一般成本每邊再加 0.3%，即買 0.4855%、賣 0.7855%。
- 只報告的情境：
  - 舊口徑 flat：買賣各 0.5%。
  - cap10m：帳戶固定 1,000 萬，每檔每邊另加 2.5%×sqrt(成交額／60 日均成交值)。
  - 全額手續費 0.1425%。

5. 期間（訊號日）
- 挑選期：2006-01～2018-12，實際從 2006-04 開始，共 153 窗。
- 驗證期：2019-01～2023-12，60 窗，只報告。
- 保留期：2024-01～2026-10-12，32 窗，只報告。
- 跨循環：2006-04～2026-10-12，245 窗。

6. 統計
- 配對差＝版本與原版的訊號窗報酬差，月份分群 t。
- 對 0050 用訊號窗 t；另報日曆月 t。

7. 最終判定門檻（全部要通過；通過時上限為待觀察）
- T1：挑選期一般成本配對差 > 0。
- T2：挑選期壓力成本配對差 ≥ 0。
- T3：損益兩平單邊滑價提高 ≥ 0.05%。損益兩平定義為挑選期對 0050 月均超額＝0 的滑價。
- T4：挑選期拿掉最好 3 個月後，配對差 ≥ 0。
- T5：挑選期扣成本前配對差 ≥ −0.05pp／月。

8. 事前登記的模型族（20 版，全部報告）
- A（純排名緩衝）：K=15、20、25、30。
- B（新鮮度條件續抱）：K∈{20, 30} × h∈{0.85, 0.90, 0.95}。
- C（換股門檻差）：δ=0.03、0.05、0.10。
- D（權重漂移帶）：±20%、±40%。
- E2：兩期重疊。
- X1：B30h95＋D40。
- X2：C03＋D40。
- X3：B30h95＋兩期重疊。
- X4：B30h95＋C03。

## 反方審查

[
 {
  "final": "不可用",
  "issues": [
   "I keep the 不可用 verdict, but the reason is overstated. The selection-period paired gap for A20 minus V3 at normal cost is −0.154pp/month in total, and almost all of it comes from 2006 and 2007. Summed paired gaps by year are 2006 −15.3pp and 2007 −19.1pp, against −23.5pp for the whole selection period. With 2006–2007 removed, the gap is +0.083pp/month (t 0.49), and T1 would pass. Dropping 2006 alone gives t −0.32, and dropping 2007 alone gives −0.18. Block bootstrap (6-month blocks) puts the 90% interval for the normal-cost selection gap at [−0.43, +0.18]pp. The honest description is that A20 cannot be distinguished from V3 in any period: |t| < 1 throughout, Newey-West t is about the same, and the cross-cycle gap is −0.051 (t −0.34). It is not 'proven worse'. The verdict still stands because the pre-registered thresholds were not met, and removing years after the fact cannot be used to upgrade it.",
   "The mechanism rejection and the permanent closure of the buffer question rest on a borderline t and on the choice of comparison group. P2a is −1.64pp (t −2.05), just past the −2 line. Dropping the worst 1, 3 or 5 months gives t −1.81, −1.44, −1.09, and 2015–2018 is −0.08. Also, what A20 actually pushes out is the lowest-ranked new entrant, not the average of the new top 10. Against the lowest-ranked 1, 2 or 3 new entrants of each month, P2a is −1.89, −1.40, −1.32pp with t −1.69, −1.57, −1.62, so it does not reach the −2 closure condition. New-entrant returns are not monotonic in rank (ranks 1–3: 2.10%, 4–6: 3.73%, 7–10: about 1.43%). The direction is consistently negative and the stock-level median is −2.64pp, so 機制不支持 holds. Calling it '否定' or permanently closing the topic should be downgraded to 'not supported; the closure condition only just triggers and is sensitive to the comparison group'.",
   "The 'freeze, then run out-of-sample only once' claim cannot be verified. The PREREG code_sha256 covers only hb_core, validate, plan, explore and iterate; oos.py and finalize.py are not hashed. oos.py was last modified at 23:01:01, after the PREREG timestamp of 22:59:59, and oos.json was written at 23:02:02. Nothing proves oos.py was not edited and re-run after the freeze. Since the verdict is negative this does not reverse anything, but the process claim does not hold. Related: hb_core.py and iterate.py were both modified at 22:58:15 (after it4 at 22:57:43 and before it5 at 22:59:01), and validate.json was rewritten at 22:58:27. This is the disclosed planner change, but it means PLAN, the 20 versions and the OOS run were all done within about 10 minutes, and the freeze happened 58 seconds after it5.",
   "The drawdown description is inaccurate. The 7pp shallower drawdown does not mainly come from 2008. Over the full period, both strategies have their maximum drawdown from 2007-07 to 2009-02 (−65.0 vs −58.0). But after 2009-06, the largest drawdown is −43.7 vs −36.3, bottoming on 2025-04-09, which is also 7pp shallower. So there are two episodes, both very few in number, and one of them is in the holdout period. FINDINGS says '2019 年以後也沒有一致改善', which conflicts with this. It is still not a reason to change the strategy, but the wording should be corrected.",
   "The 2024–2026 result rests on only 32 windows and is concentrated in 2024: the 2024 paired sum is +11.6pp and 2025 is +0.3pp. Bootstrap P(>0) of 0.95–0.97 does not reach a usable level, and it does not agree with the validation period (normal +0.006). FINDINGS lists it as report-only, which is correct. It must not be cited as evidence in favour of A20.",
   "On multiple testing and data snooping: explore.json holds 70 cell tests. The h=0.95 threshold was found after looking at the data, and it was then used to build X1, X3 and X4. The researcher disclosed this, and the 20-version Bonferroni p > 1 covers it. It does not change the verdict. Note that X4's holdout t 2.04 is post-hoc plus a single short period and must not be used to reopen the question.",
   "Look-ahead and survivorship checks found no new problems. Rank and the 52-week ratio are computed at the signal-day close and trades happen at the next day's open. Prices of delisted stocks and the merger exit rules are both included. The AB check (using only revenue already published) gives a selection-period paired gap of −0.203 (t −1.26), the same direction. The exploration statistics do not apply the merger exit rules, and old holdings that could not trade on the trade day are excluded from P2a (only a small effect on the descriptive statistics), as already disclosed."
  ],
  "note": "My check script is at /home/user/-/research/tw_stock_selection/agents/o_hold_buffer_fresh/review_stat_o4.py (run with python3 -I -B). It only reads navs_oos.parquet and explore_rows.csv and does not modify any of the researcher's files. It reproduces the researcher's paired gaps exactly: selection period normal −0.1536 (t −0.881), stress −0.0549 (t −0.318), before costs −0.2645 (t −1.50); P2a −1.635 (t −2.05); holding month 2 −1.483 (t −2.05). I also added Newey-West t, a 6-month block bootstrap, the year concentration, the drawdown episodes and a 'lowest-ranked new entrant' comparison group. Conclusion: the 不可用 verdict holds, because the pre-registered T1, T2, T4 and T5 failed and A20 shows no difference from V3 in any period. The negative selection-period gap, though, comes almost entirely from 2006–2007, so this is 'no evidence it is better', not 'proven worse'. The mechanism rejection and permanent closure rest on t −2.05, which falls back to −1.6 to −1.7 against the correct comparison group, so it should be written as 'not supported, closure condition only just triggered'. Process gap: oos.py was changed after the freeze and has no hash, so 'run only once' cannot be verified. The drawdown description also needs correcting: in 2025 A20 was also 7pp shallower."
 },
 {
  "final": "不可用",
  "issues": [
   "我從成本與可成交的角度沒能推翻「不可用」。挑選期 A20 對原版的配對差在一般成本 −0.154pp／月，每邊再加 0.3% 的壓力成本仍是 −0.055，1,000 萬容量情境 −0.077。我用 oos.json 的 gross／normal／stress 三點做線性外推：每邊額外滑價 1% 只把配對差改善約 0.33pp／月；挑選期要在一般成本之外每邊再多 0.47% 滑價配對差才轉正，跨循環要多 0.15%。所以小額散戶（滑價接近 0.1%）用 A20 是吃虧的。",
   "可成交性對兩版的影響差不多，不構成偏誤（review_exec_o4.py，跨循環）。出場股開盤跌停、賣不掉的比例：原版 0.20%、A20 0.37%，跨循環合計只有 3～4 個月出現。新買股開盤已達漲停價 98% 以上的比例：0.73% 對 0.56%。凍結（停牌）股仍在入選名單的次數兩版都是 0。出場部位超過 60 日均成交值 1% 的比例：23.1% 對 22.5%。新買部位超過 1% 的比例：29.0% 對 29.2%。A20 少換股，但沒有改善每筆交易的流動性結構，1,000 萬容量的問題原封不動。",
   "周轉沒有達到題目要的 35～40%。A20 挑選期 43.1%，但跨循環 45.1%、2019 後 48.5%（原版 66.6%）。FINDINGS 只報挑選期的周轉，沒寫實盤期的周轉比挑選期高 5pp。",
   "資料時點對 A20 略為不利。只用已公告營收的 AB 情境下，跨循環 A20 25.55% 對原版 26.47%，配對差 −0.086（t −0.61），比全資料下的 −0.051 更負。這和判定方向一致。",
   "處置股被迫出場要預收券，這一點沒有建模。原版 2019 後約每年 3.1 檔、A20 2.7 檔。分盤撮合下實際成交價不等於回測用的開盤價，成本沒計入。量小、兩版相近，不改結論。",
   "SOP 建議「改用限價單、避免追開盤市價」和回測的開盤價成交假設矛盾。本策略選的是剛突破新高的股票，限價單沒成交的往往是最強的名額，這會造成逆選擇，損益沒有量化過。建議刪掉這句，或標註為未驗證。",
   "「緩衝題永久關閉」寫得偏強。帶外的 A15（周轉 52%）在挑選期、2019 後、跨循環的一般與壓力配對差全部是小正值（跨循環壓力 +0.13，t 1.33）。但它不顯著，而且這個觀察用到了已看過的 2019 年後資料，依不利方向原則不能據此翻案。只建議在紀錄中寫明「A15 為小幅、未顯著的正向，不採用」，不要讓人以為所有緩衝版本都是負的。"
  ],
  "note": "我從成本與可成交的角度審查，同意研究員的判定：不可用。理由一：省下的成本贏不過扣成本前的報酬流失。在挑選期，A20 扣成本前每月少 0.264pp，一般成本只省回約 0.11pp。理由二：可成交性的檢查對兩版影響相近，看不出偏誤，包括開盤跌停賣不掉、開盤貼近漲停、停牌股、處置股、新買與出場部位占 60 日均成交值的比例。理由三：A20 在每筆交易上沒有比較好成交，1,000 萬容量的問題依舊；實盤期周轉 48.5%，沒有達到 35～40% 的目標。理由四：AB 情境下 A20 更差。2019 年後配對差略正但不顯著，而且這段資料已經看過，不能翻案。\n\n需要修的有兩處。第一，刪掉或標註「改用限價單」的建議：它和回測的開盤價成交假設矛盾，損益沒有量化。第二，把 A15 小幅、不顯著的正向寫進紀錄。\n\n我的驗證程式是 /home/user/-/research/tw_stock_selection/agents/o_hold_buffer_fresh/review_exec_o4.py（python3 -I -B，約 14 秒）。它只讀 import 研究員的 hb_core.py，數字取自 oos.json，研究員的檔案沒有改動。"
 }
]

## 獨立複製

{
 "reproduced": true,
 "my_numbers": "I rewrote the A20 buffer stock picker, the daily engine and all the statistics myself. The picker: last month's holdings are kept if they are still eligible, tradable the next day and ranked 20 or better; the remaining slots are filled by rank, skipping stocks that open at the limit-up price; everything goes back to equal weight. The engine solves the equal-weight target with costs exactly on a piecewise-linear equation. The code is in /home/user/-/research/tw_stock_selection/agents/o_hold_buffer_fresh/replicate_o4/ (r4_core.py, r_main.py, r_ana.py, r_ab.py, r_explore.py, r_explore2.py), run with python3 -I -B. Results are in r_results.json, r_ab.json and r_explore.json in that folder. I imported only the earlier-round infrastructure named in the frozen spec for data and the base cross-section (v2_core, drj_core with the E1 disposal list, o3_core for R_DEL, the delisted-price patch and the X exit rules). I did not read hb_core or any of the researcher's code for this topic.\n\nSelf-checks: with the buffer turned off, my picker produces the same monthly name sets as V2.plan for all 245 windows. My engine matches o3_core.simulate with a maximum relative difference of 4.7e-15.\n\nNormal cost (annual return / t vs 0050 / max drawdown):\n- Selection period: base 18.02% / 2.375 / −65.0%; A20 16.28% / 2.131 / −58.0%; paired difference −0.154 pp/month (t −0.88); 0050 6.47%.\n- Validation period: base 34.26% / 1.40 / −36.8%; A20 34.47% / 1.387 / −33.2%; paired +0.006 (0.02); 0050 16.43%.\n- Holdout: base 60.09% / 0.376; A20 67.05% / 0.678; paired +0.33 (0.54); 0050 59.23%.\n- 2019 onward: base 42.99% / 1.289; A20 45.33% / 1.514; paired +0.119 (0.42).\n- Full cycle (ALL, 245 windows): base 26.95% / 2.549 / −65.0%; A20 26.56% / 2.546 / −58.0%; paired −0.051 (−0.34); 0050 14.94%.\n\nStress cost (+0.3% per side):\n- Base: selection 13.07%, t 1.477; ALL 21.42%, t 1.539.\n- A20: selection 12.72%, t 1.433; ALL 22.53%, t 1.770.\n- Paired: selection −0.055 (−0.32); 2019 onward +0.232 (0.82); ALL +0.053 (0.35).\n\nOther cost scenarios (paired difference):\n- Before costs: selection −0.264 (−1.50); 2019 onward −0.008; ALL −0.168 (−1.09).\n- Flat 0.5%: selection −0.100 (−0.57).\n- cap10m: selection −0.077 (−0.44).\n- Full commission, ALL: base 25.88% / t 2.357; A20 25.79% / 2.399.\n\nTurnover (monthly, sell side): base 59.35% selection, 66.58% 2019 onward, 62.08% ALL. A20 43.07%, 48.49%, 45.11%.\n\nA family, selection-period paired difference at stress cost, with selection turnover:\n- A15: +0.114 (t 0.93), turnover 49.5%.\n- A20: −0.055, turnover 43.1%.\n- A25: +0.013 (t 0.07), turnover 37.8%.\n- A30: −0.138, turnover 34.4%.\n- So within the 35–45% band A25 is highest, but its neighbours A20 and A30 are negative. A20's neighbours A15 and A25 are not negative. The rule therefore falls back to A20, matching the researcher.\n\nThresholds:\n- T3: break-even slippage 0.8946% → 1.0174%, +0.123%. Alternative definition (slippage where ALL t vs 0050 falls to 2.3): 0.1739% → 0.1951%, +0.021%.\n- T4: dropping the 3 selection-period windows with the largest paired difference gives −0.247 (t −1.46).\n- T1, T2 and T5 are −0.154, −0.055 and −0.264, as above.\n\nAB (announced revenue only, Monte Carlo seeds 0–19, average NAV):\n- ALL: base 26.47% / t 2.463 / −65.1%; A20 25.55% / 2.371 / −59.5%.\n- Holdout: base 58.13% / 0.318; A20 67.66% / 0.695.\n\nEndpoint sensitivity (ALL t vs 0050, base / A20):\n- No adjustment: 2.549 / 2.546.\n- Drop last window: 2.419 / 2.469.\n- Drop best window by excess over 0050: 1 window 2.375 / 2.367; 3 windows 2.043 / 2.017; 5 windows 1.734 / 1.743.\n- Drop best year (2021 in both): 2.154 / 2.138.\n- Stop at end of 2025: 2.384 / 2.399.\n- Calendar-month t: 2.464 / 2.471.\n\nBonferroni (one-sided normal): m=30 gives base 0.162 and A20 0.163; m=10 gives 0.054 for both.\n\nExploration (selection period, before costs, base picks):\n- Turnover breakdown: sells = 56.61% exited slots + 2.63% trims on held names; buys = 58.75% new slots + 0.50% top-ups. With no reweighting at all, turnover is 56.30%.\n- Last month's holdings by current rank: top 10 41.2%, ranks 11–15 11.6%, 16–20 8.0%, 21–30 10.3%, 31–50 12.9%, 51+ 12.8%, ineligible 3.2%. This matches exactly.\n- Next-window return gap vs new entries (open to next execution open, pp with t, then months and records):\n\n| Group | Gap (t) | Months | Records |\n|---|---|---|---|\n| Still in top 10 | −0.96 (−1.67) | | |\n| Ranks 11–30, 52-week ratio ≥ 0.90 (P2a) | −1.63 (−2.05) | 122 | 326 |\n| Ranks 11–30, ratio ≥ 0.95 | +0.04 (0.05) | 87 | 176 |\n| Ranks 11–30, ratio 0.90–0.95 | −4.28 (−4.15) | | |\n| Ratio < 0.85 and rank > 10 | −3.19 (−2.99) | | |\n| Rank 31+ | −2.58 (−3.64) | | |\n| Kept top-10 names, 2nd month held | −1.51 (−2.08) | | |\n| Kept top-10 names, 3rd month held | −0.78 (−0.96) | | |\n| Kept top-10 names, 4th month or later | −0.08 (−0.09) | | |\n\n- New entries' average next-window return: 2.16% pooled, 2.33% averaged by month.\n- Capacity: share of new-buy slots where the position exceeds 1% of 60-day average traded value is 40.6% selection, 11.8% 2019 onward, 29.0% ALL. New buys per month over ALL: base 6.15, A20 4.40, and A20's share stays about the same (29.2%).",
 "diff_explained": "Almost all key numbers match to the published decimal places. That covers base and A20 at normal and stress cost for every period, the 0050 returns, the before-costs, flat 0.5%, cap10m and full-commission scenarios, turnover, the A-family selection logic, T1 to T5, the alternative T3, AB, endpoint sensitivity, Bonferroni, the turnover breakdown, the rank distribution and capacity. Small differences, with what explains them:\n\n1. Trims on held names: I get 2.63% against a reported 2.7%, and new-slot buys 58.75% against 58.5%. This is likely how the denominator or the first window is handled; the conclusion does not change.\n\n2. The exploration return gaps need two definitions the spec does not state:\n   - Returns run from the execution-day open to the next execution-day open.\n   - The \"52-week ratio < 0.85\" group is limited to rank > 10. Without that limit it gives −2.93 (−2.80) instead of −3.19.\n   - The holding-month groups are limited to kept top-10 names. Without that limit the numbers do not match.\n   With these definitions most groups match exactly. P2a and the ratio ≥ 0.95 group have one record more than reported (326 and 176 against 325 and 175). Still-in-top-10 is −0.96 (−1.67) against −0.93 (−1.64), and 2nd-month-held is −1.51 (−2.08) against −1.48 (−2.05). The researcher probably excludes one or two records that cannot trade or have a missing return.\n\n3. New entries' average next-window return: I get 2.16% pooled or 2.33% by month, against a reported +2.40%. Signal-close to next signal-close gives 2.34% / 2.45%. The exact averaging is not stated, so I could not match this one exactly. It does not affect any threshold.\n\n4. Not checked: I did not rebuild the B, C, D, E2 and X families of the 20-version set. The spec does not define how B's freshness condition, C's swap-score gap, D's weight-drift band or E2's two-period overlap work. So \"X4 has the highest selection-period stress paired t, 1.42\" is not checked. I did rebuild the A family, which decides the version choice.\n\nThe finding holds up: A20 fails T1, T2, T4 and T5 and passes only T3."
}

**最終判定（審查後）：不可用**
