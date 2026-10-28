# FINDINGS：實盤執行規格：30 萬～1,000 萬資金級距的可執行回測（o_exec_capacity_tiers）

資料夾：/home/user/-/research/tw_stock_selection/agents/o_exec_capacity_tiers/
（子代理不能寫 .md，請主研究員把這份全文存成 FINDINGS.md。）

## 結論

**判定：待觀察**（歷史回測的上限）。四個級距的判定如下：
- 主欄（GROW、手續費 A）：30 萬「可行但打折」；100 萬、300 萬、1,000 萬「不建議」。
- FIX：30 萬、100 萬「可行但打折」；300 萬、1,000 萬「不建議」。

**機制：部分支持**

**1. 方向對了，交叉點卻和預測相反**
- 四個摩擦確實隨資金規模往相反方向變動：零股價差與最低手續費隨規模下降，開盤集合競價衝擊隨規模上升。
- 但兩者的交叉點不在 100～300 萬。依事前規格，衝擊的增加遠大於零股成本的下降，所以績效隨資金單調下降，最好的級距是 30～100 萬。
- 零衝擊 AB_V4 是 27.00%、t 2.51。可執行 AB 如下：

| 級距 | GROW 年化 | GROW t | FIX 年化 | FIX t |
|---|---|---|---|---|
| 30 萬 | 24.39% | 2.05 | 25.26% | 2.20 |
| 100 萬 | 22.18% | 1.66 | 25.10% | 2.18 |
| 300 萬 | 19.88% | 1.23 | 23.65% | 1.92 |
| 1,000 萬 | 17.39% | 0.75 | 20.88% | 1.42 |

**2. 預測的「小帳戶被零股制度懲罰」被反證**
- 和 300 萬相比，30 萬 FIX 反而多 +1.61pp／年（配對 t 2.69，MDE 1.37pp）；GROW 多 +4.50pp。
- 依事前規則，SOP 可以拿掉「30 萬以下大量買零股」的資金下限警示。
- 但小帳戶本身仍然付出零股成本：30 萬 FIX 比零衝擊少 1.74pp／年。其中零股價差約 1.7%／年，最低 20 元的手續費約 1.2pp／年（手續費 C 的 t 是 2.41）。

**3. 最重要的限制：衝擊公式沒有經過校準，結論主要由它決定**
- 規格寫死的是 0.5σ√(Q÷開盤集合競價量)。開盤量只有全日的約 2～5%，所以同一筆單的衝擊約是全日量口徑的 4～6 倍。
- 事後把分母改成全日 V60（o4 的口徑），FIX 四級的 t 是 2.24、2.44、2.50、2.39，全部接近零衝擊。
- 換句話說，「容量從幾百萬就開始吃掉超額」的結論，取決於一個從來沒有用台股開盤成交驗證過的係數。
- 依事前登記，要用 live_tracking 2026-11 起的實際成交價校準。在那之前，本表只能當作保守上限，不能當作預期值。

**4. GROW 的級距意義會隨時間漂移**
- GROW 用 20 年複利（約 ×70～×120），終值權益中位數：
  - 30 萬 → 1,992 萬
  - 100 萬 → 4,628 萬
  - 300 萬 → 9,349 萬
  - 1,000 萬 → 2.0 億
- 所以 GROW 的「級距」其實是「起始資金」。後段實際上是用千萬～億級帳戶在下單，這也是 GROW 比 FIX 差很多的原因。
- 對「我現在有 X 元」這個問題，FIX 才是對的口徑。依導言，判定仍然用 GROW，兩者並列。

## 1. 研究問題
1. 主策略的優勢（零衝擊 AB_V4，t 2.51）能變成多少帳戶報酬？分 30 萬、100 萬、300 萬、1,000 萬四級。
2. 四個摩擦各付了多少：整股取整與現金、零股制度（2020-10-26 以前只有盤後零股）、最低手續費、開盤集合競價衝擊？
3. 有沒有甜蜜區間？SOP 的容量表該怎麼寫？

## 2. 機制（誰在賺、對手是誰）

**對手**
- 流動性提供者（零股造市與掛單者、開盤集合競價的對手盤），以及稅費。
- 主策略年周轉 7.4 倍（雙邊 14.8 倍）。每多 0.05% 單邊成本，一年少約 0.74pp。

**預期的四個摩擦**
- (1) 取整：小帳戶最大。
- (2) 零股成交價與零股價差：小帳戶最大。
- (3) 最低手續費：小帳戶最大。
- (4) 開盤集合競價衝擊：大帳戶最大。

**新發現：零股市場是「散戶買方擁擠」的市場**
- 收盤時零股 ask 平均比整股收盤高 0.40%（中位數），bid 則約等於收盤（賣成本中位數 0，部分價位甚至是負的）。
- 也就是說，買零股的人付了溢價，賣零股的人沒有吃虧。
- 對小帳戶來說，成本集中在「買進」這一邊：每筆零股買單約 33～40bp，賣單約 −6～−13bp。

## 3. 資料
- **名單**：用 v4lib 逐 seed 重建 AB_V4 的 40 組名單，並以零衝擊引擎重現 navs_v4 的 AB_V4|normal，逐日差 0（gen_plans.py）。
- **執行面板**：曾入選的 521 檔，從共用面板匯出到本題 data/，PREREG 雜湊涵蓋。
- **Yahoo 5 分 K**（voidful/tw_stocker，2024-02-15～2026-04-01）：
  - 只下載曾入選的 498 檔，sha256 全部和 manifest 一致。
  - 去除重複時間戳後，只保留 09:00～09:25 的 K 棒與每日彙總，存在 raw/。原檔共 1.8GB，沒有放進研究資料夾。
  - Yahoo 的 09:00 那根 K 棒有 95% 的成交量是 0，所以不能直接當成開盤量。Yahoo 全日量約為官方的 0.873。
- **vol_curve**（2026-10 共 12 天、每 3 分鐘的累積量）：TWSE 股中有 192 個股日能對上官方日 K。
- **oddlot_hist**（2026-07-13～10-27）：有零股 bid、ask。
- **制度史**：WebFetch 對新聞網站 DNS 失敗，證交所與 web.archive 被 proxy 擋下（403），所以只存到搜尋摘要（raw/regulation_sources.json），需要日後用公告原文核對。
  - 2020-03-23：整股改逐筆交易，開盤與收盤仍是集合競價；之前全天約每 5 秒集合競價一次。
  - 2020-10-26：盤中零股上路，9:10 第一次撮合，之後每 3 分鐘撮合一次。
  - 2022-12-19：撮合間隔改為 1 分鐘。
  - 2024-12-02：撮合間隔改為 5 秒。
  - 規劃中：零股改 9:00 開盤（2026-12-07 或 2027-07，報導說法不一）、撮合間隔 1 秒、千元以上股票的升降單位改 1 元。
  - 多數券商整股最低手續費 20 元；不少券商零股最低 1 元。
- **CPI**：2008、2022、2025 三年經查證，其餘年份來自記憶。只用來換算幣值。

## 4. 探索發現（只看執行量，explore.json）

**(1) 拆分（FIX 金額，40 seed × 全部名額）**

| 級距 | 只能買零股的名額 | 2020-10 前 | 2020-10 後 | 零股金額占每檔 |
|---|---|---|---|---|
| 30 萬 | 86.2% | 83.1% | 93.9% | 89% |
| 100 萬 | 33.5% | 24.6% | 55.9% | — |
| 300 萬 | 13.2% | — | — | — |
| 1,000 萬 | 3.9% | — | — | — |

- 不足 1 股的現金殘差很小：30 萬為 0.35%，1,000 萬為 0.01%。
- 幾乎每一個名額都有零股餘額，所以每月每檔都會多一筆零股委託。
- 高價股（> 300 元）的名額占 12%。
- 近年股價上升，零股比例跟著上升：30 萬在 2024～2026 是 97～99%。

**(2) 開盤量**
- vol_curve 的開盤集合競價量占全日：中央估計中位數 3.4%（p25 1.0%、p75 6.4%），上限口徑 5.1%。
- κ（開盤量 ÷ 9:05～9:10 量）中位數 0.78。
- Yahoo 的 9:05～9:10 量 ÷ ADV60，中位數 2.4%：
  - 跳空 > 4% 時 7.2%，跳空 < 0.5% 時 1.8%。
  - 入選股換股日 6.3%，其他日 2.4%。
- 粗估開盤量約為 ADV60 的 1.9%；入選換股日約 4.9%。

**(3) 零股價差（前 300 池）**
- 買成本中位數 0.40%：價位 < 50 元為 0.22%，> 100 元為 0.43～0.48%。
- 賣成本中位數 0。
- 成交價相對收盤 +0.17%。

**(4) 9:10 價格相對開盤**
- 平均 −0.005%。依跳空分桶：跳空 < −3% 時為 +0.31%（開低後回升），跳空 > 1% 時為 −0.04%。
- 換股日中位數 −0.13%，標準差 2.0%。
- 9:10 時點本身沒有系統性成本，主要是雜訊。

**(5) 參與率預覽**
- 用粗估開盤量，1,000 萬 FIX 的新進股中有 68% 超過開盤量 10%。
- 2006～2018 是 78%，2019 年後是 53%。

## 5. 模型迭代紀錄（iterations.json，5 版，只用校準資料的擬合度選版）

**選版方法**
- 三個元件（開盤量回歸、9:10 drift 分桶、零股成本分桶）一起分版。
- 樣本切分：5 分 K 用 2024-02～2025-06 擬合、2025-07～2026-04 檢驗；零股用 7～8 月擬合、9～10 月檢驗。
- 分數＝三個元件的檢驗損失各除以 v1 的損失後相加。

**各版分數**

| 版本 | 分數 | 開盤量換股日 log RMSE |
|---|---|---|
| v1 | 3.000 | 1.364 |
| v2 | 2.905 | — |
| v3 | 2.869 | — |
| v4 | **2.746** | 1.052（偏誤 −0.12） |
| v5 | 2.765 | — |

- drift 與零股成本在各版幾乎沒有差別。
- **選 v4**：加入換股日的入選與剔除旗標，入選旗標係數 +0.77（換股日的開盤量約是平日的 2.15 倍）。
- 殘差 log 標準差約 1.06～1.13。開盤量的個別估計誤差約是 ×3 倍，這是後面衝擊估計的主要不確定性。

**PREREG 之前看過的東西（誠實揭露）**
- 模擬器除錯時，曾用佔位參數（OA＝5% ADV 常數）看過 seed 0 三個級距的 GROW 終值，以及 seed 3 的成本比例。
- 這些沒有用來決定任何參數。

## 6. 凍結規格
- 見 frozen_spec 欄位。
- PREREG sha256 6946fd75…，2026-10-28T02:39:10Z。
- 共 3,688 次模擬一次跑完（run_oos.py），之後沒有改過規格。

## 7. 樣本外結果

**本題的「樣本外」是什麼**
- 本題不產生新訊號，各期間都已經被 main_v2 和 o1～o4 看過，沒有盲測期。
- 這裡的樣本外是指：同一份凍結規格跑全期，再分 SEL／VAL／HOLD 報告。

**主格（手續費 A，40 seed 平均；年化／t）**

| 級距 | 版本 | 全期 | SEL t | VAL t | HOLD | 起點 2008 t | 拿掉最好 1／3／5 個月的 t | 逐 seed t 中位數（範圍；< 2.0 的 seed 數） |
|---|---|---|---|---|---|---|---|---|
| 30 萬 | GROW | 24.39／2.05 | 2.13 | 1.08 | 52.2%／0.06 | 1.55 | 1.83／1.48／1.15 | 2.02（1.85～2.24；15） |
| 30 萬 | FIX | 25.26／2.20 | 2.12 | 1.28 | 54.5%／0.15 | 1.72 | 1.99／1.64／1.33 | 2.18（1.96～2.40；1） |
| 100 萬 | GROW | 22.18／1.66 | 1.83 | 0.95 | 46.6%／−0.17 | — | — | — |
| 100 萬 | FIX | 25.10／2.18 | 2.09 | 1.24 | 56.0%／0.21 | 1.69 | 1.97／1.62／1.29 | 2.17（2 個 < 2.0） |
| 300 萬 | GROW | 19.88／1.23 | — | — | — | — | — | — |
| 300 萬 | FIX | 23.65／1.92 | 1.77 | 1.17 | 55.3%／0.18 | 1.44 | — | 1.90（36 個 < 2.0） |
| 1,000 萬 | GROW | 17.39／0.75 | — | — | — | — | — | — |
| 1,000 萬 | FIX | 20.88／1.42 | 1.35 | 0.92 | 51.0%／0.01 | 0.96 | — | 1.40（40 個 < 2.0） |

**對照組**
- 零衝擊 AB_V4：27.00%／2.51，SEL 2.39、VAL 1.44、HOLD 56.7%／0.24。
- 0050：14.94%（HOLD 59.2%）。
- 同池等權：11.19%。
- 主策略 main_v2 EXFIN n10：28.15%／2.64。
- 各級對同池等權的 t：FIX 3.40、3.36、3.05、2.45。對同池的選股力仍然在。

**手續費與壓力情境（t）**

| 情境 | 版本 | 30 萬 | 100 萬 | 300 萬 | 1,000 萬 |
|---|---|---|---|---|---|
| 手續費 B | GROW | 1.94 | 1.55 | 1.11 | 0.62 |
| 手續費 B | FIX | 2.18 | 2.05 | 1.77 | 1.26 |
| 手續費 C | — | 30 萬 FIX 2.41（26.47%） | — | — | — |
| 壓力 | GROW | 1.52 | 1.15 | 0.70 | 0.26 |
| 壓力 | FIX | 1.58 | 1.47 | 1.33 | 0.67 |

- 手續費 C（零股最低 1 元）只對 30 萬有明顯差別。

**與零衝擊的配對差（月）**

| 級距 | 版本 | 配對差 | t | MDE80（pp／年） |
|---|---|---|---|---|
| 30 萬 | FIX | −0.120pp | −2.54 | 1.59 |
| 100 萬 | FIX | −0.132pp | −4.53 | 0.98 |
| 300 萬 | FIX | −0.230pp | −9.17 | 0.84 |
| 1,000 萬 | FIX | −0.424pp | −11.3 | 1.26 |
| 30 萬 | GROW | −0.181pp | −4.88 | — |

**拆解（ΔCAGR，pp，GROW／FIX）**

| 級距 | 取整 d1 | 零股時點 d2 | 零股價差 d3 | 手續費 A d4 | 衝擊 d5 | 參與上限 |
|---|---|---|---|---|---|---|
| 30 萬 | +0.64／+1.50 | +0.15／−0.28 | −0.87／−2.21 | +0.51／−0.50 | −2.99／−0.26 | — |
| 100 萬 | — | — | −0.32／−1.31 | — | −5.84／−1.69 | — |
| 300 萬 | — | — | — | — | −9.18／−4.15 | +1.24／+0.15 |
| 1,000 萬 | — | — | — | — | −13.60／−8.67 | +3.12／+1.77 |

- d1 為正，是因為 d1 拿掉了零股的 0.1% 基礎滑價。d1＋d2＋d3 合計才是零股淨效果：30 萬 FIX 約 −0.98pp，GROW 約 −0.08pp。
- d4 手續費 A 對大多數級距為正，因為 2.8 折低於引擎假設的 6 折；30 萬 FIX 為負，是最低 20 元的影響。
- 參與上限為正（降低衝擊），代價是順延與現金閒置。

**年化成本（佔權益 %／年，GROW／FIX）**

| 項目 | 30 萬 | 100 萬 | 300 萬 | 1,000 萬 |
|---|---|---|---|---|
| 證交稅 | 2.2 | 2.2 | 2.2 | 2.2 |
| 手續費 | 0.89／1.66 | 0.68／0.84 | 0.62／0.67 | 0.59／0.61 |
| 衝擊 | 2.22／0.20 | 4.11／1.34 | 5.72／3.10 | 6.97／5.21 |
| 零股價差 | 0.75／1.73 | 0.29／1.03 | 0.13／0.47 | 0.05／0.16 |

- 現金拖累：平均現金 0.2～3.7%；取整追蹤誤差 0.24～0.53pp。

**損益兩平**
- 各級的 t 都已低於 2.3，所以額外滑價空間是 0。
- 要守住 t 2.0：30 萬 FIX 只容得下再多 0.061%，100 萬 FIX 0.054%。

**執行量（FIX）**

| 級距 | 新進股只能買零股 | 整股單超過開盤量 10% | 超過 20% |
|---|---|---|---|
| 30 萬 | 83.9% | 0 | 0 |
| 100 萬 | 33.2% | 5.3% | 0.9% |
| 300 萬 | 13.4% | 24.8% | 11.1% |
| 1,000 萬 | 3.7% | 48.7% | 33.7% |

- 1,000 萬超過 10% 的比例：2006～2018 為 57.3%，2019～2026 為 35.1%。

**預測核對**

| 預測 | 內容 | 結果 | 判斷 |
|---|---|---|---|
| P1 | 30 萬零股名額 75～85% | 83.9% | 成立 |
| P1 | 盤後零股使 t 降 0.15～0.3 | d2 的 Δt 為 −0.045（FIX）、+0.026（GROW） | 不成立 |
| P1 | 最低 20 元每年多付 0.5～1.5pp | A 對 C 差 1.2pp | 成立 |
| P1 | 年化少 1.5～3pp | 1.74（FIX）、2.61（GROW） | 成立 |
| P2 | 100 萬零股 40～60% | 33.2%（2020-10 後 54%） | 不成立 |
| P2 | 年化少 0.5～1.5pp | 1.90（FIX） | 不成立 |
| P3 | 300 萬年化少 0.3～1.0pp、t 2.3～2.45 | 3.35pp、t 1.92 | 不成立 |
| P4 | 1,000 萬超過 10% 的比例 15～35% | 48.7% | 不成立（更嚴） |
| P4 | t 2.0～2.25 | 1.42 | 不成立 |
| P4 | 2019 年後超標比例較低 | 是 | 成立 |
| P5 | 零股成本在 > 300 元較大 | 12.4bp 對 13.5bp | 不成立 |
| P5 | 零股成本在 2020-10 前較大 | 12.8bp 對 14.4bp | 不成立 |
| P5 | 衝擊集中在名次 201～300 | 衝擊占 43%，成交值只占 33% | 成立 |
| P5 | 衝擊集中在跳空大的日子 | 衝擊占 5% < 成交值占 6% | 不成立（模型使跳空日的開盤量變大，是模型內生結果） |
| P6 | 開盤量占全日 2～6% | 3.4% | 成立 |
| P6 | 中小型股開盤量占比較低 | vol_curve 方向相反（樣本多為爆量股）；Yahoo 回歸的 lnrank 係數 −0.05，只有弱支持 | 不成立 |

## 8. 穩健性
- **分年**（FIX 對零衝擊的年差）：
  - 300 萬：21 年中 21 年為負，2017 年 −6.2pp、2019 年 −6.4pp 最大；2024～2026 年縮小到 −1.0～−1.7pp。
  - 30 萬：2009 年 +7.8pp，其他多數年份為 −1～−5pp。
  - GROW 越到後段越差（帳戶越大）。
- **分產業**（300 萬、1,000 萬 FIX，seed 0～9）：衝擊成本的產業占比和成交值占比相近。建材營造 14.1%（成交值 12.4%）、電子零組件 10%、電腦週邊 8.7%、半導體 7.8%。沒有單一產業主導。
- **去掉最大成本股**：最大的是 3030，占衝擊成本 1.2%；前 10 檔合計 9%，共 498 檔。成本很分散。
- **敏感度**（GROW、A，t）：

| 情境 | 30 萬 | 100 萬 | 300 萬 | 1,000 萬 |
|---|---|---|---|---|
| κ 低 | 1.81 | 1.34 | 0.89 | 0.39 |
| κ 高 | 2.10 | 1.74 | 1.33 | 0.85 |
| 2020-03 前 OA 減半 | 1.97 | 1.52 | 1.08 | 0.54 |
| 零股壓力 | 1.80 | 1.57 | 1.18 | 0.73 |
| KEEP（續抱不調整，SOP 第 6 步） | 2.18 | 1.78 | 1.36 | 0.98 |

- KEEP 對各級都略好，但在本規格下仍不改變判定。
- **V4 非 AB 主名單**：FIX 2.26、2.25、1.99、1.52。
- **事後比較（不是判定欄）**：衝擊分母改用全日 V60 時，FIX 的 t 為 2.24、2.44、2.50、2.39，GROW 為 2.49、2.43、2.27、1.98。和 o4 的「1,000 萬 t 2.21」同一量級。
- **獨立複製**：replicate/rep_sim.py 依 frozen_spec 另寫，跑 300 萬與 30 萬（GROW、A）共 40 seeds，逐日最大相對差 1.9e−14，通過 1e−6 門檻。但複製程式是同一位研究員寫的，建議審查員再做一次真正獨立的複製。

## 9. 機制是否被支持：部分支持
- **支持的部分**：
  - 四個摩擦隨資金規模往相反方向變動，這點被量化確認：零股價差與最低手續費隨規模下降，衝擊隨規模上升。
  - 零股市場有「買方溢價」：買成本約 0.40%，賣成本約 0。
  - 開盤量只占全日的約 3%。
  - 衝擊集中在名次 201～300 的中小型股。
  - 2019 年後流動性改善，超標比例下降。
- **不支持的部分**：
  - 甜蜜區間不在 100～300 萬。在本規格下，30～100 萬最好，300 萬以上明顯打折。
  - 小帳戶相對 300 萬沒有被懲罰，反而勝出。
  - 2020-10 前的盤後零股（收盤代理）幾乎沒有時點成本。
- **無法判斷的部分**：
  - 衝擊係數套用在開盤量上是否正確。這一點決定整個結論。

## 10. 散戶怎麼執行（SOP 容量表，只寫事前規格的結果；本表為保守上限）

| 資金（2026 幣值） | GROW 年化／t | FIX 年化／t | 新進股只能買零股 | 判定 GROW（FIX） |
|---|---|---|---|---|
| 30 萬 | 24.4%／2.05 | 25.3%／2.20 | 84% | 可行但打折（可行但打折） |
| 100 萬 | 22.2%／1.66 | 25.1%／2.18 | 33% | 不建議（可行但打折） |
| 300 萬 | 19.9%／1.23 | 23.7%／1.92 | 13% | 不建議（不建議） |
| 1,000 萬 | 17.4%／0.75 | 20.9%／1.42 | 4% | 不建議（不建議） |

- 零衝擊回測是 27.0%、t 2.51；0050 是 14.9%。
- **下單方式**：
  - 整股：成交日開盤市價，參加集合競價。
  - 零股：2020-10-26 起在盤中零股 9:10 首次撮合，用限價當日有效（盤中零股只有這一種委託方式）。2020-10 以前是盤後零股。
  - 手續費選 2.8 折以下，零股最低 1 元的券商。對 30 萬帳戶來說，這一項就差約 1.2pp／年（t 2.20 → 2.41）。
- **可以拿掉的警示**：依事前規則，「資金太小大量零股」的資金下限警示可以拿掉（30 萬沒有比 300 萬差）。但要改寫成：「零股的實際成本約為買進時 0.4%，請選零股最低 1 元的券商。」
- **只列為假說，不寫進 SOP**，交給 live_tracking 前瞻記錄：
  - 分 3 個開盤下單、限價單；
  - 300 萬以上忽略零股餘額；
  - 大額部位改在盤中或收盤成交；
  - 續抱不調整（KEEP）。
- **事前登記的校準規則**：2026-11 起，滿 6 個換股日後，如果「實際成交價相對開盤價」的價差中位數比模型高 > 0.05%，容量表重新發布。
- **反方向也要檢查**：如果實際價差明顯低於模型的 0.1%＋0.5σ√(Q/OA)（例如 100 萬帳戶的整股單衝擊遠低於模型的 20～30bp），也應重新發布，屆時本表會大幅上修。
- **盤中零股若改成 9:00 開盤**（報導說法是 2026-12-07 或 2027-07），零股時點參數要重估。

## 11. 限制
1. **衝擊公式沒有實證校準。** 導言指定的 0.5σ√(Q÷開盤量) 是把平方根法則從全日量搬到開盤集合競價量。這會讓衝擊放大 4～6 倍，而判定幾乎完全取決於它（事後改全日量口徑時，四級 t 都在 2.2～2.5）。台股沒有可下載的個股開盤集合競價委託簿，只能用前瞻實盤校準。
2. **開盤量是兩段推估。** 第一段用 Yahoo 9:05～9:10 量回歸，第二段乘上 κ（來自 vol_curve 的 12 天觀察名單股，多為爆量日）。個別估計誤差約 ×3。2020-03-23 以前是每 5 秒集合競價的舊制，定義不同；回推誤差另做了 OA 減半的敏感度。
3. **零股成本只有 2026-07～10 的收盤時點簿。** 套用在 9:10 首次撮合與 2006～2020 的盤後零股，都是代理。
4. **GROW 的級距在 20 年後已經不是原本的級距**（終值 ×70～×120）。FIX 才是「目前資金」的口徑。
5. **手續費情境全期固定。** 2015 年以前 2.8 折並不普遍，所以手續費 B 是早期較合理的口徑；在 B 下各級 t 低 0.1～0.15。
6. **用「已知開盤價」決定每檔目標金額**（與引擎相同的理想化）；實盤需要保留 1～2% 現金。處置股預收款券、開盤跌停賣不掉等也沒有建模。
7. **制度史只查到搜尋摘要**，原文頁面全部連不到，需要用證交所公告核對。CPI 有 17 年的數字來自記憶。
8. **複製由同一研究員另寫**；建議審查員重做。
9. **全期都已被其他研究看過，沒有盲測期**；前瞻驗證從 2026-11 開始。results.json 的 zero_impact_AB_V4_stress 欄是程式誤用（沿用一般成本），壓力成本請以 V4 正式數字的 23.93%／1.97 為準。

**主要檔案**（都在本題資料夾）
- 程式：gen_plans.py、build_exec_data.py、fetch_5m.py、xlib.py、ecore.py、explore.py、iterate.py、prereg.py、run_oos.py、robust.py、replicate/rep_sim.py。
- 結果：explore.json、iterations.json、results.json（PREREG、RESULTS、ROBUST、REPLICATION）、navs_main.parquet（主格逐 seed 淨值）、navs_abavg.parquet（各變體 40 seed 平均）、replicate/rep_result.json。
- 資料：data/（名單、執行面板、model_spec.json）、raw/（5 分 K 摘要與下載紀錄、screener 零股／vol_curve／日 K 副本、regulation_sources.json、cpi_tw.json）。

## 凍結規格

PREREG：sha256 6946fd75c4b7495f408258c4261dcf544a23d11e86db8be5532161a94084b1d0，2026-10-28T02:39:10Z（UTC），存在 results.json。雜湊涵蓋 57 個檔案：本題所有 .py（含 replicate/rep_sim.py）、explore.json、iterations.json、data/ 與 raw/ 全部檔案，以及引擎庫 exec_rebuild/exlib.py、o_v4_revenue_patch_official/v4lib.py、main_v2/v2_core.py、independent_audit/ia_core.py、o_infra_v3_rebuild/o3_core.py、o_disposal_rejudge/drj_core.py、data_freeze_v4.json、navs_v4.parquet、plan_V4.csv。run_oos.py 在計算前會斷言 PREREG 存在、沒被改過、所有檔案雜湊相同，而且 RESULTS 還不存在。
資料夾：/home/user/-/research/tw_stock_selection/agents/o_exec_capacity_tiers/

〇、範圍
- 目的：用同一張 AB_V4 名單（逐 seed），模擬 30 萬、100 萬、300 萬、1,000 萬四個資金級距的可執行回測，交出 SOP 容量表。選股規則完全不改。
- 名單：data/plans_ab.csv（seed 0～39，欄位 seed、signal、exec、exec_idx、picks）。
  - 由 gen_plans.py 以 v4lib 產生：V4(R_FIX, Y5_FIX, 處置 trade 口徑, unavail=v2_core.make_unavail('mc', seed)).plan(10,'official')。
  - 零衝擊引擎（O3.simulate、一般成本）重現 navs_v4.parquet 的 AB_V4|normal，40 seed 平均日報酬淨值逐日差 0。
  - data/plan_v4.csv 是非 AB 的主名單（seed −1），只報告。
- 期間：從 2006-04-11（第一個訊號日）收盤起算，到 2026-10-12 收盤。
  - 年化＝(淨值_t1 ÷ 淨值_t0)^(365.25 ÷ 天數) − 1。
  - 用簡單報酬；現金不計息。

一、資料（全部在本題 data/、raw/）
- 價格檔：px_open、px_close（原始價，下市股價格 P 已補）、px_grow（1＋ret，無成交時＝1）、px_traded、prevclose（前一筆收盤，跨停牌）。
  - 只含曾經入選的 521 檔，日期是面板全期。
- 衍生欄位：
  - CF＝收盤價前值補。
  - ADJ_t：有成交時＝G_t × PC_t ÷ C_t（非有限值→1）；無成交＝1。
  - GAP_t＝O_t ÷ PC_t − 1，只在有成交而且有開盤時有值；需要用到而缺值時當 0。
- feat_daily：
  - V60＝gapfix 口徑、60 個 GOOD 日的平均成交值。
  - sig60＝60 個 GOOD 日的 ret 標準差（至少 40 筆，向後補值）。
  - vrank＝上市股中「V60 ≥ 1e7 而且收盤 ≥ 10」者，依 V60 由大到小的名次。
  - 成交日 t 一律用 t−1 的值。
- xrules.json：X 出清規則。
- model_spec.json：執行量模型。
- bench.parquet：0050 含息、AB_V4|normal、AB_V4|EW、AB_V4|stress。
- raw/cpi_tw.json：CPI 換算用（2006 → 2026 為 ×0.765）。

二、帳戶與級距
- 級距 tier 以 2026 幣值計：30 萬、100 萬、300 萬、1,000 萬。換成當年幣值時乘 CPI[年] ÷ CPI[2026]。
- GROW（判定用）：
  - t0 收盤持有現金＝tier × CPI[2006] ÷ CPI[2026]。
  - 之後隨自身淨值成長，不提領、不加碼。
- FIX（另報）：
  - 每個換股日開盤前，把全部股數與現金乘上 sc＝tier × CPI[該年] ÷ CPI[2026] ÷ 權益，不計成本，然後才換股。
  - 權益＝可交易持股 × 開盤價＋凍結持股 × 前收＋現金。
  - 換股日淨值用時間加權：unit ×＝(權益_pre ÷ 前日收盤權益) × (收盤權益 ÷ (權益_pre＋flow))，flow＝target − 權益_pre。
- 非換股日：unit ×＝收盤權益 ÷ 前日收盤權益。收盤權益＝現金＋Σ 股數 × CF_t。

三、執行量模型（iterations.json 5 版中選 v4，以全部校準資料重擬）
- 開盤集合競價成交值 OA＝V60_{t−1} × κ × exp(b·x)。
  - κ＝0.77886（vol_curve 中「OA_central ÷ 9:05～9:10 量」的中位數）。
  - OA_central＝V(09:01) − (V(09:04) − V(09:01)) ÷ 3。
  - x＝[1, ln max(vrank,1), ln PC_t, ln max(σ,1e−4), |GAP_t|, pick, drop]。
  - b＝[−3.05646, −0.05438, 0.03689, 0.21549, 20.5129, 0.76624, −0.05387]，以 model_spec.json 的全精度為準。
  - pick＝換股日當天的入選股（含續抱）；drop＝換股日被剔除的持股；順延日兩者都是 0。
- drift（9:10 價格 ÷ 開盤 − 1）：
  - 依 GAP_t 分桶，邊界 [−3%, −1%, 0, 1%, 3%]，用 searchsorted 'right'。
  - 各桶的值為 [+0.308%, +0.105%, −0.0004%, −0.038%, −0.040%, +0.062%]。
- 零股成本 c（相對同時點的整股價，正值＝不利）：
  - 依 PC_t 分桶 [50, 100, 300]，再依 vrank_{t−1} 分桶 [150]（vrank 缺值當 1e9），各取中位數。
  - 買：[[0.125%, 0.283%], [0.302%, 0.391%], [0.433%, 0.527%], [0.424%, 0.455%]]。
  - 賣：[[0, 0], [−0.101%, −0.123%], [0, −0.178%], [−0.176%, −0.195%]]。

四、逐日流程（t＝t0+1 … t1）
1) 股數 S ← S × ADJ_t。
2) 若 t 是成交日 e：
   a) 清掉尚未成交的順延委託，並記錄未成交金額。
   b) 判定凍結：持有而且當日無成交或無開盤。凍結股若有 X 規則，而且 release ≤ t < valid_until，則現金＋＝S × CF_{t−1} × factor，S 歸 0。
   c) FIX 版在此縮放。
   d) 可交易持股的小數股（S − floor(S + 1e−9)）以 O_t 轉成現金，不計成本。
   e) 換股（第五節）。
3) 否則，若有順延委託，執行第六節。
4) 計算收盤權益與淨值。

五、換股（t＝e，入選名單 pk，k＝檔數）
- cur＝可交易而且 S > 0 的持股，股數取整。
- 目標股數（入選股）：
  - p＝PC_t。
  - 張數 L＝floor(T ÷ (1000p) + 1e−12)。
  - 零股 q＝floor((T − 1000Lp) ÷ p + 1e−9)。
  - 目標＝1000L＋max(q, 0)。不足 1 股的部分留現金。
- 非入選的可交易持股：目標＝0。
- delta＝目標 − cur；整股張數＝|delta| // 1000，零股股數＝|delta| mod 1000。
- 單檔成本 cost(j, delta)：買＝現金支出，賣＝負的淨收入；side＝±1。
  - 整股部分：
    - imp＝0.001＋0.5 × σ_{t−1} × √(1000 · L · PC_t ÷ OA)。第二項只在 OA 有限且 > 0、σ 有限時才加。
    - val＝1000 · L · O_t · (1 + side · imp)。
    - cost 加上 side · val＋fee(val)；賣出再加 0.003 · val。
  - 零股部分：
    - 基準價 b：t < 2020-10-26 用 C_t（盤後零股，以成交日收盤代理）；t ≥ 2020-10-26 用 O_t × (1 + drift(GAP_t))（盤中零股 9:10 首次撮合）。
    - val＝q · b · (1 + side · c)。
    - cost 加上 side · val＋fee(val)；賣出再加 0.003 · val。
- 每檔目標金額 T：
  - hi＝(Σ_cur S × O_t＋現金) ÷ k。
  - need(T)＝Σ_j cost(j, delta_j(T))。
  - 若 need(hi) ≤ 現金，T＝hi。否則在 [0, hi] 上二分 40 次：need(mid) ≤ 現金時 lo＝mid，否則 hi＝mid；最後 T＝lo。
  - 估計時用當日實際開盤價與收盤價。這和引擎一樣，是「已知開盤價」的理想化。
- 執行順序：
  - 依 delta 由小到大排序，所以賣單先執行。
  - 買單若 cost(j, delta) > 現金＋1e−9，整筆列入順延。
  - 每檔先下零股，再下整股。
- 整股參與上限：
  - 當日最多成交 max(1, floor(0.20 × OA ÷ (1000 · PC_t))) 張，超出的張數列入順延。
  - 衝擊用實際成交的張數計算。
- 手續費 fee(val)：
  - 情境 A（主欄）：max(20, 0.001425 × 0.28 × val)，整股與零股都是最低 20 元。
  - 情境 B：0.6 折，最低 20 元。
  - 情境 C：0.28 折，整股最低 20 元，零股最低 1 元。
  - 證交稅：賣出 0.3%。
  - 每檔每日的整股與零股各算一筆，各自適用最低手續費。
- KEEP（敏感度）：已持有的入選股不下單。

六、順延日
- 依待成交股數由小到大處理，賣單先。
- 當日無成交或無開盤的股票，原委託留著。
- 賣：數量＝min(待賣股數, round(S))。
- 買：
  - 張數上限＝floor(現金 ÷ (O_t × 1000 × 1.05))。
  - 若零股成本 > 現金 − 張數 × O_t × 1050，當日不下零股。
- 先下零股，再下整股；整股同樣受參與上限與衝擊限制，pick＝drop＝0。
- 沒成交的部分留到隔日。到下一個換股日時清掉。

七、變體（事前登記，一次跑完）
- 主格：40 seeds × 4 個級距 × {GROW, FIX} × 手續費 {A, B, C}。
- 拆解（手續費 A），依序加入：
  - d1：取整（整股＋零股，全部在開盤成交；手續費 0.6 折、無最低；零股無價差；整股 0.1%）。
  - d2：零股時點。
  - d3：零股價差。
  - d4：手續費 A。
  - d5：平方根衝擊。
  - 主格：再加 20% 參與上限與順延。
- 敏感度（手續費 A、GROW）：
  - κ 低（p25＝0.310）、κ 高（V(09:01) 口徑＝1.038）。
  - 2020-03-23 以前 OA × 0.5。
  - 零股壓力：買成本 × 2；賣成本 ≤ 0 的改為 0。
  - KEEP。
- 壓力（GROW 與 FIX）：手續費 B＋κ 低＋零股壓力。
- V4 主名單（seed −1）：手續費 A，只報告。

八、統計（每個變體都把 40 seed 的日報酬平均後連乘，得到一條「可執行 AB」淨值；另報逐 seed）
- 對 0050 的 t：訊號窗報酬（訊號日收盤 → 下一個訊號日收盤）減 0050 含息同窗報酬，用一般 t。
- 分段：
  - S1：2006-04～2013；S2：2014～2020；S3：2021-03～2026-10-12。
  - SEL：2006-04～2018；VAL：2019～2023；HOLD：2024～2026-10-12。
  - FROM2008：2008 起。
  - PRE_ODD：到 2020-10-25；POST_ODD：2020-10-26 起。
- 端點：拿掉最好 1／3／5／10 個月；拿掉最後 1 個月；拿掉第一年；起點改 2007。
- 逐 seed：t 的中位數、範圍、t < 2.0 與 t < 2.3 的 seed 數；年化的中位數與範圍。
- 配對差（對零衝擊 AB_V4）：月差的平均、t、MDE80＝2.8 × SE。
- 成本拆解：每筆成本金額 ÷ 當日開盤前權益，加總後除以年數。
- 損益兩平：在每日報酬中扣 x × 當日成交金額 ÷ 權益，找出 t 跌到 2.3、2.0 時的 x。
- 執行量統計：零股名額比例、超過開盤量 10% 與 20% 的委託比例、衝擊集中度。
- 取整追蹤誤差：d1 對零衝擊的年化追蹤誤差，取逐 seed 中位數。

九、判定（每級一格；主欄＝GROW、手續費 A 的可執行 AB；歷史回測的上限是待觀察）
- 建議：t ≥ 2.3，而且年化比零衝擊 AB_V4（27.00%）少 ≤ 1.5pp。
- 可行但打折：t ≥ 2.0。
- 不建議：t < 2.0，或對 0050 的年化超額 < 2pp。
- 另報 FIX、手續費 B、壓力情境下的判定。
- 可證偽：30 萬與 300 萬（FIX）年化差 < 0.5pp（附 MDE）時，「小帳戶被零股制度懲罰」不成立。GROW 另報。

十、SOP 容量表（只寫事前規格的結果）
- 每級寫：預期年化（GROW／FIX）、t、零股名額比例、下單方式。
  - 整股：成交日開盤市價（參加集合競價）。
  - 零股：2020-10-26 起在盤中 9:10 首次撮合；之前是盤後零股。
- 事後想到的執行建議一律不進 SOP，只列為假說。
- 校準規則：2026-11 起，live_tracking 累積滿 6 個換股日後，如果「實際成交價 − 開盤價」價差的中位數比模型高 > 0.05%，容量表要重新發布。
- 盤中零股若改成 9:00 開盤，要重估零股時點參數。

## 反方審查

[
 {
  "final": "不可用",
  "issues": [
   "唯一「可行但打折」的級距（30 萬 GROW，t 2.05）只比門檻高一點，經不起任何單一擾動：拿掉最好的 1 個月 t 1.83；起點改 2007 是 1.79；起點改 2008 是 1.55；手續費 B（2015 年以前較合理的口徑）是 1.94；40 個 seed 中 15 個 t<2.0（中位數 2.02）。我的逐年剔除檢驗（review_stat_o5.py）：21 個年份中，拿掉任何一個年份就有 12 個會讓 t 跌破 2.0；拿掉 2021 年 t 剩 1.64。這不是穩健的可行，往不利方向判應該算不建議。",
   "顯著性全部來自早期與少數幾年。可執行 AB 在 2019～2026 的 t：30 萬 GROW 0.85、FIX 1.06，100 萬 FIX 1.07，300 萬 FIX 0.99；2014～2026 全部 <1.45；HOLD 期 30 萬 GROW 年化 52.2%，低於 0050 的 59.2%。超額有 55～60% 來自 2021、2015、2006 三年。連零衝擊 AB_V4 本身，拿掉 2006～07 後 t 也只剩 1.99，2019～2026 是 1.24。也就是說，近 8 年的資料對任何一個級距都沒有支持。",
   "多重檢定：AB_V4 是 main_v2 → o1～o4 → v4 修補這條研究線的倖存版本（o_v4 有 6 版，本題 5 版，agents 底下約 148 個題目資料夾）。即使只用 N＝5 的 Bonferroni，門檻也是 t 2.58，N＝10 是 2.81。四個級距（最高 2.20）全部不過，零衝擊 2.51 也不過。判定欄用 t≥2.0、2.3 而不做打折，會高估可行性。",
   "40 seed 平均的 t 不是單一帳戶的 t。seed 是營收公告時點的蒙地卡羅抽樣，實盤只會走其中一條路徑。逐 seed 來看：30 萬 GROW 只有 62.5% 的 seed t≥2.0，300 萬 FIX 只有 10%。主判定應改看逐 seed 分布，或至少把 seed 不確定性一起報。",
   "『小帳戶沒被懲罰、可以拿掉資金下限警示』這條 SOP 修改不穩健。(1) 30 萬對 300 萬的 FIX 配對優勢只在 2020-10 以前顯著（t 2.55）；在現行盤中零股制度下（2020-10-26 以後），t 只有 0.96、每年 +0.35pp，2025、2026 兩年反而是負的（−0.44pp、−2.35pp）。(2) 2020 年以前的零股成本是用 2026-07～10 的收盤簿套回 2006～2020，加上收盤價代理，是全研究最不確定的一段。(3) 這個優勢完全來自未校準的開盤量衝擊：事後把分母改成全日 V60，30 萬 FIX 是 25.48%，300 萬是 26.97%，30 萬反而落後 1.49pp，依事前的 falsify 規則（差 ≥0.5pp），結論變成『小帳戶被懲罰』。同一份資料、換一個分母，結論就翻轉，不能寫進 SOP。",
   "『隨資金單調下降、沒有甜蜜區間』同樣是衝擊係數決定的結果，不是資料告訴我們的。事後全日量口徑下，FIX 的 t 是 2.24、2.44、2.50、2.39，呈倒 U 形，100～300 萬最好，正好是原本預測的甜蜜區間。研究員自己也承認係數沒有經過校準，所以『單調下降』只能算模型假設的推論，不能算發現。",
   "把容量表稱為『保守上限』沒有根據：有幾個偏差是往有利方向（低估成本）。(a) 零股委託完全沒有衝擊、也沒有成交量上限：999 股的高價股零股單也照單全收，2020 年以前的盤後零股市場又非常薄。(b) OA 回歸用了當日實際的 |GAP_t|（係數 20.5），以及從 2024～26 自家入選股估出來的 pick 旗標（係數 +0.77，等於把 OA 放大 2.15 倍、衝擊降約 32%）；兩者都是成交當下才知道的資訊，而且是在 9:05～9:10 的量上擬合，再乘上單一常數 κ。(c) 用已知的開盤價決定每檔目標金額，沒有保留現金。(d) 零股賣成本中位數 ≤0（部分為負），只來自 3.5 個月的簿。方向不明，最多只能稱為『中央估計，誤差 ×3』，不能稱為上限。",
   "前視與資料來源：零股成本表（2026-07～10）、9:10 drift 與 OA 回歸（2024-02～2026-04）都是用未來的資料套回 2006 年以後的歷史；這不是訊號前視，卻是成本面的時期錯置，2020-03-23 以前的舊制（每 5 秒集合競價）口徑也不同。CPI 有 17 年的數字來自記憶，直接影響 GROW 的起始資金（2006 年 ×0.765）。制度史只查到搜尋摘要。這些都還沒核對。",
   "非問題（已核對）：月窗不重疊，自相關約 0（acf1 在 −0.003～0.025），Newey-West t 反而略高於一般 t，所以不是重疊樣本的問題。區塊 bootstrap 的 t 和一般 t 一致。gen_plans 逐日重現零衝擊的差是 0。sig60 是向前補值，V60、σ、vrank 都用 t−1，選股訊號本身沒有發現新的前視。"
  ],
  "note": "驗證程式：/home/user/-/research/tw_stock_selection/agents/o_exec_capacity_tiers/review_stat_o5.py（用 python3 -I -B 執行；只讀，沒有改研究員的任何檔案）。輸出副本：/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/review_o5_out.json。\n\n結論：總判定應從「待觀察」下修為「不可用」，理由有四。\n(1) 沒有任何級距同時過得了多重檢定門檻（N≥5 時 t≥2.58）和近期子樣本（2019～2026 的 t 全部 ≤1.07）。\n(2) 唯一「可行」的 30 萬 GROW（t 2.05）是邊緣值：拿掉 21 年中任何一年，有 12 年會讓 t 跌破 2.0；40 個 seed 中有 37.5% 的 t<2.0；手續費 B 下只有 1.94。\n(3) 兩個結構性結論（單調下降、小帳戶不被懲罰並據此拿掉資金下限警示）在事後換成全日 V60 分母時方向會翻轉。30 萬對 300 萬的優勢在 2020-10 以後也不顯著（t 0.96）。這兩條不能寫進 SOP。\n(4) 「保守上限」的說法沒有根據。零股沒有衝擊、也沒有量上限；OA 用了當日的實際跳空和 pick 旗標放大；用已知開盤價定目標金額。這些都會低估成本。\n\n容量表可以保留作為成本量級的參考，但判定欄和「拿掉警示」應該撤回，等 2026-11 以後的實盤成交校準後再重做。"
 },
 {
  "final": "待觀察",
  "issues": [
   "Main column (GROW, fee A): with four frictions the spec leaves out, all four tiers rate 不建議 under the researcher's own rule. The frictions are: limit-up/limit-down open with retry next day, odd-lot sell cost floored at 0, a 2% cash buffer, and fee B before 2015. With them, 30萬 GROW drops from t 2.05 to 1.94 (CAGR 24.39 to 23.78%). The researcher's own fee B (1.94), odd_stress (1.80), stress (1.52) and start-2008 (1.55) runs all land below 2.0 as well, and 15 of 40 seeds are below 2.0. The 「可行但打折」 rating for 30萬 GROW does not hold up.",
   "FIX survives only barely: with all four frictions, 30萬 is t 2.06 and 100萬 is t 2.05, and 300萬 drops further to 1.78. From 2008 these cells are 1.58, 1.57 and 1.31. In the HOLD period (2024-2026) every tier's t against 0050 is only 0.03-0.21, so the most recent 2.8 years show no excess over 0050. For someone who has X 元 today, the table's 「可行」 cells rest on 2006-2018.",
   "The table calls itself a 「保守上限」 (conservative upper bound), but only the impact coefficient may be conservative. Other frictions are missing or optimistic. The odd-lot sell cost table has negative values (-0.10% to -0.20%), so selling odd lots earns more than whole lots. That alone costs 30萬 FIX 0.5pp/yr and t 2.20 to 2.11. Having to buy at the known open price assumes zero cash buffer, and a 2% buffer costs another 0.3pp/yr. Fee A's 2.8折 applied over the whole period is optimistic for early years (the researcher's limitation 5). The headroom before t falls below 2.0 is only 0.054-0.061% per trade, so these items can each flip the verdict.",
   "The SOP says 「整股開盤市價集合競價、可以不盯盤」 (buy whole lots at the open auction, no need to watch the screen). That conflicts with the plan rule: an AB_V4 pick that opens limit-up is skipped and the next-ranked stock bought instead. V4 does this 13 times in 245 rebalances (seed 0: 11 times). The backtest gives the substitute that day's open price, but the swap is only known at 9:00 (before 2020-03 there was no trial-match disclosure). In practice someone must watch at 9:00 and buy during continuous trading. Also, at 300萬 and above, 11-34% of whole-lot orders exceed the 20% participation cap and roll over across days, so execution needs daily monitoring. The SOP should state both requirements.",
   "The odd-lot model has gaps. Fill rates and odd-lot volume capacity are not modeled: every odd-lot limit order is assumed to fill in full at 9:10, or for pre-2020-10 after-hours odd lots at the close price. The odd-lot spread comes from just 3.5 months of 2026 closing-time order books and is applied to the 9:10 first match and to 2006-2020, which is a proxy. For 30萬, 84-94% of positions are odd-lot only, so this is the main risk for the small tier.",
   "The impact model sets the conclusion (the researcher admits this). The opening-auction volume estimate rests on 192 stock-days from vol_curve, mostly volume-spike stocks. The OA share is 0 at p10 and only 0.28-0.76% for vrank 0-200, so κ is fragile. The ×3 error is a big uncertainty in both directions. The post-hoc full-day V60 version (t 2.24-2.50) must not be used to raise the table. Under the adverse rule the frozen spec stands.",
   "Checks that found no problem: AB_V4 picks already exclude stocks under disposition (處置) on the trade date, and new entries opening limit-up are skipped and backfilled (0 of 60,740 new entries opened limit-up). Exits opening limit-down are 0.16% (99 of 60,337), and retrying the next day changes almost nothing (blocked trades come to 0.01% of equity). My independent rewrite of the rebalance and pending-order steps (reusing ecore's cost functions) reproduces all four main cells exactly."
  ],
  "note": "驗證程式：/home/user/-/research/tw_stock_selection/agents/o_exec_capacity_tiers/review_exec_o5.py（python3 -I -B，只 import ecore.py，沒有改研究員的檔案）。完整輸出在 /tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/rev_exec.log。\n\n以下是各格在不同情境下的結果，每格為「年化%／對 0050 的 t」，40 seed 平均、手續費 A 為底：\n\n| 格子 | base | 只加 ODD0 | 只加 CASH2 | 只加 FEEMIX | ALL | ALL 起點 2008 t | ALL HOLD t |\n|---|---|---|---|---|---|---|---|\n| 30萬 GROW | 24.39／2.05 | 2.00 | 2.00 | 2.02 | 23.78／1.94 | 1.45 | 0.03 |\n| 30萬 FIX | 25.26／2.20 | 2.11 | 2.15 | 2.20 | 24.47／2.06 | 1.58 | 0.07 |\n| 100萬 FIX | 25.10／2.18 | 2.11 | 2.15 | 2.13 | 24.38／2.05 | 1.57 | 0.15 |\n| 300萬 FIX | 23.65／1.92 | 1.87 | 1.87 | 1.86 | 22.91／1.78 | 1.31 | 0.13 |\n\n情境定義：\n- base：照研究員的凍結規格，四格都和研究員的數字完全一致。\n- LIM／LOCK：開盤（或開收都）在漲跌停時不成交、隔日重試，影響小於 0.05 個 t。\n- ODD0：零股賣成本下限設為 0。\n- CASH2：每次換股保留 2% 現金。\n- FEEMIX：2015 年以前用手續費 B（0.6 折），之後用 A。\n- ALL：以上四項一起加。\n\n我的判斷：「待觀察」本身成立，但 SOP 容量表要下修。\n- 主欄 GROW 四級都應該是「不建議」；30萬 GROW 不應該再寫「可行但打折」。\n- FIX 的 30萬、100萬只能寫「臨界可行」，並且要註明三件事：從 2008 起算 t 約 1.6、HOLD 期對 0050 沒有超額、零股與換股都需要 9:00 盯盤。\n- 300萬以上維持「不建議」，而且要加註需要多日分批成交。\n- 「保守上限」的說法要改成「中性偏樂觀，衝擊項除外」。"
 }
]

## 獨立複製

{
 "reproduced": true,
 "my_numbers": "I wrote my own engine from the frozen spec in /home/user/-/research/tw_stock_selection/agents/o_exec_capacity_tiers/replicate_o5/ (rsim.py, run_grid.py, analyze.py, extra.py, rstats.py; run with python3 -I -B). I did not read the researcher's code. Results are in replicate_o5/out/*.pkl and *_stats.json. I ran 3,688 sims: main grid A/B/C, stress, d1–d5, five sensitivities and V4 seed −1.\n\nBaseline (from bench.parquet and zero_impact_seed_navs): AB_V4 27.00%, t 2.512, MDD −63.5%. 0050 14.94%, same-pool EW 11.19%. Per-seed t median 2.498 (2.299–2.745), 0 seeds below 2.0. The average of the 40 seed navs equals bench AB_V4|normal with diff 0.\n\nMain grid, fee A. Columns: CAGR / t | SEL | VAL | HOLD t | 2008 t | drop best 1/3/5 | MDD | per-seed t median (range; seeds <2.0) | paired diff pp/month (t; MDE).\n- 30萬 GROW: 24.386 / 2.046 | 2.130 | 1.084 | 0.061 | 1.550 | 1.832/1.480/1.149 | −63.88 | 2.020 (1.848–2.245; 15) | −0.181 (t −4.88; 1.247)\n- 30萬 FIX: 25.263 / 2.203 | 2.124 | 1.281 | 0.152 | 1.718 | 1.994/1.643/1.331 | −64.43 | 2.183 (1.959–2.397; 1) | −0.120 (t −2.54; 1.592)\n- 100萬 GROW: 22.179 / 1.655 | 1.828 | 0.948 | −0.172 | 1.148 | 1.439/1.067/0.727 | −64.39 | 1.652 (40)\n- 100萬 FIX: 25.101 / 2.179 | 2.091 | 1.236 | 0.211 | 1.689 | 1.969/1.617/1.290 | −64.22 | 2.165 (1.949–2.398; 2) | −0.132 (t −4.53; 0.978)\n- 300萬 GROW: 19.884 / 1.225 | 1.432 | 0.825 | −0.374 | 0.732 | 0.989/0.590/0.252 | −65.63 | 1.211 (40)\n- 300萬 FIX: 23.652 / 1.915 | 1.769 | 1.167 | 0.184 | 1.442 | 1.696/1.334/1.001 | −64.80 | 1.900 (1.678–2.149; 36) | −0.230 (t −9.17; 0.841)\n- 1000萬 GROW: 17.387 / 0.751 | 0.911 | 0.667 | −0.467 | 0.311 | 0.486/0.088/−0.269 | −67.29 | 0.725 (40)\n- 1000萬 FIX: 20.878 / 1.415 | 1.353 | 0.922 | 0.014 | 0.962 | 1.177/0.794/0.449 | −66.24 | 1.401 (40) | −0.424 (t −11.29; 1.262)\n\nHOLD CAGR, measured from the 2023-12-29 close:\n- GROW: 52.20 / 46.62 / 42.05 / 40.38\n- FIX: 54.52 / 55.97 / 55.26 / 51.00\n\nMedian GROW final equity: 1,992萬, 4,628萬, 9,349萬, 2.002億.\n\nt by fee and scenario (30萬 / 100萬 / 300萬 / 1000萬):\n- Fee B: GROW 1.935/1.547/1.112/0.616; FIX 2.180/2.047/1.771/1.258\n- Fee C: 30萬 FIX 2.415 (26.466%), 30萬 GROW 2.094; other tiers within 0.05 of A\n- Stress: GROW 1.524/1.150/0.698/0.276; FIX 1.579/1.470/1.327/0.673\n\nCost split, % of equity per year (GROW/FIX):\n- Securities tax about 2.2 in every cell\n- Commission: 0.89/1.66, 0.68/0.84, 0.62/0.67, 0.59/0.61\n- Impact above the 0.1% base: 2.22/0.20, 4.11/1.34, 5.71/3.09, 6.96/5.20\n- 0.1% base slippage: 0.18–1.40\n- Odd-lot spread: 0.75/1.73, 0.29/1.03, 0.13/0.47, 0.05/0.16\n- Average cash: 0.20–3.70%\n- d1 rounding tracking error, per-seed median: 0.24–0.53pp\n\nExecution (FIX):\n- Odd-lot-only share of new entries: 30萬 83.9% (80.3% before 2020-10, 92.1% after); 100萬 32.9%; 300萬 12.1%; 1000萬 2.9%\n- Round-lot orders above 10% / 20% of opening volume: 30萬 0/0; 100萬 5.3/0.9; 300萬 26.0/11.4; 1000萬 51.3/35.6\n- 1000萬, above 10%: 60.0% in 2006–18, 37.0% in 2019–26\n- GROW, above 10%: 17.7/35.6/54.6/70.4\n\nBreakeven: t is already below 2.3 everywhere. Extra slippage that keeps t at 2.0: 30萬 FIX 0.0612%, 100萬 FIX 0.0538%, 300萬 and 1000萬 FIX 0, 30萬 GROW 0.0139%.\n\nFalsification test, 30萬 minus 300萬:\n- FIX: +1.612pp per year; paired +0.1093pp per month, t 2.689, MDE80 1.366\n- GROW: +4.503pp, t 6.146\n\nOther runs:\n- d1–d5 CAGR for 30萬 GROW: 27.645 / 27.792 / 26.922 / 27.427 / 24.439\n- V4 plan seed −1, GROW: 24.75 / 22.33 / 20.16 / 17.80\n- Of the 92 variant cells in results.json, 88 match the researcher's ALL CAGR within 0.01pp. The largest gap is 0.077pp.",
 "diff_explained": "Every headline number reproduces: CAGR, t, SEL/VAL/HOLD t, 2008 start, drop-best-month t, MDD, per-seed t, paired differences and MDE, fees B/C, stress, cost split, cash share, tracking error, the 30萬-vs-300萬 test, FIX breakeven and final equity. They agree to the 2nd or 3rd decimal. The verdicts come out the same: 30萬 GROW/FIX and 100萬 FIX are \"可行但打折\", everything else is \"不建議\", and 30萬 beats 300萬. The differences I found:\n\n(1) HOLD CAGR: the researcher measures from the 2023-12-29 close. I first measured from the first 2024 signal date and got 55.2%. Switching to their start point gives an exact match, so this is a definition difference, not an error.\n\n(2) Execution statistics are slightly lower than the researcher's in the large tiers:\n- Odd-lot-only share for FIX: 300萬 12.1% vs 13.4%, 1000萬 2.9% vs 3.7%, 100萬 32.9% vs 33.2%. The 30萬 tier matches exactly.\n- Above-10%-of-opening-volume share: mine is 0–3pp higher, e.g. 1000萬 FIX 51.3% vs 48.7%, GROW 70.4% vs 67.2%.\n- The spec does not pin down the denominator. Open questions are whether deferred orders count, whether lots are measured before or after the cap, and whether value uses PC or O. Either way the ordering by tier and the conclusions stay the same.\n\n(3) Breakeven: 30萬 GROW has t 2.046, which is at least 2.0. I find it can absorb 0.014% more slippage, but the researcher reports 0. That looks like a small omission in their table. It does not change the verdict.\n\n(4) Four sensitivity or stress cells differ by 0.01–0.08pp: κ low at 100萬/1000萬, stress 1000萬 GROW and KEEP 100萬. The likely cause is details the spec leaves open: how deferred orders are ordered within a group, and that I apply no participation cap when OA is NaN.\n\nNot replicated: the calibration figures (κ, opening-auction share of the day) need the 5-minute raw data. The post-hoc full-day V60 comparison is not a verdict column, so I skipped it."
}

**最終判定（審查後）：不可用**
