# FINDINGS：換股日分批錯開——時點運氣、營收新鮮度與單日開盤容量（o_stagger_tranche）

資料夾：`/home/user/-/research/tw_stock_selection/agents/o_stagger_tranche/`
（子代理不能寫 .md，請主研究員把這份全文存成 FINDINGS.md。）

## 結論

**判定 A（必定產出）：主策略級別加註「時點敏感」。**
- 21 個單日版本（期限後 j＝−8～+12，都只用已公告營收）的全期對 0050 t：
  - 最小 0.23、p10 **0.79**、中位數 **1.23**、p90 2.18、最大 2.37。
- 官方日的 t 高於全部 21 版，百分位 1.00：
  - V3 正式 t **2.55**。
  - 官方日曆加 AB（只用已公告營收）t 2.42～2.54。
- 即使只看「期限後第一週」J0～J4，t 中位數也只有 2.18。
- 正式數字改附區間：**跨循環年化 16.9～24.4%，t 0.79～2.18**（p10～p90）。

**判定 B：分批不可用，SOP 不採用。**
- 選中版本 V08：4 批，分別在 J+1、+6、+11、+16 換股，每月在官方日調回等額。
- 五個條件只過三個：
  - 挑選期配對差 **−0.04pp／月**（t −0.24）：通過。
  - 壓力成本：通過。
  - 容量：通過，**38.6% → 1.9%**。
  - **月超額標準差只降 6.9%**（門檻 10%）：沒過。
  - 拿掉最好 1 個月後配對差變成 −0.078：沒過。
- 樣本外更差：
  - 2019 年後對 J+1 配對差 **−0.56pp／月**（t −1.69）。
  - 全期年化 **21.74%**、對 0050 t **1.62**。
  - 對主策略 V3 正式配對差 **−0.39pp／月**（t −2.37）。

**機制：部分支持。**
- 時點運氣：支持。
- 容量：支持。
- 營收新鮮度：
  - 挑選期**被否定**：期限後 1～5 日只比期限前高 0.12pp（t 0.32），未達 0.5pp。
  - 2019 年後方向成立但不顯著：+0.64pp（t 0.89）；S4 +2.45pp（t 1.91）。

## 1. 研究問題與機制

**研究問題**
- 主策略的正式數字有多少是「固定在每月第一個 ≥11 日交易日換股」的時點運氣？
- 營收腿的優勢是否靠公告剛出來時的資訊新鮮度（對應本國投信的月營收追蹤）？
- 把 10 檔分 2～4 批在不同週換股，能不能在不損失均值的前提下，降低時點雜訊和單日開盤集合競價的容量壓力（對應造市者「不集中付流動性」）？

**機制**
- M1 時點運氣：固定單一換股日時，月報酬取決於那一天前後的路徑。
- M2 營收資訊衰退：
  - 期限前換股只能用上上月營收。
  - 期限後越晚換股，營收資訊越可能已反映在價格裡。
- M3 單日成交量占比：分 k 批，每批部位約 1/k。

**事前預測**（PLAN，sha256 ffbdedb2…，2026-10-27T23:14:35Z）
- P1：J 版本的 t 中位數約 2.1～2.4。官方日如果落在前 1/4，就要加註。
- P2：J+1～+5 的月超額比 J−8～−1 高 ≥0.5pp／月。
- P3：分批讓月超額標準差和追蹤誤差下降，均值差在 ±0.05pp 以內，單筆占比約 1/k。
- P4：高波動段（2008、S3、S4）的時點分散較明顯，S1 最小。
- P5：月初批次只能用舊營收，拖累分批組合 0.1～0.3pp／月。

## 2. 資料與引擎

**資料：主策略 V3，全部只讀 import**
- o_infra_v3_rebuild：P（下市股價格）、X（併購出清 166 條）、R_DEL 營收。
- o_disposal_rejudge：官方處置名單 E1（全部排除）。
- main_v2/v2_core：gapfix 訊號、正式同分序、排除金融。
- 成本照 exlib 與 ia_core。

**驗證**（validate.py）
- `plan_single` 用官方日曆、不做 AB 時，每期名單與 V2.plan 相同。
- 重現 V3 正式數字 ALL **26.9459%、t 2.5486**，逐位相同。
- 新引擎 tsim：k=1 時對 o3_core.simulate 的逐日差是 0；k=2、c0 時等於兩個獨立模擬的平均，差也是 0。

**錨定日曆**
- D_m＝m 月 10 日起第一個交易日。
- 2019-01、2023-03 營收依延長期限。
- J{j}＝D_m 之後第 j 個交易日收盤算訊號，隔日開盤成交。
- 官方日在 245 個錨定月裡的 j：+1 有 155 次、0 有 88 次；另有 2 次落在延長期限之前（j＝−4、−1），也就是 V3 正式在這兩個月用了還沒公告的營收。

**AB：只用已公告營收**
- 期限前用最近已公告值（m−2..m−4）。
- 期限當天依 rev_announce 的產業延後機率抽樣。
- 2026 年 INS15 名單的期限是 15 日。
- rev_5y 沒有逐檔公告日（rev_hist 的 ann_date 只有 2026 年 10.6%），所以全期都用規則；不需要新的外部資料。

**期間**
- 挑選期 SEL：錨定月 2006-04～2018-12（153 個月），只用它選版。
- 驗證期 VAL：2019～2023（只報告）。
- 保留期 HOLD：2024～2026-10（只報告）。

**V4**：尚未凍結（o_v4_revenue_patch_official 沒有 results），沒有報。

## 3. 探索發現（只用挑選期，explore.json）

**3.1 單日版本掃描**

挑選期，一般成本，對 0050 t（自己的窗）：

| 版本 | t | 版本 | t | 版本 | t |
|---|---|---|---|---|---|
| J−8 | 2.01 | J−1 | 1.20 | J+6 | 2.49 |
| J−7 | 1.58 | J0 | 2.32 | J+7 | 2.37 |
| J−6 | 2.13 | J+1 | 2.14 | J+8 | 1.93 |
| J−5 | 0.99 | J+2 | 1.21 | J+9 | 1.96 |
| J−4 | 0.65 | J+3 | 1.67 | J+10 | 2.37 |
| J−3 | 0.47 | J+4 | 1.75 | J+11 | 1.91 |
| J−2 | 1.76 | J+5 | 1.57 | J+12 | 2.04 |

- 中位數 **1.91**、p10 0.99。
- 官方日（V3 正式）2.37 落在第 95 百分位；官方 AB 2.27 落在第 81 百分位。
- 挑選期年化 7.7～18.5%。

**3.2 營收新鮮度**
- 扣成本前：J+1～+5 平均 1.09pp，J−8～−1 平均 0.96pp，差 **+0.12pp（t 0.34）**。
- 這條曲線看起來是鋸齒，不是單調：J−5～−3 偏低，J+6～+10 偏高。

**3.3 拆腿**
- 只用價格腿：新鮮減不新鮮 −0.17pp（t −0.49）。
- 只用營收腿：+0.11pp（t 0.29）。
- 只用營收腿的組合，在挑選期對 0050 的超額每個 j 都在 −0.25～+0.07pp 之間。換句話說，挑選期營收腿的價值是「和價格腿交互作用」，不是單獨的新鮮資訊。

**3.4 時點雜訊**
- 同一錨定月、不同 j 的月超額橫斷面標準差：挑選期 3.38pp、S1 3.66、2014～2018 2.94、**2008 年 5.14**。
- 只在 J0 這一天改變 AB 抽樣 seed（期限當天約 17% 的股票改用舊營收），挑選期 t 就在 1.97～2.32 之間變動。這是「小擾動就能改變 t」的直接證據。

**3.5 容量**（1,000 萬帳戶，挑選期）

| 批數 k | 新買名額超過 60 日均成交值 1% 的比例 | 單筆占比中位數 |
|---|---|---|
| 1 | 38.6% | 0.74% |
| 2 | 15.7% | 0.36% |
| 3 | 5.1% | 0.24% |
| 4 | 2.0% | 0.18% |

## 4. 模型迭代紀錄（iterations.json，5 輪 12 版，只用挑選期）

共同窗、一般成本，配對差對 J+1（pp／月），標準差是對 0050 月超額的標準差變化：

| 版本 | 設定 | 配對差 | 標準差變化 | 輪次 |
|---|---|---|---|---|
| V01 | T2 a0 b0 c0 | −0.074 | −5.7% | 1 |
| V02 | T3 a0 b0 c0 | −0.048 | −1.7% | 1 |
| V03 | T4 a0 b0 c0 | −0.051 | −7.4% | 1 |
| V04 | T2 a1 | −0.254 | −11.3% | 2 |
| V05 | T3 a1 | −0.435 | −14.7% | 2 |
| V06 | T4 a1 | −0.535 | −21.1% | 2 |
| V07 | T2 c1 | −0.062 | −5.4% | 3 |
| V08 | T4 c1 | −0.038 | −6.9% | 3 |
| V09 | T2 b1 | −0.012 | −4.5% | 4 |
| V10 | T4 b1 | −0.006 | −6.1% | 4 |
| V11 | T4 b1 c1 | +0.015 | −5.9% | 5 |
| V12 | J+1、+3、+5 | −0.122 | −3.1% | 5 |

- 批次位置：T2＝J+1、+11；T3＝J+1、+8、+15；T4＝J+1、+6、+11、+16。
- 第 5 輪的 V11、V12 是看完前 4 輪才寫入的，理由記在 round5_reason。
- a1（跳過他批持股）把組合稀釋成 20～40 檔，標準差降最多，但均值大降。
- a0 的分批組合均值約等於各批單日版的平均。標準差降不多，因為共同窗上各批的相關性很高：J+1 對 J+6 是 0.78、對 J+11 是 0.73。
- **選版**：先篩配對差 ≥ −0.05，再取標準差最低，結果是 V08（標準差 4.492）。
  - **門檻邊緣**：鄰居 V03（c0）是 −0.0507，剛好沒過；V11 是 +0.015。
  - 有贏家詛咒風險。
- 選項 b 的解讀：b1 把晚批次的營收腿百分位凍結在同一錨定月 J+1 那批的橫斷面，只更新價格腿。在 AB 下，期限後各批本來就用同一個營收月，所以 b 的差異很小，結果也印證這一點。

## 5. 凍結規格

- PREREG sha256 2277654…，2026-10-27T23:19:40Z。
- 4 批各 1/4 資金，分別在 D_m 之後第 1、6、11、16 個交易日收盤換成該批前 10 名，隔日開盤成交。
- 重複持股照買（a0），每批完整重排、只用已公告營收（b0）。
- 每月在官方換股日開盤調回等額，只對淨買賣計成本（c1）。
- 其餘與主策略 V3 相同。
- 全文見 frozen_spec 欄位。

## 6. 樣本外與全期結果（oos.json；PREREG 存在才會執行）

**6.1 單日版本全期（判定 A）**

| 版本 | ALL 年化 | 對 0050 t | HOLD t |
|---|---|---|---|
| J−8 | 20.7% | 1.23 | −0.76 |
| J−7 | 18.2% | 0.86 | −1.01 |
| J−6 | 19.7% | 1.14 | −2.07 |
| J−5 | 18.1% | 0.89 | −1.10 |
| J−4 | 16.5% | 0.57 | −1.63 |
| J−3 | 14.9% | 0.23 | −1.14 |
| J−2 | 22.6% | 1.78 | −0.95 |
| J−1 | 20.2% | 1.32 | −0.69 |
| J0 | 26.8% | 2.37 | 0.23 |
| J+1 | 24.6% | 2.18 | 0.12 |
| J+2 | 22.2% | 1.87 | 0.83 |
| J+3 | 24.4% | 2.19 | 1.59 |
| J+4 | 22.1% | 1.76 | 0.48 |
| J+5 | 18.5% | 1.11 | −0.28 |
| J+6 | 23.1% | 1.93 | 0.18 |
| J+7 | 21.2% | 1.54 | −0.59 |
| J+8 | 19.6% | 1.31 | −0.65 |
| J+9 | 19.0% | 1.21 | −0.63 |
| J+10 | 18.3% | 1.01 | −1.18 |
| J+11 | 16.9% | 0.79 | −0.84 |
| J+12 | 17.6% | 0.91 | −1.31 |

- 分布：中位數 **1.23**、p10 **0.79**、p90 2.18。日曆月 t 中位數 1.32；共同窗 t 中位數 1.26。
- **官方日 2.55 高於全部 21 版。**
- 各分段的分布（中位數，官方日的值與百分位）：
  - S1：1.95（官方 2.06，第 62 百分位）。
  - S2：0.07（官方 1.15，第 100 百分位）。
  - S3：0.16（官方 1.17）。
  - S4：−0.69（官方 0.38，第 86 百分位）。
  - HOLD 年化 15.9～94.9%，中位數 38.3%；0050 同期 59.2%。**多數換股日在 2024～2026 輸給 0050。**
- 最大回撤 −45.9%～−63.7%。
- 拿掉最好 10 個月後 t 中位數 −0.66。
- 期限後第一週 J0～J4：全期 t 1.76～2.37（中位數 2.18）。

**6.2 選中的 V08**（一般成本，共同窗）

| 期間 | 年化 | 對 0050 t | 0050 年化 |
|---|---|---|---|
| SEL | 15.84% | 2.13 | 6.45% |
| VAL | 25.66% | 0.74 | 17.78% |
| HOLD | 45.37% | −0.51 | 58.54% |
| ALL | 21.74% | 1.62 | 14.94% |

- ALL：最大回撤 −57.7%，對同池等權 t 2.97。
- 配對差：
  - 對 J+1：VAL −0.44、HOLD −0.80、ALL −0.23（t −1.47）。
  - 對 V3 正式：ALL −0.39（t −2.37）。
- 不同成本：
  - 壓力 flat：18.72%（t 1.01）。
  - 壓力 +0.3：16.28%（t 0.51）。
  - 扣成本前：28.12%（t 2.86）。
- 其餘 11 版樣本外也全部輸給 J+1：VAL −0.26～−0.64、HOLD −0.06～−0.80。唯一例外是 V12（期限後第一週集中），HOLD +0.65。

**6.3 並列比較**（ALL，一般成本）

| 版本 | 年化 | 對 0050 t | 最大回撤 |
|---|---|---|---|
| V3 正式（不做 AB） | 26.95% | 2.55 | −65.0% |
| 官方 AB（seed 0～4） | 26.3～26.9% | 2.42～2.54 | −65% |
| J+1 | 24.79% | 2.15 | −63.7% |
| V08 | 21.74% | 1.62 | −57.7% |
| 0050 | 14.94% | — | — |

- 對同池等權 t：V3 正式 3.76。

## 7. 穩健性

**端點敏感度**

| | V3 正式 | V08 |
|---|---|---|
| 全期 t | 2.55 | 1.62 |
| 拿掉最後 1 月 | 2.42 | 1.58 |
| 拿掉最好 1 月 | 2.37 | 1.38 |
| 拿掉最好 3 月 | 2.04 | 1.04 |
| 拿掉最好 5 月 | 1.73 | 0.74 |
| 拿掉最好 10 月 | 1.03 | −0.04 |
| 拿掉 2021 年 | 2.15 | 1.37 |
| 日曆月 t | 2.46 | — |

**成本損益兩平**（單邊滑價 s）

| | t 降到 2.3 | t 降到 2.0 | 月超額歸零 |
|---|---|---|---|
| V3 正式 | 0.174% | 0.263% | 0.857% |
| J+1 | 0.055% | 0.143% | 0.728% |
| V08 | — | 一般成本下已低於 2.0 | 0.538% |

- 手續費全額、沒有折扣：V3 正式 t 2.36，V08 t 1.41。

**分年**：V08 對 J+1 的配對差（pp，年加總）
- 正：2008 +10.6、2018 +5.3、2010 +3.8。
- 負：2014 −11.7、2015 −8.4、2021 −18.6、2026 −30.9（2026-09 那一個訊號窗，分批只吃到一部分）。
- 分批在 2008 確實分散了時點；但在領漲集中的月份會錯過。

**分產業貢獻**（扣成本前）
- V08：電子零組件 13.1%、半導體 12.9%、其他電子 10.8%、通信網路 7.5%、航運 6.6%。
- V3 正式：半導體 16.8%、電子零組件 12.5%。
- 兩者結構相近。

**去掉最大貢獻股**
- 兩者的最大貢獻股都是川湖：占 V08 的 2.8%、V3 正式的 3.5%。
- 拿掉後扣成本前年化：V08 28.25% → 27.48%；V3 正式 33.41% → 32.30%。
- 結論不靠單一股票。

**位置平移的 t 分布**（事前加列的描述性分析）
- T4 c1 整組平移 6 種位置：t 1.19～1.71（中位數 1.45，標準差 0.17）。
- T2 平移 11 種位置：t 0.90～1.93（標準差 0.30）。
- 單日版本：標準差 0.57。
- **分批確實把時點運氣壓縮到約 1/3，而「去掉運氣」後的全期 t 約 1.45。**

**多重檢定**
- m＝主策略改版 10＋J 掃描 21＋分批 12＝43。
- V08 單尾 p 0.052，Bonferroni 後 1.0。
- V3 正式單尾 p 0.0054：m＝10 時 0.054，m＝43 時 0.23。

**周轉**：每次換股平均
- V3 正式 62.1%、J+1 62.9%、V08 63.6%。
- 分批不會降低周轉，總成交額相同，只把單筆切小。

## 8. 機制是否被支持

**M1 時點運氣：支持。**
- 同一規則換一天，全期 t 從 0.23 到 2.37。
- 官方 2.55 高於所有單日版。
- 只改期限當天的 AB 抽樣，t 就在 2.03～2.37 間變動。
- V3 的「t 2.55」應理解為一次幸運的抽樣，系統性水準約 **1.2～1.5**：
  - 單日版中位數 1.23。
  - 分批平移中位數 1.45。
  - 只看期限後第一週，約 2.2。

**M2 營收新鮮度：部分支持。**
- 挑選期沒有差異（+0.12pp，t 0.32），P2 被否定。
- 2019 年後：期限後 0～3 日比 9～12 日高 1.24pp（t 1.68），S4 +2.45pp（t 1.91），方向一致但不顯著。
- 可能的解讀：近年市場對營收的反應變快（投信與量化競爭加劇）。新鮮度只在近年有價值，而且這一點只能從已看過的資料得到，不能當確認證據。
- 和 main_v2 的發現一致：用 V2 機制、不做 AB 的事後檢查（scratchpad/st/chk.py，非正式）顯示，「每月第一個 ≥18 日」全期 t 1.60，「≥25 日」t 0.86；「≥4 日」若用還沒公告的營收（前視）t 2.2，但 AB 版只有約 0.6～0.9。這代表「最新一個月營收」在公告前後的價值很高。

**M3 容量：支持。**
- 單筆占 60 日均成交值的比例約降為 1/k。
- 1,000 萬帳戶超過 1% 的名額：38.6% → 1.9%。

**預測檢查**
- P1 被否定：中位數只有 1.23。官方日落在最頂端，所以要加註。
- P3 部分成立：均值在挑選期 ±0.05pp 以內，但標準差只降 2～7%；單筆約 1/k 成立。
- P4 大致成立：2008、S3、S4 的分散度 5.1～5.4pp，高於 S1、S2 的 3.3～3.7pp。但 S1 不是最小，S2 才是。
- P5：挑選期沒有拖累。2019 年後晚批次的拖累約 0.4pp／月（V03 −0.50 對 V12 −0.09），比預測的 0.1～0.3 大。b1（只更新價格腿）只改善 0.05pp。

## 9. 散戶怎麼執行

- **維持單一官方換股日**（每月第一個 ≥11 日交易日）。不要把換股延後到期限後第 5 個交易日以後，也不要提前到期限前。近年營收腿的價值集中在公告後一週內。
- **主策略預期要下修**：對外引用 t 2.55 時，同時附「換股日分布」：年化 16.9～24.4%、t 0.79～2.18。系統性 t 約 1.2～1.5。這不改變方向，但主策略已經不是穩定第 1 級。
- **1,000 萬以上帳戶的容量**：
  - 不要用整月分批，它在 2019 年後每月少 0.4～0.8pp。
  - 改成把同一張名單在期限後第 1～3 個交易日的開盤分次下單。V12 的事後觀察：2019 年後對 J+1 −0.09pp、容量 3.3%。這只是事後線索，需要前瞻紙上追蹤，不能當規則。
- 限價單、手續費折扣都必要：V3 只容得下約 0.07% 的額外滑價就會跌破 t 2.3。

## 10. 限制

- **AB 偏保守**：期限前把所有公司都視為未公告，實際上很多公司在 10 日前就已公告。J−8～J−1 因此偏低；這只影響期限前版本的解讀。
- 期限當天的抽樣機率只用 2026-09 一個月估計。
- 營收截斷（R_DEL 仍缺下市股營收）沒有修；V4 凍結後要重跑 J 掃描。
- 2019～2023 價格面板只有存活者。
- 2019 年後的主策略已經被多題看過，所以本題判定上限是待觀察。2019 年後的新鮮度型態是已看過資料上的發現。
- 容量只計算「新買名額占 60 日均成交值」，沒有模擬衝擊成本；分批後的總成交額不變。
- 選中版本落在門檻邊緣（V03 −0.0507）。iterations 共 12 版，第 5 輪是看過結果後加入的。
- 有一處統計瑕疵已修正：explore 與 oos 裡依日曆月加總的周轉，在 J−8、J−7 會被高估；finalize 已改成每次換股的平均。

**主要檔案**（都在 `/home/user/-/research/tw_stock_selection/agents/o_stagger_tranche/`）
- 程式：st_core.py、validate.py、plan.py、explore.py、iterate.py、prereg.py、oos.py、finalize.py。
- 結果：validate.json、explore.json、explore_rows.csv、iterations.json、oos.json、navs_oos.parquet、results.json（PLAN、PREREG、DECISION、SUMMARY、FILES_sha256）。

## 凍結規格

PREREG（o_stagger_tranche/results.json；sha256 2277654749052ebe6d4d46df72f42cccfaa5c8e3000ea464d40717e3d3161a44；2026-10-27T23:19:40Z。之前的 PLAN sha256 ffbdedb2…，2026-10-27T23:14:35Z）

選中版本：V08＝T4 a0 b0 c1

1. 資料：主策略 V3，全部只讀 import。
- 營收：o_infra_v3_rebuild 的 R_DEL（2021-03 起改用 rev_5y）。
- 下市股價格 P：apply_patch。
- 併購出清 X：exit_rules 'main'，共 166 條。
- 處置：o_disposal_rejudge 官方處置名單 E1，全部排除，以訊號日認定。
- 訊號：main_v2/v2_core 的 gapfix 滾動窗；0050 含息。
- validate.py 逐位重現 V3 正式數字：ALL 26.9459%，t 2.5486。

2. 錨定日曆
- 錨定月 m 的營收期限 D_m＝m 月 10 日起第一個交易日（加權指數日曆）。
- 特例：2019-01 營收的期限是 2019-02-15，2023-03 營收的期限是 2023-04-12。
- J{j} 的訊號日＝D_m 之後第 j 個交易日，在該日收盤算訊號。
- 成交日＝下一個「有值檔數 ≥300」的面板日，以開盤價成交。
- 訊號日不在面板裡，該月就不換股、續抱。
- 錨定月集合和 V3 官方訊號月相同：2006-04～2026-09，共 245 個月。

3. 選股：與主策略相同。
- 前 300 流動性池（60 日均成交值 ≥1,000 萬、股價 ≥10），排除金融，排除 E1 處置股，隔日要有開盤價。
- 分數＝0.5×52 週比百分位＋0.5×近 3 月營收年增率百分位。
- 同分依序比：營收高者優先，再比代碼。
- 選前 10 名。開盤漲停而且該批沒持有的股票，換下一名遞補。
- 停牌部位凍結；X 出清；P 下市價格。

4. AB：只用已公告的營收。
- 訊號日 < D_m：m−1 月營收視為未公告，改用 m−2..m−4（也就是最近已公告值）。
- 訊號日 = D_m：依 rev_announce 的產業延後機率抽樣（seed 0），金融的機率是 1。
- 2020-01 營收：2/14 前也抽樣。
- 2026 年 INS15 名單的期限是 15 日。
- 2021 年後 rev_5y 沒有逐檔公告日，所以全期都用上面的規則。

5. 分批 V08
- 共 4 批，各分到 1/4 資金。
- 批 i 在 J+1、J+6、J+11、J+16 的訊號日收盤算訊號，隔日開盤換成「該批自己的」前 10 名。持有到下個月同一個 j 日。
- a0：其他批持有的股票照樣可以重複買。
- b0：每批都完整重排，營收用 AB 規則下最近已公告的值。
- c1：每月在官方換股日（每月第一個 ≥11 日交易日）的隔日開盤，在任何一批換股之前，把各批的可交易部位與現金等比例縮放到等額。只對全帳戶每檔的淨買賣計成本，成本依價值比例分攤。所有批都開始之後才執行。
- 引擎是 st_core.tsim。k=1 時與 o3_core.simulate 逐日相同；k=2、c0 時等於兩個獨立模擬的平均（驗證差 0）。

6. 成本
- 一般：買 0.1855%、賣 0.4855%（手續費 0.0855%、證交稅 0.3%、滑價 0.1%）。
- 壓力 flat：買賣各 0.5%。
- 壓力 +0.3：一般成本每邊再加 0.3%。
- 扣成本前：零成本。

7. 統計
- 分批與配對比較一律用官方日曆窗：官方訊號日收盤到下一個官方訊號日收盤，終點 2026-10-12。按錨定月分群算 t。
- 單日 J 版本用自己的窗，另報日曆月 t 與官方共同窗 t。
- 期間：SEL＝錨定月 2006-04～2018-12，只用它選版；VAL＝2019～2023；HOLD＝2024～2026-10，只報告。

8. 選版規則（只用 SEL、一般成本）
- 先篩對 J+1 的配對差 ≥ −0.05pp／月。
- 再取對 0050 月超額標準差最低者。
- 結果：V08，配對差 −0.0375，標準差 4.492。
- 鄰居 V03（只差在 c0）是 −0.0507，剛好落在門檻外，屬於門檻邊緣，有贏家詛咒風險。

9. 判定規則（事前登記）
- A：全期 21 個 J 版本對 0050 t 的 p10 < 2.0 時，主策略級別加註「時點敏感」，正式數字附 p10～p90 區間。
- B（全部要過）：
  - B1：SEL 配對差 ≥ −0.05pp
  - B2：月超額標準差（或追蹤誤差）下降 ≥10%
  - B3：1,000 萬帳戶超過 60 日均成交值 1% 的新買名額比例下降 ≥ 一半
  - B4：兩種壓力成本的配對差都 ≥ −0.05pp
  - B5：拿掉配對差最好的 1 個月後，B1、B2 仍成立
- 全部通過時，上限也只到待觀察。

10. 多重檢定：m＝主策略改版 10＋本題 21 個 J 版本＋12 個分批版本＝43。

## 反方審查

[
 {
  "final": "不可用",
  "issues": [
   "判定 A 的解讀不成立，問題在「把時點運氣當原因」。官方日其實就是 J0 和 J+1 的混合（88 次與 155 次）。在共同窗配對比較：官方 AB（seed 0）對 J0 是 −0.06pp／月（t −0.38），對 J+1 是 +0.12pp（t 1.10），兩者都分不出差別。拿「21 版裡百分位 1.00」當運氣證據也不成立：21 版月超額的平均相關 0.79，用特徵值算的有效版本數只有約 1.6，等於不是 21 次獨立抽樣。",
   "2.55 和中位數 1.23 之間的差距，大部分是結構性的資訊時點，不是運氣。21 版裡有 8 版在期限前，用的是過舊的營收，研究員自己也承認那段的 AB 偏保守。另外，期限後越晚換股的版本在 2019 年後顯著變差：J0～J4 減 J8～J12 為 +1.00pp／月（t 2.98）；挑選期則沒有這個結構（−0.09，t −0.41）。只看 j≥0 的 13 版，t 中位數是 1.54、p10 0.93，所以 p10<2.0 的旗標照樣成立。但「系統性 t 約 1.2～1.5」是把期限前的舊資訊版和 2019 年後的衰退一起平均進去。整組平移的中位數 1.45 也一樣：6 種位置每一種都有一批落在期限前（j0＝−8～−3），不能叫「去掉運氣」。",
   "V3 正式的 2.55 含有前視，研究員只寫了 2 個月（j＝−4、−1），實際範圍更大。另外兩個來源：88 個 J0 換股日用了約 17% 還沒公告的營收；2026 年 INS15 名單的期限是 15 日，官方日都在這之前。V3 正式減官方 AB：2019 年後 +0.11pp／月（t 2.06），HOLD +0.22pp（t 1.69），2026 年加總 +4.35pp。公平的正式數字應該用 AB 的 t 2.42～2.54，HOLD 的 60.09% 也偏高。拿 V08 對 V3 正式的 −0.39（t −2.37）也部分來自這個前視基準；改對官方 AB 是 −0.36（t −2.15）。結論方向不變，但數字要改。",
   "少數年份集中。V3 正式對 0050 的累積超額有 50% 來自 2021、2015、2006 三年；拿掉這三年，t 剩 1.41（官方 AB 1.24，V08 0.78）。研究員只報了拿掉 2021 的結果（2.15），低估了年份集中度。Newey-West t 是 2.64，和普通 t 差不多，自相關不是問題。",
   "B2 沒過，這點本身並不確定。V08 月超額標準差只降 6.9%，月份 bootstrap 的 90% 區間是 −12.5%～−0.7%，降幅達 10% 的機率約 19%。這不足以翻案：B5 也沒過（拿掉最好 1 個月後配對差 −0.078）；2019 年後對 J+1 −0.56pp（t −1.69）；V08 落在門檻邊緣（V03 −0.0507）有贏家詛咒。所以「不可用」成立，沒有往上修的空間。",
   "第 9 節的執行建議都建立在已看過的 2019 年後資料和事後挑選上：「不要延後到期限後第 5 日以後」「近年營收價值集中在公告後一週」「同一張名單在第 1～3 日分次下單」（V12 是第 5 輪看完結果才加的）。J 掃描本身也不單調（J+5 是 1.11、J+6 是 1.93）。這些只能當假說，不能寫進 SOP。",
   "挑選期不是真正沒看過的資料：主策略 V3 本身就是在全期資料上一路改版出來的，2006～2018 早已被多個題目看過。所以挑選期配對差 −0.04pp 只是「沒有變差」，不是獨立確認。",
   "存活者偏誤：2019～2023 的價格面板只有存活股。所有版本的 VAL 水準都偏高，配對比較受的影響比較小。研究員已經揭露。",
   "流程與多重檢定：iterate.py 的修改時間（23:21:33）晚於 PREREG（23:19:40）。oos 有斷言 SPECS 和 PREREG 一致，影響有限。5 輪 12 版在 90 秒內跑完，「看完前 4 輪才寫第 5 輪」的先後無法獨立驗證。m＝43 的 Bonferroni 對高度相關的 J 版本過嚴，但不影響判定。",
   "容量（38.6% → 1.9%）大致成立。c1 在官方日開盤調回等額的交易沒有算進容量，但這筆交易平均只占帳戶的 0.37%／月（p90 0.74%），影響可以忽略。"
  ],
  "note": "判定維持「不可用」，但研究員有兩個解讀要改。（1）分批本身不可用：按事前規則，B2 和 B5 都沒過；2019 年後對 J+1 −0.56pp／月（t −1.69），對官方 AB 全期 −0.36pp（t −2.15）。這個結論在統計上站得住。（2）判定 A 的旗標（p10<2.0）是按規則觸發的，但「2.55 是一次幸運抽樣、系統性 t 約 1.2～1.5」這個解讀不成立。官方日和 J0、J+1 配對都分不出差別，t 分別是 −0.38 和 1.10；2.55 和 1.23 之間的差距主要來自期限前的舊資訊版本，以及 2019 年後晚換股版本顯著衰退（t 2.98）。對主策略更大的問題有兩個：一是 V3 正式含前視，2019 年後 +0.11pp（t 2.06），2026 年加總 +4.35pp，應改用 AB 的 t 約 2.42；二是年份集中，拿掉最好 3 年後 t 只剩 1.41。驗證程式在 /home/user/-/research/tw_stock_selection/agents/o_stagger_tranche/review_stat_o4.py（只讀，用 python3 -I -B 執行）。執行途中我在檔尾補了兩段檢查：前視逐年拆解和調回等額的交易量。研究員的檔案都沒有修改。"
 },
 {
  "final": "不可用",
  "issues": [
   "我同意「分批不可用」，而且從成本與可成交的角度看更差。V08 的成本壓力是對 J+1 比配對差才算通過（B4）。看絕對數字：一般成本 ALL t 1.62、2019 年後 0.32；單邊滑價 0.2% 時 ALL t 1.25、2019 年後 0.12；手續費不打折加 0.2% 滑價時 ALL t 1.04、2019 年後 0.01、HOLD −0.70。V08 在任何現實成本下都沒有對 0050 的優勢。",
   "容量是 V08 唯一的好處，但被高估了。38.6%→1.9% 是挑選期 2006～2018 的數字。2019 年後（研究員的 oos.json selected_detail.capacity.J1_POST），J+1 單日 1,000 萬帳戶超過 60 日均成交值 1% 的名額只有 11.1%（中位數 0.26%），全期是 27.6%。在現在的市場，單日換股的容量壓力本來就小，分批能改善的空間有限。另外，這個指標只算新買、不算賣出，分母是全日均成交值而不是開盤集合競價量（資料裡沒有），所以容量的機制只能算間接證據。",
   "小帳戶（30 萬～100 萬）的可成交性在範圍內卻沒有檢查，這也直接打掉分批方案。分批後每檔金額只有 1/40：30 萬帳戶每檔約 7,500 元，100% 的名單一張超過配置；100 萬帳戶每檔 2.5 萬，86.8% 一張超過。2020-10-26 以前沒有盤中零股，零股只能在盤後撮合，回測假設的「隔日開盤成交」根本做不到。主策略本身也有這個問題：30 萬帳戶單日換股時，79.5% 的名單一張超過 3 萬。另外 30 萬帳戶分 4 批時，若最低手續費 20 元，每邊多 0.18% 成本。FINDINGS 寫的容量範圍是 30 萬～1,000 萬，卻只分析了 1,000 萬的情況。",
   "第 9 節的散戶 SOP「同一張名單在期限後第 1～3 個交易日的開盤分次下單」沒有經過回測，引用的 V12 也不是同一件事：V12 是 J+1、+3、+5 各批重新排名。另外 V12 的「容量 3.3%」是全期數字，不是 2019 年後。我自己做了驗證（review_exec_o4.py）：官方訊號名單在官方成交日起連續 3 個開盤各下 1/3，每批持有到下月同一位置，c0。結果對官方單日的配對差是 ALL −0.005pp／月（t −0.08）、SEL −0.068（t −1.01）、2019 年後 +0.10（t 0.79），容量從 29.0% 降到 3.2%；AB 版的結果差不多。所以這個建議大致可行，但目前是事後補的，挑選期還略負，只能算待觀察，不能寫進 SOP 當規則。",
   "V08 的操作負擔沒有計入。每月要在 5 個交易日下單（4 個批次日加上官方日調回等額），每批都要處理開盤漲停遞補。開盤漲停遞補要等開盤後才知道，遞補股實際只能在盤中買，回測卻用開盤價成交。官方日有 4.5% 的換股事件出現這種情況；分批後要看盤的早上變成 4～5 倍。調回等額在開盤時按比例加減碼，沒有排除開盤漲停或處置股，不過它的成本只有約 0.0025%／月，影響很小。",
   "跌停時賣出照開盤價成交，沒有處理跌停鎖死賣不掉的情況。不過賣出時開盤就是跌停的比例只有 0.13～0.32%，各版本差不多，不影響結論。",
   "判定 A 我大致同意，並補一點不利的證據：主策略在只用已公告營收（AB）的版本，2019 年後對 0050 t 只有 1.12、HOLD 0.22。在單邊 0.2% 滑價下，全期 t 2.09；手續費不打折加 0.2% 滑價時是 1.90。「V3 t 2.55」除了時點運氣，還要再扣成本敏感度。另一方面我也驗證了，那兩個前視月（2019-02、2023-04）改成 AB 後 t 反而升到 2.567，所以前視沒有灌高 V3，這一點可以排除。",
   "周轉方面：每次換股的周轉三者相同（約 63%），但分批把下單次數變成 4 倍、單筆金額切小。對大帳戶這有利於容量，對小帳戶則是整股與最低手續費的問題（見上）。研究員說「總成交額相同」是對的，但沒有算到小帳戶這一面。"
  ],
  "note": "從成本與可成交的角度審查，我同意研究員「不可用」的判定，而且更不利。\n\n**V08 的成本**\n- 一般成本：全期對 0050 t 1.62，2019 年後只有 0.32。\n- 手續費不打折加單邊 0.2% 滑價：2019 年後 t 0.01，等於沒有超額。\n\n**容量（研究員認定唯一的好處）**\n- 2019 年後，J+1 單日版的 1,000 萬帳戶只有 11.1% 的名額超過日均成交值 1%，不是挑選期的 38.6%。單日換股在現在的市場本來就沒有多大容量問題。\n- 指標用的是全日均成交值，不是開盤集合競價量，所以這個機制只是間接證據。\n\n**小帳戶**\n- 30 萬～100 萬帳戶分 4 批後，有 87～100% 的名單一張超過每檔配置。2020-10 以前沒有盤中零股，回測用的開盤成交在實際上做不到；另外有最低手續費的負擔。這點直接否定把分批用在小帳戶。\n\n**第 9 節的散戶 SOP**\n- 「同一張名單分 3 個開盤下單」研究員沒有測過，引用的 V12 也不是同一件事。\n- 我實際驗證：對官方單日的配對差是 −0.005pp／月，容量從 29% 降到 3.2%。這個方向可行，但屬於事後補做、挑選期還略負，只能算待觀察的紙上追蹤，不能當成規則。\n\n**判定 A**\n- 成立。AB 版主策略在 2019 年後 t 只有 1.12；加 0.2% 滑價後，全期 t 是 2.09。\n- 兩個前視月沒有灌高 t：改成 AB 後 t 反而升到 2.567。\n\n**影響不大的項目**\n- 跌停開盤賣出比例約 0.2%，開盤漲停遞補約 4.5% 的換股事件，影響都很小。\n\n驗證程式：/home/user/-/research/tw_stock_selection/agents/o_stagger_tranche/review_exec_o4.py\n輸出：/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/rev_exec.txt"
 }
]

## 獨立複製

{
 "reproduced": true,
 "my_numbers": "I rewrote everything from scratch: anchor calendar, AB, ranking, the daily engine with tranches and c1, and all stats. The code is in replicate_o4/ind_core.py, ind_base.py, ind_jscan.py, ind_tr.py, ind_detail.py, ind_misc.py and ind_jan.py. The only imports are read-only data layers: o3_core (P/X/R_DEL plus v2_core gapfix signals, FINMASK, PCOL, IS15) and drj_core.load_list (official disposal list E1). Run with python3 -I -B.\n\n[Baseline] V3 official calendar, no AB: ALL 26.945918746008978%, t 2.5486212673374764, max drawdown −64.97%. These match R_DEL|E1 in v3_results.json digit for digit. Relative difference vs o3_core.simulate on the same plan is 0.0, both for NAV and for daily returns. Official-day position relative to the deadline: j=+1 155 times, j=0 88 times, j=−4 and j=−1 once each. All four counts match.\n\n[J scan] Revenue source is chosen by anchor month (rev_5y from 2021-03). Windows are all in-panel signal days, including months with no trade; the end point is that J day in 2026-10, or the last panel day if it is not in the panel.\n- J−8 20.68/1.234/−0.76, J−7 18.16/0.863/−1.01, J−6 19.66/1.14/−2.07, J−5 18.06/0.885/−1.10, J−4 16.53/0.573/−1.63, J−3 14.95/0.232/−1.14, J−2 22.61/1.776/−0.95, J−1 20.20/1.322/−0.69.\n- J+1 24.64/2.185/0.12, J+2 22.31/1.885/0.83, J+3 24.45/2.194/1.60, J+4 22.11/1.763/0.48, J+5 18.51/1.111/−0.28, J+6 23.09/1.926/0.18, J+7 21.23/1.543/−0.59, J+8 19.61/1.305/−0.65, J+9 18.98/1.209/−0.63, J+10 18.32/1.008/−1.18, J+11 16.93/0.790/−0.84, J+12 17.60/0.913/−1.31.\n- All 20 of these match the table.\n- J0 is the exception. My seed 0 gives 24.02/1.964/0.14 against the researcher's 26.8/2.37/0.23. My J0 seeds 0–4 give t 1.96, 2.42, 2.26, 2.07, 2.18.\n\n[Distribution] min 0.23, p10 0.79, p25 0.91, median 1.23, p75 1.78 all match. p90 1.96 and max 2.19 differ (researcher 2.18 and 2.37). Annualized return p10 16.9 and median 19.7 match; p90 24.0 differs (researcher 24.4). The J−8..−1 median of 1.01 matches. The J0–J4 median is 1.96 (researcher 2.18). Max drawdown −45.9 to −63.7 matches. Median t after removing the best 10 months is −0.657, which matches. Official percentile 1.00 matches. Selection-period t median 1.91 (0.47–2.49) matches. Official selection-period t 2.3746 sits at the 95th percentile (it is just above J+7's 2.3737), which matches.\n\n[Official AB seeds 0–4] t 2.427, 2.388, 2.541, 2.578, 2.376, mean 2.462. The mean matches but the range does not: mine is 2.38–2.58, the researcher's 2.42–2.54.\n\n[Freshness]\n- Selection period gross +0.124 (t 0.34) and normal +0.119 (0.32) both match.\n- 2019+ +0.645 (0.90), S4 +2.452 (1.91) and full period +0.315 (0.89) all match.\n- Price leg −0.165 (−0.49) and revenue leg +0.108 (0.29) both match.\n- Revenue-only portfolio's selection-period excess per j runs −0.245 to +0.074, which matches. That is the zero-cost version; the brief labels this item \"一般成本\", so the cost label is the brief's, not mine.\n- The post-hoc J0..J3 minus J9..J12 is 2019+ +1.18 (t 1.58) and selection period −0.155. It differs slightly because it includes J0.\n\n[Timing noise] Cross-sectional sd: selection period 3.38, S1 3.65, S2 3.35, 2008 5.13, S3 5.42, S4 5.43. Correlations of selection-period excess: J+1 with J+6 0.784, J+1 with J+11 0.728. All match.\n\n[Tranches, selection period, paired diff vs J+1 / sd change]\n- V01 −0.0737/−5.73%, V02 −0.0483/−1.65%, V03 −0.0507/−7.42%.\n- V04 −0.2543/−11.26%, V05 −0.4346/−14.73%, V06 −0.5352/−21.13%.\n- V07 −0.0619/−5.41%, V08 −0.0375/−6.95% (sd 4.492), V09 −0.0115/−4.49%.\n- V10 −0.0060/−6.05%, V11 +0.0154/−5.91%, V12 −0.1223/−3.11%.\n- All 12 match to the digit, and the selection rule picks V08.\n\n[V08]\n- Periods (annualized / t vs 0050 / 0050 annualized): SEL 15.84/2.13/6.45, VAL 25.66/0.74/17.78, HOLD 45.37/−0.51/58.54, ALL 21.74/1.62/14.94.\n- Selection-period calendar-month t is 2.434 when the window ends at 2018-12.\n- Max drawdown −57.7, t vs equal-weight pool 2.97, mean excess 0.569pp, tracking error 19.0%.\n- Paired diff vs J+1: SEL −0.0375 (−0.24), 2019+ −0.562 (−1.69), ALL −0.234 (−1.47).\n- Paired diff vs V3 official: ALL −0.388 (−2.37), 2019+ −0.736 (−2.20).\n- Stress costs: flat 18.72/1.01, +0.3 16.28/0.51, zero cost 28.12/2.86.\n- Endpoint sensitivity: drop last month 1.58, best 1/3/5/10 months 1.38/1.04/0.74/−0.04, best year 2021 1.37, calendar-month t 1.67.\n- One-tailed p 0.052.\n- Every V08 number matches.\n\n[Comparison runs]\n- J+1: 24.79/2.15, max drawdown −63.7, paired vs V3 official −0.154 (−1.50).\n- V3 official: SEL 17.88 (2.37), VAL 34.05 (1.40), HOLD 63.24 (0.38).\n- V3 official endpoints: 2.42/2.37/2.04/1.73/1.03; drop 2021 2.15; calendar-month t 2.46; p 0.0054.\n- All match.\n\n[Test B] B4: flat −0.0397, +0.3 −0.0415. B5: best month 2009-11; after removing it B1 is −0.0781 and B2 is −7.59%. B3 shows up in the capacity results below.\n\n[Capacity] Share of new buys above 1% of 60-day average traded value, selection period: k=1 38.6%, k=2 15.7%, k=3 5.1%, k=4 (V03) 2.0%, V08 1.9%. Medians 0.74/0.36/0.24/0.18%. Full period: J+1 27.6%, V08 1.15%. New-buy counts: 918 and 3,695. All match.\n\n[Break-even slippage] V3 official 0.174/0.263/0.857%. J+1 0.055/0.143/0.728%. V08 is below 2.0 already at normal cost; mean excess reaches zero at 0.538%. Full commission: V3 official t 2.357 (25.88%), V08 1.41. All match.\n\n[Turnover] V3 official 62.1%, J+1 62.9%, V08 63.6%. Match.\n\n[Position shift] T4 c1 at 6 positions: t 1.20–1.71, sd 0.176, median 1.39 (researcher 1.45). T2 at 11 positions: c0 gives 0.90–1.81, sd 0.26; c1 gives 0.95–1.69. Single-day versions: sd 0.54, common-window sd 0.52.\n\n[Tests] Test A holds (p10 0.79 < 2.0). Test B fails, because B2 (−6.9%) and B5 fail.",
 "diff_explained": "1) All the core numbers reproduce exactly. That covers the baseline at full precision, zero engine difference, 20 of the 21 single-day versions, all 12 tranche selection-period paired diffs and sd changes, every V08 and comparison-run statistic, capacity, break-even, full commission, B1–B5, and the conclusions of tests A and B.\n\n2) The single real difference is the AB random-draw sequence. It affects J0 and the official AB runs.\n- My AB masks for official days are bit-for-bit identical to v2_core.make_unavail('mc', seed): I checked every month for seeds 0–4. Even so, my official AB seed 0 differs from the researcher's OFF_AB0 NAV in navs_oos.parquet in 99 windows. My t is 2.427, theirs 2.423.\n- The first sampled month, 2006-06, matches. The second, 2006-09, does not use the next draw in the stream. So the researcher's draws are ordered or seeded differently.\n- I tried consecutive draws, a shared object between the official and J0 runs, per-date seeds, and the v2 handling of the 2019-01 and 2023-03 extensions. None reproduced their J0.\n- The rule itself matches: my J0 across seeds 0–4 gives t 1.96–2.42, which contains their seed-0 value of 2.37. Their seed 0 sits near the top of that range; mine sits near the bottom.\n\n3) Every differing distribution number traces back to that J0 gap:\n- t p90 and max: 1.96/2.19 vs 2.18/2.37.\n- Annualized-return p90: 24.0 vs 24.4.\n- J0–J4 median: 1.96 vs 2.18.\n- Post-hoc J0..J3 minus J9..J12: +1.18/t 1.58 vs +1.24/1.68.\n- Single-day sd: 0.54 vs 0.57.\n- T4 c1 shift median 1.39 vs 1.45, because the j0=−5 position contains a J0 tranche.\n- T2 shift maximum 1.81 vs 1.93, because the j0=0 position contains a J0 tranche.\n- Official AB selection-period t: my seeds give 2.10–2.32, not their 2.27, which is a different random stream. Substituting the researcher's J0 value 2.37 into my set gives back their p90 2.18, max 2.37, annualized p90 24.4 and J0–J4 median 2.18.\n\n4) Two wording ambiguities, now settled:\n- The selection-period calendar-month t of 2.43 uses calendar months ending 2018-12. With the end at 2019-01 it would be 2.51.\n- The correlations 0.78 and 0.73 are for selection-period excess returns. The full-period values are 0.75 and 0.74.\n- Also, the revenue-only per-j range (−0.25 to +0.07) matches the zero-cost version, not normal cost (normal cost gives −0.47 to −0.14), so the brief's \"一般成本\" label on that item is off.\n\n5) Conventions I settled by matching numbers:\n- Revenue source is chosen by anchor month. Choosing by signal date shifts J−7 and J+8.\n- A J version's own windows include in-panel months with no trade.\n- The end point is that J day in 2026-10.\n\n6) On independence:\n- replicate_o4 already contained an earlier replicator's rcore.py and related scripts. I did not use them; all my code is in the ind_*.py files.\n- I did not read the researcher's code. I read only their results.json PLAN and PREREG for version definitions, and navs_oos.parquet's J+0 and OFF_AB0 NAVs to diagnose the AB difference."
}

**最終判定（審查後）：不可用**
