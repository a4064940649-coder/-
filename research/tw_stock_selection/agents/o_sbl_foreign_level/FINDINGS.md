# FINDINGS：資訊型放空與外資持股水準（o_sbl_foreign_level）

資料夾：`/home/user/-/research/tw_stock_selection/agents/o_sbl_foreign_level/`

## 結論
- **借券濾網 H3a：無法測試。**
  - 本環境只連得到 GitHub。借券賣出餘額（TWT93U）唯一找到的全市場存檔是 2026-07-30 起的 59 個交易日，挑選期和驗證期都沒有資料。
  - 已事前登記 2026-11 起的前瞻追蹤規格，2028-11 評估。
- **外資持股水準 H3b：待觀察。** 只取得 2021-10 起的資料，判定上限就是待觀察；2015 年以前的資料無法取得。
  - **第二條腿 B1：不可用。**
    - B1 的規則：市值三組內，各取外資持股比率最高 10 檔。
    - 挑選期對同池等權 +0.68%／月、t 2.39；驗證期 2024～2026 降到 +0.51%／月、t 0.98，沒達到 t ≥ 1.0。
    - 驗證期對 0050 t −1.67，對主策略 t −1.83。
  - **主策略第三分數 C2（權重 0.2）：待觀察，只符合替代條件。**
    - 兩期都是年化較高、回撤較淺、追蹤誤差較低。
    - 但對主策略的配對差 t 只有 0.54（挑選期）和 0.15（驗證期）。
    - 在官方實值的保留段，C2 的回撤反而比主策略深（−29.0% 對 −20.0%）。
- **機制：部分支持。**
  - 支持的部分：市值組內「外資持股高減低」的長短價差兩期都顯著，挑選期 +1.68%／月 t 3.63，驗證期 +1.86%／月 t 3.37。
  - 不支持的部分：
    - 效果主要來自外資持股低的那一側表現差，做多高外資持股賺得少。
    - 效果所在的市值組從挑選期的小型股，換成驗證期的大型股。
    - 6 個月增量沒有顯著效果。
    - P7（中型股較強）、P9（主策略右尾）都不成立。

## 1. 研究問題與機制
- **H3a：** 透過借券放空的外資和避險基金，有研究和評價上的優勢，也願意付借券成本，所以借券賣出餘額快速增加帶有負面資訊。融券以散戶為主，當對照：融券增加不應該有負的預測力。
- **H3b：** 外資持股比率是多年累積的存量，反映長期研究和治理篩選（Huang & Shiu 2009）。
  - 這個訊號變化慢、和大型股高度相關，追蹤誤差大，所以沒被套利掉。
  - 效果可能只是規模溢酬，所以必須在市值組內比較。
- **和已測研究的差別：** 研究中心過去測的外資訊號都是 20 日或連續買賣超的「流量」，延後一天就失效。這題測的是「存量」，以及借券。

## 2. 資料（第 0 步，詳見 results_data_audit.json）
- **能連的來源：** 本環境只有 GitHub 可連（git 匿名 clone 和 MCP 搜尋）。twse、openapi.twse、FinMind API、Hugging Face、Kaggle、TPEx、MOPS 全部連不上。
- **查過的來源：** code search 用了 TWT93U、MI_QFIIS、FinMind 欄名、「借券賣出餘額」、「外資持股比率」等關鍵字，約 1,000 筆結果。另外逐一檢查 18 個候選 repo，包括題目列的 5 個。幾乎全是抓取程式，或只有單一個股的小檔。

| 來源 | 欄位 | 起訖 | 上市檔數 | 缺漏日 | 重疊期一致率 |
|---|---|---|---|---|---|
| BlackBoxOP/tw_stock：TWT93U | 融券餘額、借券賣出前日／賣出／還券／調整／當日餘額 | 2026-07-30～10-27（59 日） | 1,062 | 0 | 融券對本機 margin_hist 100%；恆等式 99.997%；前日＝上一筆 100% |
| BlackBoxOP：MI_MARGN | 融資、融券（張） | 2024-01-02～2026-10-27 | 1,069 | 0 | 對 margin_hist（2025-09～）100% |
| BlackBoxOP：T86 | 外資（不含自營）、投信、自營 | 2024-01～2026-10 | 1,094 | 1 | 對 inst_5y 100% |
| 本機 inst_5y | 外資淨買超（張） | 2021-10-08～2026-10-07 | 1,110 | 0 | — |
| voidful/tw-institutional-stocker：T86 | 外資淨買超（股） | 2025-09-11～2026-10-21 | 1,128 | 0 | 對 BlackBoxOP 99.27% |
| voidful：MI_QFIIS | 發行股數、外資股數、外資比率 | 2025-10-07～2026-10-21（254 日） | 1,088 | 0 | 日變動對淨買超：誤差在股數 0.02% 內 70.6%，相關 0.92 |

- **H3b 的資料作法：** 用 2025-10-07 的 MI_QFIIS 快照，在比率空間扣掉之後每天的「外資淨買超 ÷ 股數」，回推到 2021-10-08。
- **回推驗證：** 從 2026-10-21 回推到 2025-10-07，和官方值比較。
  - 回推 1 年：MAE 0.89pp、p90 2.09pp、Spearman 0.981。
  - 回推 0.5 年：MAE 0.67pp。
  - 6 個月增量：Spearman 0.91。
- **偏誤檢查：** 回推誤差和同期報酬的 Spearman 是 −0.01，沒有看得出來的偏誤。
- **回推誤差的來源：** 現增、可轉債轉換、員工認股、ADR 轉換等不經市場交易的股數變動。誤差大致隨回推年數線性增加，2021 年的數字估計 MAE 約 3pp。
- **資料凍結：** 2026-10-27T19:30:04Z（data_freeze.json），早於任何報酬計算。

## 3. 探索（只看挑選期，訊號日 2021-10-12～2023-12，27 期，無成本）
- **H3a：** 挑選期沒有借券或融券資料，探索做不了。
- **P6 成立。** 市值三組內依外資持股比率分 5 組：
  - Q5−Q1 平均 +1.68%／月，t 3.63（sd 2.41%）。
  - 五分位均值依序 0.28、0.74、1.64、1.51、1.96%，大致單調。
  - 不控市值 +1.41（t 1.82）；市值本身 +0.19（t 0.22），所以不是規模溢酬。
- **增量弱。** 6 月增量 +0.67（t 1.06）、12 月增量 +0.81（t 1.35）。P6 後半「增量也為正」只有方向成立。
- **P7 不成立。** 最強的是小型股：小 +2.66（t 2.94）、中 +1.34（t 1.75）、大 +1.05（t 1.25）。
- **0050 成分內外的差別：** 近似 0050 成分以外 +2.17（t 2.98），成分內 +0.96（t 1.44），符合「大型權值股的外資部位比較不帶資訊」。
- **相關矩陣（每月 Spearman 平均）：**
  - 外資比率和市值 0.67、近似 0050 成分 0.35、52 週比 0.26、營收 −0.09、6 月增量 0.08。
  - 外資比率的月對月排名相關 0.985，訊號非常持久。
- **P9 不成立。**
  - 主策略前 20 大貢獻股，入選前 6 個月外資比率上升的比例是 69%（49 筆入選）；全部入選股是 81.5%（200 筆）。
  - 主策略入選股中，外資高的一半減低的一半：−0.18%／月，t −0.12。

## 4. 模型迭代（iterations.json；共 6 版＋8 個鄰近版本）
選版規則事先寫在 iterate.py：第二條腿取「對同池等權 t 最高」；組合版取「對主策略配對差 t 最高」。

| 版本 | 內容 | 挑選期結果（一般成本） |
|---|---|---|
| B1 | 市值組內外資比率最高 10×3 | 對等權 +0.68%／月 t 2.39（壓力 1.96）；年化 22.1%、回撤 −22.6%；和主策略超額相關 0.23 |
| B2 | 市值組內 6 月增量最高 10×3 | +0.61 t 1.14 |
| B3 | 池內外資比率最高 30（不控市值） | +0.55 t 1.46 |
| C1 | 主策略 80%＋B1 20% | 配對差 +0.08 t 0.49；回撤 −33.5 對 −36.8；追蹤誤差 18.8 對 20.9 |
| C2 | 第三分數＝組內外資比率百分位（0.2） | 配對差 +0.34 t 0.54；年化 17.4 對 14.0；回撤 −24.4 對 −36.8 |
| C3 | 第三分數＝6 月增量百分位 | −0.24 t −0.46 |

- **B1 鄰近版本：** 5×3 的 t 2.03、15×3 的 t 2.42；選中的 10×3 不在格點邊緣。
- **C2 鄰近版本：** 權重 0.1 的 t 0.89、0.3 的 t 1.10；選中的 0.2 反而較低，沒有贏家詛咒，但也看不出規律。
- **同期對照：** 主策略 vs 等權 t 0.33；主策略年化 14.0%、等權 13.8%、0050 7.4%。
- **挑選期穩健性（selftest）：**
  - B1 去掉前 5 大貢獻股後 t 只剩 0.72；只看非電子股 t 0.51。挑選期就已經很依賴少數股票。
- **選中的版本：** B1 和 C2。C2 對 C1 只贏一點點（t 0.54 對 0.49）。

## 5. 凍結規格
- 見 frozen_spec 欄位。PREREG 建立於 2026-10-27T19:37:12Z，spec sha256 是 5662cc8d…。

## 6. 樣本外結果（只跑一次）

| 期間 | 主策略 | B1 | C2 | C1 | 同池等權 | 0050 |
|---|---|---|---|---|---|---|
| 驗證 A：2024-02～2025-09（回推持股） | 24.3%／−43.7% | 12.8%／−38.1% | 35.3%／−40.8% | 22.2%／−42.4% | 6.2% | 39.4% |
| 保留 B：2025-10～2026-09（官方實值） | 159.3%／−20.0% | 81.7%／−22.7% | 142.3%／−29.0% | 143.2%／−19.8% | 65.6% | 99.6% |
| 合併 2024～2026 | 63.9%／−43.7% | 34.9%／−38.1% | 68.4%／−40.8% | 58.2%／−42.4% | 26.2%／−37.1% | 59.2%／−27.5% |

（格內是年化／最大回撤）

### B1 第二條腿
- **對同池等權（一般成本）：**
  - A 段 +0.34%／月，t 0.63；B 段 +0.78，t 0.73。
  - 合併 +0.51，t 0.98；壓力成本 t 0.69。
- **對 0050：** −1.34%／月，t −1.67。
- **對主策略：** −2.19%／月，t −1.83。
- **和主策略超額的相關：** 合併 0.44，A 段 0.575。P8 只在合併期成立。
- **市值組內（對各組等權）：** 小 −0.18、中 −0.19、大 +1.88（t 2.01），三組平均 +0.50。
  - 效果從挑選期的小型股換到大型股，代表沒有一個穩定的市值區段。
- **門檻判斷：** (1) 只差 t ≥ 1.0 這一項沒過 → 依事前規則判不可用。

### C2 第三分數
- **配對差：** 合併 +0.13%／月，t 0.15；A 段 +0.59，t 0.84；B 段 −0.60，t −0.28。
- **替代條件，兩期都成立：**
  - 驗證期：年化 68.4 ≥ 63.9 −1，回撤 −40.8 淺於 −43.7，追蹤誤差 26.1 < 28.7。
  - 挑選期：17.4 對 14.0，−24.4 對 −36.8，追蹤誤差 20.1 對 20.85。
- **但保留段相反：** 年化 142.3 對 159.3，回撤 −29.0 對 −20.0。
- **對 0050：** t 0.73。
- **t 的拆解（對等權）：** 挑選期 0.33 → 0.76，主要是均值上升（0.28 → 0.62%），波動幾乎不變（4.29 → 4.16%）。驗證期 2.03 → 2.15，均值 2.70 → 2.83%，波動 7.41 → 7.34%。
- **C1（次要）：** 驗證期配對差 −0.44%／月，t −1.83；回撤 −42.4 對 −43.7；追蹤誤差 24.9 對 28.7。報酬下降的幅度大於風險下降。

### 全期 2021-10～2026-10-12
- 主策略 37.9%（−43.7%），C2 42.1%（−40.8%），B1 29.4%，等權 20.6%，0050 32.1%（−34.0%）。
- 對 0050 t：主策略 0.57、C2 0.96、B1 −0.46。

### 長短價差（P6 的描述性複核，無成本）
- 合併 +1.86%／月，t 3.37；A 段 +1.38（t 2.41）、B 段 +2.66（t 2.38）。
- 價差在樣本外仍然存在，但多頭那一側賺不到：B1 對等權只有 +0.5%／月。所以效果主要是「外資持股低的股票表現差」。

## 7. 穩健性（樣本外）
- **逐年（B1 對等權）：** 2024 +0.57、2025 +0.25、2026 +0.78%／月，每年 t 都 < 0.7。
- **C2 對主策略逐年：** 2024 +1.29、2025 −0.81、2026 +0.11。
- **去掉最大貢獻股（B1，毛）：** 全部 t 1.60；去掉第一大（2360）t 1.18；去掉前 5 大（2360、6442、2308、2454、3711）−0.15，t −0.34。
- **分產業：** B1 名單 29% 是半導體、22% 電子零組件。只看非電子股 −1.64%／月，t −1.65。效果集中在電子股。
- **成交可行性：** B1 有 30 檔。30 萬資金每檔約 1 萬元，100% 要用零股；100 萬資金也有 93% 要零股。實際成交會變成盤中零股，和開盤價有落差。
- **回推誤差：** 和報酬無關（Spearman −0.01）。2025-10 以後的保留段用的是官方實值，結論和回推段方向一致（都弱）。

## 8. H3a 描述性試算（h3a_pilot.json；登記之後才跑，不構成檢定）
- **借券 20 日增量最高 1/5 減其餘：**
  - 2026-09-11：+3.34%，方向和假說相反，248 檔。
  - 2026-10-12：只到 10-22，池子只有 65 檔（營收還沒公告完），不能用。
  - 8 日增量版（2026-08-11）：−1.35%。
- **橫斷面相關：**
  - 借券增量和融券增量只有 0.07、−0.04。這支持「兩種放空者是不同的人」這個前提。
  - 借券增量和 52 週比 −0.25～−0.49。借券是在跌深、遠離高點的股票增加，和「在高點布局」的 P3 前提不一致。但只有兩個橫斷面。
- **P2 對照（融券）：** 2024-02～2026-09 共 30 個月，融券增量最高 1/5 減其餘 +1.29%／月，t 2.19。融券增加沒有負效果，甚至是正的，符合 H3a 對散戶融券或軋空的預期。少了借券資料，這個對照無法和借券比較。

## 9. 機制是否被支持
- **H3b 部分支持：**
  - 成立：市值組內的水準效果兩期都在，不是規模溢酬，0050 成分以外較強。
  - 不成立：
    - 做多側效果小，不足以當第二條腿。
    - 沒有穩定的市值區段（小型股和大型股互換）。
    - 增量（「外資最近在加碼」）沒有效果，所以看不出是外資的「長期資訊」。比較像「外資不碰的股票」有負面特性，可能是品質或治理篩選，而不是外資能挑贏家。
    - P9：主策略右尾和外資加碼無關。
- **H3a 無法判斷：** 機制前提有一點跡象（借券和融券幾乎不相關），但沒有報酬證據。

## 10. 散戶怎麼執行（目前只建議前瞻追蹤，不建議實盤）
- **資料：** 證交所每天公布「外資及陸資投資持股統計」（MI_QFIIS）。換股日用前一交易日的外資持股比率。
- **C2：** 在主策略池內，把每檔的外資持股比率在市值三組內轉成百分位。分數改成 0.4×52 週比百分位＋0.4×營收年增率百分位＋0.2×外資持股百分位，其餘同主策略。
- **追蹤：** 2026-11 起，每月和原版主策略並列記錄名單和紙上淨值。
- **H3a：** 每天從 TWT93U 存借券賣出餘額，2028-11 依登記的規格評估。

## 11. 限制
1. **資料只有 2021-10 起。** 挑選期只有 27 個月，涵蓋 2022 年空頭；驗證期 31 個月。所有 t 的檢定力都低，判定上限是待觀察。
2. **2025-10 以前是回推值。** 誤差隨回推年數變大，2021 年估計 MAE 約 3pp。市值是用 2025-10 的股數近似。
3. **存活者偏誤。** 價格面板 2019～2023 只有存活股；外資持股只涵蓋 2025-10 仍上市的股票。
4. **TWT93U 只能前瞻。** 外資持股和借券的鏡像都是個人 GitHub repo（不可信，已逐筆對帳），未來可能停更。研究中心應該從現在開始自己存檔。
5. **多重檢定。** H3b 共 6 版＋8 個鄰近版本；挑選期 t 2.39 經多重檢定折扣後的證據力更弱。
6. **只測上市股，沒有上櫃。**

## 12. 檔案
- 程式：fetch_consolidate.py、data_audit.py、build_freeze.py、sbl_lib.py、explore.py、iterate.py、prereg.py、oos.py、h3a_pilot.py、finalize.py
- 結果：results.json（PREREG、results、verdict）、results_data_audit.json、explore.json、iterations.json、results_oos.json、results_oos_selftest.json、h3a_pilot.json
- 淨值：oos_navs.parquet
- 面板：fo_panel.parquet、sbl_panel.parquet、short_panel.parquet
- 原始資料（不可信）：raw/，含 manifest.json 的來源 commit 與 sha256
- 沒有修改其他資料夾，沒有 git commit。

## 凍結規格

PREREG（凍結規格全文）
- 寫入位置：/home/user/-/research/tw_stock_selection/agents/o_sbl_foreign_level/results.json 的 PREREG 鍵
- 建立時間：2026-10-27T19:37:12Z
- 規格 sha256：5662cc8daf794f826db836b0ecca5caccaa55e9818e17f359a9a91c3f0385d95
- 資料凍結：data_freeze.json，2026-10-27T19:30:04Z，早於任何報酬計算
- 程式 sha256（oos.py 會先斷言 PREREG 存在、這些 hash 都沒變，才計算樣本外）：
  - sbl_lib.py 1dd3717f…
  - oos.py c3ed99f0…
  - data_freeze.json 284d1f65…
  - iterate.py 0f9c3114…
  - explore.py 7b03540d…

一、資料
1. 外資持股比率
   - 2025-10-07～2026-10-21：用 MI_QFIIS 實值。來源是 GitHub 鏡像 voidful/tw-institutional-stocker 的 data/twse_foreign.csv，彙整成 raw/vf_twse_foreign.parquet。
   - 2021-10-08～2025-10-06：在比率空間回推，r(t) = r(2025-10-07) − Σ_{s∈(t, 2025-10-07]} 外資（不含自營）淨買超(s) ÷ 2025-10-07 發行股數。
   - 淨買超資料：本機 extras/inst_5y（2021-10-08～2026-10-07）；2026-10-08 起改用 BlackBoxOP/tw_stock data/institutional。兩者在重疊期逐筆 100% 一致。
   - 回推值截在 [0,1]。小於 −0.5pp 的格子約占 0.4%。
   - 整理後的面板：fo_panel.parquet。
2. 時點：訊號日 t 只用 t−1 收盤已公布的值。
3. 市值近似：2025-10-07 的 MI_QFIIS 發行股數 × 訊號日收盤（shares_anchor.csv）。
4. 限制：2015 年以前、甚至 2021-10 以前的全市場外資持股比率和借券賣出餘額都取不到。判定上限是「待觀察」。

二、H3b 主版本 B1（第二條腿）
1. 股池：和主策略同池（main_v2 gapfix）。條件是 60 日均成交值 ≥ NT$1,000 萬、收盤 ≥ 10，依 60 日均值取前 300 名，再排除金融保險（金融保險業＋28xx、5876、5880）、處置股，以及隔日沒有開盤成交的股票。
2. 選股：池內依近似市值分成小、中、大三組，每組取外資持股比率最高的 10 檔，共 30 檔，等權。
3. 換股：每月第一個日期 ≥ 11 日的交易日收盤算訊號，隔日開盤成交（M2）。新買進遇到開盤漲停就跳過，往下一名遞補；已經持有的不受影響。停牌的持股凍結。
4. 成本：
   - 一般：手續費 0.1425%×0.6＋滑價 0.1%，賣出另加 0.3% 證交稅。
   - 壓力：每邊再加 0.3% 滑價。
5. 對手：
   - 同池等權（無成本，main_v2 口徑）
   - 0050 含息
   - 主策略 n10 排除金融（重建後和 main_v2 的 EXFIN|n10|normal 逐日報酬差 < 1e-15）

三、H3b 組合主版本 C2（依挑選期規則「對主策略配對差 t 最高」選出）
- 主策略分數改成：0.4×52 週比百分位＋0.4×營收 3 月年增率百分位＋0.2×市值組內外資持股比率百分位（缺值給 0.5）。
- 其餘規則同主策略 n10 排除金融。
- 次要版本 C1（只報告，不用於判定）：主策略 n10 80%＋B1 20%，每個成交日以前一日收盤近似，把權重重設回 80/20。

四、期間
- 挑選期：訊號日 2021-10-12～2023-12-11，共 27 期（B1 有 26 個完整月窗）。
- 驗證期 A：2024-02-15～2025-09，持股比率是回推值。
- 保留段 B：2025-10～2026-09，持股比率是官方 MI_QFIIS 實值，評價到 2026-10-12。
- 判定用 A＋B 合併（2024～2026），各段另外報告。
- 2026-11 起做前瞻紙上追蹤。

五、門檻
1. B1：
   - 扣一般成本後，對同池等權的月超額在挑選期 t ≥ 2。
   - 驗證期均值 > 0 且 t ≥ 1.0。
   - 和主策略超額的相關 < 0.5。
   - 市值組內仍成立：三組各自對組內等權的月超額，三組平均 > 0。
2. C2 要滿足以下其中一項：
   - 對主策略的配對差兩期都 > 0 且 t ≥ 1.0；或
   - 替代條件：兩期年化都 ≥ 主策略 −1pp，而且最大回撤較淺、對 0050 的追蹤誤差較低。
3. 判定：
   - 資料只有 2021-10 起，最多判待觀察。
   - B1 驗證期均值 ≤ 0 或 t < 1.0 → 第二條腿不可用。

六、H3a 前瞻追蹤（現在無法測試）
1. 訊號：
   - 借券賣出當日餘額(t−1) − 借券賣出當日餘額(t−21)，除以 t−1 的 MI_QFIIS 發行股數。
   - 池內最高 1/5 是「借券增加」名單。
   - 融券（MI_MARGN 今日餘額）用同樣方法算，當對照。
2. 2026-11 起每個換股日記錄：
   - P1：借券增加名單和其餘股票的下一期報酬差（成交日開盤 → 下一成交日開盤）。
   - P2：融券同法。
   - P3：近似 0050 成分以外（近似市值前 50 以外）、且 52 週比 ≥ 池中位數的子樣本。
   - P4：近似 0050 成分。
   - P5：主策略入選股中命中者 vs 其他入選股，以及「命中就跳過、往下遞補」版的紙上淨值。
3. 2028-11 累積滿 24 個月才評估，門檻用原計畫：
   - P1 毛價差 ≤ −0.5%／月且 t ≤ −2，P2 沒有同樣的負效果。
   - 主策略層級：配對差 t ≥ 1.5，每月平均被濾掉 ≥ 0.5 檔。
4. 2026-08～10 的描述性試算（h3a_pilot.py）不用於任何決策。

七、多重檢定
- H3b 共 6 版：B1、B2、B3、C1、C2、C3。
- 另有 8 個鄰近版本：每組 5／15 檔、權重 0.1／0.3。
- H3a 0 版。

八、執行順序（全部用 python3 -I -B，在 o_sbl_foreign_level/ 下）
1. fetch_consolidate.py <BlackBoxOP clone> <voidful copy>
2. data_audit.py
3. build_freeze.py
4. explore.py
5. iterate.py
6. oos.py selftest（只用挑選期資料除錯）
7. prereg.py
8. oos.py
9. h3a_pilot.py
10. finalize.py

## 反方審查

[
 {
  "final": "不可用",
  "issues": [
   "機制「部分支持」的主要證據其實是 52 週比（動能）的影子。研究員拿市值組內外資持股 Q5−Q1 價差兩期都顯著（t 3.63／3.37）當支持證據。我在每個月先把外資持股比率對池內 52 週比百分位做迴歸、取殘差，再在市值組內分五組：挑選期剩 +0.70%／月、t 1.42，驗證期剩 +0.75%／月、t 1.29，兩期都不顯著。再加控營收、log 市值、電子業後，驗證期只剩 +0.44%／月、t 0.70；其中 VAL_A 段 +0.03%（t 0.04）。主策略本身已經用 52 週比，所以這個價差對主策略沒有增量資訊。研究員的相關矩陣列了外資比率和 52 週比相關 0.26，卻沒有做這個控制。",
   "「不是規模溢酬」只在粗的市值三分組下成立。改用連續 log 市值加電子業虛擬變數做殘差後，驗證期價差降到 +0.67%／月、t 1.16，A 段 t 0.85、保留段 t 0.77。外資比率和市值的相關有 0.67，三分組內殘留的規模差異足以造成大部分樣本外價差。挑選期控制後還有 t 3.26，所以這是挑選期的特性，樣本外沒有保住。",
   "B1 挑選期 t 2.39 經多重檢定後不顯著。登記的試驗有 6 版加 8 個鄰近版本，共 14 個；explore 另外還看了 lvl、d6、d12、不控市值、市值、0050 成分內外、三個市值組等十幾個切面。df 25 時，m=6 的 Bonferroni p 是 0.15，m=14 是 0.35，m=14 需要 t ≥ 3.22。驗證期 t 0.98 的單邊 p 是 0.17。整體看，B1 從頭到尾都沒有統計證據。",
   "挑選期和驗證期都靠少數股票。挑選期去掉前 5 大貢獻股後 t 剩 0.72，這在登記前的 selftest 就已經知道。驗證期去掉 2360、6442、2308、2454、3711 後，B1 對等權是 −0.15%／月（t −0.34）；只看非電子股是 −1.64%／月（t −1.65）。驗證期 B1 的效果就是 2024～2026 AI 題材大型電子股，和外資資訊優勢無關。",
   "回推持股有未來資訊：2025-10 以前的外資比率是用 2025-10-07 的快照回推。算出來的值其實是 F(t)/S(2025-10)，也就是用未來的股數去除，市值近似也用同一個未來股數。現增、可轉債轉換這類股數變動，常發生在股價大漲之後。研究員做的誤差對報酬檢查只有一個橫斷面（保留段，Spearman −0.01），挑選期回推 2～4 年、MAE 約 3pp 那段完全沒驗證。驗證期 Q1 的平均外資比率只有 6.29%，3pp 的誤差足以讓 Q1、Q2 大量錯分。",
   "存活者條件：挑選期池內外資比率缺值的比例是 0%。價格面板有 155 檔在 2024 年前停止交易，但沒有一檔出現在挑選期池中且缺外資值。這代表池子實際上只剩活到 2025-10 的股票，等權基準和 Q1 都被存活者條件化。兩邊受影響的方向不同，偏誤無法量化。",
   "C2 的判定理由不夠，沒有統計檢定支撐。對主策略的配對差，挑選期 t 0.54、驗證期 t 0.15（標準誤約 0.87%／月，而均值只有 0.13%）。分段看，A 段 +0.59、保留段 −0.60，逐年 2024 +1.29、2025 −0.81、2026 +0.11，正負交替。替代條件「年化不低 1pp、回撤較淺、追蹤誤差較低」只比點估計、不檢定，而且在官方實值的保留段反過來：年化 142.3 對 159.3，回撤 −29.0 對 −20.0。C2 勝過 C1 只靠 t 差 0.05。在可信的官方實值段落，C2 反而比主策略差，這應該判不可用。",
   "驗證期不是全新的資料：C2 是在主策略上加一個分數，而主策略（main_v2 gapfix）本身就是研究中心在包含 2024～2026 的資料上建出來、看過的。所以 C2 對主策略的比較不是乾淨的樣本外；保留段主策略年化 159% 本身就是事後結果。",
   "長短價差裡顯著的是空方那一側，散戶做不到。驗證期 Q1 減組均值 −1.17%／月、t −4.01，Q5 減組均值 +0.69%、t 1.94，而且是未扣成本的毛報酬。控制 52 週比之後，Q5 那一側驗證期只剩 +0.21%（t 0.66）。逐年看，2025 年價差也只有 t 0.95。",
   "H3a 無法測試的判定成立。h3a_pilot 只有兩個橫斷面，其中一個方向還和假說相反（+3.34%），不能拿來當機制前提的跡象。"
  ],
  "note": "我在 /home/user/-/research/tw_stock_selection/agents/o_sbl_foreign_level/review_stat_o2.py 寫了驗證程式（用 python3 -I -B 執行；研究員的檔案沒有改）。結果和研究員的數字一致：Q5−Q1 挑選期 +1.68%／月、t 3.63，驗證期 +1.86%／月、t 3.37。去掉最好的 3 個月、去掉回推值接近 0 的股票、只看電子或非電子股，價差都還在。\n\n但只要在月度橫斷面先控制 52 週比百分位，兩期的 t 就降到 1.42／1.29。再加控營收、連續 log 市值和電子業，驗證期剩 t 0.70。所以「外資持股水準有獨立資訊」這件事在樣本外不成立，它大致是主策略已經用到的動能訊號，加上組內殘留的規模和 AI 電子題材。\n\nB1 判不可用，我同意，而且比研究員說的更弱：挑選期 t 2.39 經 14 個試驗的 Bonferroni 修正後 p 0.35；去掉前 5 大貢獻股後，驗證期是 t −0.34。C2 對主策略配對差 t 0.15，保留段的官方實值年化和回撤都比主策略差，替代條件只比點估計，不能撐起待觀察。\n\n我的判定：\n- **H3b：** 不可用，機制應從「部分支持」改成「不支持」（或者只說價差描述性存在，但大多被動能解釋）。\n- **H3a：** 維持無法測試。\n- **整體：** 我往不利方向判為不可用。"
 },
 {
  "final": "不可用",
  "issues": [
   "Costs are not what sinks this topic, and the researcher's numbers on that point hold up. Turnover per rebalance (fraction of the book sold): main 0.64, C2 0.58, B1 0.24 (validation period). With a 0.671% round trip, the annual cost drag is 5.2%, 4.7% and 2.0%; under stress costs (1.27% round trip) it is 9.8%, 8.8% and 3.7%. Rebalances that sell at a limit-down open: 0 in validation (1 in the selection period, never locked all day). At 10 million NTD, each name's allocation is at most 1.04% of its 60-day average traded value (median 0.05–0.06%). Capacity and limit-move problems are not the issue. The problem is that the edge itself is noise.",
   "C2's 'alternative condition' (annual return not below the main strategy's minus 1pp) rests on a single month. The best relative month is signal date 2026-04-13 (C2 minus main +10.08pp). Replace that month's C2 return with the main strategy's and, on monthly windows, C2 falls to 63.64% a year against the main strategy's 66.31%, which breaks the 'not below minus 1pp' condition. Under stress costs it also fails: 56.92% against 58.82%. Monthly paired difference without the best month: −0.20%/month, t −0.23; without the best two: −0.42, t −0.48. Only 58% of months are positive.",
   "The backcast data error is about as large as C2's edge. In the holding segment, I replaced the official MI_QFIIS values with a one-year backcast anchored on 2026-10-21. C2 changes 0.92 of its 10 names per month, and its monthly return shifts by −1.41%/month (t −1.55). Validation segment A runs on backcast holdings with 0–4 years of error (researcher's estimate: MAE about 3pp for 2021). C2's edge there, +0.59%/month (t 0.84), is smaller than the noise the data error alone produces. That segment cannot stand for the names a live trader would see.",
   "In the only segment that uses official values (2025-10 to 2026-09), C2 loses: 142.3% vs 159.3% a year, drawdown −29.0% vs −20.0%. The result also swings on one month. The first month, 2025-10-13, is −15.95pp for C2. If my run starts inside that segment and drops that month, the gap flips to +0.80%/month, t 0.46. Either way it is noise, and it gives no support for adopting C2.",
   "C2 tilts toward high-priced stocks, which hurts small accounts. Validation picks: median open price 433.5 NTD for C2 vs 257 for main; 25.6% of C2 picks are above 1,000 NTD vs 20.6%. At 300,000 NTD, 98% of C2 picks need odd lots; at 1 million NTD, 78.8% (main 70.3%). B1 needs odd lots for 100% of picks at 300,000 NTD and 95.6% at 1 million. Intraday odd-lot orders are not matched at the opening auction, but the backtest assumes every buy and sell fills at the open, so that slippage is not modeled. C2's edge over main is only +0.13%/month (t 0.15; 0.19 under stress costs), so even a little odd-lot slippage would wipe it out.",
   "The only significant result is the within-size-group 'high minus low foreign holding' long-short spread (+1.86%/month, t 3.37), and it is mostly short-side alpha. Retail traders can hardly capture it. Margin short selling (融券) runs into limited share availability, the uptick rule (平盤下限制), forced buy-backs before shareholder meetings and ex-dividend dates, and disposal-stock prepayment rules (處置預收). Securities borrowing (借券) is mostly closed to retail. So the 'mechanism partly supported' finding cannot be turned into a strategy retail can trade, and it should not be read as a reason to keep C2 on watch.",
   "Implementing this means watching the screen at the open. A new buy that opens limit-up is skipped and replaced by the next candidate, so the trader must see the 9:00 open before placing the replacement order. With odd lots that means being at the screen after 9:00 to place intraday orders; it cannot be done with pre-open orders alone. The main strategy has the same requirement, but the backtest does not include the extra price slippage this causes.",
   "Small look-ahead and tradability details (not quantified, and the main strategy shares them): (1) cross_section uses the execution day's TRADED and open price to decide eligibility, which assumes on signal day that the stock will trade the next day. (2) The disposal (處置) filter only checks the signal day, so stocks that enter disposal on the execution day can still be bought, and the model still assumes an opening fill even though disposal stocks require prepayment (預收款券) and batched matching. (3) Size groups use 2025-10 share counts times price, so the backcast period knows future share counts. (4) The foreign holding data only covers stocks still listed in 2025-10, so B1 can only pick survivors."
  ],
  "note": "I tried to overturn the conclusion from the cost and tradability side. I did not modify any of the researcher's files. My verification script is /home/user/-/research/tw_stock_selection/agents/o_sbl_foreign_level/review_exec_o2.py (python3 -I -B).\n\nVerdicts:\n- **H3a (borrowing filter):** I agree it is untestable.\n- **B1 (second leg):** I agree it is not usable. Validation t is 0.98 under normal costs and 0.69 under stress costs. Turnover is low (2%/year cost drag), so costs are not what kills it; the alpha simply isn't there.\n- **C2 (third score):** I disagree with \"watch\" and would downgrade it to \"not usable\", for four reasons:\n  1. The alternative condition depends on one month, 2026-04-13. Neutralize it and C2 trails the main strategy by 2.7pp a year (63.64 vs 66.31), which fails the condition; under stress costs it also fails (56.92 vs 58.82).\n  2. The data error alone moves C2's return by −1.41%/month, more than its edge in the backcast segment (+0.59%/month).\n  3. In the only segment that uses official values, C2's return and drawdown are both worse than the main strategy's.\n  4. C2 tilts toward high-priced stocks, so small accounts need more odd lots, and the open-price fill assumption becomes more optimistic.\n- **Mechanism:** The significant long-short spread is mostly short-side alpha, which retail cannot capture.\n\nOverall verdict on what is tradable: not usable. Keeping the H3a daily borrowing-data archive for forward tracking is still worthwhile. If C2 is tracked forward, run it as a paper record only. Before any re-evaluation it should first:\n- record odd-lot fills at the actual intraday prices, not the open price;\n- clear the bar with t ≥ 1 on official data only."
 }
]

## 獨立複製

{
 "reproduced": true,
 "my_numbers": "I rebuilt everything independently in /home/user/-/research/tw_stock_selection/agents/o_sbl_foreign_level/replicate_o2/ and ran it all with python3 -I -B. I did not read any researcher code or their processed files. Data came from the original voidful csv, the BlackBoxOP parquet files, extras/inst_5y and margin_hist. Pool and engine are the shared main_v2/v2_core + ia_core. My main n10 ex-financials rebuild matches main_v2 EXFIN|n10|normal with a daily return difference of 0.\n\n[Data] (researcher → mine)\n- Foreign net-buy agreement: inst_5y vs BlackBoxOP 100% → 100%; voidful vs BlackBoxOP 99.27% → 99.27%.\n- Short balance vs margin_hist 100% → 100%.\n- TWT93U identity 99.997% → 99.997%.\n- MI_QFIIS daily change within 0.02% of shares: 70.6% → 71.0%. Correlation 0.92 → 0.89 (same-day lag; lag 1/2 is only 0.07).\n- 1-year backcast MAE/p90/Spearman: 0.89/2.09/0.981 → 0.88/2.05/0.981.\n- 0.5-year MAE 0.67 → 0.66; 6-month increment Spearman 0.91 → 0.92.\n- Backcast error vs return Spearman −0.01 → −0.005; share of cells below −0.5pp 0.4% → 0.49%.\n- Lending balance: 59 days, 1,062 listed stocks → same. Holdings ratio: 254 days, panel starts 2021-10-08 → same.\n\n[Exploration, selection period, 27 windows]\n- Q5−Q1 within size groups +1.68 t3.63 → +1.63 t3.38.\n- Small / mid / large 2.66 / 1.34 / 1.05 → 2.74 / 1.18 / 0.97.\n- Without size control +1.41 t1.82 → +1.42 t1.84; size itself +0.19 t0.22 → +0.19 t0.23.\n- 6-month increment +0.67 t1.06 → +0.74 t1.13; 12-month +0.81 t1.35 → +0.87 t1.45.\n- Outside approx-0050 +2.17 t2.98 → +2.16 t2.97 (top 50 by market-wide approx market cap). Inside 0.96 t1.44 → 1.45 t1.26.\n- Correlations with market cap / 52-week ratio / revenue 0.67/0.26/−0.09 → 0.67/0.26/−0.10.\n- P9 top 20 / all picks 69% / 81.5% → 69.4% / 81.5%.\n\n[B1]\n- Selection, 26 windows: +0.68 t2.39 (stress 1.96) → +0.69 t2.29 (stress 1.85). Annualized 22.1 → 21.7–23.1 depending on endpoints; drawdown −22.6 → −22.5.\n- A, 19 windows: +0.34 t0.63 → +0.37 t0.71. B, 12 windows: +0.78 t0.73 → +0.69 t0.62.\n- Validation combined, 31 windows from 2024-03: +0.51 t0.98 (stress 0.69) → +0.49 t0.94 (stress 0.66). Annualized / drawdown 34.9 / −38.1 → 36.0 / −37.6. B annualized 81.7 → 79.9.\n- Validation t vs 0050 / vs main −1.67 / −1.83 → −1.68 / −1.84.\n- Correlation with main strategy excess, selection / validation / A: 0.23 / 0.44 / 0.575 → 0.29 / 0.445 / 0.555.\n- Validation size groups −0.18 / −0.19 / +1.88 t2.01 → −0.17 / −0.23 / +1.86 t2.00.\n- Gross excess without top 5 contributors −0.15 t−0.34 → −0.25 t−0.52.\n\n[C2]\n- Selection annualized C2 / main 17.4 / 14.0 → 19.5 / 14.0 (from 2021-10-13 to 2024-02-15). Drawdown −24.4 / −36.8 → −24.4 / −36.8.\n- Selection paired difference +0.34 t0.54 → +0.47 t0.72.\n- Validation annualized 68.4 / 63.9 → 68.2 / 63.2; drawdown −40.8 / −43.7 → same; paired difference +0.13 t0.15 → +0.01 t0.01.\n- B annualized 142.3 / 159.3 → 137.2 / 159.3; drawdown −29.0 / −20.0 → −27.1 / −20.0.\n- Tracking error vs 0050, validation 26.1 / 28.7 → 25.8 / 28.7. Selection 20.1 / 20.85 → 20.6 / 20.5 (27 windows) or 18.0 / 18.7 (26 windows).\n- Names per month that differ from the main strategy: 4.15 → 3.93.\n\n[Full period, 2021-10-12 to 2026-10-12]\n- Main 37.9 / −43.7 / t0.57 → 37.4 / −43.7 / t0.53.\n- C2 42.1 / −40.8 / t0.96 → 42.5 / −40.8 / t0.93.\n- B1 29.4 / −38.1 → 29.1 / −37.6; C1 36.6 → 36.1; equal weight 20.6 → 20.3; 0050 32.1 / −34.0 → 32.06 / −33.96.\n\n[H3a]\n- 2026-09-11 top fifth by lending increase minus the rest +3.34% → +3.29%.\n- Lending increase vs short increase correlation 0.07 → 0.066 (Spearman; Pearson 0.118).\n- Short-sale top fifth minus the rest, 30 months +1.29 t2.19 → +1.28 t2.19 (2024-04 to 2026-09).\n\n[Thresholds]\n- B1: selection t ≥ 2 passes. Validation t < 1.0 (0.94), so it fails. Correlation < 0.5 passes. Size-group average > 0 passes.\n- C2: paired t is below 1 in both periods. The alternative condition holds: annualized ≥ main −1pp in both periods, shallower drawdown, lower tracking error.\n- These match the researcher's conclusions.",
 "diff_explained": "1. Data audit, exploration, B1 validation, full-period figures and H3a reproduce almost exactly, mostly within 0.1pp and t within 0.1.\n2. Window counts. Researcher's \"A = 19, validation = 31\" equals dropping the 2024-02-15 window and starting from 2024-03. With that definition, validation +0.49 t0.94, t vs 0050 −1.68, vs main −1.84, correlation 0.445 and the size groups all line up. With 32 windows it would be +0.64 t1.21, so the B1 \"fail / pass\" verdict depends on that choice. Selection uses 26 complete windows, ending 2023-11. Short-sale \"30 months\" equals 2024-04 to 2026-09.\n3. Annualized endpoints. Selection CAGR matches the researcher's main 14.0 when measured from 2021-10-13 to 2024-02-15 (mine 13.95). B1 shifts by about ±1pp with the endpoints (21.7–23.1 vs 22.1).\n4. C2 is the largest gap: selection annualized 19.5 vs 17.4, B section 137 vs 142, validation paired difference +0.01 vs +0.13. C2 differs from the main strategy by only about 4 names per month, so a few picks swapping moves the result. Shifting the holdings-ratio date by one day changes selection CAGR by 0.2–2pp and the validation paired t by −0.4 to 0 (r_c2_sens.py). The likely source is small implementation details (holdings-ratio date alignment, size-group boundaries, tie handling), not a logic error. Drawdowns, validation annualized, tracking errors and the verdict match.\n5. Main strategy full-period 37.4 vs 37.9: the starting point differs by one day or the start is defined differently; drawdown and 0050 are identical.\n6. Holdings-ratio daily change correlation 0.89 vs 0.92 and the in-0050 split 1.45 vs 0.96 likely come from a different stock universe or definition of the approx-0050 constituents. Other definitions match exactly: market-wide top 50 by approx market cap, and P9 measured over the selection period.\n7. Removing the top 5 contributors gives −0.25 vs −0.15. Same direction (excess turns negative); the difference is in how the gross per-window calculation replaces those 5 stocks.\n\nKey files:\n- r_data.py, r_lib.py, r_strat.py, r_explore.py, r_h3a.py, r_scan.py, r_c2_sens.py\n- my_data_audit.json, my_strat.json, my_explore.json, my_h3a.json, my_navs.parquet\n- my_compare.json (side-by-side comparison of the researcher's and my numbers)"
}

**最終判定（審查後）：不可用**
