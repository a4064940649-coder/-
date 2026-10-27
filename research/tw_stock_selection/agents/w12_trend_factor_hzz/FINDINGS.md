# FINDINGS：多均線趨勢因子（Han–Zhou–Zhu 2016 JFE；中國 A 股版 Liu–Zhou–Zhu）

資料夾：`/home/user/-/research/tw_stock_selection/agents/w12_trend_factor_hzz/`
- `prereg.py`：事前設定。在看任何結果之前就寫進 `results.json` 的 `PREREG`，之後沒有改過。
- `run.py`：回測程式。成交價格陣列（AO/AC）、漲停判定、成本都直接用 `exec_rebuild/exlib.py`（M2），沒有改原檔。執行紀錄在 `run_log.txt`，約 15 秒跑完。
- `results.json`：PREREG、meta、6 個組合 × 6 個期間、配對差、診斷（IC、五分位、β̄、相關）、判定。
- `nav.parquet`：各組合的日頻淨值（基本成本、毛、0.5%、壓力）以及同池等權（毛、淨）。
- `monthly_counts_TF11_Q20.csv`：主版本每月的股池大小、持股檔數、有 1000 日歷史的比例。
- 來源：HZZ 2016（wustl profile）、Liu–Zhou–Zhu 中國版（AUT ACFR PDF、WRDS WARSP）、r-bloggers 2014、AEA 2021。

## 結論：不可用

- **主版本 TF11_Q20 扣成本（滑價 0.1%，來回約 0.67%）後，兩段都沒過判定。**
  - 主版本設定：11 條價格均線、長均線缺值用可得最長均線替代、買前 20%。
  - 挑選期 2009-02～2019-04（123 個月）：
    - 淨年化 7.48%，毛年化 12.48%。0050 13.58%，同池等權 12.52%（扣成本後 11.75%）。
    - 對等權月超額 −0.37%（t −2.38），顯著輸。對 0050 月超額 −0.39%（t −1.27）。
    - 日頻最大回撤 −35.3%（0050 −23.0%，等權 −34.3%）。
  - 驗證期 2019-05～2026-10（89 個月）：
    - 淨年化 17.47%，毛年化 22.85%。0050 28.27%，等權 17.41%。
    - 對等權月超額 +0.16%（t 0.46）。對 0050 t −1.03。最大回撤 −37.8%。
  - 2024～2025（23 個月）：
    - 年化 4.1%，0050 42.4%，等權 8.0%。
    - 對等權 t −0.73，對 0050 t −2.31。
- **毛報酬也只和等權打平，訊號在台股沒有排序能力。**
  - 挑選期扣成本前對等權 t 0.09，驗證期 1.58。
  - 截面 IC：挑選期 −0.002（t −0.14），驗證期 0.006（t 0.42）。
  - 挑選期 Q5−Q1 毛價差每月 −0.03%（t −0.09）。
  - 每月換手約 56% 市值，每年大約吃掉 5 個百分點。毛報酬原本就沒有優勢，扣成本後就變成顯著落後。
- **驗證期的五分位價差來自輸家那一端，只做多吃不到。**
  - 驗證期 Q5−Q1 每月 +1.34%（t 2.48）。
  - 其中 Q1 相對全池 −0.70%（t −2.92），是主要來源；Q5 相對全池只有 +0.64%（t 1.85）。
  - 價差集中在 2019–2023 只有存活者的面板（Q5−Q1 t 2.44）。到了 2024-01～2026-10 的完整面板，t 只剩 1.06，IC 是 −0.018。
- **回歸學出來的均線權重不穩定，兩段之間會翻號。**
  - 挑選期：β_50 = +0.065（t 2.11），β_20 = −0.072（t −1.58）。
  - 驗證期：β_20 = +0.156（t 2.01），β_10 = −0.20（t −1.85），β_200 = −0.060（t −2.40）。
  - 20 日均線的係數從負翻成正。滾動 12 個月平均追的是上一年的型態，所以「機器學到的均線組合」在台股是在追雜訊。
- **長均線（400～1000 日）沒有幫助，反而略差。**
  - 用 TF11 減只用 L≤200 的 S200（都買前 20%）看扣成本月差：挑選期 −0.06%（t −0.67），驗證期 −0.15%（t −1.58）。
  - 看前 30 檔：驗證期 −0.41%（t −1.74）。
- **中國版加上成交量均線也沒有改善。**
  - CN_Q20 減 TF11_Q20 的月差：挑選期 +0.03%（t 0.23），驗證期 +0.09%（t 0.88）。
  - 8 個量均線係數的 t 全部小於 1.1。
- **六個版本沒有一個在任何一段過判定。**
  - 所有版本挑選期對等權的 t 都在 −1.88～−2.44 之間。
  - 驗證期最好的是 S200_N30：淨年化 22.1%，對等權 t 1.09，仍輸 0050 的 28.3%。
- **最大的虧損月份是動能崩盤。**
  - 主版本的趨勢分數和 12-1 動能的截面等級相關約 0.25～0.33。分數高的股票通常站在長均線之上很遠；2026-06 選到的股票 A_200 平均 0.61，全池是 0.88。
  - 2026-07：組合 −23.9%，同月等權 −10.5%。
    - 主要是被動元件類股票單月重挫：2492 −62%、3090 −58%、2327 −56%。
    - 已檢查，這段是官方來源（src 4）、flag 0、價格逐日下滑，不是資料錯誤。
  - 其他大虧月份（2018-10 −15.6%、2020-03 −17.5%）和等權相近。

## 事前規則（PREREG 摘要，看結果前固定）

- **資料：**
  - 有值檔數 < 300 的日子（2018-01-13）整列剔除。剔除之後才算月底日曆、均線、V60 和其他回看。
  - 只用 4 碼代碼，排除 91xx。
  - 月報酬用 adj 月底對月底。|r| ≥ +200% 或 ≤ −80% 設成缺值，全期共 3 筆。
- **均線：**
  - A_L = MA_L(adj) / adj_t，L ∈ {3,5,10,20,50,100,200,400,600,800,1000}。adj 在上市後向前填補（停牌日沿用最後價），月底當天必須有 adj。
  - 個股至少要有 200 日歷史才給訊號。L 超過歷史長度的均線，用已有真值的最長均線替代（論文作法）。
  - 中國版：價和量各取 L ∈ {3,…,200,400} 共 8 條，合計 16 個變數。量 = value / close_raw，月底量必須 > 0。量的 400 日均線缺值時用 200 日替代。
  - 每個月、每個變數在當月樣本內做 1%/99% 截尾。教學沒有給這個參數，這是事前固定的常見作法。
- **回歸與預期報酬：**
  - 回歸：r_{i,m} = β0 + Σβ_L·A_{L,i}(上月底) + ε。樣本是上月底基礎股池，至少 50 檔才記錄係數，平均每月約 522 檔。
  - 第一個回歸月是 2008-02。那時面板第一次累積到 1000 個交易日（上月底 2008-01-31）。
  - 預期報酬 E = Σβ̄_L·A_L(本月底)，β̄ 是最近 12 個已實現月份係數的平均。
  - 第一個訊號日 2009-01-21（農曆年前的 1 月最後交易日），所以挑選期實際從 2009-02-02 開始持有。
- **股池：**
  - 月底條件：V60 ≥ NT$1,000 萬、收盤 ≥ 10 元、flag ∈ {0,1,2}，隔日要有開盤價與 adj。
  - 三種訊號都要有值，所以所有組合共用同一個股池，平均 531 檔。
  - 股池中真有 1000 日歷史的比例每年在 82%～94% 之間，其餘靠替代補值。
- **組合：**
  - PRIMARY TF11_Q20：買前 floor(20%)，平均 106 檔，等權。
  - 另報：TF11_N30、S200_Q20、S200_N30、CN_Q20、CN_N30。
- **成交：**
  - 月底收盤算訊號，隔日開盤成交（M2），持有 1 個月。
  - 開盤漲停（2015-06 前 7%、之後 10%，容差 0.6%）而且原本沒持有的股票視為買不到，由下一名遞補。主版本全期共擋掉 79 次。
  - 共 212 次換股，最後持有到 2026-10-22。
- **成本：**
  - 基本情境（判定用）：買 0.1855%、賣 0.4855%。
  - 另報：0.5% 情境、壓力情境（來回 1%）、毛報酬。
- **對手與統計：**
  - 同池等權用同一引擎、不扣成本（毛），比扣成本的策略嚴格。另有 0050 含息。
  - t 值用日曆月超額報酬（月份分群），挑選期與驗證期分開算。
- **判定：** 主版本在挑選期和驗證期都要同時滿足三項：對等權月超額 > 0、t ≥ 2、淨年化 ≥ 0050。

## 1. 主要結果（基本成本；年化／對等權 t／對 0050 t／最大回撤）

| 版本 | 挑選期 2009-02～2019-04 | 驗證期 2019-05～2026-10 | 2024～2025 |
|---|---|---|---|
| **TF11_Q20（主）** | 7.48% / −2.38 / −1.27 / −35.3% | 17.47% / 0.46 / −1.03 / −37.8% | 4.1% / −0.73 / −2.31 |
| TF11_N30 | 3.56% / −2.44 / −1.59 / −48.7% | 17.06% / 0.47 / −0.61 / −43.6% | 5.5% / −0.09 / −1.70 |
| S200_Q20 | 8.30% / −1.88 / −1.09 / −36.1% | 19.42% / 0.88 / −0.82 / −38.1% | 6.3% / −0.25 / −2.30 |
| S200_N30 | 3.92% / −2.30 / −1.55 / −42.5% | 22.13% / 1.09 / −0.13 / −49.4% | 9.0% / 0.29 / −1.46 |
| CN_Q20 | 7.94% / −2.16 / −1.19 / −36.7% | 18.88% / 0.78 / −0.90 / −36.7% | 5.6% / −0.40 / −2.15 |
| CN_N30 | 4.52% / −2.38 / −1.58 / −52.5% | 16.71% / 0.42 / −0.61 / −44.8% | −0.5% / −0.71 / −1.95 |
| 0050 | 13.58% / — / — / −23.0% | 28.27% / — / — / −34.0% | 42.4% |
| 同池等權（毛） | 12.52% / −34.3% | 17.41% / −33.8% | 8.0% |

- **主版本在其他成本情境下：**
  - 挑選期：毛 12.48%、0.5% 情境 8.95%、壓力 5.13%（對等權 t −3.57）。
  - 驗證期：毛 22.85%、0.5% 情境 19.05%、壓力 14.93%（對等權 t −0.09）。
- **驗證期拆兩段：**
  - 2019-05～2023-12（只有存活者）：年化 17.67%，等權 17.33%，0050 14.66%。對等權 t 0.39。
  - 2024-01～2026-10（完整面板）：年化 17.14%，等權 17.55%，0050 54.48%。對等權 t 0.26，對 0050 t −2.13。
- **全期 2009-02～2026-10：** 年化 11.59%，0050 19.57%，等權 14.56%。對等權 t −0.88，最大回撤 −47.7%。

## 2. 逐年（TF11_Q20 扣成本／0050／同池等權，%）

- 2009：106.5 / 86.3 / 135.2（2 月起）
- 2010：8.2 / 13.1 / 9.7
- 2011：−30.9 / −15.8 / −27.7
- 2012：16.5 / 11.9 / 14.1
- 2013：24.7 / 11.7 / 21.5
- 2014：4.7 / 16.7 / 4.7
- 2015：−12.8 / −6.3 / −8.8
- 2016：−1.9 / 19.6 / 7.6
- 2017：23.4 / 18.1 / 23.4
- 2018：−24.7 / −4.9 / −11.1
- 2019：20.4 / 33.5 / 26.4
- 2020：44.7 / 31.1 / 24.8
- 2021：38.0 / 21.9 / 32.0
- 2022：−21.5 / −21.4 / −14.3
- 2023：27.0 / 27.5 / 35.6
- 2024：16.7 / 48.6 / 11.9
- 2025：−7.1 / 36.8 / 4.3
- 2026（到 10-22）：44.0 / 67.2 / 35.1

18 年中只有 7 年勝過等權（2012、2013、2020、2021、2024、2026 以及接近打平的 2017），而且勝出的都是趨勢延續的多頭年。

## 3. 診斷

- **IC 與五分位（毛，隔日開盤到下一個隔日開盤）：**
  - 挑選期：IC −0.002（t −0.14）。Q5−Q1 −0.03%（t −0.09）。
  - 驗證期：IC 0.006（t 0.42）。Q5−Q1 +1.34%（t 2.48），其中 Q1 −0.70%（t −2.92）、Q5 +0.64%（t 1.85）。
  - 2024-01～2026-10：IC −0.018。Q5−Q1 +1.16%（t 1.06）。
- **β̄ 的符號在兩段之間不一致**（見結論）。11 條均線高度共線，係數噪音很大，12 個月平均也壓不住。
- **截面等級相關：**
  - TF11 與 S200 為 0.80～0.87，TF11 與 CN 為 0.68～0.79。
  - TF11 與上月報酬為 0.10～0.20，和 12-1 動能為 0.25～0.33。
  - 也就是說，學到的組合偏向溫和的趨勢／動能，沒有學出美股論文中短均線帶來的反轉成分。
- **替代比例：** 股池中有 6%～18% 的股票沒有 1000 日歷史，靠替代補值。2019 年以後超過 92% 有完整歷史，但這部分是因為面板只有存活者。

## 4. 限制

- 只做多、等權。論文用的是全市場十分位多空，所以本題不能直接否定「多空價差」，只能說台股只做多前 20% 吃不到。
- 2019–2023 面板只有存活者，長均線樣本因此偏樂觀；即使如此，驗證期仍然沒過。
- 2024-01-02～2024-02-14 沒有資料，所以沒有 2024-01 月底，那一期持有跨兩個月。
- src 3（2024-02-15～2025-09-23）的收盤和成交量都是 Yahoo 5 分 K 近似值。
- 只有上市股，沒有上櫃。處置股名單只有 2021-09 起，本題不排除處置股。

## 判定

- PRIMARY TF11_Q20：挑選期（對等權 t −2.38，淨年化 7.48% < 0050 13.58%）不過；驗證期（t 0.46，淨年化 17.47% < 0050 28.27%）不過。
- 結論：**不可用**。6 個變體在兩段也都沒過，不支持改判。


## 反方審查

{
 "final": "不可用",
 "issues": [
  "Lookahead check: I found no future data in the signal. At month-end m the factor uses A_L(ME[m-1]) to regress r[m], β̄ averages the 12 realized months m-11..m, the forecast uses A_L(ME[m]), and trades fill at the next day's open (M2). The pool, V60, winsorizing and the 1000-day start date all use data known on the signal date.",
  "Recomputed t-stats from nav.parquet with my own code. Selection period: net vs gross EW t -2.38, net vs 0050 t -1.27. Validation: vs EW 0.46, vs 0050 -1.04. All match results.json. The calculation is calendar-month excess returns.",
  "Wording of 'significantly worse' overstates the case. The -2.38 compares the net strategy with a gross EW benchmark, so the benchmarks are not symmetric. Net vs net EW gives t -2.01 (-0.31%/month). Gross vs gross EW gives t 0.09, about +0.01%/month. The fair reading: in the selection period the signal has essentially zero gross alpha, and costs (turnover 56.5% per month, round trip about 0.67%) turn it into a significant loss. The signal itself is not reliably negative. The verdict does not change.",
  "The validation period's positive gross alpha (+0.54%/month, t 1.57) depends on a few extreme months. Removing the 3 best months (2026-04 +12.4%, 2021-06 +12.1%, 2026-09 +6.7%) drops t to 0.68. Even taken as given it is not significant, and after costs t is 0.46.",
  "Survivorship bias, 2019-2023: the panel holds only survivors, which inflates the validation period's absolute returns. Even with that boost, 2019-05 to 2023-12 only gives t 0.39 vs EW. The bias works in the strategy's favor, so it cannot overturn the negative verdict.",
  "src=3 approximate closes (Yahoo 5-minute bars, 2024-02 to 2025-09): these feed the short moving averages (A_3, A_5), the volume moving averages (value/close_raw) and execution prices. That adds noise exactly in the weakest stretch, 2024-2025 (4.1%/yr vs 0050 42.4%), so the size of that collapse is not reliable evidence. But 2019-05 to 2023-12, with no approximate data, also fails, so the verdict stands.",
  "Several simplifications make the backtest optimistic, so the real result is likely worse. Delisted or suspended stocks are carried forward at their last price instead of written down. Sells are assumed to fill even when a stock is locked limit-down. Disposal stocks (處置股) are not excluded. Only 79 of 22,436 buys (0.35%) were blocked by an opening limit-up, so that effect is negligible.",
  "Possible post-hoc tuning: prereg was written at 13:49:19 and run.py was edited again at 13:50:32, so I cannot rule out repeated runs or changes. However, all 6 pre-specified variants (TF11, S200 and CN, each as top 20% and top 30) fail in both periods. That is the opposite of cherry-picking a winner, so it does not affect a negative verdict.",
  "How far the conclusion reaches: this tests long-only top 20% in a liquid listed-stock (上市) pool. It is not the paper's long-short decile spread. In validation, the gross Q5-Q1 spread is +1.34%/month (t 2.48), mostly from the bottom quintile's -0.70% (t -2.92), and the mean IC is only 0.006 (t 0.42). In the selection period Q5-Q1 is about 0. So the result rules out long-only use. It does not show the trend factor is useless in Taiwan in general: OTC stocks (上櫃) and a short side were never tested.",
  "Data quality: the 0050 file has a Sunday row on 2026-09-20 (-7.1%, then +9.2% the next day). The backtest reindexes 0050 onto the panel's calendar, which skips that row, so the results are unaffected. The 2024-01 gap in the panel merges two months into one holding period and has little effect.",
  "Sample size: the selection period has 123 months and validation 89, about 120 to 700 stocks per month, which is enough. 2024-2025 is only 23 months and should be read as a stress window, not a basis for judgment."
 ],
 "note": "I agree with the researcher's verdict that the multi-moving-average trend factor is 不可用 (unusable). I recomputed the key t-stats independently from nav.parquet and they match. I found no lookahead bias or cost-calculation errors. The main survivorship, data and simplification issues all bias results upward, so they cannot overturn a negative verdict.\n\nThe one claim to correct is \"significantly worse than the equal-weight pool\" (t -2.38). That number compares the net strategy with a gross benchmark. A fair reading is that gross alpha is about zero (t 0.09) in the selection period, and turnover costs push it to significantly negative (net vs net t -2.01). The validation period's gross alpha depends on 3 extreme months.\n\nThe conclusion covers only the long-only top-20% use on listed (上市) stocks. It does not show the trend factor is useless in Taiwan.\n\nFiles: /home/user/-/research/tw_stock_selection/agents/w12_trend_factor_hzz/run.py, /home/user/-/research/tw_stock_selection/agents/w12_trend_factor_hzz/prereg.py, /home/user/-/research/tw_stock_selection/agents/w12_trend_factor_hzz/results.json, /home/user/-/research/tw_stock_selection/agents/w12_trend_factor_hzz/nav.parquet"
}