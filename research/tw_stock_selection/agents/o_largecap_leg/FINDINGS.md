# FINDINGS：大型股池第二條腿（o_largecap_leg）

子代理不能寫 .md 檔，請主研究員把這份全文存成 `agents/o_largecap_leg/FINDINGS.md`。
資料夾：`/home/user/-/research/tw_stock_selection/agents/o_largecap_leg/`

## 結論

- **判定：不可用。機制：不支持。**
- **最終規格**：照事前規則凍結 V01，即成交值前 50 池選 5 檔、0.5／0.5 分數、等權、組合比例 30%。
- **SOP 和 live_tracking 都不改。**
  - 「2330 30%」維持現狀，也就是 TSMC30_R，待觀察。
  - 想降低追蹤誤差，可以用「2330 30%」或「0050 30%」；大型股腿沒有提供兩者以外的東西。
- **為什麼不可用（事前判定 (1)～(4) 全部不過）**
  - **(1)／P1 選股力不成立。** 挑選期 L 腿對同池等權只有 +0.26pp／月，t 0.62，門檻是 1.5。壓力成本下是 −0.04。
  - **(2) 追蹤誤差沒有下降。** 組合的追蹤誤差 VAL_A 27.2% 對主策略 27.3%，HOLD 29.1% 對 28.8%，門檻要低 3pp。
  - **(3) 輸 TSMC30_R。**
    - 挑選期對 0050 的 t：1.55 對 3.09。
    - 2330 漂移歸零的反事實：SEL 1.53 對 2.28，只有 VAL 1.54 對 1.43 略高。
    - 池剔除 2330 後，SEL t 0.89。
  - **(4) 驗證期的正超額只集中在 5 檔。** 萬海、創意、長榮、川湖、金像電剔除後，淨超額 −0.48pp。
- **組合比主策略差。**
  - 主策略 70%＋L 腿 30% 的跨循環年化 25.2%，主策略 27.7%；對 0050 t 2.18 對 2.56；最大回撤 −66.2% 對 −65.0%。
  - 挑選期對主策略的配對差 −0.34pp／月，t −2.77。
  - 驗證期 +0.13pp（t 0.61）與 2024～2026 +0.02pp 都只是雜訊，而且不是盲測。
- **結構上的原因（本題最重要的發現）**
  - 「60 日均成交值前 50 名」不是大型股池，是高周轉／注意力股池。
    - 只有 33～44% 是臺灣50 成分。
    - 2024～2026 的 L 腿名額有 54% 不在臺灣50，例如川湖、台光電、亞翔、晟銘電、創意。
  - 在這個池裡用 52 週高點＋營收選股，選到的仍是「中型動能股」：
    - 超額和主策略高度相關（0.60～0.77，P2 要求 ≤ 0.5）。
    - 對（2330−0050）的 β 是負的（SEL −0.23、VAL −0.43、HOLD −0.74；P2 要求 ≥ 0）。
    - 所以它補不了「權值股領漲就輸 0050」的缺口。
  - 權值股領漲的年份，L 腿反而更差：挑選期組合減主策略 −0.41pp／月（t −3.13）。這和機制 2 的預測相反。
- **V3 資料重跑結論相同。** V3 包含下市股價格、併購對價出清、全期處置濾網。
  - 主策略跨循環 26.15%，t 2.39；組合 24.05%，t 2.05；TSMC30 26.37%，t 3.31。
  - L 腿對池等權：SEL +0.29（t 0.69），VAL +0.89（t 1.12）。

## 1. 研究問題與機制

- **問題**：主策略是等權的中型股組合，它對 0050 的超額對（2330−0050）的 β 是負的，所以權值股領漲的年份一定輸 0050。用事前可知、規則化的大型股動能腿，能不能取代「固定買 2330 30%」？
- **原本假設的贏家**
  - 外資：在大型高流動股有研究優勢，而且是順勢交易者。
  - 市值加權的被動資金：持續買進權值股。
- **原本的機制假設**
  1. 大型股也有較小的反應不足，所以同一組分數在大型池內仍有選股力。
  2. 被動資金的順風：領漲權值股同時接近 52 週高點、營收也在成長。
  3. 大型股腿的超額和主策略低相關，所以追蹤誤差下降。
  4. 為什麼沒被套利掉：主動基金有單一持股上限；被動資金不看動能；大型股的順勢超額小，大機構不值得專門做。
- **已看過的期間（事前寫明）**
  - o_capacity_tsmc_gap 看過 1～50 名入選股兩期的報酬。
  - integrate_o1／o2 看過 TSMC30。
  - o_sbl 看過 2021 年以後的大型組。
  - 2024～2026 被台積電行情污染。
  - 所以 2019 年以後不算盲測，判定上限是待觀察。

## 2. 資料與程式

- **基準**：MAIN 是 integrate_o2 的 MAIN_R（R_DEL 營收、排除金融 n10）；TSMC30 是 TSMC30_R。兩者都在程式內斷言，和 integrate_o2/navs.parquet 逐日相同（最大相對差 0.0）。
- **引擎**：只讀 import o_rev_survivorship/osurv_core 和 o_capacity_tsmc_gap/ocore（→ main_v2/v2_core → independent_audit/ia_core）。
  - 成本、跳動點漲停遞補、停牌凍結都沿用，沒有重寫。
  - 加權組合用 ocore.simulate_w。
  - 0050 對照組用袖子層混合，因為面板沒有 0050 開盤價。移轉金額扣成本，賣出 0050 加 ETF 稅 0.1%。
- **大型股池**
  - 主版本：合格股中 60 日均成交值前 P 名，合格條件見凍結規格。
  - 輔助版本：臺灣50 成分。
    - 從 etf0050_w.csv 的 2026-10-09 成分出發，用 etf_rebalance/events.csv 倒推。
    - 另加三個非定期事件：2016-12 中壽替華亞科、2018-04 國巨替矽品、2018-04 日月光投控承接日月光。
    - 36 步倒推都一致：每步成分 50 檔，沒有矛盾。
    - 只用 2015-04 以後的資料。
  - 不用市值排名（shares_hist 回推等於前視），也不用外資持股。
- **程式與檔案**
  - 依執行順序：plan.py → tw50.py → explore.py → iterate.py → freeze.py → oos.py → robust.py → oos_v3.py → posthoc.py。
  - 共用核心：lcore.py。
  - 輸出：plan.json、tw50_members.json、explore.json、iterations.json、results.json（PREREG＋RESULTS）、results_robust.json、results_v3.json、posthoc.json、navs.parquet。
  - 全程 `python3 -I -B`；沒有下載資料（raw/ 是空的，已刪除）；只寫本資料夾；沒有 git commit。
- **程序上的誠實揭露**
  1. **PREREG 的 score 欄位被覆蓋。** freeze.py 寫 frozen_spec 時，'score' 鍵被文字說明蓋掉了（文字內容就是 0.5／0.5）。
     - oos.py 改從 sha256 已鎖定的 iterations.json 讀代碼 '55'，並斷言和文字一致。
     - 規格沒有任何改變。
  2. **oos.py 漏存穩健性結果。** 穩健性字典 rob 算了但沒寫進 RESULTS。
     - robust.py 用同一段程式重算，存成 results_robust.json。
     - 程式內斷言 L 腿 VAL 月均和 RESULTS 一致。
     - 這些結果只是描述，不參與判定。
  3. **lcore.py 凍結後沒有改動**（sha256 比對：lcore_changed_after_freeze=False）。

## 3. 探索（只用 SEL：2006-04～2018-12，153 個訊號）

- **(a) 大型池（P50）內的五分位價差**：下一期「開盤→開盤」報酬，Q5−Q1。

  | 分數 | Q5−Q1 | 月 rank IC |
  |---|---|---|
  | 合成 0.5／0.5 | +0.62pp（t 1.23） | 0.029（t 1.73） |
  | 只用 52 週比 | +0.55（t 0.93） | — |
  | 只用營收年增率 | +0.60（t 1.29） | — |

  - 營收分數「前 1/5 減池等權」是 0.00，價差全部來自最差組。
  - 電子業 +0.87（t 1.43），非電子 +0.73（t 1.02）。
  - 池子大小：P30 +0.16（t 0.28），P80 +0.47（t 1.02）。
  - 分段：2006～2012 +0.22（t 0.28），2013～2018 +1.08（t 1.78）。
- **(a2) 資訊傳遞速度**：「前 1/5 減池等權」，換算成每月。

  | 持有期 | 大型池 | 主策略池中第 51 名以後 |
  |---|---|---|
  | 1 個月 | +0.39（t 1.38） | +0.86（t 3.97） |
  | 3 個月 | +0.32（非重疊 t 1.64） | +0.64（t 3.31） |
  | 6 個月 | +0.27 | +0.40 |

  - 大型股的反應不足大約只有中型股的一半，方向和機制 1 一致。
  - 但幅度小於換股成本：L 腿每月換掉約 50% 名單，成本拖累約 0.3～0.4pp／月。
- **(b) L5 的持股結構**
  - 每月替換 49.5% 名單；153 個月共 141 檔不同的股票；名稱集中度 HHI 0.016。
  - 2330 入選月份占 16.3%。
  - 入選最多：大立光 39 次、可成 33、聯發科 27、台積電 25、上銀 20。
  - 名額中電子業占 65%，池內電子業占 64.5%。
- **(c) 相關與 β**
  - L 腿超額和主策略超額（都對 0050）的月相關 0.60；L 腿相對池等權的超額和主策略超額的相關 0.49。
  - 對（2330−0050）的 β：L 腿 −0.23，主策略 −0.30，L 腿相對池等權 +0.02。
  - 池等權對 0050：−0.46pp／月（t −1.91）。13 年中有 11 年是 0050 贏池等權。
- **(d) t 的拆解**
  - V01 組合：SEL Δt −0.79，其中均值 −0.86、波動 +0.07。
  - TSMC30：Δt +0.75，其中均值 −0.28、波動 +1.03。
- **(e) 子樣本**
  - 權值股領漲年份：L 腿對池等權 +0.33（t 0.72），組合減主策略 −0.41（t −3.13）。
  - 其他年份：L 腿 −0.12。
  - 臺灣50 輔助池（2015-04～2018，45 個月）：L 腿對池等權 +0.12（t 0.25）。
  - 臺灣50 合格成分中，只有 71.5% 落在 P50 池內。
- **探索階段就看得到的問題**：大型池內的選股力弱，而且組合的 t 下降。

## 4. 模型迭代（只用 SEL；事前列出 12 版，全部報告）

| 版本 | 規格 | 組合對 0050 t | 月均超額減主策略 | 追蹤誤差減主策略 | L 腿對池等權 |
|---|---|---|---|---|---|
| V01 | P50 n5 0.5／0.5 等權 w30 | 1.554 | −0.336 | −0.59 | +0.259（t 0.62） |
| V02 | P30 | 1.244 | −0.484 | −1.59 | −0.078 |
| V03 | P80 | 1.753 | −0.231 | +0.08 | +0.528（t 1.16） |
| V04 | n3 | 1.590 | −0.276 | +0.66 | +0.486 |
| V05 | n8 | 1.481 | −0.404 | −1.74 | +0.032 |
| V06 | 只用 52 週比 | 1.243 | −0.496 | −1.97 | −0.257 |
| V07 | 成交值加權 | 1.601 | −0.327 | −0.84 | +0.284 |
| V08 | w20 | 1.843 | −0.223 | −0.57 | +0.259 |
| V09 | w50 | 0.937 | −0.562 | −0.08 | +0.259 |
| V10 | P80 n8 | 1.655 | −0.299 | −0.63 | +0.296 |
| V11 | P30 n3 | 1.693 | −0.279 | −0.51 | +0.598（t 1.18） |
| V12 | 只用 52 週比＋成交值加權 | 1.176 | −0.524 | −2.15 | −0.353 |

- **基準**：主策略 SEL 對 0050 t 2.343，追蹤誤差 17.09%。
- **對照組（SEL）**

  | 對照組 | 對 0050 t | 追蹤誤差 |
  |---|---|---|
  | TSMC30 | 3.091 | 11.69 |
  | 主策略 70%＋0050 30% | 2.329 | 11.97 |
  | 安慰劑（池等權 30%） | 1.574 | 13.41 |
  | 成交值第 1 名 30% | 1.399 | 14.96 |

- **選版**：12 版的月均超額都比主策略低 0.15pp 以上，沒有合格版本，依事前規則凍結 V01。
- **版本較多的影響**
  - 不論哪一版，都低於主策略的 2.34，所以多重檢定不影響結論。
  - 若忽略 0.15pp 條件，t 最高的是 V08（w20），它落在格點邊緣，只是把 L 腿的權重降低。

## 5. 樣本外結果（凍結後；一般成本）

| 組合 | 期間 | 年化% | 對 0050 t | 追蹤誤差% | 月均超額 pp | 對主策略配對差 pp（t） |
|---|---|---|---|---|---|---|
| 主策略 | VAL | 46.06 | 1.400 | 27.72 | 1.168 | — |
| | VAL_A | 38.72 | 1.471 | 27.29 | 1.496 | — |
| | HOLD | 60.09 | 0.376 | 28.83 | 0.553 | — |
| | ALL | 27.66 | 2.560 | 21.65 | 1.022 | — |
| **COMBO** | VAL | 47.89 | 1.557 | 27.80 | 1.303 | +0.135（0.61） |
| | VAL_A | 41.14 | 1.670 | 27.23 | 1.695 | +0.199（0.69） |
| | HOLD | 60.72 | 0.383 | 29.11 | 0.568 | +0.016（0.05） |
| | ALL | 25.23 | 2.183 | 21.43 | 0.863 | −0.159（−1.40） |
| TSMC30_R | VAL | 46.09 | 1.912 | 18.57 | 1.069 | −0.099 |
| | HOLD | 64.98 | 0.513 | 19.63 | 0.514 | −0.038 |
| | ALL | 27.47 | 3.441 | 14.62 | 0.928 | −0.094 |
| 主策略 70%＋0050 30% | VAL | 42.00 | 1.385 | 19.41 | 0.809 | −0.359 |
| | HOLD | 60.85 | 0.365 | 20.24 | 0.377 | −0.176 |
| 安慰劑（池等權 30%） | VAL | 43.55 | 1.423 | 22.06 | 0.945 | −0.223 |
| 成交值第 1 名 30% | VAL | 44.45 | 1.519 | 22.73 | 1.039 | −0.129 |
| L 腿單獨 | VAL | 48.89 | 1.518 | 34.74 | 1.587 | — |
| | HOLD | 59.21 | 0.321 | 35.24 | 0.578 | — |

- 0050 同期年化：VAL 30.40%、VAL_A 16.57%、HOLD 59.23%、ALL 14.94%。
- 驗證期 t 的變化幾乎全是均值造成的：VAL Δt +0.16，其中均值 +0.16、波動 −0.004。也就是說，L 腿沒有降低追蹤誤差。TSMC30 的 t 提升則有 129% 來自波動下降，平均超額其實是下降的。
- **壓力成本 stress3（每邊再加 0.3% 滑價）**
  - COMBO：VAL 41.44%（t 1.10）、HOLD 54.07%（t 0.12）、ALL 20.10%（t 1.28）。
  - 主策略：ALL 22.10%（t 1.61）。TSMC30：ALL 23.50%（t 2.44）。
- **L 腿對同池等權**
  - VAL +1.07pp（t 1.29），VAL_A +1.45（t 1.43），HOLD +0.36（t 0.25）。
  - 池剔除 2330：VAL +0.93（t 1.12）。
- **判定逐條**
  - (1) SEL t 0.62 < 1.5 → 不過。
  - (2) 追蹤誤差沒降 3pp → 不過。配對差與 HOLD 年化兩項則有過。
  - (3) SEL t 1.55 < 3.09，反事實 SEL 1.53 < 2.28，剔除 2330 後 SEL t 0.89 → 不過。
  - (4) 剔除前 5 大貢獻股後 −0.48pp → 不過。

## 6. 穩健性

- **分年**：L 腿對池等權的年度超額加總正負交錯。
  - 正的年份：2006 +16、2010 +22、2011 +22、2015 +20、2019 +16、2021 +60、2025 +18pp。
  - 負的年份：2008 −17、2016 −11、2018 −18pp。
  - 2021 年的 +60pp 主要來自航運股，占 VAL 正超額的一大半。
- **剔除最大貢獻股**
  - VAL：萬海 +37.1pp、創意 +32.7、長榮 +30.1、川湖 +25.2、金像電 +17.9。剔除後毛超額 −0.07pp（t −0.10）。
  - SEL 的最大貢獻股是上銀、可成、美律、大立光。
- **分產業**
  - VAL 電子業入選股減池等權 +1.77pp（t 1.98），非電子 −0.66（t −0.34）。
  - SEL 兩者差不多：電子 +0.89，非電子 +0.78。
  - 這只是事後描述。
- **權值股領漲年份**（0050 贏池等權）
  - VAL 只有 2020、2022、2024 三年。
  - 這三年 L 腿對池等權 −0.15，組合減主策略 −0.30；其他年份分別是 +1.82 和 +0.40。
  - 方向和機制 2 相反。
- **外資買超月份**（2021-11～2026-09，買超 25 個月、賣超 33 個月；同期分組，不可交易）
  - 組合減主策略：買超月 +0.83pp（t 2.13），賣超月 −0.53（t −1.72）。
  - L 腿對池等權：買超月 +1.26（t 0.99）。
  - 這和「外資順勢」的方向一致，但用的是事後資訊。
- **臺灣50 輔助池**（合格成分平均 31～32 檔）
  - L 腿對池等權：2015-04～2018 +0.12（t 0.25），VAL +0.84（t 1.42），HOLD +1.73（t 1.56）。
  - 組合：VAL 42.7%（t 1.28，追蹤誤差 23.2%），HOLD 62.7%（t 0.50，追蹤誤差 26.6%）。
  - 挑選期太短、太弱，不能據以採用。
- **12 版全部攤開看驗證期**（只揭露，不能拿來選版）
  - 沒有任何一版的組合追蹤誤差比主策略低 3pp。
  - 事後最好的是 V03（P80）：L 腿對池等權 VAL +2.03（t 1.99），但 HOLD 只有 +1.80（t 0.94），SEL 只有 t 1.16。
- **V3 資料重跑**（主策略淨值和 V3 正式淨值逐日相同）
  - 所有判定方向都不變。
  - COMBO：ALL 24.05%（t 2.05）；反事實 SEL t 1.40，TSMC30 2.07。

## 7. 機制是否被支持

- **機制 1（大型股也有反應不足）：方向一致，但太弱。**
  - 大型池的訊號強度大約是中型股的一半，扣成本後沒有選股力（P1 失敗）。
- **機制 2（被動資金的順風）：不支持。**
  - 成交值排名的大型池不是權值股池，L 腿很少持有 2330（16%）。
  - 在權值股領漲的年份，L 腿反而輸。
- **機制 3（分散效果）：不支持。**
  - 相關 0.60～0.77，β 是負的；組合超額和主策略超額的相關高達 0.95～0.97。
- **機制 4（沒被套利掉的理由）**：前三點都不成立，所以不必再討論。
- **真正能降低追蹤誤差的，只有「直接持有權值股」**，也就是 2330 或 0050。
  - 它的代價是平均超額下降：0050 30% 的 t 不升反降；TSMC30 的 t 提升全來自波動下降，而且依賴 2330 本身的漂移。
- **整體判定：不支持。**

## 8. 散戶怎麼執行

- **不採用 L 腿**，維持主策略 n10。追蹤誤差選項維持「2330 30%」（待觀察），或改用 0050 30%。
- **可行性**（若照凍結規格執行，以 2024～2026 入選股估計）

  | 資金 | 每檔分配 | 一張買得起的比例 | 零股剩餘現金 |
  |---|---|---|---|
  | 30 萬：L 腿 | 1.8 萬 | 0%（中位股價 617.5 元） | 3.1% |
  | 30 萬：主策略 | 2.1 萬 | 0.3% | 1.6% |
  | 100 萬：L 腿 | 6 萬 | 0.6% | 0.95% |
  | 100 萬：主策略 | 7 萬 | 19% | — |

  - L 腿和主策略幾乎都必須用零股。
  - 2330 30% 的部位：30 萬約 0.08 張，100 萬約 0.25 張。
  - L 腿每月約 5.7 筆換股交易。
- **和處置股策略 20% 合併的總配置**（處置股上限 20%）
  - 現行建議：處置股 20%、主策略 56%、2330 24%。
  - 只用主策略：處置股 20%、主策略 80%。
  - 0050 降追蹤誤差版：處置股 20%、主策略 56%、0050 24%。
  - L 腿版（未通過，不建議）：處置股 20%、主策略 56%、L 腿 24%。

## 9. 限制

- **不是盲測。** 2019 年以後的期間已被看過，2024～2026 被台積電行情污染，判定上限本來就是待觀察。
- **大型股的定義。** 事前選定用成交值當代理，事後才確認它主要抓到高周轉股。真正的市值池需要歷史股數，但 shares_hist 只有 2025-09 以後。臺灣50 輔助池只回推得到 2015-04，而且排除金融、KY 後只剩約 31 檔。
- **其他資料缺口**
  - V2 面板 2019～2023 只有存活股；已用 V3 重跑處理。
  - 處置名單 2021-09 以前，V2 沒有、V3 是近似。
  - 外資資料只有 2021-10 以後。
- **0050 對照組的近似。** 用袖子層混合，換股日是以收盤再平衡，不是開盤。
- **檢定總數。** 12 個模型版本、7 個對照組、4 條判定門檻（8 個子條件）、預測 P1～P4。結論是全面不過，所以檢定數多不影響方向。
- **程序瑕疵**（第 2 節已揭露）：PREREG 的 score 鍵被覆蓋、oos 漏存穩健性結果。兩者都沒有影響規格或判定。

## 凍結規格

【凍結規格：o_largecap_leg V01（results.json PREREG）】
PREREG sha256 96f1b71621f08655fa360d13d1558ced5ba99497ce58af33df82db84a977bf05；凍結時間 2026-10-27T21:37:05Z。
- 依據：plan.json 的研究計畫，sha256 5c1b7ebb…，2026-10-27T21:30:40Z，寫在任何計算之前。
- 凍結前的淨值都只模擬到 2019-01-11 收盤。
- 樣本外結果在 2026-10-27T21:40:15Z 寫入，只寫一次。oos.py 先斷言 PREREG、iterations.json、tw50_members.json 的 sha256 相符，才計算。
- lcore.py 凍結後沒有改動（lcore_changed_after_freeze=False）。

選版
- 12 版全部都不符合「SEL 月均超額不比主策略低 0.15pp 以上」。
- 依事前規則凍結主版本 V01。

1. 主策略（MAIN）
   - 等於 integrate_o2 的 MAIN_R，程式內斷言逐日相同（差 0.0）。
   - 規格：main_v2 正式規格（gapfix 窗、正式次排序、排除金融保險，含 28xx／5876／5880）。
   - 營收：2021-03 以前用 o_rev_survivorship 的 R_DEL，之後用 rev_5y。
   - 持股：n10 等權。

2. 大型股池 P＝50
   - 時點：每月第一個日期 ≥ 11 的交易日，以收盤資料判斷。
   - 範圍：上市股。
   - 合格條件：
     - 60 日均成交值（gapfix 窗）≥ 1,000 萬，收盤 ≥ 10。
     - 排除金融保險、KY／DR（FOREIGN 114 檔）。
     - 近 3 月營收年增率和 52 週比都有值。
     - 非處置股（2021-09 起才有名單）。
     - 隔日有開盤成交。
   - 合格股依 60 日均成交值由大到小，取前 50 名。

3. 分數
   - 池內百分位：0.5×52 週比（還原收盤 ÷ 252 日最高）＋0.5×近 3 月營收年增率（m−1～m−3 平均）。
   - 同分時，營收年增率高者優先，再比代碼小者。

4. L 腿：選分數前 5 檔，等權
   - 隔日開盤成交。
   - 開盤漲停、而且原本沒有持有的股票，遞補下一名。
   - 停牌部位凍結。每月換股。

5. 組合（COMBO）：單一帳戶
   - 主策略 10 檔各 7%，L 腿 5 檔各 6%；兩邊都選到的股票，權重相加。
   - 每次換股成交日回到目標權重，換股之間不再平衡。

6. 成本
   - 一般：買 0.0855%＋滑價 0.1%；賣 0.0855%＋稅 0.3%＋滑價 0.1%。
   - 壓力 stress3：每邊再加 0.3% 滑價。
   - 另報研究中心舊壓力：買賣各 0.5%。

7. 期間
   - SEL：訊號窗起點 2006-04～2018-12。
   - VAL：2019-01～2026-09 訊號，評價到 2026-10-12。
   - VAL_A：2019～2023。HOLD：2024-01～2026-10-12。

8. 對照組
   - MAIN。
   - TSMC30：主策略 70%＋2330 30%，等於 TSMC30_R，已斷言。
   - 主策略 70%＋0050 30%：袖子層混合。
   - 主策略 70%＋池等權 30%（安慰劑）。
   - 主策略 70%＋成交值第 1 名 30%。
   - L 腿單獨。
   - 另報池剔除 2330 的版本。

9. 判定門檻
   - (1) SEL：L 腿淨值對同池等權（零成本）月超額 t ≥ 1.5；VAL：月均 > 0。
   - (2) VAL：COMBO 對主策略的配對差 ≥ −0.1pp／月；HOLD 年化 ≥ 主策略；VAL_A 和 HOLD 的追蹤誤差都比主策略低 ≥ 3pp。
   - (3) 和 TSMC30 比，三項都要成立：
     - SEL 對 0050 的 t ≥ TSMC30。
     - 2330 漂移歸零的反事實下（SEL、VAL 兩段各自歸零），SEL 和 VAL 的 t 都 > TSMC30。
     - 池剔除 2330 後，(1) 仍成立。
   - (4) VAL 剔除貢獻最大的 5 檔後（期報酬換成池等權），扣成本拖累後仍 > 0。
   - (5) 結論對應：
     - 全部通過 → 待觀察（上限）。
     - P1 不成立，或剔除 2330 後超額 ≤ 0 → 不可用。
     - (3) 不過 → 維持 TSMC30_R。

10. 已知缺陷（照實揭露，沒有改動判定）
    - freeze.py 的 frozen_spec 中，'score' 鍵被文字說明覆蓋（文字內容是「0.5×52 週…＋0.5×營收」）。
    - oos.py 改從 sha256 已鎖定的 iterations.json 讀代碼 '55'，並斷言兩者一致。

## 反方審查

[
 {
  "final": "不可用",
  "issues": [
   "I could not overturn the verdict. P1 fails under every cost treatment I tried, and Newey-West changes nothing. Leg net vs a zero-cost pool, SEL: +0.26pp, t 0.62 (NW 0.63). With the pool also paying normal costs (turnover 14% vs the leg's 53%), the fairer net-vs-net version is +0.35pp, t 0.83. Even gross (zero cost) it is +0.60pp, t 1.43, still below 1.5. Year-block bootstrap of SEL leg vs pool: P(mean ≤ 0) = 0.21. So the cost asymmetry built into the convention does not change the outcome.",
   "The SEL damage from the combo is robust. Combo minus main is −0.336pp/month, t −2.77, NW t −3.23. Year-block bootstrap 95% CI is [−0.52, −0.16], P(mean ≤ 0) = 1.0, and 9 of 13 years are negative. Replacing 30% of the main strategy with the leg lowered returns in the selection period; this is not noise.",
   "The claim 'loses to TSMC30' is weaker than written. Paired combo minus TSMC30 is SEL −0.245pp (t −1.33), VAL +0.234 (t 0.69), ALL −0.065 (t −0.38): none significant. TSMC30 minus main is also only −0.09pp (t −0.54). TSMC30's higher t against 0050 comes from lower tracking error, not higher return. Condition (3) compares two t-ratios, not a paired test of means, so FINDINGS should not describe it as the combo returning less than TSMC30. In the drift-zeroed counterfactual the combo's VAL t is 1.54 vs TSMC30's 1.43. Overall (3) still fails because of SEL, so the verdict is unchanged.",
   "The 'cap-weight-led years' subsample cannot discriminate in SEL. 0050 beat the pool equal-weight in 11 of 13 SEL years (129 of 153 months). So combo minus main −0.41 (t −3.13) is essentially the full-sample −0.34 and is not evidence against mechanism 2. In those same SEL years the leg vs pool was +0.33 (t 0.72), higher than the other years' −0.12. FINDINGS sections 1 and 7 say 'the leg does worse in cap-weight-led years' and cite this combo-minus-main number for SEL, which overstates it; it should be corrected. VAL has only 3 such years (35 months, L vs pool −0.15, t −0.16), which is pure noise.",
   "VAL's positive excess depends on one year and a few stocks. 2021 (shipping) is 61% of the VAL leg-vs-pool sum. Ex-2021, leg vs pool is +0.48pp (t 0.62, NW 0.88) and combo minus main is +0.095 (t 0.42). The year-block bootstrap for VAL (p 0.005) is not credible with only 8 blocks and one dominant block; I do not count it in favour of the leg. HOLD has only 32 months: combo minus main t 0.05, so that period cannot support any conclusion either way.",
   "Test (4) is overly harsh by design. The total contribution is 136.7pp and the top 5 stocks account for 143pp (more than 100%), so dropping the 5 best contributors would kill almost any strategy. Its failure should not be the main evidence. The leg is ruled out by P1, by tracking error not falling (VAL_A 27.23 vs 27.29, HOLD 29.11 vs 28.83), and by the SEL paired difference.",
   "Multiple testing: across the 12 versions, the best leg-vs-pool SEL t is 1.18 (V11) and the best combo t against 0050 is 1.84 (V08). None reaches 1.5 or beats the main strategy's 2.34. The conclusion is that every version fails, so the large number of versions does not affect it. The one version with the best VAL result after the fact, V03, cannot be used: that would be selecting after seeing the out-of-sample data.",
   "Look-ahead and survivorship: I found no look-ahead in the pool definition. It uses 60-day average traded value, the 52-week ratio and m−1 to m−3 revenue, all public at the signal date. Requiring an open-price trade the next day is the same rule the main strategy uses. The TW50 reconstruction uses effective dates and returns the pre-change constituents on the effective day itself, so no look-ahead there either. The V2 panel holds only survivors for 2019 to 2023 and has no disposal list before 2021-09. The V3 rerun points the same way (combo ALL t 2.05 vs main 2.39), so this does not affect the verdict. Neither the 2019+ VAL nor HOLD is a blind test, so even if everything had passed, the ceiling would have been 待觀察 (watch list).",
   "Process and reproducibility: lcore.py's modification time (21:35:15) is later than explore.json (21:34:06), so exploration may have run on an earlier version of lcore. Re-running with the current lcore, I reproduce SEL leg vs pool 0.259 / t 0.622, identical to the explore output, so there is no material effect. The explore_sha256 in PREREG matches, and lcore was not changed after the freeze. The overwritten score key in PREREG and the robustness results that were never saved are both disclosed and do not affect the spec."
  ],
  "note": "I could not overturn the verdict: 不可用 (not usable) stands. The leg has no stock-picking power and only drags down returns in the selection period. VAL's positive excess comes from 2021 and five stocks, and the leg does not reduce tracking error. My verification script is /home/user/-/research/tw_stock_selection/agents/o_largecap_leg/review_stat_o3.py (python3 -I -B; it writes nothing, results print to the screen).\n\n- **Selection power:** In SEL, leg vs pool is t 0.62 (NW 0.63) when the pool pays no costs. With the pool also paying normal costs (net vs net) it is t 0.83, and gross it is t 1.43. All are below the 1.5 threshold.\n- **Selection-period drag:** Combo minus main in SEL is −0.336pp, NW t −3.23. The year-block 95% interval is −0.52 to −0.16, and 9 of 13 years are negative.\n- **VAL:** 2021 is 61% of the leg-vs-pool sum, and the leg does not reduce tracking error. Ex-2021, leg vs pool is +0.48pp (t 0.62).\n\nThree points in FINDINGS should be corrected:\n1. **Loses to TSMC30:** This only means a lower t against 0050. The paired difference between combo and TSMC30 is not significant: SEL t −1.33, ALL t −0.38. TSMC30's advantage is lower tracking error, not higher return.\n2. **Worse in cap-weight-led years:** This has no discriminating power in SEL, because those years cover 11 of 13 years. In them the leg actually beat the pool, +0.33. The −0.41 (t −3.13) cited is just the full-sample effect.\n3. **Test (4), dropping the top 5 contributors:** This test would sink almost any strategy, so it should not be the main evidence.\n\nNone of these changes the verdict. P1 fails, tracking error does not fall, and the SEL paired difference is significantly negative, which together make it 不可用. The mechanism is not supported. I kept the SOP and live_tracking unchanged and did not modify any of the researcher's files."
 },
 {
  "final": "不可用",
  "issues": [
   "Disposition stocks drive the positive VAL excess, and the V2 run does not filter them out. The V2 disposition list only starts in 2021-09. Checked against the V3 approximate full-period list: Evergreen (2603) on the 2021-06-11 pick was already in disposition at signal or execution. Wan Hai (2615) and Yang Ming (2609) on the 2021-05/06 picks, plus Gold Circuit (2368) in 2020-03, went into disposition while held. Wan Hai and Evergreen are #1 and #3 of the top 5 VAL contributors. In 2021, buying or selling them meant prepaid cash/shares and batch matching, not the opening price. So part of the +1.07pp VAL excess and of the (4) top-5 concentration cannot be traded as backtested. On V3 data it falls to +0.89 (t 1.12). This only makes the leg worse.",
   "Turnover cost is roughly the whole signal. The L leg turns over 0.531 one-way per month. Normal round trip is 0.671%, a 0.356pp/month drag; under stress3 (1.271%) the drag is 0.675pp/month. Under stress3 the L leg vs the pool equal-weight is -0.04pp (t -0.10) in SEL, -0.003pp in HOLD, and only +0.71pp (t 0.86) in VAL. Under the old stress costs SEL is +0.09 (t 0.22). Even adding all costs back to gross, SEL is only about 0.615pp, t about 1.48 (estimated), still below 1.5. Charging the pool equal-weight its own rebalancing cost to make it fair does not save P1 either.",
   "Small accounts cannot reproduce the SEL period as executed. Intraday odd-lot trading did not exist before 2020-10-26. Share of L-leg slots affordable as one whole lot: 2.4% at NT$300k, 33.6% at NT$1M, 94.7% at NT$10M. Accounts under NT$1M could only use after-hours odd lots at the close. I measured switching the trade from the rebalance-day open to the close: SEL +0.04pp, VAL_A -0.12pp, HOLD +0.34pp (all t < 1.2). It does not change the direction, but SEL is effectively a backtest for a NT$10M account only.",
   "Limit-down at open blocking a sale, and limit-up at open forcing a substitute (which needs someone watching at 9:00 to place a second order), are almost negligible for the L leg. Limit-down sells: 1 of 639 (2008-11, 3231). Months needing a substitute: 1 of 153 in SEL, 0 in VAL_A, 1 of 32 in HOLD. The main strategy also had only 11 limit-down sells. This does not affect the conclusion.",
   "Capacity is not a constraint. At NT$10M each name gets NT$600k, a median of 2.4% (max 5.3%) of 60-day average traded value × 5% (a rough proxy for opening-auction value). At NT$30M the max is 15.9%. Capacity therefore cannot be an argument for the leg either.",
   "The researcher says the leg needs about 5.7 trades per month. 0.531 one-way turnover × 5 names gives about 5.3 buys and sells, plus weight rebalancing, which is consistent. The cost model (0.0855% fee + 0.3% tax + 0.1% slippage, ETF tax 0.1% on the 0050 sleeve) matches the engine (ia_core.rates, lcore.sleeve_mix)."
  ],
  "note": "I confirm the researcher's verdict of 不可用 (not usable) and 不支持 (mechanism not supported). From the cost and tradability angle nothing argues for the L leg; the checks only make it worse.\n\n1. The V2 run counts unfiltered disposition stocks. Part of the VAL positive excess comes from 2021 shipping stocks that were in disposition at execution or went into it while held. In 2021 those had to be traded with prepaid cash/shares and batch matching, not at the opening price.\n2. After costs there is no edge. Under stress3 the L leg vs the pool equal-weight is about zero in both SEL and HOLD. Even with all costs added back, SEL t is only about 1.48 (estimated), below the 1.5 threshold.\n3. For accounts under NT$1M, the SEL backtest cannot be executed as written, because intraday odd-lot trading only started in 2020-10. Moving the trade to the after-hours close changes little, though.\n4. Limit-up/limit-down blocking and capacity are not material.\n\nNone of these changes the side conclusions: keep TSMC30_R as 待觀察 (on watch) and allow 0050 30% to cut tracking error. The researcher's files were not modified.\n\nVerification script, run with `python3 -I -B`, output to stdout only: /home/user/-/research/tw_stock_selection/agents/o_largecap_leg/review_exec_o3.py\n\nInputs read: /home/user/-/research/tw_stock_selection/agents/o_largecap_leg/results.json and /home/user/-/research/tw_stock_selection/agents/o_infra_v3_rebuild/disposal_hist.csv"
 }
]

## 獨立複製

{
 "reproduced": true,
 "my_numbers": "我沒有讀研究員的程式，自己從頭寫了一套重做。程式放在 /home/user/-/research/tw_stock_selection/agents/o_largecap_leg/replicate_o3/：\n- rlib.py：資料、gapfix 滾動窗、跳動點漲停、可成交引擎、統計\n- build.py、stats.py：一般資料重跑\n- v3data.py、v3run.py：V3 資料重跑\n- t13～t18.py：口徑核對\n- 輸出：stats_rep.json、navs_rep2.parquet、v3_rep_c2.json\n\n全部用 python3 -I -B 執行。\n\n【引擎驗證】\n- 我的選股、成交、成本引擎重建的主策略淨值，和 integrate_o2 的 MAIN_R 逐日相同，最大相對差 2.4e-10。\n- TSMC30 和 TSMC30_R 相同，差 8.6e-10。\n- 我自己的分數與股池重排主策略名單，245 個訊號日全部和 plans_R_DEL 相同。\n\n【一般成本：年化%／對 0050 t／追蹤誤差】\n- 0050：SEL 6.45、VAL 30.40、VAL_A 16.57、HOLD 59.23、ALL 14.94。\n- 主策略：\n  - SEL 17.63／2.343／17.09；VAL 46.06／1.400／27.72；VAL_A 38.72／1.471／27.29；HOLD 60.09／0.376／28.83；ALL 27.66／2.560／21.65。\n  - 最大回撤 −64.97%。\n- COMBO：\n  - SEL 13.19／1.554／16.50；VAL 47.89／1.557／27.80；VAL_A 41.14／1.670／27.23；HOLD 60.72／0.383／29.11；ALL 25.23／2.183／21.43。\n  - 最大回撤 −66.22%。\n- TSMC30：\n  - SEL 17.32／3.091／11.69；VAL 46.09／1.912／18.57；HOLD 64.98／0.513／19.63；ALL 27.47／3.441／14.62。\n  - 最大回撤 −58.19%。\n- 主策略 70%＋0050 30%：\n  - 換股訊號日收盤再平衡、不計成本：SEL 14.50／2.343／11.96；VAL 42.14／1.400；HOLD 61.04／0.376；ALL 24.25／2.560／15.15。\n  - 再平衡金額計來回成本：SEL 14.43／2.324；VAL 42.01／1.387；HOLD 60.87／0.367；ALL 24.16／2.538。\n- 安慰劑（主策略 70%＋池等權 30%）：SEL 12.05／1.574／13.41；VAL 43.55／1.423／22.06；ALL 23.05／2.094。\n- 主策略 70%＋成交值第 1 名 30%：\n  - SEL 11.36／1.399；VAL 44.45／1.519；HOLD 64.98／0.513；ALL 22.87／2.066。\n  - VAL 期第 1 名是 2330 的比例 88.0%。\n- L 腿單獨：SEL 2.21／−0.41（最大回撤 −71.38%）；VAL 48.89／1.518；HOLD 59.21／0.321；ALL 17.83／0.946。\n\n【月均超額（pp／月）】\n- 對 0050：\n  - 主策略 SEL 0.934、VAL 1.168、HOLD 0.553、ALL 1.022。\n  - COMBO SEL 0.598、VAL 1.303、HOLD 0.568、ALL 0.863。\n- COMBO 對主策略配對差：SEL −0.336（t −2.77）；VAL +0.135（0.61）；HOLD +0.016；ALL −0.159（−1.40）。\n- t 拆解（Shapley）：\n  - COMBO：SEL Δt −0.789，其中均值 −0.857、波動 +0.068；VAL Δt +0.158，其中均值 +0.162、波動 −0.004。\n  - TSMC30：SEL Δt +0.749，其中均值 −0.280、波動 +1.029；VAL Δt +0.513，其中均值 −0.148、波動 +0.660。\n\n【L 腿對同池等權】\n- 口徑：L 腿扣一般成本，對照組是池等權零成本，詳見下方差異說明第 1 點。\n- 結果：SEL +0.259（t 0.62）；VAL +1.070（1.29）；VAL_A +1.449（1.43）；HOLD +0.359（0.25）；ALL +0.563（1.39）。\n- 壓力成本 stress3：SEL −0.043；VAL +0.707；HOLD −0.003。\n- 池剔除 2330：SEL +0.367（0.89）；VAL +0.929（1.12）；HOLD +0.154。\n- 池等權對 0050：SEL −0.459（t −1.91）。\n\n【相關與 β】\n- L 腿超額和主策略超額的相關：SEL 0.603、VAL 0.712、HOLD 0.772。\n- 對（2330−0050）的 β：L 腿 SEL −0.23、VAL −0.43、HOLD −0.74；主策略 −0.30、−0.88、−0.81。\n- COMBO 和主策略超額的相關：SEL 0.953、VAL 0.965、HOLD 0.973。\n\n【其他檢定】\n- 2330 漂移：\n  - 漂移定義為 2330 減 0050 的日報酬算術平均：SEL 3.48bp／日、VAL 4.20bp／日。\n  - 歸零後對 0050 t：COMBO SEL 1.53、VAL 1.54；TSMC30 SEL 2.28、VAL 1.43。\n- 剔除 VAL 貢獻最大 5 檔：\n  - 貢獻用（持有期報酬−池等權）÷5 加總：萬海 37.1、創意 32.7、長榮 30.1、川湖 25.2、金像電 17.9。\n  - 毛超額 1.486 → −0.067pp。\n  - 成本拖累 0.407～0.425，扣掉後 −0.47～−0.49pp。\n- 事件數與周轉：\n  - 訊號窗 SEL 153、VAL 92、VAL_A 60、HOLD 32。\n  - L 腿 SEL 月替換 49.5%。\n  - 2330 入選 L 腿的月份占比：SEL 16.3%、VAL 13.0%。\n  - 周轉：主策略 0.618、COMBO 0.580。\n\n【壓力成本 stress3（年化%／t）】\n- COMBO：SEL 8.74／0.68、VAL 41.44／1.10、HOLD 54.07／0.12、ALL 20.10／1.28。\n- 主策略 ALL 22.10／1.61；TSMC30 ALL 23.50／2.44。\n\n【V3 資料重跑】\n- 我按 o_infra_v3_rebuild 的 PREREG 文字自己實作 P、X、D_hi。\n- 主策略淨值和 V3 正式欄 P1|R1|X1|Dhi|EXFIN|n10|normal 逐日相同，差 2.7e-10。\n- 主策略：SEL 16.46／2.129、VAL 43.90／1.313、HOLD 60.09／0.376、ALL 26.15／2.393。\n- COMBO：SEL 12.52／1.425、VAL 45.65／1.474、HOLD 60.72／0.383、ALL 24.05／2.047。\n- TSMC30：ALL 26.37／3.312。\n- L 腿對池等權：SEL +0.286（t 0.69）、VAL +0.885（1.12）。\n\n【判定】\n- 一般資料：\n  - (1) SEL t 0.62，未達 1.5，不過。\n  - (2) 配對差、HOLD 兩項都過；但追蹤誤差和主策略幾乎一樣（VAL_A 27.23 對 27.29、HOLD 29.11 對 28.83），不過。\n  - (3) SEL t 1.554，低於 TSMC30 的 3.091，不過。\n  - (4) 剔除 5 檔後扣成本為負，不過。\n- 結論和研究員相同：L 腿不可用，維持 TSMC30_R。\n- V3 資料下 (1)(2)(3) 也全部不過。",
 "diff_explained": "研究員報的關鍵數字幾乎全部重現到小數末位。剩下的差異都很小，也都找得到原因；判定和結論都不變。\n\n1. 規格寫法和實作口徑不一致，請研究員補註：\n   - 「L 腿對同池等權（扣一般成本）」實際算法是：L 腿扣一般成本，對照組池等權是零成本。\n   - 池等權的實作還有兩點：\n     - 不做開盤漲停遞補，50 檔全買。\n     - 永久停止交易的部位一直凍結、不出清。例如 V2 面板中 2448 晶電從 2019-05 起就沒有資料。\n   - 我起初讓池等權也扣成本、也擋漲停、也把下市部位出清，得到 SEL +0.343（t 0.82）、VAL +1.153。\n   - 改成上述口徑後完全吻合，安慰劑組合也逐位相同。\n   - 對照組零成本會讓 L 腿看起來比較差，所以這個口徑對 L 腿偏保守，不影響「不過」的判定。\n\n2. 主策略 70%＋0050 30%（袖子層混合）：\n   - 換股訊號日收盤再平衡、不計成本：t 2.343，和主策略本身的 t 相同。\n   - 研究員是 2.329，顯然有計再平衡成本。我把漂移金額乘上來回成本後得到 2.324；年化差 ≤0.05pp，t 差 ≤0.014。\n   - 規格沒有寫明成本怎麼算，屬於慣例差異。\n\n3. 剔除 VAL 貢獻最大 5 檔：\n   - 貢獻要算成「（持有期報酬−池等權）÷5」的加總，名單和數字才會完全吻合；我起初用絕對報酬，名單變成創意、萬海、川湖、長榮、亞翔。\n   - 毛超額 1.486 → −0.067 相同。\n   - 成本拖累：研究員 0.416；我用 L 腿零成本減一般成本得到 0.407，用持有窗毛報酬減淨值月報酬得到 0.425。\n   - 扣成本後是 −0.47～−0.49（研究員 −0.483），都小於 0，判定 (4) 一樣不過。\n\n4. 周轉：研究員報主策略 0.621、COMBO 0.583，我是 0.618、0.580。差 0.003 來自周轉的定義，不是淨值差異，因為兩邊淨值逐日相同。\n\n5. V3 的實作細節：\n   - D_hi 推估的漏網處置名額，要從「排除金融後的主策略合格池」中，依 52 週比由高到低、同值時代碼小者優先排除，主策略淨值才逐日相同。\n   - L 腿的 50 檔池只套近似處置事件，不套這些推估名額，COMBO 和 L 腿對池等權的數字才吻合。如果 L 腿池也套推估名額，COMBO SEL 會是 12.57／1.427。\n   - 建議 FINDINGS 寫明這兩點。\n\n6. 判定門檻 (1) 的文字有歧義：\n   - 「L 腿淨值對同池等權（零成本）」的「零成本」可以讀成指對照組，也可以讀成指 L 腿。研究員採前者：L 腿扣一般成本。\n   - 如果讀成 L 腿也零成本：V2 的 SEL t 是 1.43，仍不過；V3 的 SEL t 是 1.51，剛好過門檻。\n   - 但 V3 的 (2) 追蹤誤差那項和 (3) 本來就不過，所以結論不變。建議把文字改清楚。\n\n其他數字都在小數末位相同：0050 各段、主策略、COMBO、TSMC30、主策略 70%＋成交值第 1 名、L 腿單獨、配對差、Shapley 拆解、相關與 β、2330 漂移反事實、事件數、壓力成本、V3 主要數字。\n\n驗證期年化的計算區間：VAL_A 從 2019-01-11 到 2023-12-29；HOLD 從 2023-12-29 到 2026-10-12。用這組端點，0050 的 16.57% 和 59.23% 才對得上。"
}

**最終判定（審查後）：不可用**
