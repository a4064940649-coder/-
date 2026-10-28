# FINDINGS：營收面板截斷修補與主策略單一正式數字 V4（o_v4_revenue_patch_official）

資料夾：/home/user/-/research/tw_stock_selection/agents/o_v4_revenue_patch_official/
（子代理不能寫 .md，請主研究員把這份全文存成 FINDINGS.md。）

## 結論

**判定：待觀察。**
- 正式級別為**第 1 級邊緣**。
- 這份 V4 還沒有經過獨立逐日複製，依規定不得發布。

**V4 單一正式數字**（以 AB、一般成本為準）
- 跨循環年化 **27.00%**，對 0050 t **2.51**，對同池等權 t 3.77，最大回撤 −63.5%。
- 落在 2.30～2.58 之間，所以是第 1 級邊緣。
- m=10 的 Bonferroni p 是 **0.060**，沒有過 0.05。
- 一般成本、非 AB 的主欄是 27.47%、t 2.59。

**端點敏感度**
- 拿掉最好的 1 個月（2021-06-11，萬海入選，單月 +26.9pp）後 t 是 **2.316**，沒有跌過一級，所以依規則不加註「端點敏感」。
- 但只差 0.016。而且如果把唯一一筆「首次見報晚於訊號日」的修補格拿掉（嚴格 point-in-time），t 會變成 2.489，再拿掉最好 1 個月就是 **2.293**，會跌到第 2 級。
- **建議總監對外寫成「第 1 級邊緣（端點與 point-in-time 邊緣敏感）」。**

**保守口徑（AB＋B10）**
- 24.65%，t 2.13。
- 這是假設補不到的缺營收股全部是營收最高的最壞情況。
- SOP 預期年化依規則仍用 V4（第 1 級邊緣不觸發降級），保守口徑放在附註。

**營收截斷修補本身幾乎補不到**
- 可用來源只有 GitHub 上的鉅亨新聞鏡像。web.archive.org、MOPS、goodinfo 全部連不到。
- 通過 99% 一致率閘門的新聞營收只補了 **45 個年增率格**。
- 池內缺營收的下市股票月共 **1,194 個**，只補到 **6 個**。主策略因此只多選進 **1 個股月**：2014-05 的華亞科。
- 補不到的部分只能用 B05／B10 界限，加上新做的季營收代理估計 B_Q 來估計。

**偏誤的量級**
- 合理估計是 **−0.2～−0.4pp／年**：B_Q −0.19pp、B05 −0.37pp。最壞是 −2.6pp（B10）。
- 方向是「截斷讓回測略為高估」。
- 兩種相反的機制都看得到，但都不顯著。所以機制判定是**部分支持**。

**V4 比 o3 官方名單版（26.95%）高 0.52pp，原因不是併購溢價。** 拆解如下：
- 依規格明文排除下市金融股（2008-04 原本選到 6012 金鼎證）：+0.28pp。
- 處置改成交日口徑：−0.03pp。
- R_FIX 單一筆華亞科：+0.27pp。
- 逐月配對差 +0.036pp／月，t 0.74，不顯著。

**2024～2026 仍沒有贏 0050**
- V4 是 56.8%，0050 是 59.2%，t 0.23。
- 成交日處置口徑讓這段比訊號日口徑少 3.3pp（60.1% → 56.8%）。

## 1. 研究問題
1. R_DEL 的營收面板在下市股上大量截斷（例如矽品、日月光缺 64 個月以上）。補上之後，主策略的跨循環優勢還剩多少？
2. 補不到的部分，偏誤的方向與界限是什麼？
3. 把 o3 留下的口徑全部整合成一個 V4 正式數字，包括：
   - 官方處置名單；
   - 處置日定義（rem=0，以及公告日＝訊號日）；
   - P 和 X；
   - AB 實盤基準、成本損益兩平點、端點敏感度、多重檢定。

## 2. 機制（誰在賺、對手是誰）

**誰在賺**
- 主策略的營收腿，是「本國投信月營收追蹤」的公開版。

**本題的對手是回測自己的前視存活者偏誤**
- 缺營收的股票不會進合格池。
- 下市股在下市前 1～5 年營收就從面板消失。
- 結果等於事先知道誰會被併購或下市，然後把它們排除掉。

**修補後的方向不確定，有兩條相反的通道**
- (a) 併購溢價右尾：被併購的標的股價貼近 52 週高點，如果營收不差，就可能被選進，而且之後有收購溢價。這條通道讓偏誤對主策略不利，也就是截斷反而壓低了回測。
- (b) 地雷：下市前營收衰退或經營惡化的股票被放回，可能被選進。這條通道讓偏誤對主策略有利。
- (c) o3 的 B05 沒有選進任何缺營收股，年化卻少了 0.75pp。這表示還有第三條通道：股票被放回後，擾動了其他股票的池內百分位排名。

## 3. 資料

**資料凍結（步驟 0，總監統一資料）**
- data_freeze_v4.json 固定了以下檔案的 sha256：
  - 官方處置名單（2c171924…）、P（bc4526b4…）、X（262e9256…）、delist_official（41f51cb7…）；
  - R_DEL（3e4b72e4…）、tw_revenue、rev_5y、disposal_5y、taiwanan 月營收（6eacc673…）；
  - 日線面板 8 檔、0050、stock_list、AB 用的 p_delay 和 firstseen；
  - 本題的 R_FIX（759a6831…）、Y5_FIX（216941c6…）、季代理（8cd666b4…）、新聞抽取結果。
- o4 其他三題請以 v3_base 部分為底。

**連不到的來源**（都記在凍結檔裡）
- web.archive.org 和 mops／mopsov：proxy 403，WebFetch 也拒絕。
- goodinfo、twse：DNS 失敗。
- gh api search：這個 session 只能用 repo-scoped endpoint，所以改用 GitHub MCP 的程式碼搜尋。
- 搜尋結果：t21sc03 的封存只有 ChiJiun（只有上櫃，而且是 2026 年重新產生的頁面，沒有下市公司）。2013～2018 年含下市公司的上市月營收表，在 GitHub 上找不到。

**新來源 N：鉅亨新聞鏡像**
- 來源是 aRthur08701244/Intern-in-CTBC @258f42aa，已 clone 到 raw/ctbc，視為不可信、只當 CSV 讀。
- 用正規式抽取「X 月（合併）營收 Y 億／萬元」，連同同一句裡的年增率和月增率。共 13,426 則提及、10,095 個股票月。

**新來源 Q：taiwanan 的季損益表營收**
- 取 git 全歷史（raw/taiwanan），共 903 檔，2013Q1～2018Q3。
- 季營收和月營收三個月合計的一致率（差 < 0.5%）只有 **84.6%**，沒過 99% 閘門。所以不併入 R_FIX，只拿來做 B_Q 代理估計。

**單位與 IFRS 斷點**
- 全部換算成千元。新聞只採 2013 年以後（IFRS 合併口徑）的資料。
- 2013 年各月的年增率，用新聞的 2013 年水準除以面板的 2012 年個體營收。這和面板對存活股的作法相同（同樣是合併對個體），目的是讓修補股和存活股在同一個橫斷面上可以比較。
- 另做了敏感度 NO2013，結果和 R_DEL 版相同。

## 4. 探索發現（不跑淨值；報酬類統計只用 2006～2018）

**(1) 缺營收月數表**（gap_delisted_before_after.csv）
- 2006 年起終止上市、在面板內的 134 檔股票，全部至少缺 1 個月營收：
  - 13 檔完全沒有營收；
  - 133 檔缺 ≥ 3 個月；
  - 71 檔曾經「在池內但缺營收」。
- 依池內股票月加權，缺口最大的是：
  - 2325 矽品、2311 日月光：各 59 個池內換股日（缺營收水準 66 個月）；
  - 6286 立錡 47、3474 華亞科 46、2448 晶電 43、3598 奕力 38、3061 璨圓 36、1729 必翔 35、3519／3514／3561（太陽能三家）各 34；
  - 2473 思源 30。
- 修補後，前 5 優先股票的池內缺口：
  - 矽品 59 → 59、日月光 59 → 59、立錡 47 → 47、思源 30 → 30；
  - 華亞科 46 → 43。
- 新發現：2006～2012 年下市的股票，在下市前約 12 個月也普遍缺營收。所以那段期間每個換股日仍有 **5.27 檔**合格股缺營收（預測是 < 1 檔，預測錯了）。

**(2) 新來源和面板的一致率**
- 只看資料品質，不看報酬，6 個版本如下：

  | 版本 | 規則 | 一致率 | 新增可算 y3 的池內股票月 | 結果 |
  |---|---|---|---|---|
  | M1 | 只接受 ≥ 2 則共識 | 99.22% | 0 | 通過 |
  | M2 | 共識，或單則但內部一致 | 99.26% | 6 | 通過，**選用** |
  | M3／M4 | 加用新聞自報的年增率 | 新聞年增率一致率 94.7% | — | 沒過 |
  | M5 | 精度放寬到 0.5% | 98.90% | — | 沒過 |
  | M6 | 全部單則 | 98.37% | — | 沒過 |

- 新聞數字是第一次公告的值；有些和面板（更正後的最新值）的差異來自更正，例如台泥 2018-04。

**(3) 合格池增加幾檔、落在什麼分數位置**（每個換股日平均）
- R_FIX 新增：2013～2018 年 **0.069 檔**（預測 4～6 檔，完全沒做到）；2006～2012 年 0 檔。
- 仍缺營收的合格股：2013～2018 年 8.5 檔，2006～2012 年 5.27 檔，2019 年後 0.61 檔。
- 季代理可以覆蓋其中 5.35 檔（2013～2018）。季代理相對真實 y3 的 Spearman 相關是 0.80，百分位差中位數 0.073。
- 這些缺營收股如果放回，綜合分數會不會進前 10 名：
  - B05：0 個股月會進前 10。這證實 B05 只透過「排名擾動」影響結果。
  - B10：252 個股月會進前 10。
  - B_Q：14 個股月會進前 10（日月光 5、必翔 5、永大 2、全智科 2）。
- 矽品在 2017-11～2018-04（日月光公開收購、合併前）的 52 週比百分位是 0.96～1.0，但季代理營收百分位只有 0.23～0.31，B_Q 名次 72～85。也就是「貼近高點，但營收不強」。

**(4) 機制：缺營收下市股的下月報酬**
- 定義：成交日開盤 → 下一成交日開盤，減同池等權，按訊號月分群；樣本是挑選期 1,039 筆、135 個月。
- 結果：

  | 子樣本 | 每月超額 | t |
  |---|---|---|
  | 全部 | −0.38pp | −0.64 |
  | 52 週比在前 1/3 | +0.43pp | 0.75 |
  | 52 週比在後 2/3 | −0.93pp | −1.34 |
  | 現金併購標的 | +0.64pp | 0.90 |
  | 換股或轉換 | +0.46pp | 0.73 |
  | 經營惡化 | −4.57pp | −1.61 |
  | B10 下會進前 10 | +0.51pp | 0.77 |

- 個股：矽品 +1.23pp（t 1.53）、日月光 +1.03pp（t 1.38）。
- 解讀：兩條通道都看得到。併購標的偏正，經營惡化股偏負。但都不顯著。

**(5) 處置日口徑**
- 成交日口徑和訊號日口徑結果不同的名額共 155 個：
  - rem=0（處置在訊號日當天結束）81 個；
  - 公告日＝訊號日、隔天才開始處置的 74 個。
- 其中 10 個名額的分數在前 10 名，大多在 2021 年以後，例如 2021-06-11 的萬海。

## 5. 模型迭代紀錄

**策略規格完全不動。**

**資料合併規則共 6 版**（見第 4 節第 2 點）
- 選版規則是事前寫定的：
  - 先過閘門：水準一致率 ≥ 99%；用到新聞年增率的版本，年增率一致率也要 ≥ 99%。
  - 通過者中，取池內缺口新增覆蓋最多者；同分取新增格子較少者。
- 結果選 **M2**。
- 「更正公告取最新或取首次」的處理：新聞是首次公告值，只用來填面板的缺格；面板既有的格子（更正後的最新值）一律保留，程式內有斷言。

**冒煙測試（PREREG 之前，只重現已發布的數字）**
- 重現 o3 的 R_DEL|E1：26.9459%、t 2.5486，逐位相同。

**時間序（UTC）**
- R_FIX 23:0x → data_freeze 23:12:09 → PREREG 23:14:50 → 第一次跑報酬。

**多重檢定**
- 本題的資料規則 6 版、策略 1 版。
- 主策略歷來已改版 ≥ 6 輪，取 m=10，Bonferroni 門檻 t 2.58。

## 6. 凍結規格
- 見 frozen_spec 欄位。
- PREREG 的 sha256 是 3797d0fd…，存在 results.json。run_v4.py 會先斷言 PREREG 與 data_freeze_v4 都存在而且沒有被改過，才開始計算。

## 7. 樣本外結果（PREREG 後一次跑完，規格不改）

**本題的「樣本外」是什麼**
- 本題不產生新訊號，各期間都已經被 main_v2 和 o1～o3 看過，沒有盲測期。
- 所以「樣本外」指的是：同一份凍結規格跑全期，再分 SEL／VAL／HOLD 報告。

**主要欄位**（年化／對 0050 t；壓力成本放括號）

| 欄 | 一般成本 | 壓力成本 |
|---|---|---|
| V4 | **27.47／2.59**（S1 19.4／2.17、S2 23.7／1.23、S3 41.5／1.11、S4 56.8／0.23） | 24.40／2.05 |
| AB（判定主欄） | **27.00／2.51** | 23.93／1.97 |
| AB worst | 24.98／2.14 | — |
| B05 | 27.10／2.52 | — |
| B10 | 24.88／2.16 | — |
| AB＋B05 | 26.81／2.48 | — |
| AB＋B10（保守） | **24.65／2.13** | 21.64／1.56 |
| B_Q | 27.28／2.55 | — |
| AB＋B_Q（最佳估計） | 26.98／2.50 | — |

- 0050 為 14.94%，同池等權 11.19%，V4 最大回撤 −63.1%。
- SEL 18.9／2.51、VAL 35.5／1.45、HOLD 56.8／0.23（0050 分別是 6.5、16.4、59.2）。

**統計檢定**
- NW t（lag 3／6／12）：V4 2.61／2.72／2.74；AB 2.52／2.64／2.66。
- 12 個月區塊 bootstrap 單尾 p：V4 0.0039，AB 0.0052。
- m=10 Bonferroni p：V4 0.048，AB 0.060。

**端點敏感度**（V4／AB）

| 項目 | V4 | AB |
|---|---|---|
| 拿掉最後 1 個月 | 2.46 | 2.38 |
| 拿掉最好 1 個月 | 2.40 | 2.32 |
| 拿掉最好 3 個月 | 2.07 | 1.98 |
| 拿掉最好 5 個月 | 1.76 | 1.67 |
| 拿掉最好年份（2021） | 2.17 | 2.11 |
| 日曆月 t | 2.50 | 2.43 |

**成本損益兩平**（單邊滑價總值）

| 欄 | t 跌到 2.3 | t 跌到 2.0 |
|---|---|---|
| V4 | 0.188% | 0.279% |
| AB | 0.164% | 0.254% |

- 手續費全額時：26.40%、t 2.40。

**同分序區間**（隨機同分 20 次）
- 年化 26.93～27.70%，t 2.50～2.63。

**和對照組比較**
- 對 main_v2 的逐月配對差：−0.062pp（t −0.83）。
- 對 o3 官方名單版：+0.036pp（t 0.74）。
- 2024～2026：V4 56.8%，0050 59.2%，同池等權 25.5%。

## 8. 穩健性

**分年**
- 21 年中輸 0050 的有 5 年：2008、2016、2019、2022、2024。
- 2024 年是 12.3%，0050 是 48.6%。
- 2021 年單年 104.2%，是最大的貢獻年。

**分產業**（貢獻占比）
- 半導體 16.4%、其他電子 12.2%、電子零組件 12.0%、航運 8.5%、通信網路 7.4%。沒有單一產業過半。

**去掉最大貢獻股**
- 全期最大貢獻是 3443 創意（22.2pp），去掉後毛年化少 0.88pp。
- HOLD 期間最大貢獻是 2059 川湖，去掉後少 8.5pp。這段的集中度很高。

**處置口徑**
- 成交日口徑和訊號日口徑：全期差 −0.03pp（t −0.05），影響 11 個換股月。
- 但影響集中在 S4：60.1% → 56.8%。
- 而且成交日口徑會把 rem=0 的萬海（2021-06）選進來。這一個月就是全期最好的月份。

**IFRS 敏感度**
- NO2013 等於 R_DEL 版：27.20%、t 2.55。原因是唯一的修補入選股，用到的是 2013 年新聞水準當基期。

**point-in-time**
- R_FIX 的 50 個新聞水準格裡，有 3 個首次見報晚於使用它的訊號日。
- 其中 3474 的 2014-04 營收首次見報是 2014-05-20，晚於 2014-05-12 的訊號日。這一格正是唯一修補入選股月（華亞科 2014-05，當月對其他入選股 +33.9pp）的來源。
- 嚴格版（拿掉這一筆）的 AB：26.86%、t 2.489，拿掉最好 1 個月後 2.293。

**可證偽檢定（預測 5）**
- R_FIX 的修補股只有 1 個入選月，無法檢定。
- 用界限情境當代理：
  - B10 選進的缺營收股，對同月其他入選股是 −1.68pp／月，t −2.18，117 個月。
  - B_Q 是 −2.33pp，t −1.97，14 個月。
- 也就是說，被迫放回的缺營收股，表現比主策略自己選的股票差。
- 「選股力不受偏誤影響」在 B10 下被反證（|t| > 2）。但 B10 是人造的最壞情境；在最佳估計 B_Q 下只是邊緣（t −1.97）。

**預測核對**

| 預測 | 內容 | 結果 | 判斷 |
|---|---|---|---|
| P1 | 2013～2018 每換股日合格池增加 4～6 檔；2006～2012 < 1 檔 | 0.07 檔／日；2006～2012 仍缺 5.27 檔 | 不成立，資料補不到 |
| P2 | 選進的修補股股月 ≤ 10，集中在 S2；S1、S3、S4 不變 | 1 個股月，在 S2；S1、S3、S4 變動 0 | 成立 |
| P3 | 年化落在 24.27～26.95%，t 2.3～2.5 | 27.47%、t 2.59 | 不成立，超出上緣；但原因是 fin2 和單一修補股，不是併購溢價通道 |
| P4 | 處置口徑 ≤ 0.2pp；AB 少 0.3～0.5pp；損益兩平滑價 0.11～0.14% | 全期 −0.03pp（但 S4 −3.3pp）；AB −0.47pp；損益兩平 0.19%／AB 0.16% | 前兩項成立，損益兩平不成立 |

## 9. 機制是否被支持：部分支持

**支持的部分**
- 營收截斷確實是前視濾網：134 檔下市股全部有缺口，池內有 1,194 個股票月缺營收。
- 被併購的標的確實貼近 52 週高點：矽品下市前 6 期的 52 週比百分位是 0.96～1.0。
- 兩條相反的通道都看得到：
  - 現金併購標的 +0.64pp（t 0.90）；
  - 經營惡化股 −4.57pp（t −1.61）。
- 「排名擾動」通道成立：B05 沒有選進任何缺營收股，但 −0.37pp／年。

**不支持或無法判斷的部分**
- 量級和方向的最佳估計（B_Q −0.19pp）在統計上和 0 無法區分。
- R_FIX 本身補不到可以量的樣本。
- V4 高於 V3，並不是併購溢價通道造成的。

## 10. 散戶怎麼執行

**選股規則不改，只改文字與數字。**

**SOP 第 2 步**
- 「排除處置股」寫成成交日口徑：換股日隔天（成交日）還在處置期間的股票，排除；已經公告、隔天才開始處置的，也排除；處置在換股日當天結束的（例如萬海那類），照常買。
- 「排除金融」明文加上「已下市的證券、銀行」。這對實盤沒有影響。

**SOP 第 10 節正式數字改成 V4**
- AB 年化 27.0%，對 0050 t 2.51，第 1 級邊緣，Bonferroni p 0.06。
- 保守 24.7%（t 2.13）。
- 最大回撤約 −63%～−64%。
- 2024～2026 是 56.8%，0050 是 59.2%。

**實盤預期**（0050＋超額×0.5）
- 約 21.0%；保守約 19.8%（0050 14.9%）。

**成本**
- 滑價上限改寫為「單邊 0.16%（AB）」。手續費要打折；全額時 t 是 2.40。
- 下單仍用限價單。

## 11. 限制

1. **新來源只是新聞抽樣**（每月約 3,000 則鉅亨新聞），對中小型下市股幾乎沒有覆蓋。2006～2012 和 2019 年以後沒有任何新來源。
2. **R_FIX 的唯一入選股，資料可得時點有疑慮**（首次見報晚於訊號日）。凍結規格沒有 point-in-time 條件，這是事後發現的；嚴格版數字已在第 8 節揭露。
3. **季代理是估計，不是界限。** 它和真實 y3 的相關只有 0.80，而且季損益表營收和月營收合計本來就有約 15% 的不一致。
4. **還沒有經過獨立逐日複製。** 規定複製員不讀本題的 .py，只讀 data_freeze_v4.json 和 frozen_spec。
   - 要比對的檔案：navs_v4.parquet（欄 V4|normal、AB_V4|normal、V4_B05|normal、V4_B10|normal、AB_V4_B05|normal、AB_V4_B10|normal 等）、plan_V4*.csv（逐月名單）。
   - 門檻：逐日最大相對差 < 1e-9。
   - 在複製完成之前，V4 不得發布。
5. **處置成交日口徑仍沒有處理預收款券的 T+2 現金限制。** 這一點 o3 已經指出。
6. **全期都已被看過，沒有盲測期。** 前瞻盲測要從 2026-11 起算。
7. **本題曾意外修改另一個資料夾。** 在 o_index_deletion_flow/raw 的 partial clone 上執行 git ls-tree -l 時，觸發了 lazy fetch：.git 物件增加了，工作樹沒有變（git status 乾淨），我發現後已經中止。之後所有原始資料都改成在本題的 raw/ 重新 clone。

**主要檔案**（都在本題資料夾）
- 程式與設定：v4lib.py、extract_news.py、build_quarterly.py、build_rfix.py、build_qproxy.py、freeze_v4.py、explore_gap.py、explore.py、prereg.py、run_v4.py、diag_decomp.py、diag_post.py、smoke_o3.py。
- 資料與結果：data_freeze_v4.json、iterations.json、iterations_data.json、explore.json、results.json（PREREG＋RESULTS＋POST_DIAG）、navs_v4.parquet、yoy_long_R_FIX.parquet、rfix_cells.csv、gap_delisted_before_after.csv。

## 凍結規格

V4 凍結規格（PREREG sha256 3797d0fdf9606e3cb3e32cb34329dccd54a2e0b12dddc474a4015252e57cff4e，2026-10-27T23:14:50Z；data_freeze_v4.json 的 sha256 寫在 PREREG.data_freeze_v4_sha256）
資料夾：/home/user/-/research/tw_stock_selection/agents/o_v4_revenue_patch_official/

一、資料（全部列在 data_freeze_v4.json，含 sha256；複製員只讀這些檔，不讀本題 .py）
- 日線面板 scratchpad/panel/{adj,close_raw,open,high,value,ret,flag,src}.parquet，加 P＝o_infra_v3_rebuild/delist_prices.parquet（sha bc4526b4…）。只補原本缺值的格子：close/open/high/value/ret 照檔填入，flag＝0。補完後重算：TRADED＝close 有值；GROW＝有收盤時 1＋ret（ret 缺值當 0），否則 1；自算還原價＝GROW 連乘；NOBS＝每日有值檔數，GOOD＝NOBS ≥ 300；前一筆收盤＝close 前值補後 shift(1)。
- X＝o_infra_v3_rebuild/merger_consideration.csv（sha 262e9256…）加 delist_official.csv（sha 41f51cb7…）。凡是「面板最後一筆之後再也沒有成交、而且最後一筆早於面板終點 30 天」的持股：
  - 現金併購：價值乘以（現金對價 ÷ 最後收盤），在付款日（缺值時用終止日＋14 天）之後第一個換股日轉成現金。
  - 換股、轉換，或沒有現金對價的現金案：按最後收盤，在終止日之後第一個換股日出清。
  - 經營惡化：歸零。
  - 型態未知：按最後收盤，在最後交易日＋14 天後出清。
  - 代碼重用（終止上市後同一代碼由新公司使用）：舊證券視為在終止日前最後一筆下市；這條規則只在新證券第一筆成交之前有效。
- 官方處置名單＝o_disposal_rejudge/disposal_hist_rejudge.csv（sha 2c171924…，只取 in_panel 列），聯集 disposal_5y.csv.gz 中 (code, start) 不在前者的列（以 date 當公告日）。
- 營收：
  - 訊號日 < 2021-03-01：用 yoy_long_R_FIX.parquet（sha 759a6831f18954f333b0acb3882bf9a31a64210a550a8c8b9e093790d77c20c2）。它等於 R_DEL（sha 3e4b72e4…）原封不動，再加本題 45 個年增率格（rfix_cells.csv）。
  - 訊號日 ≥ 2021-03-01：用 yoy_rev5y_FIX.parquet（sha 216941c6…）。它等於 rev_5y 的 yoy 欄，本題修補 0 格。
  - y3（訊號月 m）＝m−1、m−2、m−3 三個月年增率的平均，三個都要有值，否則 y3 缺值。
- 界限與估計（只在界限欄使用）：
  - gap 股＝delist_official.csv 中 four==True 的代碼 → 終止日，排除名稱含 KY／DR、以 F- 開頭或代碼 91xx 者，也排除金融股（v2 的 FIN，或名稱含 銀、證、金控、證金、產保、壽、票券、產險、人壽、投信、合庫、信託）。
  - B_Q 用 yoy_qproxy.parquet（sha 8cd666b4…）。

二、策略規格（與 main_v2 v2 相同，只改處置口徑並把排除規則寫明）
- 換股訊號日 d：從 2005-01-01 起，每月第一個「日 ≥ 11」的面板交易日。保留條件：(m−1) ≤ 營收表的「最後完整月」（當月有值檔數 ≥ 全表中位數的 0.8），而且當月 y3 有值檔數 > 100；只取 d < 2026-10-12。2026-10-12 只當評價終點。成交日 e＝d 的下一個面板交易日。
- 滾動窗 gapfix：只用 GOOD 日計算。窗內缺值時該值作廢，但 2017-08-03、2018-07-12 與 2024-02-15～03-15 的缺值豁免。52 週比＝自算還原價 ÷ 252 個 GOOD 日內最高值；V60＝60 個 GOOD 日的平均成交值。
- 股池：
  - V60 有值、V60 ≥ 1,000 萬、收盤 ≥ 10，按 V60 由高到低取前 300（stable 排序）。
  - 52 週比有值；e 日有開盤、有成交。
  - 排除金融：v2 的 FINMASK（stock_list 產業＝金融保險業、代碼 28xx、5876、5880），再加終止上市的金融股（判定規則同上；實際影響 6012 金鼎證、5854 合庫、6004 元京證）。
  - 排除 KY／DR／F-／91xx（o_rev_survivorship 的 foreign_codes，加上 delist_official 中名稱以 F- 開頭或含 KY 者）。
  - 處置（主口徑 trade）：公告日 ≤ d，而且 start ≤ IDX[e] ≤ end，就排除。
  - y3 有值。
  - 合格股 < 50 檔的換股日跳過。
- 分數＝0.5×（52 週比的池內百分位，pandas rank pct）＋0.5×（y3 的池內百分位）。
- 排序：分數四捨五入到 1e-10 後由高到低；同分時 y3 高者優先；再同分時依面板欄位序。依序取 10 檔。e 日開盤漲停（開盤 ≥ 漲停價−1e-6）而且目前沒有持有的股票跳過、往下遞補。
  - 漲停價＝ref×(1＋lim) 再依 tick 無條件捨去。lim：2015-06-01 前 7%，之後 10%。tick：<10 元 0.01、<50 元 0.05、<100 元 0.1、<500 元 0.5、<1000 元 1、其餘 5。
  - ref＝close÷(1＋ret)；flag==1 的日子改用前一筆收盤。
- 成交與成本：
  - e 日開盤調倉成等權 10 檔。持股從前一日收盤到 e 日開盤的報酬＝GROW[e]×open/close。e 日沒有成交的持股凍結，不能賣。
  - 用二分法 60 次求每檔目標金額 T，使 10T＋成本 ≤ 可用總值。成本＝買方費率×正向調整量＋賣方費率×反向調整量。開盤到收盤乘 close/open；非換股日乘 GROW。
  - 一般成本：買＝0.001425×0.6＋0.001；賣＝同＋0.003。壓力成本：每邊 0.005。
- 同池等權（EW）：合格池等權、零成本，用同一個引擎（凍結規則相同、沒有 X）。
- 統計：
  - 年化：從 2006-04-01 起第一筆日報酬（基期取前一日）算到 2026-10-12；最大回撤用日頻。
  - t＝訊號窗報酬（訊號日收盤 → 下一訊號日收盤，最後一窗到 2026-10-12）減 0050 含息同窗報酬，窗起點落在區間內，算一般 t。
  - 分段：S1 2006-04～2013、S2 2014～2020、S3 2021-03～2026-10-12、S4／HOLD 2024～2026-10-12、SEL 2006-04～2018、VAL 2019～2023。
- AB（判定主欄）：
  - 用 v2_core.make_unavail('mc', seed)，seed 0～39。訊號日落在法定期限日（10 日之後第一個交易日）的情況：
    - D>S（訊號日早於期限日，含 2019-01、2023-03 的營收月）：全部視為未公告。
    - D=S：依 rev_announce/p_delay.json 的產業延後機率抽樣（金融股＝1）。
    - 2026-09 營收用 firstseen_2026-09.csv 的 A_before 判定；2026-01～08 的營收月，INS15 名單一律視為未公告。
  - 未公告的股票改用 m−2、m−3、m−4 三個月的 y3。
  - 40 條日報酬取平均，連乘成一條淨值；另報 worst（D=S 全部視為未公告）。
- 界限：
  - B05／B10：gap 股在 d ≤ 終止日、營收以外條件都合格、但 y3 缺值時放回股池，y3 百分位固定為 0.5／1.0。B10 的同分序視為 y3＝+∞。
  - B_Q：同一批股票改用季營收代理 y3 排名。代理值＝k＝m−1、m−2、m−3 各月「k 月（含）以前最後一個完整季」的季營收年增率的平均。沒有代理值的股票仍排除。
- 敏感度：
  - SIG：處置改用訊號日口徑（公告日 ≤ d，而且 start ≤ d ≤ end）。
  - RDEL：營收用 R_DEL 加原始 Y5。
  - NO2013：拿掉 2013 年的修補格，以及基期為 2013 年新聞水準的 2014 年修補格。
  - SPEC0：不用 fin2，也不明文排除 KY。
  - O3REP：R_DEL＋訊號日口徑＋SPEC0，必須等於 26.9459%／t 2.5486（已斷言通過）。
  - 隨機同分：rand:0～19。
- 判級：以 AB 一般成本的 t 判級。t ≥ 2.58 為第 1 級；2.30～2.58 為第 1 級邊緣；< 2.30 降為第 2 級，SOP 改用 AB＋B10。拿掉最好 1 個月後若跌過一級，加註「端點敏感」。

三、R_FIX 資料合併規則（選版只看資料品質，不看報酬；6 版中選 M2）
- 來源 N：鉅亨網新聞鏡像（github aRthur08701244/Intern-in-CTBC @258f42aa，EB_EventExtraction2＋EB_SegAndRawNews，2013-01～2021-12）。抽取「公司 X 月（合併）營收 Y 億／萬元」，單位換算成千元。
- 採用條件：
  - 有效精度 ≤ 0.25%；文章日期落後營收月 ≤ 1 個月；營收月 ≥ 2013-01（IFRS 合併口徑）；同股同月多則提及彼此差 ≤ 0.5%，取中位數。
  - 接受「≥ 2 則共識」，或「單則但同句的月增率或年增率，與已知上月或去年同月水準（面板或已接受的 N）一致，差 ≤ 1pp」，迭代到收斂。
- 年增率＝水準 ÷ 12 個月前水準（與面板同法；2013 年各月的基期是面板 2012 年個體營收，與存活股的斷點處理相同）。
- 只補 R_DEL／Y5 的缺格；只補終止上市名單內、非 KY／DR／F-、非金融的股票，而且營收月 ≤ 終止月。既有格子一律不覆寫（已斷言）。N 是首次公告值，面板是更正後的最新值；兩者並存時保留面板。
- 閘門：重疊月水準一致率（差 < 0.5%）≥ 99%。M2 為 99.26%（5,419 個重疊月）。

## 反方審查

[
 {
  "final": "待觀察",
  "issues": [
   "【未複製不得發布】依規定 V4 須先通過獨立逐日複製才能發布，目前沒有做。我只重算到研究員自己的 navs_v4.parquet（AB t 2.512、V4 t 2.589，逐位相同），這不算獨立複製，所以現在不能判「可用」。",
   "【第 1 級邊緣標籤偏寬，不利方向應寫第 2 級】(a) 嚴格 point-in-time 版（拿掉 2014-05-20 才首次見報、晚於 2014-05-12 訊號日的華亞科修補格）的 AB 拿掉最好 1 個月後 t 2.293，低於 2.30。(b) 拿掉第一個不完整年份 2006 後 t 2.216，拿掉 2007 後 2.291，都低於 2.30；研究員的端點規則只檢查最好月、最後月和最好年（2021），沒檢查起點。(c) 起點改為 2008-01 時 t 1.99。(d) 壓力成本下 AB t 1.97。不確定時往不利方向判，正式級別應為第 2 級，而不是加註後的第 1 級邊緣。",
   "【多重檢定 m=10 低估】agents/ 下有 142 個研究資料夾，主策略本身就是從第三輪大範圍訊號搜尋挑出來的，之後又改版 6 輪以上，m=10 只算了主策略改版。即使只用 m=10，Bonferroni p 已是 0.060；m=20 時 p 0.12，m=50 時 0.30，要過 0.05 需要 t 2.81～3.09。V4 規格裡的 fin2（排除 2008-04 的金鼎證，+0.28pp，t 2.549→2.598）和成交日處置口徑，都是在已看過 o3 名單與報酬後才決定的，這是規格自由度，沒有算進 m。",
   "【顯著性只來自 2006～2013】AB 分段 t：2006～2013 為 2.04；2014～2026 只有 1.67（152 個月）；2019～2026 為 1.24；HOLD（2024～2026）0.24。全期都已被 main_v2 和 o1～o3 看過，沒有盲測期。勉強算樣本外的近期段落沒有顯著超額，2024～2026 還輸 0050（56.8% 對 59.2%）。",
   "【右尾少數月份】最好 3 個月占全期超額總和 27.4%，拿掉後 t 1.98；最好 5 個月占 40.6%，拿掉後 t 1.67；最好 10 個月占 67.8%，拿掉後 t 0.96。同時拿掉 2021 和 2015 兩年，t 1.72。公平起見要說明：左右各修剪 3～10 個月時 t 反而升到 2.59～2.98，所以這不是單純靠右尾；但單側端點規則只拿掉 1 個月，確實太寬。",
   "【AB 是 40 條路徑的平均】我逐 seed 重跑：t 中位數 2.498（範圍 2.299～2.745）。拿掉最好 1 個月後，40 個 seed 中有 19 個低於 2.30；嚴格 PIT 版是 25/40。2.316 這個「不算端點敏感」的結論，大約只是擲銅板的結果。",
   "【存活者偏誤大多沒補，最佳估計不完整】R_FIX 只覆蓋 1,194 個缺營收池內股票月中的 6 個。B_Q 只涵蓋 2013～2018 年平均每日 8.5 檔缺口中的 5.35 檔，2006～2012 年平均每日 5.27 檔缺口完全沒有代理，偏偏這段是全期顯著性的主要來源（2006 一年就貢獻 31.7pp 超額）。所以「合理偏誤 −0.2～−0.4pp」不是全期估計。B10 下 S2 的 t 從 1.24 掉到 0.45，SEL 從 2.39 掉到 1.95。",
   "【地雷通道其實有證據，機制判定偏樂觀】依研究員自己的可證偽檢定，B10 硬放回的缺營收股對同月其他入選股是 −1.68pp／月（t −2.18，117 個月），B_Q 是 −2.33pp（t −1.97）；探索中的經營惡化股是 −4.57pp。這些都指向「截斷讓回測高估」，不能只用「兩條通道都不顯著」帶過。",
   "【前視：營收用的是更正後最新值】面板是更正後的最新值，新聞是首次公告值；M2 一致率 99.26%，表示約 0.7% 的重疊格與首次公告不同。修正值的前視問題影響全體股票，AB 只模擬公告延遲、不模擬更正，這部分沒有量化。另外 2013 年的年增率是合併對個體，橫斷面不一致，也沒有量化。",
   "【萬海 2021-06 的效應】全期最好的一個月（2021-06，AB +26.9pp），是因為成交日口徑把萬海換進來而變大；同月訊號日口徑是 +21.6pp。萬海也是全期第 3 大貢獻股（19.9pp）。口徑合理，但端點邊緣性有一部分是這次規格改動造成的。",
   "【最後一個窗】最後一個窗是 2026-09-11 起，超額 +14.3pp，是全期第 7 好的月份，用的是最新、尚未經過時間檢驗的資料。拿掉後 t 從 2.51 降到 2.38；拿掉整個 2026 年是 2.365。"
  ],
  "note": "我同意「待觀察」，但不同意「第 1 級邊緣（正式不算端點敏感）」。依不利方向，正式級別應寫第 2 級，SOP 預期年化應改用保守口徑 AB＋B10（24.65%／t 2.13），依據如下：\n- 嚴格 point-in-time 版拿掉最好 1 個月後 t 2.293；\n- 拿掉 2006 年後 t 2.216；\n- 起點改為 2008 年時 t 1.99；\n- 壓力成本下 AB t 1.97；\n- m 只要合理大於 10，Bonferroni 就過不了 0.05。\n\n我也同意機制判定「部分支持」，但要補一句：可證偽檢定在 B10 和 B_Q 下都顯示放回的缺營收股表現較差，這指向截斷讓回測高估。\n\n我重算的數字和研究員一致（AB t 2.512、V4 t 2.589、嚴格 point-in-time 2.489／2.293），沒有發現程式錯誤；問題在判讀與門檻。\n\n驗證程式：/home/user/-/research/tw_stock_selection/agents/o_v4_revenue_patch_official/review_stat_o4.py\n- 只讀研究員的檔案，結果只印在畫面，不寫任何檔案，研究員的檔案都沒有改。\n- 一般執行：python3 -I -B review_stat_o4.py\n- 加 --seeds 會逐 seed 重跑 AB，約需數分鐘。\n\n依規定，獨立逐日複製完成之前不得發布。"
 },
 {
  "final": "待觀察",
  "issues": [
   "Under realistic costs the grade falls from '第 1 級邊緣' to '第 2 級'. Annual one-way turnover is 7.43x (monthly turnover 0.622, about 6.2 of 10 names newly bought each month). Normal-cost drag is already about 5.0% a year, and every extra 0.05% of one-sided slippage costs another 0.74% a year. In my rerun, AB with 0.15% one-sided slippage gives 26.06%, t 2.346, and t 2.144 after dropping the best month. With 0.20% it gives 25.12%, t 2.18. Either way it is endpoint-sensitive.",
   "The '不加註端點敏感' call rests on a margin of only 0.016 in AB drop-best-1 (2.316 vs 2.30). One extra basis point of slippage per side takes that away. Back-calculated, only one-sided slippage up to about 0.105% keeps drop-best-1 at or above 2.3. The SOP's 'slippage cap of 0.16% one-sided (AB)' only covers the full-sample t reaching 2.3. At 0.16%, drop-best-1 is about 2.11, so the cap and the grade wording contradict each other. The cap should be about 0.10%, or the result should be written as level 2.",
   "Capacity: I used a participation impact model, slippage = 0.1% + 0.5·σ·sqrt(Q/ADV), with Q = capital/10 held fixed in real terms. Results: 300萬 gives AB 26.05%, t 2.345, drop-best-1 2.143. 1,000萬 gives AB 25.28%, t 2.207, drop-best-1 2.00, so even the full-sample t is below 2.3. 3,000萬 gives non-AB t 2.06. At 1,000萬, 27.6% of stock-months have a position above 1% of 60-day ADV (maximum 8.45%). The thinnest pick averaged only 11.8M a day, and the 2006–2013 median was 164M. The strategy also fills only at the opening auction, which carries far less volume than the full day. So these impact numbers are optimistic, and the upper end of the stated capacity range (1,000萬) is not supported.",
   "Small accounts (30萬, 3萬 per name): 79.6% of stock-months have a board-lot price above the position size, so they need odd lots. Before 2020-10-26 that was 75.1%, and odd lots then could only be traded after the close at the closing price, so opening-price fills were impossible. With close execution, AB gives 26.76%, t 2.465, drop-best-1 2.265, which is endpoint-sensitive. After 2020, intraday odd-lot trading has wider spreads and thinner liquidity than the 0.1% slippage assumes.",
   "The live-trading expectation in SOP section 10 (about 21.0%, conservative 19.8%) uses the zero-impact AB 27.0% as its base. Rebuilt from a realistic AB under cost (about 25–26%), it is roughly 20–20.5%, and the 2024–2026 shortfall versus 0050 widens. Under the 1,000萬 impact model, S4 is 55.3% vs 0050's 59.2%.",
   "The 13 limit-up-open blocks require watching the market at 9:00 and placing replacement orders. The backtest has the replacement bought at that day's opening price, but in practice the opening auction is already over by then. Also, the SOP's 'use limit orders' conflicts with the assumption of a full fill at the open. This strategy buys stocks near 52-week highs, and a conservative limit price systematically misses the gap-up winners.",
   "Under stress cost (0.5% per side), AB t is 1.97, below level 2. Full brokerage fee (no discount) is only reported for the non-AB V4 (t 2.40). Extrapolated, AB with full fee is about 2.32, with drop-best-1 about 2.12, which is also endpoint-sensitive.",
   "Data timing: the only patched pick (Hynix, 2014-05, +0.27pp) depends on a news figure first reported on 5/20, after the 5/12 signal day. In the strict point-in-time version, AB drop-best-1 is 2.293, which is level 2. Stacked with any cost increase this is a further downgrade. The official MOPS figure was probably already public on 5/10, but this review could not verify that, so it is judged on the unfavourable side.",
   "Points that check out: locked limit-down sells are negligible (3 sells opened limit-down, only 1 was locked all day, 1463 on 2017-07-12; deferring that sale changes t from 2.589 to 2.581). Under the trade-day disposal rule, Wanhai 2021-06-11 is executable on the official list (disposal 5/31–6/11). However, this rule change is exactly what brings in the best month of the whole sample."
  ],
  "note": "I agree with keeping the verdict at 待觀察, but '第 1 級邊緣' only holds at zero market impact with one-sided slippage of about 0.105% or less. Under any realistic retail execution cost it should be written as level 2.\n\nReproduction: V4|normal, AB_V4|normal and plan_V4.csv all match the researcher's output exactly (maximum relative difference 0). My custom vectorised-fee engine reproduces V4|normal with difference 0.\n\nMain reasons:\n(1) Turnover is 7.4x a year, so each 0.05% of slippage costs 0.74% a year. AB endpoint robustness (drop-best-1 2.316) disappears with one more basis point of slippage per side.\n(2) With the impact model, AB t is 2.345 at 300萬 and 2.207 at 1,000萬; drop-best-1 is 2.14 and 2.00. The 1,000萬 upper bound is not supported, and opening-auction execution makes this optimistic.\n(3) A 30萬 account needs odd lots for about 80% of stock-months. Before 2020-10 it could only trade at the close; with close execution, AB drop-best-1 is 2.265.\n(4) The SOP's 0.16% slippage cap contradicts the level-1-edge wording. It should be about 0.10%, or the result should be written as level 2 for capital of 300萬 or more.\n\nLocked limit-down sells and the T+2 disposal cash issue have negligible effect on the numbers.\n\nVerification script: /home/user/-/research/tw_stock_selection/agents/o_v4_revenue_patch_official/review_exec_o4.py (python3 -I -B, read-only, output to stdout only; runtime about 6 minutes)."
 }
]

## 獨立複製

{
 "reproduced": true,
 "my_numbers": "我只依凍結規格與 BRIEF 從頭寫了一套程式，放在 /home/user/-/research/tw_stock_selection/agents/o_v4_revenue_patch_official/replicate_o4/：rcore.py 是引擎，t1.py 跑主欄，ab.py 跑 AB，batch.py／rep1～rep4.py 跑各欄、配對差與事件數，全部用 python3 -I -B。我沒有讀本題研究員的 .py。只 import 了規格點名的 main_v2/v2_core.make_unavail，用來產生 AB 的抽樣。所有輸入檔的 sha256 都已核對，與規格一致。R_FIX 恰好比 R_DEL 多 45 格，沒有覆寫任何既有格子。\n\n一、主欄 V4（一般成本）\n- 年化 27.473%，t 2.5889，對同池等權 t 3.856，最大回撤 −63.1%。\n- 月均超額 0.993pp，標準差 6.01pp，共 245 個窗。\n- 0050 為 14.94%，同池等權 11.19%。X 出清 0 次，凍結 0 次。月周轉 61.9%（研究員 62.2%）。\n- 分段（年化／t，0050 年化）：\n\n  | 分段 | 年化 | t | 0050 年化 |\n  |---|---|---|---|\n  | S1 | 19.4% | 2.17 | 5.5% |\n  | S2 | 23.7% | 1.23 | 14.5% |\n  | S3 | 41.5% | 1.11 | 28.0% |\n  | S4 | 56.8% | 0.23 | 59.2% |\n  | SEL | 18.9% | 2.51 | — |\n  | VAL | 35.5% | 1.45 | — |\n\n- 統計檢定：NW t 2.61／2.72／2.74；Bonferroni p 0.048；12 個月區塊 bootstrap p 0.0032～0.0043（依抽樣方式不同）。\n- 端點敏感度：\n  - 拿掉最後 1 個月 2.46；拿掉最好 1／3／5 個月 2.40／2.07／1.76。最好的月是 2021-06-11，+26.89pp，選股含 2615 萬海。\n  - 拿掉 2021 年 2.17；只算到 2025 年底 2.47；日曆月 t 2.49～2.50。\n- 輸 0050 的年份：2008、2016、2019、2022、2024。\n- 隨機同分 20 次：年化 26.93～27.70%，t 2.495～2.632。\n\n二、判定主欄 AB（mc 40 次）\n- 年化 27.001%，t 2.512，對同池等權 t 3.77，最大回撤 −63.5%。\n- 統計檢定：NW t 2.52／2.64／2.66；Bonferroni p 0.060；bootstrap p 0.0035～0.0057。\n- 端點敏感度：拿掉最後 1 個月 2.38；拿掉最好 1／3／5 個月 2.316／1.98／1.67；拿掉 2021 年 2.11；日曆月 t 2.43；只算到 2025 年底 2.37。\n- 分段：S1 18.6%（2.04），S2 t 1.24，S3 t 1.12，S4 56.7%（0.24）。\n- 前 300 池內平均每期 26.25 檔改用舊營收。\n\n三、其他欄（一般成本／壓力成本）\n- V4：27.47%／2.589；24.40%／2.05\n- AB：27.00%／2.51；23.93%／1.97\n- AB worst：24.98%／2.14；21.94%／1.60\n- B05：27.10%／2.52；24.03%／1.98\n- B10：24.88%／2.16；21.87%／1.60\n- AB＋B05：26.81%／2.48；23.74%／1.93\n- AB＋B10：24.65%／2.125；21.64%／1.565\n- AB＋B10 worst：23.39%／1.875\n- B_Q：27.28%／2.55\n- AB＋B_Q：26.985%／2.503\n- 手續費全額：26.40%／2.40\n- 敏感度：\n  - SIG：27.50%／2.64，S4 60.1%\n  - RDEL：27.20%／2.548\n  - NO2013：27.20%／2.548\n  - SPEC0：27.15%／2.534\n  - O3REP：26.9459%／2.5486（逐位相同）\n  - AB_RDEL：26.858%／2.489，拿掉最好 1 個月 2.293\n\n四、成本損益兩平\n- V4：t 跌到 2.3 時 0.188%，跌到 2.0 時 0.279%。\n- AB：0.164%、0.254%。\n\n五、配對差（每月 pp／t）\n- V4 − main_v2：−0.062（−0.83）；SEL +0.056（1.04）；VAL −0.282（−1.27）；HOLD −0.213（−0.75）。\n- V4 − o3：+0.036（0.74），24 個月不為零。\n- R_FIX 效果：+0.020（0.98），2 個月不為零。\n- 成交日 − 訊號日處置口徑：−0.002（−0.05）。\n- AB − V4：−0.032（−1.26）；B05 − V4：−0.025（−2.15）；B10 − V4：−0.188（−2.18）；B_Q − V4：−0.012（−0.96）。\n\n六、o3 → V4 拆解：26.946／2.549 → 26.946／2.549 → 27.228／2.598 → 27.201／2.548 → 27.473／2.589。\n- 2008-04 原本選到 6012 金鼎證。\n- 成交日口徑讓 11 個月的名單不同，含 2021-06。\n- R_FIX 唯一的入選股月是 2014-05 的 3474 華亞科。\n\n七、事件數\n- 處置口徑不同的名額 155 個：81 個 rem=0、74 個公告日＝訊號日；其中 10 個名額分數在前 10 名。\n- 界限欄選進缺營收股的股月數：B10 229 個（2325 矽品 32、2311 日月光 21）；B_Q 15 個；B05 0 個。\n- 每換股日的合格池變化：\n  - R_FIX 新增：2013～2018 年 0.069 檔，2006～2012 年 0 檔。\n  - 修補後仍缺營收：2006～2012 年 5.27 檔，2013～2018 年 8.50 檔，2019 年後 0.61 檔。\n  - 季營收代理可覆蓋：2013～2018 年 5.35 檔。\n- R_FIX 共 45 格、14 檔。\n- 可證偽檢定：B10 −1.74pp（t −2.29，117 個月）；B_Q −2.50pp（t −2.29，14 個月）；R_FIX 只有 1 個月。",
 "diff_explained": "能比的關鍵數字幾乎全部重現。主欄 V4、AB、SIG、RDEL、NO2013、SPEC0、B05、B_Q、AB＋B05、AB＋B_Q、AB worst、壓力成本、手續費全額、成本損益兩平、全部配對差、o3→V4 拆解、隨機同分範圍、處置名額 155／81／74／10、界限欄股月數、池內缺口統計，都與研究員一致到報告的位數。O3REP 26.9459%／2.5486 逐位相同，所以兩邊的選股計畫應該完全相同。\n\n不一致或需要說明的地方：\n1. 年化的起點與規格文字不同。規格寫「從 2006-04-01 起第一筆日報酬，基期取前一日」，照字面做會得到 V4 27.43%、0050 14.99%。研究員的數字其實要從第一個訊號日 2006-04-11 收盤起算到 2026-10-12（日數 ÷ 365.25）才對得上：V4 27.473%、0050 14.94%、同池等權 11.19%。其他分段（S2～S4、SEL、VAL）則照規格，用分段起點前一日當基期。這是研究員實作與規格文字不一致，不影響 t。\n2. 規格沒寫清楚 X 現金對價的付款日是否含當天。我改成「付款日當天的換股日就轉現金」，B10 才會從 24.85%／2.155 變成 24.88%／2.16。這只影響 B10 系列，因為 B10 會持有下市股，有 9 次 X 出清、5 次凍結；主欄 X 為 0 次，不受影響。\n3. bootstrap p 看抽樣方式。研究員是 V4 0.0039、AB 0.0052。我用循環區塊、t 統計量得到 0.0032／0.0035；用非循環區塊、平均 ≤ 0 得到 0.0043／0.0057。兩種都落在研究員的數字附近，屬於蒙地卡羅與方法差異，結論不變。\n4. 可證偽檢定的值略大：B10 我得 −1.74pp（t −2.29），研究員 −1.68pp（t −2.18）；B_Q 我得 −2.50pp（t −2.29），研究員 −2.33pp（t −1.97）。月數 117／14 相同，方向與顯著性一致；差異來自規格沒定義個股「下月報酬」怎麼算，我用的是訊號日到下一訊號日的收盤報酬。\n5. 月周轉我得 61.9%，研究員 62.2%，差異應來自周轉定義（我用買方金額 ÷ 總值），只是描述性數字。\n6. 沒有重做的項目：新聞重疊月一致率 99.26%、季代理 Spearman 0.80 與百分位差中位數 0.073、下市股缺營收的計數（134／13／133／71、1,194／6）。這些要用原始新聞或季營收資料重建，超出這次重測的範圍；我只核對了 R_FIX 等於 R_DEL 加 45 格、14 檔。\n\n判級：AB 一般成本 t 2.512，落在 2.30～2.58，是第 1 級邊緣。拿掉最好 1 個月後是 2.316，仍在 2.30 以上，沒有跌過一級，不必加註「端點敏感」。"
}

**最終判定（審查後）：待觀察**
