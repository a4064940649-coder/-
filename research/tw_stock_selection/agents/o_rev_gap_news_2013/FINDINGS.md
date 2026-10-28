# FINDINGS：2013～2021 營收缺口的 PIT 新聞代理與首次公告值擴充（o_rev_gap_news_2013）

資料夾：/home/user/-/research/tw_stock_selection/agents/o_rev_gap_news_2013/
（子代理不能寫 .md，請主研究員把這份全文存成 FINDINGS.md。）

## 結論

**判定：無法測試。** 第一道閘門 G1 沒過，依事前規則主題結案。SOP 的保守口徑維持 B10，沒有跑任何策略報酬。

**兩道閘門都沒過**
- **G1（缺口代理覆蓋 ≥ 60%）**：選定版本是 P3 解析器加 N1 代理。在有真值的非缺口格上，它的百分位 Spearman 是 0.879，品質合格；但 732 個缺口格只覆蓋 **23.9%**。
  - 覆蓋最高的 NQ（新聞＋季報法定期限）覆蓋 66.0%，但 Spearman 只有 0.722。Q 單獨用是 0.626。兩者都低於 0.80，事前寫定的規則不能選它們。
- **G2（首次值覆蓋 ≥ 20%）**：AB_V4 前 10 名股月中，三個營收月都有嚴格佐證首次值的比例只有 **0.09%**（95% 區間 0.02～0.52%）。依 o5 教訓 6，覆蓋 < 5% 只能寫「無法估計」，不得寫任何方向。

**對 SOP 的影響**
- 不改。正式數字仍是 AB_V4 27.00%（t 2.51），保守口徑 AB_V4_B10 24.65%（t 2.13）。
- 「B10 三項穩健性檢查全部不過、2014 年後不顯著」的加註照舊。

**唯一有用的新事實（描述性、條件式）**
- B10 假設缺口股的營收百分位一律是 1.0。但在 B10 實際選進的 133 個缺口股月中：
  - 有 N1 代理的 29 格，代理百分位中位數只有 0.32，沒有一格 ≥ 0.9。
  - 用品質不合格的 NQ，有代理的 99 格中位數是 0.56，只有 11% ≥ 0.9。
  - 把 N1 代理放回名單後，進前 10 名的缺口股月只剩 1 個（B10 是 133 個）。
- 這暗示 B10 的最壞情況可能嚴重高估了存活者偏誤，真實界限可能比較靠近 AB。
- 但有兩個限制：
  - 代理只覆蓋 22%，而且有新聞的多半是大型股、併購標的，有選樣偏誤。
  - NQ 的品質低於門檻。
- 依 o5 教訓 9，這只能寫成條件式：「若有新聞的缺口格可以代表全體，B10 偏悲觀」。不得改 SOP 數字。

## 1. 研究問題
1. 2013～2021 年的營收缺口格（主要是下市股，另有非下市格）能不能用「時間戳 ≤ 訊號日收盤」的公開來源代理？代理之後，AB 與 B10 之間的界限能收窄多少？
2. 前 10 名股月的首次公告值，能不能擴大佐證覆蓋？
3. 存活者偏誤主要來自經營惡化股，還是併購標的？

## 2. 機制（誰在賺、對手是誰）
- **誰在賺**：本國投信的月營收追蹤，也就是主策略的營收腿。投信當時看得到所有公司的營收，包括後來被併購或下市的公司。
- **對手**：回測資料 tw_revenue.csv 從 2013 年起只剩存活公司，等於替主策略事先刪掉一批股票。B10 用「營收百分位 1.0」的最壞情況把它們放回，是界限，不是真值。
- **預期**：
  - 併購標的的股價在併購期間被收購價鎖住，下月超額應接近 0。
  - 經營惡化股如果因營收好而入選，之後超額應該為負，是偏誤主體。

## 3. 資料
**V4 凍結版（唯讀）**
- R_FIX、Y5_FIX、處置 trade 口徑、P、X、gap_dict、merger_consideration、delist_official。
- 引擎只讀 import v4lib。主策略規格、股池、成本、換股日都沒改。

**新來源**
- 只能用 GitHub：MCP 程式碼搜尋 19 組、repo 搜尋 4 組，查過 3 個 git 鏡像。全部記在 sources_log.json，包括 0 筆的查詢。
- 實際用上的來源：

| 代號 | 來源 | 範圍 | 時間戳 | 提及數 |
|---|---|---|---|---|
| CTBC | aRthur08701244/Intern-in-CTBC（鉅亨網抽樣，已有；兩個資料夾是同一批文章） | 2013-01～2021-12，約 25.2 萬篇 | 只有日期 | 28,586 |
| PTT | ji08kk44/BDSE26_TEAM3（PTT Stock 逐日檔，新 clone） | 2019-01～2021-12 | 檔名日期 | 1,303 |
| ELD | roccqqck/stock_eland（Yahoo 股市＋PTT，新 clone） | 2016-01～2018-12，3,086 篇 | 含時間 | 209 |
| BOF | bofrankbo/LLM-Predict-Stock（日月光逐檔新聞，新 clone） | 2004～2024，551 檔 | 檔名日期 | 42 |
| Q | taiwanan 季營收（o_v4 raw_quarterly） | 2013Q1～2018Q3，903 檔 | 法定期限 | — |

**沒找到的來源**
- 2013～2018 年的 PTT「[情報] 月營收」封存。
- 經濟日報、自由、MoneyDJ、ETtoday 在 2013～2018 年的 json／csv 封存。
- 任何在 2013～2021 年有 PIT 提交時間的 MOPS 月營收鏡像：
  - mickyang12/mops 只有 2026 年 1 個提交。
  - ch-lai 的完整歷史只有 2019-09-13～14 的提交。
  - taiwanan 的月營收提交在 2018-12 之後。
- bofrankbo 雖然有 55 檔逐檔新聞，但和 48 檔缺口股只交集日月光一檔。

## 4. 探索發現（不看策略報酬）

**步驟 0：淨值跳日（o5 教訓 1）**
- AB_V4 淨值在 2013～2021 年沒有連續 > 5 個交易日的空洞。間隔 ≥ 9 日曆日的都是春節休市，加權指數同樣沒有交易。
- 全期唯一的空洞是 2023-12-29 → 2024-02-15，缺 25 個交易日，不在本題窗內。

**(1) 缺口格清冊**
- 732 格、48 檔、90 個換股日：併購 425、惡化 235、非下市 72（2013～2020 為 65 格，約等於審查說的 63 格）。
- 依年份：2013 年 245、2014 年 225、2015 年 120，三年合計 590 格（81%），到 2018 年只剩 8 格。
- 依 V60 排名：1～100 名 274 格、101～200 名 227 格、201～300 名 205 格。
- 依 52 週比百分位：≥ 0.7 的有 182 格。
- 最多格的股票：矽品 51、日月光 51、立錡 39、華亞科 39、晶電 35、奕力 34。
- B10 實際入選 133 個股月（併購 98、惡化 35），和 o5 的數字一致。最多的是矽品 29、日月光 19、必翔 11、立錡 10。

**(2) 非下市 72 格逐格原因**（data/nondelist_reasons.csv）
- 資料源漏抓 46 格：晶睿 3454 有 26 格、6288 有 15 格、2358 有 5 格。公司仍上市，但 tw_revenue 該代碼 2013 年起中斷，rev_5y 也沒有。
- 基期月營收缺或 ≤ 0：14 格。
- 新代碼或控股轉換：7 格，例如神達 3706、上緯投控 3708、日月光投控 3711。
- 個別月份缺值：5 格。

**(3) 缺口格的資料型態**
- 缺口格平均缺 2.9 個月的營收水準，基期也缺約 2 個月。
- 所以代理必須同時重建當期與去年同月；新聞直接給年增率最有用。

**(4) 機制描述**（個股下一窗報酬減同池等權，挑選期訊號日 ≤ 2019-04；不是策略報酬；月份分群 bootstrap 95% 區間）

| 子樣本 | 格數／月數 | 月均超額 | t | 95% 區間 |
|---|---|---|---|---|
| 全部 | 660／62 | −0.31pp | −0.66 | −1.24～+0.66 |
| 併購 | 377／58 | +0.31pp | 0.50 | −0.91～+1.58 |
| 惡化 | 235／48 | −0.71pp | −1.22 | −1.80～+0.51 |
| 非下市 | 48／34 | −1.86pp | −1.39 | −4.42～+0.64 |
| B10 入選 | 121／53 | +0.18pp | 0.29 | −0.97～+1.41 |
| B10 入選・併購 | 86／47 | +0.07pp | 0.12 | −1.11～+1.23 |
| B10 入選・惡化 | 35／26 | +0.54pp | 0.51 | −1.46～+2.63 |

- 方向上，惡化股比併購股差，符合預期 (ii)；但區間全部包含 0。
- B10 入選的惡化股（事件 < 30 個月）沒有地雷特徵。
- o5 量到的「−4.53pp／月」是相對於被替換的邊際強勢名額，不是相對於同池等權；兩者不矛盾。

## 5. 模型迭代紀錄（iterations.json；判準只看資料品質，不看報酬）
- 解析器 3 輪：
  - P1：o4 規則的重寫。
  - P2：名稱全名擴充，允許「昨（4）日、盤後、精密」等間隔詞。
  - P3：P2 加上「只有年增率、沒有水準」的句子。
- 每輪都評估 5 個代理版本，共 15 個設定。

| 輪 | 提及數 | N1 Spearman | N1 缺口覆蓋 | NQ Spearman | NQ 缺口覆蓋 | 選定 |
|---|---|---|---|---|---|---|
| P1 | 27,968 | 0.882 | 22.0% | 0.720 | 65.0% | N1 |
| P2 | 29,052 | 0.884 | 23.4% | 0.722 | 65.6% | N1 |
| P3 | 30,140 | 0.879 | 23.9% | 0.722 | 66.0% | N1 |

P3 的完整表：

| 版本 | 驗證覆蓋 | Spearman | 百分位誤差中位數／p75／p90 | 前 10 名一致（全名單／涉及代理格） | 缺口覆蓋 | 133 入選覆蓋 |
|---|---|---|---|---|---|---|
| N3 | 7.1% | 0.970 | 0.000／0.021／0.089 | 0.989／0.874 | 5.3% | 2.3% |
| N2 | 13.3% | 0.956 | 0.009／0.046／0.109 | 0.968／0.792 | 11.9% | 8.3% |
| N1 | 29.5% | 0.879 | 0.034／0.101／0.209 | 0.900／0.677 | 23.9% | 21.8% |
| Q | 63.3% | 0.626 | 0.100／0.228／0.412 | 0.764／0.471 | 57.7% | 69.2% |
| NQ | 77.5% | 0.722 | 0.071／0.177／0.349 | 0.733／0.536 | 66.0% | 74.4% |

**選定 P3／N1**（Spearman ≥ 0.80 中覆蓋最高）。
- 解析器改進只把覆蓋提高 1.9pp。瓶頸是新聞量：CTBC 只是鉅亨的抽樣；PTT、ELD 年份不對；BOF 只有一檔。不是解析器的問題。

## 6. 凍結規格
- 見 frozen_spec 欄位。
- PREREG sha256 8d692fe8…；data_freeze_r.json 涵蓋 54 個檔案（全部腳本、V4 凍結資料、引擎庫、提及表）與 4 個 raw 來源的串接 sha256。
- gate.py 會先斷言這些都沒變，才計算。
- 有效檢定數：15＋6＋13＋≥ 7，合計 ≥ 41。實際執行的只有 15 個迭代設定。

## 7. 正式閘門結果（PREREG 後一次跑完）

**G1：23.9%（175/732），未達 60%，不過**
- 2013～2020 的 719 格：與全體同量級。
- B10 入選 133 股月：21.8%。
- 依類型：併購 27.3%、惡化 22.6%、非下市 8.3%。
- 依年份：2013 年 26.1%、2014 年 24.0%、2015 年 21.7%、2016 年 26.8%、2017 年 14.3%、2018 年 12.5%、2019 年 0%、2020 年 14.3%、2021 年 76.9%。

**G2：0.09%，不過，無法估計**
- AB_V4 40 seeds、1,187 個不同股月。
- 嚴格、三個月：0.09%（95% 區間 0.02～0.52%）。
- 嚴格、最新月：1.81%。寬鬆、三個月：1.79%。寬鬆、最新月：11.7%。
- 首次值和最新值不一致率（描述）：嚴格 14/476 = 2.9%（區間 1.8～4.9%），寬鬆 5.2%。高於 o5 的 1.74%，原因可能是 2013 年 IFRS 合併與個體口徑差異；不做推論。

**決策：維持 B10，主題結案**
- AB_V4R、B10R、AB_V4R_N1、逐 seed、j-scan、端點、成本兩平、可執行版、SPA 都沒有執行，依規則不得跑。

**描述（名單計數，不是報酬）**
- N1 放回、未覆蓋格排除：前 10 名只有 1 個缺口股月（B10 是 133 個）。
- N1 放回、未覆蓋格以最壞情況計：1＋119 個。這表示若照 B10R 定義，界限幾乎不會收窄（120 對 133）。
- NQ 放回：19 個（以最壞情況計再加 41 個）。
- B10 入選股月的代理營收百分位：
  - N1：29 格，中位數 0.32，≥ 0.9 的 0%。
  - NQ：99 格，中位數 0.56，≥ 0.9 的 11%。
  - Q：92 格，中位數 0.60。

## 8. 穩健性
- 本題沒有報酬欄位，分年、分產業、去掉最大貢獻股都不適用。
- 覆蓋的穩健性：
  - 三輪解析器的 N1 覆蓋在 22.0～23.9% 之間，NQ 在 65.0～66.0% 之間。
  - 選定結果不受解析器輪次影響。
- 研究員自查：rep/rep_cov.py 不 import proxy.py，另行實作 N1 覆蓋，得 23.907%，與 gate 差 0.0pp，逐格一致 100%。這不是正式獨立複製。

## 9. 機制是否被支持：無法判斷
- 正式機制檢定（放回缺口股對實際被替換的邊際名額）依規則沒有執行。
- 探索階段的個股描述：
  - 惡化 −0.71pp，區間 −1.80～+0.51。
  - 併購 +0.31pp，區間 −0.91～+1.58。
  - 方向符合「惡化差於併購」，但都不顯著。
  - 判定規則要求「惡化 ≤ −2pp 而且區間不含 0」，沒有達到。
- B10 入選的惡化股 +0.54pp（35 格），看不到地雷特徵。

**預測核對**
1. 覆蓋：預測全體 45～75%、133 格 60～90%。實際選定版只有 23.9% 與 21.8%，**否定**。只有品質不合格的 NQ（66.0%／74.4%）落在區間內。預測首次值 5～25%，實際 0.09%，比最悲觀的預測還低；「G2 一半以上機率不過」**成立**。
2. 品質：N3 預測 ≥ 0.85，實際 0.970，**成立**。N1 預測 0.65～0.80，實際 0.879，比預期好。Q 預測 0.55～0.75，實際 0.626，**成立**。
3. 機制、4. 正式欄位、5. 首次值配對：沒有執行，無法檢定。
   - 「代理放回後入選 30～80 個」：N1 只有 1 個，NQ 19 個，都少於預測；但 N1 覆蓋太低，不具代表性。

## 10. 散戶怎麼執行
- 不需要改任何操作，主策略規格不變。
- 實盤預期仍依 o5 的分級距表（以 B10 可執行版為準，約 15～19%）。
- 本題唯一的提醒：B10 的「最壞情況」可能偏悲觀。但能證明這一點的資料（2013～2016 年完整月營收公告封存）目前找不到，所以不能把預期往 AB 方向上修。

## 11. 限制
1. **新聞量是硬限制**：CTBC 是鉅亨約 25 萬篇的抽樣，不是全量；2013～2016 年其他媒體在 GitHub 上沒有封存。
2. **代理覆蓋有選樣偏誤**：有新聞的缺口格偏向大型併購標的。所以「B10 偏悲觀」的描述，不能推到沒有新聞的小型惡化股。
3. **Q 代理有 2～4.5 個月滯後**：Spearman 只有 0.63，而且 2018Q3 以後沒有季報資料，所以 2019～2021 年的 Q 覆蓋是 0。
4. **時間戳只有日期**：沒有時間的文章，一律把訊號日當天的視為不合法。這會讓 2～3 號公告的營收全部可用，10 號當天公告的則要等到下一個訊號日。
5. **G2 佐證嚴格**：CTBC 兩個資料夾算同一來源，又沒有任何有 PIT 提交時間的鏡像，所以嚴格佐證幾乎只能靠「CTBC＋PTT」在 2019～2021 年同時報導。這個口徑本身就讓 G2 幾乎不可能過。寬鬆口徑下最新月也只有 11.7%。
6. **沒有正式獨立複製**：本題閘門不過，需要複製的只有覆蓋率，見 frozen_spec。

**主要檔案**
- 腳本：explore_inventory.py、explore.py、parse_news.py、proxy.py、iterate.py、freeze_r.py、prereg.py、gate.py、firstvalue.py、describe_b10.py、rep/rep_cov.py。
- 結果：explore.json、iterations.json、results.json（PREREG、GATE、DESCRIBE_B10_PICKS_PROXY_PCT、SELF_CHECK_rep_cov）、firstvalue_P3.json、sources_log.json、data_freeze_r.json。
- 資料：gap_cells_2013_2021.csv；data/ 下的 mentions_P1～P3.parquet、proxy_table_gap_cells.csv、nondelist_reasons.csv、explore_gap_stock_excess.csv、top10_firstvalue_P3.csv。
- 原始資料：raw/（bofrankbo_LLM-Predict-Stock、ji08kk44_BDSE26_TEAM3、roccqqck_stock_eland、chlai_full、0lidaxiang_NLP-for-DP）。

## 凍結規格

【凍結規格：o_rev_gap_news_2013（2013～2021 營收缺口 PIT 代理與首次值擴充）】
資料夾：/home/user/-/research/tw_stock_selection/agents/o_rev_gap_news_2013/
PREREG sha256：8d692fe8cb7baaf4f579003506ad9303dfb653b886198afa03395a9280864d84（results.json 的 PREREG；gate.py 會先斷言 PREREG、data_freeze_r.json、gate.py、firstvalue.py 的 sha256 都沒變，並逐檔重算凍結檔中 54 個檔案的 sha256，才開始計算）。
凍結順序（UTC）：iterate P1～P3 → freeze_r.py（data_freeze_r.json）→ prereg.py → gate.py → describe_b10.py → rep/rep_cov.py。

1. 策略規格（完全不改）：V4 凍結版資料（o_v4_revenue_patch_official/data_freeze_v4.json：R_FIX、Y5_FIX、處置 trade 口徑、P、X、下市與併購對價）；52 週高點百分位 0.5 ＋近 3 月營收年增率百分位 0.5；前 300 流動性池（60 日均值 ≥ 1,000 萬、股價 ≥ 10）；排除金融（FIN2）與 KY／DR；月換 10 檔；官方換股日（每月第一個日期 ≥ 11 的交易日）；隔日開盤成交；一般成本（滑價 0.1%＋手續費 6 折）／壓力成本（每邊 0.5%）。引擎只讀 import v4lib（→ drj_core、o3_core、v2_core、ia_core）。

2. 缺口格母體：訊號日 2013-01-01～2021-12-31（90 個換股日）。前 300 池中營收以外都合格（ok_other）、但 V4 非 AB 的 y3（m−1..m−3 三個月年增率平均，2021-03 前用長期營收 R_FIX，之後用 Y5_FIX）缺值的股月。共 732 格、48 檔：
   - 併購下市（代碼在 merger_consideration.csv，且訊號日 ≤ 終止上市日）425 格；
   - 惡化等其他下市（在 v4lib.gap_dict，但不在併購表）235 格；
   - 非下市 72 格（2013～2020 為 65 格）；
   - B10（plan_V4_B10.csv，非 AB）實際入選的有 133 個股月（併購 98、惡化 35）。
   清冊檔：gap_cells_2013_2021.csv（explore_inventory.py）。

3. PIT 規則（寫死）：來源時間戳 ≤ 訊號日 13:30。只有日期的文章視為當日 23:59，所以訊號日當天的文章不合法。季報以法定期限當時間戳：Q1 5/15、Q2 8/14（2013 年用 8/31）、Q3 11/14、年報（Q4＝全年−Q1−Q2−Q3）次年 3/31，期限日視為 23:59。

4. 來源（raw 一律不可信，只當文字讀）：
   - CTBC：aRthur08701244/Intern-in-CTBC @258f42aa，在 o_v4 的 raw/ctbc，唯讀。內容是鉅亨網 EB_EventExtraction2 加 EB_SegAndRawNews，兩個資料夾是同一批文章，算同一來源。
   - BOF：bofrankbo/LLM-Predict-Stock @3d2baa6e 的 history_data/tw/stock_news/2311/2311_news（日月光逐檔新聞，檔名＝日期）。
   - ELD：roccqqck/stock_eland @191e20a5 的 5pa_article.csv（Yahoo 股市＋PTT，2016～2018，post_time 有時間）。
   - PTT：ji08kk44/BDSE26_TEAM3 @13882587 的 ptt/ptt_csv(date)，只用 2019-01-01～2021-12-31，檔名就是發文日；不存在的日期檔（例如 2019-02-29）整檔不用。
   - Q：o_v4 的 raw_quarterly.parquet（taiwanan 季營收 2013Q1～2018Q3，排除 suspect 列）。
   各來源的 sha256 與 HEAD 記在 data_freeze_r.json 的 raw_sources。

5. 解析器（parse_news.py，選定第 P3 輪）：
   - 抽取「<公司> X 月（自結）（合併）營收（淨額）[動詞] Y 億／萬／千元」，數值一律換成千元。
   - 抽取其後 60 字內的「年增／年減／年成長／較去年同期（同月）（也）成長…／YoY Z%」與「月增…」。
   - P3 另外抽取只有年增率、沒有水準的句子：「X 月營收年增 Z%」。
   - 公司歸屬：取月份前 40 字內最近的名稱或代碼。名稱表合併 tw_revenue、stock_list、終止上市名單、taiwanan；做簡稱與全名擴充（P2 起：2～3 字簡稱＋科技／電子／精密／光電／能源／機電／通訊／實業／工業／企業；後綴去除也加入精密、能源、機電、通訊、化學、控股、投控）；一名對多碼的別名不用。
   - 名稱與月份之間只允許時間詞、公告詞與代碼括號。前 8 字出現「估、預期、累計、全年、季…」則排除。
   - 營收月的年份規則：營收月 ≤ 文章月 → 同年，否則前一年；落後 0～3 個月才收。
   - 輸出 data/mentions_P3.parquet：30,140 筆提及、11,121 個（代碼, 營收月）。

6. 代理（proxy.py）：
   - 各營收月的取值順序：先用官方年增率（V4 面板）；沒有時，用訊號日前最早一則新聞的年增率；新聞沒有年增率時，用新聞水準 ÷ 基期水準 − 1。基期水準先用官方 p−12，沒有再用 PIT 新聞的 p−12 水準。
   - N3：三個月都有值，平均（m−1 尚無值時，比照 AB 退一個月）。
   - N2：至少 2 個月有值，取有值月份平均。
   - N1：至少 1 個月有值，取有值月份平均。N2、N1 都至少要有 1 個新聞月。
   - Q：法定期限 < 訊號日的最近一季營收年增率。基期季沒有季報時，用官方三個月營收加總。
   - NQ：有 N1 就用 N1，否則用 Q。
   - 驗證口徑：在「有真值的非缺口格」上，把三個月的官方值全部當成缺，只用新聞與季報重建。指標是：重建值放回當日橫斷面後的百分位，對真值百分位的 Spearman；百分位誤差分布；前 10 名一致率。

7. 選版規則：Spearman ≥ 0.80 的版本中，取缺口覆蓋最高者；同分取較嚴格者（N3＞N2＞Q＞NQ＞N1）。解析器輪次取選定版本缺口覆蓋最高者，同分取較早一輪。結果選定 P3／N1（Q 與 NQ 的 Spearman 為 0.63、0.72，未達 0.80）。

8. 閘門：
   - G1：選定版本在 732 格的覆蓋率 ≥ 60%。
   - G2：2013～2021 年 AB_V4（40 seeds、mc 公告延遲）前 10 名股月中，三個營收月全部有嚴格佐證首次值的比例 ≥ 20%。
     - 嚴格佐證：在首次公告窗（營收月次月月底前）內，有 ≥ 2 個獨立來源的水準一致（相對差 ≤ 0.5%；沒有水準時，年增率差 ≤ 0.1pp），或一個來源加上 git 時間戳。本題找不到 2013～2021 有 PIT 提交時間的鏡像，這條路徑為 0。
     - 寬鬆佐證只供描述：≥ 2 則不同文章一致。

9. 決策樹（事前寫定）：
   - G1 不過：維持 B10，主題結案；不跑任何策略淨值；首次值寫「覆蓋 x%，無法估計」。
   - G1 過、G2 不過：跑 AB_V4R（代理放回、未覆蓋格排除）、B10R（代理放回、未覆蓋格以營收百分位 1.0 計）、AB_V4R_N1，一般與壓力成本，附全套穩健性與 SPA。
   - 本次 G1 不過，第 9 項的淨值部分沒有執行。

10. 有效檢定數：迭代設定 15（3 輪解析器 × 5 個代理版本）＋欄位×成本 6＋j-scan 13＋主策略改版 ≥ 7，合計 ≥ 41。後三項因 G1 不過而沒有執行，但照實列出。

獨立複製要點：
- 拿凍結的 raw 檔，自己寫解析器重建代理表，比對 data/proxy_table_gap_cells.csv 與 GATE.G1。標準：缺口格覆蓋率差 ≤ 3pp，代理值逐格一致率 ≥ 95%。
- 若 G1 不過，報酬欄位不需複製。
- 研究員自查 rep/rep_cov.py：不 import proxy.py，另行實作 N1 覆蓋，得到 23.907%，與 gate 差 0.0pp，逐格一致 100%。這不算正式獨立複製。

## 反方審查

[
 {
  "final": "無法測試",
  "issues": [
   "我同意結案、SOP 不改（AB_V4 27.00%、B10 24.65%）；但「品質合格的 PIT 代理 P3／N1」這個說法不成立。Spearman 0.879 是用全期驗證格算的，各年分布和缺口格不同。逐年看，2013 年 Spearman 只有 0.682，2014 年 0.805；而 2013～2014 正是缺口格的 64%（470/732）。2013～2014 合計 0.752，其中只有 1 個新聞月的格只有 0.680。如果按照 N1 實際覆蓋的 175 個缺口格的「年份 × 新聞月數」分布加權，Spearman 是 0.782，低於事前門檻 0.80。175 格中，115 格只靠 1 個新聞月、140 格完全沒有官方月份；驗證格只有 1 個新聞月時，Spearman 是 0.818。所以 N1 在缺口母體上和 NQ 一樣是品質不合格的版本，不能稱為「品質合格」。",
   "解析器有系統性漏抓，而且漏的正好是本題要找的母體，FINDINGS 卻說「瓶頸是新聞量，不是解析器」，這不對。parse_news.py 會丟掉對應到多個代碼的簡稱（alias 要求 len(cs)==1）。下市舊代碼和控股新代碼常用同一個簡稱，例如日月光 2311 和日月光投控 3711（去掉「投控」後也是「日月光」），另外還有奕力、昇陽光電、中化、定穎、神達 3706。這些簡稱都被丟掉，影響 137 個缺口格（19%），其中 24 格是 B10 入選。實際例子：CTBC 在 2013～2018 年至少有 70 則「日月光 X 月營收 Y 億」的句子，但 2311 只抽到 8 個提及，N1 覆蓋 4%。日月光在 B10 入選 19 次（第 2 多），覆蓋卻是 0 格。驗證格都是存活股，不受這個問題影響，所以這是只壓低缺口覆蓋的選擇性缺陷。就算這 137 格全部補上，G1 上限約 40%，仍然到不了 60%，所以「G1 不過」的結論不變；但 23.9% 只是有缺陷解析器量到的下界，限制第 1 點和 §5 的歸因要改寫。",
   "「B10 入選 29 格代理百分位中位數 0.32、沒有一格 ≥ 0.9」幾乎只是單一股票的結果，沒有統計資訊量。29 格只來自 8 檔股票，矽品 2325 就占 18 格（該股中位數 0.315）。以股票為群做 bootstrap，中位數的 95% 區間是 0.26～0.72，包含 0.5 以上。若以 8 檔股票當作獨立樣本，在均勻分布下「沒有一格 ≥ 0.9」的機率是 0.9^8 = 0.43。此外，29 格中 26 格完全沒有官方月份、20 格只有 1 個新聞月、10 格在 2013 年（Spearman 只有約 0.68）。B10 第 2 大的日月光又因為上一點的缺陷完全不在裡面。建議把「B10 的最壞情況可能嚴重高估存活者偏誤，真實界限可能比較靠近 AB」整段刪掉，或降級為「樣本量不足以推論」。連條件式寫法都不足以支持「嚴重高估」這種程度用語。",
   "選樣偏誤是確定的，不只是可能。B10 入選 133 格中，有 N1 覆蓋的 29 格流動性排名中位數 72，沒覆蓋的 104 格是 112；有覆蓋的只有 8 檔，沒覆蓋的有 25 檔。必翔 1729（11 格）、永大 1507（6 格）、全智科 3559（5 格）等小型或惡化股完全沒有代理。",
   "名單計數的描述也不能支持往 AB 移動。N1 放回後，若未覆蓋格以最壞情況計，缺口股月是 1＋119 = 120 個，對比 B10 的 133，界限幾乎沒收窄。「只剩 1 個」是 ABR 口徑，等於直接把未覆蓋格排除，本身就是 AB 的假設，不是新證據。",
   "PREREG 的時間（05:24:12）在三輪迭代完成（iterations.json 05:22）之後，而且 PREREG 裡已經記錄 iter_G1_cov_seen=0.239。所以「PREREG 後一次跑完的正式閘門」在跑之前結果就已知，只是形式。結論是負面的，不會造成膨脹，但不能把它說成事前檢定。",
   "探索階段的機制描述（表 §4(4)）只按月份分群做 bootstrap，沒有處理同一檔股票跨月的相關。例如矽品、日月光各 51 格，又是事件型持有，因此區間被低估。目前全部不顯著，方向判斷不受影響；但「惡化差於併購」連方向性的描述都不應該寫。",
   "PIT 時間戳本身沒有發現前視問題。只有日期的文章都設為當日 23:59，訊號日當天的文章不合法。營收月當月就出現該月營收的提及（lag 0）有 15 筆，都沒有被 N1 覆蓋格用到。lag 1 提及中，水準和官方值相差超過 5% 的比例是 1.6%。"
  ],
  "note": "驗證程式在 /home/user/-/research/tw_stock_selection/agents/o_rev_gap_news_2013/review_stat_o6.py（只讀研究員檔案、未修改任何研究員檔案；python3 -I -B，約 60 秒）。審查判定維持「無法測試」：G1 不過的結論在不利方向下更穩固（本來就不過），SOP 維持 B10 正確。但需要三處修正。第一，N1 在缺口母體上用「年份 × 新聞月數」加權後，Spearman 只有 0.78，2013 年 0.68，所以不是品質合格的代理，本題實際上沒有任何合格版本。第二，解析器丟掉同名的新舊代碼（日月光 2311 對 3711 等），系統性漏掉 137 個缺口格，其中 24 格是 B10 入選；所以「瓶頸是新聞量、不是解析器」的說法要改。補上後 G1 上限約 40%，仍然不過。第三，「B10 偏悲觀」的描述性結論實際上是 8 檔股票裡矽品一檔占 18/29；以股票分群的中位數 95% 區間是 0.26～0.72，「零格 ≥ 0.9」在均勻分布下有 43% 機率發生，沒有資訊量。這段應該刪除，不能進入 SOP 加註或散戶預期。G2 的 0.09%「無法估計」沒有異議。"
 },
 {
  "final": "無法測試",
  "issues": [
   "我同意「無法測試、主題結案、SOP 維持 B10（24.65%，t 2.13）」這個判定。G1 只有 23.9%（175/732），離門檻 60% 很遠，依事前規則沒有任何空間把它改判成「可用」或「待觀察」。研究員沒有跑任何報酬，周轉、0.3% 證交稅、滑價、成本兩平這些項目也就無從審起。實盤預期仍沿用 o5 的 B10 可執行版（約 15～19%），這次沒有重新驗證。",
   "「B10 偏悲觀」這段描述，證據基礎比 FINDINGS 寫的還薄。29 個有 N1 代理的 B10 入選格只來自 8 檔股票，其中矽品 2325 就占 18 格（62%）；其餘建興 8008、華亞科 3474、華寶 8078、復航 6702 各 2 格，另有 3 檔各 1 格。先在每檔股票內取中位數、再對 8 檔取中位數，結果是 0.375，不是 0.32。所以「中位數 0.32、沒有一格 ≥ 0.9」實際上大致是在描述「矽品的營收年增率排名中等」。有 9 格矽品落在 2015-12 到 2018-01，這段期間股價先被日月光的公開收購價、後被合併對價鎖住，營收百分位高低幾乎影響不了報酬。這段描述連條件式都只能寫成「在矽品這一檔上」，不能延伸成「真實界限可能比較靠近 AB」，建議刪掉這句。",
   "研究員自己的名單計數已經顯示，即使採用 N1，界限也幾乎收不窄：B10R 口徑下是 1＋119＝120 個缺口股月，B10 原本是 133 個。而且 N1 放回後唯一進前 10 名的缺口股月是非下市格，不屬於原本的 133 個。所以就算覆蓋率再提高一點，對 SOP 的實務影響也接近 0。FINDINGS 結論段的措辭應該往「沒有可執行的新資訊」靠。",
   "資料取得時點（我的驗證：review_exec_o6.py 的 A、B 段）：2013～2021 年 90 個訊號日都落在 11～18 日，在月營收 10 日的法定公告期限之後。29 格實際用到的新聞提及，日期全都早於訊號日至少 0～1 天，沒有任何一則早於營收月結束（不是預估值）。只有日期的文章一律視為當天 23:59，而成交在次日開盤，這個口徑偏保守。PIT 部分沒有發現前視。",
   "解析器確實有錯配：同一句話列出多家公司時會配錯股票。例如 2013-10-07 鉅亨網「南科(2408)、華亞科(3473)9月營收38.05億元、59.76億元」，把南科的 38.05 億元記到華亞科 3474（mentions_P3 第 4 列與第 14357 列）。這一格的代理值剛好正確（年增率 114%，百分位 0.72，我用 59.76 億元重算得 113.5%），原因是同一時間戳還有一則直接給年增率的文章，被優先採用。也就是說，結果正確只是因為剛好有另一則文章；只有水準值、沒有年增率的格子一樣可能錯配。另外 CTBC 兩個資料夾的文章在提及表裡逐列重複；Proxy 用 sort_values 排序，預設不是穩定排序，同一時間戳的多則提及取哪一則，靠的是排序後的順序。",
   "可成交與容量（C 段，B10 入選 133 股月）：進場日開盤漲停買不到的是 0 格；V60 最低 3,177 萬元、P10 4,842 萬元、中位數 2.2 億元。1,000 萬元帳戶每檔 100 萬元，最多只占日均成交值的 3.1%，容量不是問題。持有窗內沒有鎖跌停賣不掉的日子。但有 4 格在持有窗內就因合併下市：立錡 6286 在 2016-04-29、華亞科 3474 在 2016-12-06、日月光 2311 與矽品 2325 在 2018-04-30。另有 7 格在窗內出現停止交易。散戶實際拿到的是現金對價（例如矽品每股 51.2 元，在下市後才撥款）或日月光投控 3711 的股份，這筆資金趕不上下一個換股日。B10 引擎若以下市日按對價結算、並立即再投入，會稍微高估可執行報酬。本題沒有處理這點；它會讓 B10 口徑稍微偏樂觀，不是偏悲觀。",
   "首次值 G2 的 0.09% 照「無法估計」處理是對的。首次值和最新值不一致率 2.9%（嚴格口徑）高於 o5 的 1.74%，這只是描述，不能寫成任何偏誤方向。"
  ],
  "note": "驗證程式：/home/user/-/research/tw_stock_selection/agents/o_rev_gap_news_2013/review_exec_o6.py，以 python3 -I -B 執行，只讀取研究員的檔案、沒有修改。A 段核對訊號日的日期分布；B 段逐格列出 29 格 N1 代理所用的提及（來源、日期、原文片段）；C 段檢查 133 個入選股月的漲停、容量、持有窗內下市；D 段用華亞科自己的營收重算那兩格。判定維持「無法測試」：G1 依事前規則沒過，沒有報酬可以審，PIT 時點與容量都沒發現問題。要推翻的是附帶的描述性推論：「B10 偏悲觀、界限靠近 AB」主要靠矽品一檔（18/29），其中 9 格還落在價格被收購價或合併對價鎖住的期間；而且 N1 口徑下界限只從 133 降到 120。這句應降級成只描述矽品，或直接刪除。另外有兩點會讓 B10 稍微偏樂觀、不是偏悲觀，FINDINGS 應註明：合併下市後現金對價入帳的延遲，以及解析器在多公司句子裡的錯配風險。"
 }
]

## 獨立複製

{
 "reproduced": false,
 "my_numbers": "I rebuilt everything independently under /home/user/-/research/tw_stock_selection/agents/o_rev_gap_news_2013/replicate_o6/ and ran it all with python3 -I -B. I only imported the v4lib engine and never read the researcher's code. The scripts are r_inventory.py, r_parse.py, r_proxy.py, r_validate.py and r_g2.py.\n\n1) Gap-cell inventory (r_gap_cells.csv): 732 cells in 48 stocks over 90 rebalance dates. 425 merger, 235 deterioration, 72 not delisted. 2013–2020 has 719 cells, 65 of them not delisted. By year: 2013 has 245, 2014 has 225, 2015 has 120. B10 picks are 133: 98 merger and 35 deterioration. The most-picked stocks are 2325 (29), 2311 (19) and 1729 (11). Compared cell by cell with gap_cells_2013_2021.csv, all 732 match, and type and in_B10 agree 100%.\n\n2) Parser (my own regex): 18,196 mentions and 12,460 (code, revenue month) pairs. CTBC's two folders are deduplicated as one source. By source: CTBC 16,289, PTT 1,602, ELD 256, BOF 49.\n\n3) Gap-cell proxy coverage, mine vs researcher:\n- N3: 5.9% vs 5.3%\n- N2: 14.3% vs 11.9%\n- N1: 27.2% vs 23.9% (gap 3.28pp)\n- Q: 57.7% vs 57.7%, values identical cell by cell\n- NQ: 66.7% vs 66.0%\n- N1 for 2013–2020: 26.3%. N1 by type: merger 32.0%, deterioration 24.3%, not delisted 8.3%. N1 by year: 2013 26.5%, 2014 31.6%, 2015 23.3%, 2019 0%, 2021 76.9%.\n- N1 coverage of the 133 B10 picks: 24.1% vs 21.8%. NQ: 73.7% vs 74.4%.\n- Cell-by-cell agreement: N3 97.5%, N2 94.1%, N1 90.3%, Q 100%, NQ 90.3%. For N1, coverage status agrees in 95.4% of cells; where both have a value, the values agree in 78%.\n\n4) Validation (25,618 non-gap cells, 108 signal dates; official YoY for the three months set to missing):\n\n| Version | Coverage | Spearman | Error median / p90 |\n|---|---|---|---|\n| N3 | 8.7% | 0.963 | 0.000 / 0.086 |\n| N2 | 15.5% | 0.961 | 0.008 / 0.102 |\n| N1 | 31.9% | 0.885 | 0.031 / 0.201 |\n| Q | 63.3% | 0.626 | 0.099 / 0.410 |\n| NQ | 78.4% | 0.731 | 0.067 / 0.342 |\n\nThese are all within about 0.01 of the researcher's figures, and Q matches to every reported digit. Applying the version-selection rule gives N1 again, because Q and NQ are below 0.80. G1 = 27.2% < 60%, so G1 fails and the decision is the same as the researcher's.\n\n5) G2 (AB_V4, 40 seeds): 43,200 stock-month×seed rows and 1,187 distinct stock-months, both exactly matching.\n- Strict, all three months: 40 of 43,200 = 0.0926%, exactly matching the researcher's 0.09%.\n- Strict, latest month only: 2.38% vs 1.81%.\n- Loose, three months: 1.76% vs 1.79%. Loose, latest month: 8.7% vs 11.7%.\n- Corroborated cells: strict 605 vs 506; loose 2,535 vs 4,441.\n- First vs latest disagreement: strict 28/584 = 4.8% vs 2.9%; loose 5.7% vs 5.2%.\n- G2 fails.\n\nI did not replicate the return figures, because G1 fails.",
 "diff_explained": "Against the spec's formal replication standard (gap-cell coverage gap ≤3pp, cell-by-cell proxy agreement ≥95%), N1 misses narrowly: the gap is 3.28pp and agreement is 90.3%. That is why reproduced=false. The numbers that drive the decision do reproduce: the inventory and Q match exactly, the G2 headline 0.09% matches exactly, validation Spearman is within 0.01 throughout, and both the version selection (N1) and G1 failing are the same. N1 coverage is 24–31% under any parser setting I tried, far below 60%. NQ also fails the selection rule, with Spearman 0.73 < 0.80.\n\nComparing against the researcher's output file data/mentions_P3.parquet (data, not code), the N1 differences all trace back to parser details:\n\n(a) The researcher's parser has a bug: it assigns a cumulative YoY to the single month. Example: for 2325 in 2013-10 it records yoy 5.0, but that is the YoY for \"累計前10月\". I cut the 60-character window at \"累計\", \"前N月\" and \"季\".\n\n(b) The researcher accepts things I do not, because the spec does not list them:\n- \"業績\" as revenue (e.g. \"8 月業績達65.1億元\")\n- \"的\" between company name and month (e.g. \"同業華寶的11月營收\")\n- \"3成\" as 30%\n- a level with a space before 元 (e.g. \"74億 元\")\n\n(c) The researcher has extra exclusion words (the spec's \"…\"). Judging from the misses, these appear to include \"至\" and forecast words such as \"樂觀\". As a result they do not capture sentences like \"至於晶電4月營收\" or \"接單樂觀東貝()2月份合併營收\".\n\n(d) The researcher does not recognize YoY wording like \"較去年同期則增加了\" / \"則成長\", so they fall back to level ÷ base. I aligned with the spec, which allows only \"也\", and that fixed the 3698 differences.\n\n(e) For same-timestamp mentions, which one counts as \"earliest\" depends on the parser's ordering. This affects a handful of values, mainly for 2325.\n\nMy earlier looser parser drafts gave N1 between 29.9% and 37.2%. The version I finally adopted was not tuned to match the researcher's output.\n\nThe researcher's 30,140 mentions include both CTBC folders as separate copies (CTBC and CTBCs, 14,293 each). This conflicts with the spec, which says the two folders are the same source. It also likely explains why the researcher's loose corroboration count (4,441) is far above mine (2,535): duplicate copies of the same article get counted as \"different articles\". Strict corroboration differs (506 vs 605) because the parsers extract levels and YoY differently.\n\nThe two numbers I could not reproduce:\n- The researcher's \"top-10 agreement\" figures (0.989 / 0.968 / 0.900 / 0.764 / 0.733), because the spec does not define the metric. I calculated two versions myself. Agreement over all covered cells came out 0.994 / 0.990 / 0.983 / 0.975 / 0.979. The share of true top-10 cells that stay in the top 10 came out 0.869 / 0.871 / 0.822 / 0.593 / 0.693. Neither matches.\n- G2's 95% interval (0.02–0.52%). A plain Wilson interval on 43,200 rows gives 0.07–0.13%; the researcher probably used an interval based on distinct stock-months or clusters.\n\nMy validation set has 25,618 cells versus the researcher's \"about 24,000\"; I used all 108 signal dates in 2013–2021."
}

**最終判定（審查後）：無法測試**
