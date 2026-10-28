# FINDINGS：逐檔營收公告觸發換股（閘門未過）→ 後備題：J0／J+1 固定換股日正式化與前視清理

資料夾：/home/user/-/research/tw_stock_selection/agents/o_rev_event_trigger/
資料一律用 V4 凍結版（o_v4_revenue_patch_official/data_freeze_v4.json）。引擎只讀 import v4lib；本題的程式全部放在這個資料夾。

## 1. 研究問題
- **原題**：每檔股票在自己的營收公告後第 1 個交易日開盤進出，名單每日滾動、維持 10 檔，藉此吃到「公告到期限」這段漂移，針對 2019 年後營收時效縮短。
- **閘門**：
  - 2011～2018 年池內股月中，能取得逐檔、到日精度、PIT 合法公告時間戳的比例要 ≥ 80%（扣掉來源本身缺口後的上界）。
  - 周轉增加後的損益兩平要 ≥ 0.15%。
- **後備題**（閘門不過時做）：主策略官方換股日其實是 J0（88 次）與 J+1（155 次）的混合。要改成單一、明確、沒有前視的定義，並回答兩個問題：
  - 主策略 t 2.51 有多少來自 J0 月份的前視；
  - 預期是否要下修。

## 2. 機制
- **原題的機制**：好消息傾向早公告、壞消息拖到期限。到固定換股日時，早公告股的驚喜已被消化 5～10 個交易日。散戶與受限的主動基金追不上 300 檔的逐日公告，0.3% 證交稅也讓小漂移不值得交易。
- **後備題的機制**：
  - J0（期限日收盤算訊號）約有 17～22% 的公司在收盤後才公告。
  - 回測若用最新值，就是前視；若實盤只用已公告值，名單會混入過期營收。
  - J+1 沒有這個問題，但比 J0 晚一天，在 2019 年後營收時效縮短的環境下，可能付出新鮮度成本。

## 3. 資料與閘門（gate.py，看任何報酬之前）
- **母體**：2011～2018 年 V4 官方訊號月，前 300 池、排除金融與 KY／DR，共 **25,260** 個股月。
- **(a) 鉅亨網新聞**（o_v4 凍結的 raw_news_mentions.parquet；原始 clone 在 o_v4 raw/ctbc，2013-10～2021-09）：
  - 精確到日（片段寫「今(X)日」，且 X 等於文章日）：**1.94%**。
  - 寬鬆口徑（公告月內、不晚於期限後 2 個交易日的任何營收新聞）：**8.85%**。
  - 在來源有資料的月份內，寬鬆口徑也只有 12.1%。
- **扣掉來源缺口後的上界：73.1%**，原因是 2011～2012 年完全沒有來源，已低於 80%。
- **(b)(c)** web.archive.org、archive.org API、MOPS、mopsov、FinMind、twse：本 session 全部 CONNECT 403，紀錄在 raw/access_log.json。
- **GitHub code search**：沒有任何 2011～2018 年逐檔公告時間戳的封存，只有即時爬蟲，抓到的是更正後的現值。
- **(d)** deadline_calendar／p_delay 是法定期限與機率，不是逐檔時間戳。
- **(e)** rev_hist ann_date 只有 2026-08／09，與新聞重疊 0 個月，**時間戳精度一致率無法計算**。
- **結論：閘門不通過**，損益兩平那一半沒有計算。改做後備題。產出的公告日表：data/ann_news.csv（9,909 個代碼×營收月）。

## 4. 探索發現（explore.py；只用挑選期，不看報酬）
- **日曆**：官方日（每月第一個 ≥ 11 日的交易日）中，J+1 155 次、J0 88 次，另有 2 個特例月（2019-02 早於延長期限 4 天、2023-04 早 1 天）。重現 o4 的統計。
- **公告時點**（新聞，池內 2,312 個股月；新聞日期是公告時間的上限）：
  - 有嚴格時間戳者：期限前 1～4 日公告占 73.4%，期限日當天 19.0%，期限前 ≥ 5 日 5.7%。
  - 「期限日當天 19%」和 rev_announce 用 2026-09 估出的 p = 17% 吻合，可以當作 p 模型的獨立歷史佐證。
  - 期限前 ≥ 5 日公告的比例每年只有 3～5%，2013～2018 沒有趨勢。
- **預測 2（原題，描述性）**：
  - 期限前 ≥ 5 日公告股的近 3 月年增率池內百分位 **0.577**，期限日公告股 **0.468**。
  - 按月分群差 **+0.124（t 2.54，41 個月）**，高於預測的 0.05，「好消息早公告」方向成立。
  - 嚴格時間戳子樣本 +0.185（t 1.77，12 個月）。
  - 限制：新聞涵蓋只有約 12%，而且偏向大型股。
- **J0N 的 q**：2026-09 期限日（10-12）收盤前，非金融公司完成率 82.9%；到當晚、次日 09:00 前是 **99.67%**（q = 0.33%，6/1,821 家）。因此「盤後完成率 ≥ 99% 的最早日」就是期限日當晚，記為 J0N。
- **前視暴露**（名單層）：非 AB 名單中訊號時點還沒公告的期望名額，每月 J0 **1.62**、官方混合 **0.59**、J0N 0.03、J+1 **0**。

## 5. 模型迭代紀錄（iterate.py；挑選期 2006-04～2018-12，淨值只模擬到 2019-01 的同型訊號日）
本題共 6 個版本，另外 13 個 j-scan 版本只當穩健性分布，不參與選版。

| 版本 | AB t（挑選期） | 日曆月 t | 年化 | 前視（pp／月） | 對 J+1 配對（日曆月） |
|---|---|---|---|---|---|
| it1 J+1 | **2.250** | 2.31 | 16.64% | 0.000 | — |
| it2 J0（AB 用 p） | 2.180 | 2.24 | 17.35% | **0.0996（t 2.03）** | +0.049（t 0.27） |
| it3 J0N（D 當晚營收） | **2.415** | 2.46 | 18.55% | 0.0065 | +0.143（t 0.77） |
| it4 J0 ABN（新聞時間戳＋條件 p） | 2.202 | 2.26 | 17.47% | — | +0.059（t 0.32） |
| it5 官方混合（參考） | 2.385 | 2.51 | 18.07% | 0.048 | +0.122（t 1.15） |
| it6 J0 worst（期限日全部未公告） | 2.405 | 2.45 | 19.28% | — | +0.182（t 0.92） |

- **選版**（事前規則）：前視 < 0.03 的只有 J+1 與 J0N，t 較高的是 J0N。但 J0N 對 J+1 的配對 |t| 只有 0.77，小於 2，所以**選 J+1**。
- **觀察**：it6（全部用舊營收）反而比 it2（隨機 17% 用舊營收）好。橫斷面內混用新舊營收會讓排序失真，隨機 p 模型可能高估真實的懲罰；真實的晚公告者偏壞消息，不是隨機的。這是限制，不改判定。
- ABN 幾乎沒有差別，因為新聞只涵蓋約 12%。

## 6. 凍結規格
見 frozen_spec 欄位。PREREG sha256 208462cf…（2026-10-28T01:37:07Z），涵蓋 52 個檔案，包括全部腳本、引擎庫、資料與面板。

## 7. 樣本外結果（oos.py，一次跑完；判定欄為 AB、一般成本）

| 口徑 | ALL 年化／t／日曆月 t | 最大回撤 | VAL 年化／t | 2024～2026 年化／t | 壓力成本 ALL t |
|---|---|---|---|---|---|
| **J+1（選定）** | **25.12%／2.26／2.12** | −62.2% | 33.18%／1.31 | 55.52%／0.10（SOP 口徑 **52.8% 對 0050 58.0%**） | 1.69 |
| J0 | 25.44%／2.20／2.11 | −60.8% | 30.54%／1.10 | 60.04%／0.23 | 1.67 |
| J0N | 26.72%／2.42／2.31 | −61.4% | 30.98%／1.13 | 63.79%／0.37（SOP 口徑 60.6%） | 1.90 |
| 官方 AB_V4 | 27.00%／2.51／2.43 | −63.5% | 35.11%／1.44 | 60.06%／0.24（SOP 口徑 56.7% 對 59.2%） | 1.97 |
| 0050 | 14.82% | | 17.60% | 57.72% | |

- **J+1 分段**：S1 17.19%（t 1.94）、S2 t 0.93、S3 t 1.07；POST 40.44% 對 30.04%（t 1.06）。
- **J+1 對照組**：對同池等權 t 3.52（等權 10.96%）；m=10 Bonferroni p 0.12。
- **與主策略的配對**（日曆月，pp／月）：

  | 比較 | ALL | SEL | POST | 最小可偵測效果（ALL） |
  |---|---|---|---|---|
  | J+1 − AB_V4 | −0.117（t −1.15） | −0.122（t −1.15） | −0.108（t −0.53） | 0.28 |
  | J0 − J+1 | +0.002（t 0.01） | | | 0.51 |
  | J0N − J+1 | +0.097（t 0.53） | | | 0.51 |

  三組配對差都偵測不到。
- **前視清理（全期）**：
  - J0 的前視是 +0.102pp／月（t 2.30），約 1.3pp／年；J0N − J0 是 +0.095（t 2.21）。期限日收盤前只用已公告營收，確實有實質代價。
  - 官方混合 0.030（t 1.10）；J+1 0.006（只有 2020-02）；J0N 0.007。
- **逐 seed（J+1）**：t 2.225～2.274，40/40 低於 2.3，0 個低於 2.0；拿掉最好 1 個月的中位數 2.08。
  - J0 的 seed 分散大：1.80～2.55，中位數 2.16，31/40 低於 2.3，5/40 低於 2.0。
- **換股日分布**（J0～J+12，AB 10 seeds）：全期 t 中位數 **1.68**，p10 **1.04**，最大 2.29（J+3）。
  - 2019 年後 J0～J4 平均 t 1.16，J8～J12 平均 −0.33，重現 o4「越晚越差」的結構。這段資料已被看過。
- **端點（J+1）**：

  | 情境 | t |
  |---|---|
  | 拿掉 2006 年 | 2.02 |
  | 起點改 2008 年 | 1.85 |
  | 拿掉最後 1 個月 | 2.13 |
  | 拿掉最好 1／3／5／10 個月 | 2.09／1.76／1.43／0.65 |
  | 拿掉最好年份（2021） | 1.78 |
  | 拿掉最好 3 年（2021、2015、2026） | 1.16 |
  | 日曆月 t 拿掉最好 1 個月 | 1.87 |

- **成本**：J+1 在一般成本（滑價 0.10%）下 t 就已低於 2.3；t 跌到 2.0 時的單邊滑價是 0.175%。
- **判定**：J+1 的 t 比官方低 **0.253**，超過 0.1 → **預期下修**。用 0050 加 0.5 倍超額的公式，實盤預期 **19.97%**；官方用同一公式是 20.97%。

## 8. 穩健性
- **分年**：J+1 輸 0050 的年份是 2008（−5.5pp）、2016（−23.0）、2019（−12.5）、2022（−1.3）、2024（−41.2）；2021 年 +84.8pp。
- **分產業貢獻**：半導體 13.7%、其他電子 12.2%、電子零組件 11.6%、航運 8.2%、通信網路 8.0%。
- **去掉最大貢獻股**：
  - 全期最大是 3443 創意，占總貢獻 3.6%，拿掉後毛年化少 1.0pp。
  - 2024～2026 最大是 2059 川湖，占 12.4%，拿掉後少 8.0pp。
- **事後（finalize.py，不參與判定）**：OFF_N 用官方日，但 J0 月改用期限日當晚營收：27.42%／t 2.58，對 AB_V4 +0.027pp／月（t 1.11）。這表示官方 2.51 與 J+1 2.26 的落差，不是官方日的前視造成的。落差來自兩個來源：(i) J0 月提早一天換股；(ii) 期限日當晚的營收已經可用。和 J+1 相比都不顯著。

## 9. 機制是否被支持：部分支持
- **原題的事件觸發機制**：無法測試報酬，閘門沒過。描述層有支持：早公告股營收較好（+0.124 百分位，t 2.54）。
- **後備題**：
  - J0 收盤時點的前視或過期營收懲罰是真實的，t 2.0～2.3。
  - 「早一天換股較好」：J0N − J+1 +0.097（t 0.53），最小可偵測效果 0.51pp，**檢定力不足，無法判斷**。
  - 2019 年後「越晚越差」在 j-scan 重現，但屬於已看過的資料。

## 10. 散戶怎麼執行（建議的 SOP 文字，訊號日與成交日分開寫）
- **步驟 1（訊號日）**：每月營收公告期限日的下一個交易日，當天收盤後算訊號。
  - 期限日是 10 日；遇假日順延到下一個交易日；主管機關延長期限時，用延長後的日期。
  - 10 日是交易日時，訊號日與舊規則「第一個 ≥ 11 日的交易日」相同。
  - 10 日遇週末或假日時，比舊規則晚一個交易日，例如 2026-10 是 10-13，不是 10-12。
- **步驟 5（成交日）**：訊號日的下一個交易日開盤買賣。
- **2207、2905**：2026 年起營收期限是 15 日，訊號日時還沒公告，改用上上個月為止的 3 個月。
- **工具**：main_picks_today.py 的修改建議在 main_picks_today_J1.patch（不改原檔）。已用 245 個錨定月驗證，日期 0 筆不符。
- **SOP 預期段改寫**：
  - 跨循環 2006-04-11～2026-10-13：年化 25.1%，對 0050 t 2.26（日曆月 2.12）。
  - 實盤預期約 20.0%。
  - 2024～2026：52.8% 對 0050 58.0%。
  - 換股日敏感度：j ≥ 0 各版本 t 中位數 1.68、p10 1.04。
- **小帳戶與大帳戶**：
  - 30 萬帳戶每月約 15 筆委託，77.7% 的名額連一張都買不起、只能買零股；2020-10-26 以前零股只能盤後以收盤價成交。
  - 1,000 萬帳戶每月約 24 筆委託，27.6% 的新買名額超過 60 日均成交值的 1%。
- **J0N（期限日當晚算、隔天開盤買）**：只列為假說，不寫進 SOP（o4 教訓 8）。等 2026-11 起的前瞻紀錄。

## 11. 限制
- 2019 年後已被 o4 看過，樣本外不是盲測。
- p 與 q 都只用 2026-09 一期估計。
- 隨機 p 模型可能不符合「晚公告偏壞消息」的事實（it6 的現象）。
- 新聞時間戳只是上限，而且涵蓋偏誤大。
- 面板放在 scratchpad，雖然 sha 已登記，仍不在研究資料夾。
- finalize.py 是事後腳本，不在 PREREG 雜湊內，只產出口徑換算、OFF_N 與 patch。
- 多重檢定：本題 6 個迭代版本、3 個樣本外候選、13 個 j-scan、1 個事後版本，加上主策略本身至少 6 輪改版。
- 「逐檔觸發」若要重開，必須先取得 2011～2018 年的 MOPS 逐檔公告時間封存（現在 proxy 403）。2026-11 起可以用 tw-stock-screener 的首見時間做前瞻紙上追蹤。

## 12. 檔案
gate.py、gate.json、explore.py、explore.json、validate.py、validate.json、plan.py、plan.json、iterate.py、iterations.json、prereg.py、oos.py、oos.log、finalize.py、results.json（PREREG／RESULTS／POSTHOC）、navs_oos.parquet、plan_Jp1_seed0.csv、plan_J0_seed0.csv、plan_J0N_seed0.csv、data/ann_news.csv、data/q_night.json、raw/access_log.json、main_picks_today_J1.patch、fx_core.py。全部在 /home/user/-/research/tw_stock_selection/agents/o_rev_event_trigger/。

## 凍結規格

PREREG：results.json → PREREG（sha256 208462cf804a8d1ce2dd8e524334adb4a4c086a12b95779cdc23f9914f54880c，2026-10-28T01:37:07Z）。事前計畫 plan.json（sha256 7c7a54a9…，2026-10-28T01:31:33Z）早於任何報酬計算。PREREG 雜湊共登記 52 個檔案，oos.py 執行時逐一斷言：本資料夾全部 .py、plan／gate／explore／iterations／validate.json、data/（ann_news.csv、q_night.json）、raw/access_log.json，引擎庫 v4lib、drj_core、v2_core、ia_core、o3_core，V4 凍結檔 data_freeze_v4.json 與其列出的面板及輸入檔，以及 navs_v4.parquet、plan_V4.csv、raw_news_mentions.parquet、R_FIX、Y5_FIX、p_delay.json、firstseen_2026-09.csv。

選定版本：J+1。
1. 錨定月 m：V4 官方訊號月集合（v2 的營收可用條件；2006-04～2026-10，共 245 個月；2024-01 沒有面板資料，跳過）。營收月 m−1 的法定期限 D_m 定為 m 月 10 日起的第一個交易日，日曆用加權指數交易日（finlab taiex_twii.csv）。主管機關延長期限時用延長後的日期：2019-02 → 2019-02-15，2023-04 → 2023-04-12。
2. 訊號日：D_m 之後的第 1 個交易日，當天收盤算訊號。訊號日或其次一面板日不是 GOOD 日（有值檔數 < 300）時，該月續抱（J+1 沒有發生）。成交日：訊號日的次一面板交易日開盤。開盤漲停（tick 價）且未持有時跳過，由下一名遞補。
3. 策略：完全照 v4lib.V4。
   - 股池：前 300 流動性池，60 日均成交值用 gapfix 窗，≥ 1,000 萬，收盤 ≥ 10。
   - 排除：金融 fin2；KY／DR／F-／91xx；處置股採成交日口徑（公告日 ≤ 訊號日，且成交日仍在處置期間）。
   - 營收：訊號日 < 2021-03-01 用 R_FIX，之後用 Y5_FIX；y3 = m−1、m−2、m−3 三個月年增率平均，三個都要有。
   - 分數：0.5 × 52 週比（252 日 gapfix）池內百分位 + 0.5 × y3 百分位。同分序：分數 → y3 高 → 代碼欄位序。取 n = 10，等權。
   - 下市處理照 P（下市股價格）與 X（merger_consideration 的 main 規則）。
4. 營收可得性（AB，fx_core.make_unavail 的 day 模式）：
   - 訊號日 < D：全部視為未公告。
   - 訊號日 = D：依 rev_announce 的產業延後機率 p 抽樣（金融 p = 1）。
   - 2020-02 錨定月，D < 訊號日 ≤ 2020-02-14：也抽 p（疫情延長）。
   - 2026-01～08 營收：INS15 名單（含 2207、2905）的期限是 15 日，訊號日 < D15 視為未公告。
   - 未公告者改用 m−2..m−4。
   - 隨機數：np.random.default_rng([seed, yyyymmdd, tag])。seed 0～39，每個 seed 一張名單、一條淨值，日報酬平均後連乘成正式淨值。
   - J+1 只有 2020-02 與 2026 年的 INS15 月份有隨機性。
5. 成本：
   - 一般：買 = 0.1425% × 0.6 + 0.1% 滑價；賣 = 同上 + 0.3% 證交稅。
   - 壓力：買賣各 0.5%。
   - 二分法解調倉後總值。損益兩平格點：單邊滑價 0.10%～0.40%，每格 0.02%。
6. 報酬與統計：
   - 一律用簡單報酬。全期年化從 2006-04-11 收盤起算，到 2026-10-13 收盤。分段年化從段內第一個訊號日收盤，到段後第一個訊號日收盤。
   - t：訊號窗（訊號日收盤 → 下一訊號日收盤）對 0050 含息的月超額 t，窗依錨定月歸段。分段：SEL 2006-04～2018-12，VAL 2019-01～2023-12，HOLD 2024-01～2026-10。
   - 另報日曆月 t（完整月 2006-05～2026-09）。版本之間的配對差一律用日曆月，錨定月配對差當次要指標。
   - 2024～2026 依 SOP 口徑報告時，從 2023-12-29 收盤起算。
7. 選版規則（plan.json，事前登記）：
   - 前視 = |日曆月（非 AB − AB）均值|，門檻 < 0.03pp／月；符合者中取挑選期 AB t 最高者。
   - 暫定版本不是 J+1，而且它對 J+1 的日曆月配對 |t| < 2 時，改選 J+1。
   - 結果：J+1 與 J0N 前視都夠低；暫定 J0N（t 2.415），但對 J+1 的配對 t 只有 0.77 → 選 J+1。
8. 判定規則：只改定義時判「可用」。選定版本的 AB 全期 t ≤ 官方 2.512 − 0.1 時，實盤預期下修為 0050 + 0.5 × 超額，用選定版本的 AB 年化計算。
9. 重做順序：python3 -I -B gate.py → explore.py → validate.py → plan.py → iterate.py → prereg.py → oos.py →（事後）finalize.py。全部大約 15 分鐘。validate.py 斷言 V4J 用官方日時與 o_v4 的 V4、AB_V4 淨值逐日相同（最大相對差 0）。

## 反方審查

[
 {
  "final": "待觀察",
  "issues": [
   "「J+1 沒有前視」只對公告時點成立，對資料本身不成立。V4 營收面板是更正後的最新值，不是首次公告值；o4 審查記錄的新聞重疊格不一致率約 0.7%，全體股票都受影響，AB 沒有模擬更正。存活者偏誤也幾乎沒補：1,194 個缺營收的下市股月只補了 6 個，2006～2012 年平均每日 5.27 檔缺口完全沒有代理，偏誤方向對策略有利。另外，期限日當晚仍未公告的 q＝0.33%（6/1,821）在 J+1 一律當成已公告。報告寫的「前視 0.006，只有 2020-02」只量到時點那一層，資料層的前視沒有量化，不能寫成「沒有前視」。",
   "選定版本本身沒有過門檻。J+1 AB 全期 t 2.259，日曆月 t 2.119，40 個 seed 全部低於 2.3。單尾 p 0.012：m＝10 時 Bonferroni p 0.12；照實計數（本題 6 個迭代版本＋3 個樣本外候選＋13 個 j-scan＋1 個事後版本，加主策略至少 6 輪，m 約 29）時 p 0.35，Sidak 0.29。扣掉多重檢定後，對 0050 的超額不顯著。",
   "超額集中在少數年份。逐年窗超額加總共 205pp：2021 一年 54.5pp，前 3 年（2021、2015、2026）占 55.7%，前 5 年占 76.4%。拿掉 2021 後 t 1.78，拿掉 2006 與 2021 後 1.52，拿掉 2006、2007、2021 後 1.33。起點改 2009 年 t 2.00，改 2011 年 1.89。每 12 個月取樣一次的滾動 60 月 t 只有 2015-03（2.41）與 2016-03（2.24）超過 2；2020 年以後介於 0.39～1.72。",
   "樣本外不是盲測，而且保留期等於零。o4 已在 V3 資料上看過 J0／J+1 全期（含 2019～2026）。POST 均值 0.81pp／月，t 1.06，勝率 51%；HOLD（2024～2026）均值 0.14pp，t 0.10，日曆月 t −0.10，勝率 43.8%，年化 52.8% 輸給 0050 的 58.0%。唯一的近期資料沒有任何支持證據。",
   "J+1 本身就是 j-scan 13 個版本中的第 2 名（J+3 2.29、J+1 2.26、J0 2.19、J+6 1.97、J+2 1.91、J+4 1.73），中位數只有 1.68。相鄰換股日之間 t 的標準差約 0.245（J0～J4），與「J+1 比官方低 0.253」同一量級。所以 Δt＝−0.25 本身就是雜訊：配對差 −0.117pp，t −1.15。用 t 差 0.1 機械觸發下修，不能把 20.0% 當成有統計根據的預期。挑選期 t 與 POST t 在 j 之間的相關是 −0.49，表示挑選期排名對後期沒有預測力。",
   "實盤預期 20.0% 的不確定性被隱藏了。月超額均值 0.837pp，95% CI 0.11～1.56pp；代入「0050＋0.5 倍超額」後，年化超額只在 +0.7～+9.4pp 之間，對應實盤約 15.5%～24.2%。這個區間還沒有扣掉資料前視、存活者偏誤與多重檢定。SOP 只寫「約 20.0%」是過度精確。",
   "J0 前視的大小（0.102pp／月，t 2.30）、J0N−J0（+0.095，t 2.21）與事後 OFF_N 拆解都依賴 p／q 模型：p、q 只用 2026-09 一期估計，再以隨機方式套到 2006～2018。研究員自己的 it6（期限日全部未公告）t 2.405，高於隨機 p 的 2.180，與「隨機 p 模型合理」互相矛盾。因此前視懲罰的大小與正負都沒有被識別，「官方 2.51 的落差不是前視造成」這個說法也不能成立。",
   "機制「部分支持」偏寬。原題（逐檔觸發）因閘門沒過，報酬完全無法測試。描述層的 +0.124 百分位（t 2.54，41 個月）只來自約 12% 的新聞涵蓋股，偏大型股；而且新聞偏向報導亮眼的營收，涵蓋本身與好消息相關（選擇偏誤）。嚴格時間戳子樣本只剩 12 個月，t 1.77。後備題的核心假說「早一天換股較好」：J0N−J+1 t 0.53，最小可偵測效果 0.51pp，檢定力不足。合理的寫法是「未支持／無法判斷」。",
   "壓力成本下 t 1.69；單邊滑價 0.10% 時 t 就已低於 2.3，t 跌到 2.0 時的滑價是 0.175%。若加入 o4 審查的開盤集合競價衝擊（1,000 萬帳戶 27.6% 的新買名額超過 60 日均成交值的 1%），J+1 版本會再低於主策略。"
  ],
  "note": "結論：我判「待觀察」。單就換股日定義來說，J+1 是單一、明確、沒有公告時點前視的定義，值得取代官方的 J0／J+1 混合日，這部分可以接受。但研究員把「可用」接上「跨循環 25.1%、實盤約 20.0%」，這部分在統計上撐不住，所以整體不判可用。\n\n1. **前視沒有清乾淨**：\n   - J+1 只清掉公告時點那一層前視。\n   - 營收用的仍是更正後最新值，不是首次公告值。\n   - 下市股的存活者偏誤幾乎沒補，偏誤方向對策略有利，而且都沒有量化。\n2. **J+1 本身沒過門檻**：\n   - 全期 t 2.26（日曆月 2.12），40/40 個 seed 低於 2.3。\n   - 照實際試驗數（m 約 29）做 Bonferroni，p 約 0.35，扣掉多重檢定後不顯著。\n3. **超額集中、近期無支持**：\n   - 前 3 年貢獻 56% 的超額。拿掉 2021 年 t 1.78；拿掉 2006 與 2021 年 t 1.52。\n   - 2020 年後的滾動 5 年 t 都不到 1.8。\n   - 2024～2026 年 t 0.10，52.8% 輸給 0050 的 58.0%。\n   - 2019 年後已被 o4 看過，樣本外不是盲測。\n4. **下修本身是雜訊**：\n   - 相鄰換股日之間 t 的標準差約 0.25，所以 Δt −0.25 本身就落在雜訊裡。\n   - J+1 在 j-scan 13 個版本中排第 2，中位數只有 1.68。\n   - 「約 20.0%」的 95% 區間大約是 15.5%～24.2%。\n5. **前視大小沒有被識別**：J0 前視與 OFF_N 拆解都建立在只用 2026-09 一期估計、再隨機套用的 p／q 模型上，研究員自己的 it6 結果與這個模型互相矛盾。\n6. **機制應改為未支持**：原題的報酬無法測試；描述層的證據有新聞涵蓋的選擇偏誤；「早一天換股較好」檢定力不足。機制應寫成「未支持／無法判斷」，不是「部分支持」。\n\n**建議**：\n- 換股日定義改用 J+1。\n- SOP 的預期改寫成區間，並加註以下幾點：\n  - 在 J+1 定義下，主策略的 t 低於 2.3，屬於第 2 級；\n  - 營收是更正後最新值，存活者偏誤也沒補；\n  - 2024～2026 年輸給 0050。\n\n**驗證**：我用研究員的 navs_oos.parquet 重算，ALL／SEL／VAL／HOLD／POST 的年化與 t 和研究員的數字逐位一致；官方日等於 J+1 的月數是 155/245。\n\n**驗證程式**：/home/user/-/research/tw_stock_selection/agents/o_rev_event_trigger/review_stat_o5.py。只讀研究員的檔案，結果只輸出到 stdout，執行方式是 python3 -I -B。研究員的檔案都沒有修改。"
 },
 {
  "final": "可用",
  "issues": [
   "成本敏感度太高。一般成本（單邊滑價 0.10%）下全期 t 是 2.26，已經低於 2.3。單邊滑價每多 0.1%，t 約掉 0.35；到 0.175% 時 t 降到 2.0。壓力成本下 ALL t 1.69、日曆月 t 1.61，HOLD t −0.05（51.6% 對 0050 57.7%）。周轉是每月單邊 62.9%，一般成本下每年拖累約 5pp。",
   "實盤預期 20.0% 只用了一般成本，對 30 萬帳戶偏樂觀。這類帳戶 77.7% 的名額只能買零股，2020-10-26 之後更是 89.5%。盤中零股 1 分鐘撮合一次，買賣價差和溢價都明顯大於整股，但沒有測。若零股的實際單邊滑價是 0.2% 以上，t < 2.0。改用壓力成本 CAGR 22.07% 套同一個公式，預期約 18.4%；單邊 0.2% 時約 19.2%。SOP 應分帳戶規模寫預期，不要只寫單一的 20.0%。",
   "2024～2026 年在一般成本下就輸 0050：SOP 口徑 52.8% 對 58.0%，HOLD t 0.10；壓力成本下 t −0.05。最近 2.8 年扣成本後沒有超額。",
   "容量只報了買進腿，賣出腿更差。以 1,000 萬帳戶每檔 100 萬計，占成交日全日成交值：買進腿中位數 0.30%，超過 1% 的占 20.9%，超過 5% 的占 1.8%；賣出腿中位數 0.50%，超過 1% 的占 33.4%，超過 2% 的占 18.2%，超過 5% 的占 4.9%（2006～2018 年是 6.5%）。開盤集合競價只占全日成交值的一小部分，這些名額要在開盤一次成交會有衝擊成本，實際上只能盤中分批或用條件單。FINDINGS 只寫『新買名額 27.6% 超過 60 日均成交值的 1%』，低估了賣出端。",
   "處置股：賣出腿有 1.97%（每 seed 約 30 腿／20 年）的成交日落在處置期間。這時要預收券，而且是分盤撮合（5 或 20 分鐘一次），開盤價拿不到。回測仍用開盤價成交，SOP 也沒寫怎麼處理。影響小，但屬於規格缺口。買進腿是 0，處置遮罩有效。",
   "開盤漲停會跳過並遞補，每個 seed 245 個月內有 8 次，約每 2.5 年 1 次。這一步要在 9:00 看完開盤結果才能決定遞補股，遞補股實際上是在開盤後成交，不是回測假設的開盤價。頻率很低，影響可以忽略，但 SOP 應該寫明這是唯一需要盯開盤的情況。",
   "換股日只差一天，t 就跳動很大：j-scan 的 J+1 是 2.26，J+2 1.91，J+3 2.29，J+4 1.73。散戶如果漏掉成交日、延後一天，結果不穩。J+1 的 2.26 只是 j 分布（中位數 1.68）中偏好的一個點。",
   "不利面查不到的部分（如實記錄）：開盤跌停賣不掉的腿只有 0.20%，延到下一個非跌停開盤賣出平均反而賺 4.65%（每月 +0.006pp），所以回測在這點是保守的。全期改用收盤成交，ALL t 2.34，配對 +0.03pp／月（t 0.42）。2020-10-26 前改用收盤成交（模擬 30 萬帳戶的盤後零股），t 2.19，−0.014pp／月（t −0.27）。改用（開＋收）/2 成交，t 2.27，−0.002pp／月。成交時點對 J+1 不敏感。J+1 的訊號在期限日之後一天收盤才算，資料時點沒有前視，也不需要盤中盯盤。"
  ],
  "note": "驗證程式：/home/user/-/research/tw_stock_selection/agents/o_rev_event_trigger/review_exec_o5.py（python3 -I -B，唯讀，只輸出到 stdout）。我用 10 個 seed 重現 J+1 AB：ALL 25.15%，t 2.263，日曆月 t 2.122，和研究員 40 個 seed 的 25.12%／2.26 一致。\n\n從成本與可成交的角度，J+1 這個定義本身可以執行：\n- 沒有前視，收盤後算訊號、隔天開盤下單，不需要盤中盯盤。\n- 漲跌停、處置、改成收盤或盤中成交，對結果的影響都在 ±0.03pp／月以內。\n- 周轉、委託筆數、容量都和官方混合日相同。\n所以「改用 J+1 定義」成立。\n\n但「實盤預期約 20.0%」只套用一般成本，對小帳戶偏樂觀：\n- 一般成本下 t 已經低於 2.3，單邊滑價 0.175% 時就降到 2.0。\n- 30 萬帳戶大多只能買零股，零股價差沒有測。\n- 2024～2026 年扣成本後輸 0050。\n\n建議 SOP 分帳戶規模寫預期：小帳戶 18～19%。1,000 萬帳戶約三分之一的賣出名額、約五分之一的買進名額，超過成交日全日成交值的 1%，要寫明盤中分批下單，以及處置股賣出要預收券。"
 }
]

## 獨立複製

{
 "reproduced": true,
 "my_numbers": "I rebuilt this independently in /home/user/-/research/tw_stock_selection/agents/o_rev_event_trigger/replicate_o5/ (rcore.py, s1–s9, all run with python3 -I -B). I did not read the researcher's fx_core.py or any of the other researcher .py files. The only shared engine I imported is v4lib.V4 (with its own plan and sim). I rewrote everything else myself: deadline D_m, the J days, day-mode unavail, the AB 40-seed averaging, and all the statistics.\n\nEngine check: running V4 on the official days reproduces navs_v4 'V4|normal' and 'AB_V4|normal' exactly (max relative difference 0). AB_V4 comes out at 27.00%, t 2.512, MDD -63.5%, and from 2023-12-29 (SOP basis) 56.74% vs 59.23%.\n\nGate:\n- Pool stock-months: 25,260 (exact match).\n- Loose coverage: 8.86% (reported 8.85%).\n- Exact to the day, my regex: 1.78–1.80% (reported 1.94%).\n- Source-gap upper bound, depending on definition: 74.1–76.3% (reported 73.1%). All are below 80%, so the gate fails.\n\nExplore:\n- Official days: J+1 155, J0 88, plus 2 special months (2019-02, 2023-04), exact match.\n- Strict-timestamp announcements falling on the deadline day: 18.0% (reported 19.0%).\n- y3 percentile: announced ≥5 days early 0.590 vs on the deadline 0.479. By month the gap is +0.118 (t 2.32, 33 months); reported +0.124 (t 2.54, 41 months).\n\nJ+1 with AB, normal cost:\n- ALL: 25.115%, 0050 14.816%, MDD -62.18%, t 2.257, calendar-month t 2.117, mean 0.837pp/month. Against equal weight t 3.522, with equal weight at 10.964%.\n- SEL 16.64% / t 2.25. VAL 33.15% (0050 17.60%) / t 1.30. HOLD 55.52% vs 57.72% (t 0.10).\n- S1 17.19% / t 1.94, S2 t 0.93, S3 t 1.07.\n- Stress cost: 22.06% / t 1.69, HOLD 51.58% vs 57.72%.\n- Slippage: at 0.10% the t is 2.257; t reaches 2.0 at 0.175%.\n- Per seed: t 2.239–2.274 (median 2.247), 40/40 below 2.3, none below 2.0. With the best month dropped the median is 2.07.\n- Endpoints: drop 2006 2.02; start 2008 1.85; drop last month 2.13; drop best 1/3/5/10 months 2.08/1.75/1.43/0.65; drop best year (2021) 1.78; drop best 3 years 1.16.\n- Bonferroni p 0.12. Turnover 62.85% (official 62.12%). Years behind 0050: 2008, 2016, 2019, 2022, 2024.\n- Expected live return 19.97%.\n- Top contributor: 3443 創意 with 3.6% share, removing it costs -0.99pp. For 2024–2026 it is 2059 川湖 with 12.4%, removing it costs -8.04pp.\n\nLook-ahead (non-AB minus AB, calendar months): J+1 is 0.000 in SEL and 0.006 over ALL. J0 is +0.122 in SEL (t 2.37) and +0.128 over ALL (t 2.65).\n\nComparison versions:\n- J0 non-AB: 26.815% / t 2.438.\n- J0 AB: 25.09% / t 2.14, SEL t 2.13.\n- J0 worst: SEL t 1.33.\n- J0N AB (q = 6/1821): 26.72% / t 2.422, SEL t 2.415, 2024–2026 60.61%.\n- Official days with my own day-mode AB: SEL t 2.376 (reported 2.385).\n\nPaired differences (calendar months, pp/month):\n- J+1 − official: −0.117 (t −1.16, MDE 0.28).\n- J0 − J+1: −0.023 (t −0.13, MDE 0.51).\n- J0N − J+1: +0.098 (t 0.53, MDE 0.51); in SEL +0.143 (t 0.77). This is below |t| 2, so the selection rule picks J+1, same as the researcher.\n- J0N − J0: +0.121 (t 2.58).\n\nj ≥ 0 grid (J0 to J+12, AB, 10 seeds):\n- t values in order: 2.06, 2.26, 1.92, 2.29, 1.73, 1.14, 1.94, 1.63, 1.30, 1.06, 1.11, 0.92, 0.95.\n- Median 1.63 (reported 1.68), p10 0.97 (reported 1.04), max 2.29 at J+3 (exact match).\n- After 2019: J0–J4 average t 1.15 (reported 1.16); J8–J12 average −0.31 (reported −0.33).",
 "diff_explained": "Most J+1 numbers match to the second decimal: CAGR, t, calendar-month t, MDD, equal-weight comparison, every segment, stress cost, slippage breakeven, endpoints, Bonferroni, turnover, years behind 0050 and top contributors. J0N AB and the J0N − J+1 pair (t 0.77) also match. So the version choice and the verdict both reproduce: J+1 is selected, its AB t of 2.26 is at or below 2.412, and expected live return is revised down to 19.97%.\n\nSmall differences and their likely causes:\n1. J0 AB: mine is 25.09% / t 2.14 against the reported 25.44% / t 2.20. This also makes J0 look-ahead, J0 − J+1 and J0N − J0 each about 0.02–0.03pp/month larger in absolute terms than reported. J+1 has almost no randomness, so it is unaffected. J0 is the only version where a large share of months (88) depend on p-sampling. My generator is default_rng([seed, yyyymmdd, tag]) with a tag I picked myself; the researcher's tag and draw order are unknown. Per-seed differences are amplified in the 0.5/0.5 percentile ranking. In my own 10-seed grid J0 came out at t 2.06, which shows J0 itself moves about ±0.08 with the random draws. The deadline-day handling for the 2026 INS15 group (D15) may also differ slightly. Direction and conclusions are unchanged: J0 has look-ahead of t > 2, and J0N vs J0 is t > 2.\n2. Per-seed t range: mine is 2.239–2.274, reported 2.225–2.274. The cause is the same random-generator difference (only 2020-02 and the 2026 INS15 months are random).\n3. Gate exact-to-day coverage: mine is 1.78% vs 1.94%, and the upper bound is 74–76% vs 73.1%. I recomputed from o_v4's raw_news_mentions.parquet with my own regex and month definitions; the researcher used their own ann_news.csv and a source-gap definition based on file dates. The loose coverage of 8.86% matches. Every version is far below the 80% threshold, so the gate result (fails) is the same.\n4. Explore deadline-day share: 18.0% vs 19.0%. The early-vs-deadline percentile gap is +0.118 (t 2.32, 33 months) vs +0.124 (t 2.54, 41 months). The differences come from deduplicating strict timestamps and from computing y3 percentiles at the official day; direction and size are consistent.\n5. j-grid median and p10 are about 0.05 lower than reported (10 seeds, same random-generator cause). The maximum (J+3, 2.29) and the post-2019 averages match.\n\nNot verified:\n- POST (40.44% vs 30.04%), because the segment definition is unknown. My S3 from 2021-03 is 40.26% vs 27.49%, t 1.07.\n- OFF_N, execution capacity for the 300k and 10M accounts, industry shares, and the 2026 ann_date overlap.\n- The proxy-block log (I did not test it)."
}

**最終判定（審查後）：待觀察**
