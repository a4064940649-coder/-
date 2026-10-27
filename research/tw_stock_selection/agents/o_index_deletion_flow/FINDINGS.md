# FINDINGS：被規則強迫的指數賣壓：刪除股的收盤承接，與主策略遇到納入時提前賣出（o_index_deletion_flow）

資料夾：/home/user/-/research/tw_stock_selection/agents/o_index_deletion_flow/
（子代理不能寫 .md；這份全文請主研究員存成 FINDINGS.md）

## 結論

- **刪除股收盤承接：不可用。** 結果觸發失敗條件 P5：高賣壓組在 2019～2026 樣本外淨報酬 ≤ 0。
  - 挑選期 2010-11～2018：高賣壓組每筆淨 **+2.96%**，t **2.71**。
  - 驗證期 2019～2023：**−2.54%**（n 8）。
  - 保留期 2025-09～2026-10：**−3.25%**（n 8）。
  - 合併 2019～2026：**−2.89%**，t −1.62，中位數 −2.24%。
  - 劑量反應在樣本外反轉：高減低 −2.78%（t −1.58）。
- **機制：部分支持。**
  - 2010～2018 看得到「被動賣壓 → 收盤被壓低 → 隔日回補」，而且賣壓越大，E 日成交越集中、價格壓得越低：成交倍數 4.5／7.9／13.2 倍，E 日異常 −0.65％／−0.89％／−2.42％。
  - 2019 年以後，E 日收盤的價格壓力消失。MSCI 標準刪除的 E 日異常從 −2.31% 變成 +0.17%，但 E 日成交仍是平常的 17.5 倍。
  - 也就是說，被動資金仍在收盤成交，價格卻不再讓步：有人把收盤承接的利潤搶光了。價格反應提前到公告後、生效前（刪除股前 10 日 −2.5%）。
- **P4：不加入 SOP。**
  - 持股遇到納入，生效後到下個換股日比同期其他持股少 **4.67%**，但 t 只有 **−1.71**（門檻是 −2）。2006～2015 只有 2 個事件月。
  - 「生效日收盤賣出」的改版，年化從 27.08% 變成 26.94%。
  - 事後看到：只看 MSCI 標準／小型納入的話，配對差是 −7.4%／−8.2%（t −3.4／−3.0），臺灣50 納入則是 +1.4%。這要重新事前登記、紙上追蹤，不能直接用。

## 1. 研究問題與對手

- **問題：** 指數刪除時，被動資金必須在生效日收盤賣出，不看價格。散戶能不能在收盤集合競價用同一個價格承接，等價格回補後獲利？主策略常常買到 MSCI 的納入候選股，抱著跨過生效日會不會吃虧？
- **贏家：** 指數套利者與流動性提供者，包括外資自營、量化、券商自營。他們提前建部位，再賣給被動資金。
- **對手：** 被規則綁住、不看價格的被動資金。
- **沒被套利掉的理由（事前假說）：** 收盤競價的瞬間需求太大，造市者需要補償。
- **檢驗結果：** 2019 年前成立，之後補償消失。

## 2. 資料

### 名單蒐集與凍結
- 名單在看任何報酬之前凍結：events_v2.csv，sha256 b19b78f2…，2026-10-27T17:57Z。
- WebFetch／curl 都被擋，主要靠 GitHub 上的新聞鏡像，用部分 clone 取得，存在 raw/（視為不可信）：
  - agora0/news：中央社 2021～2024。
  - aRthur08701244/Intern-in-CTBC：鉅亨／MoneyDJ 新聞 2013～2021。
  - nickchengtw/stock-llm-neuralprophet：鉅亨 2022～2025。
  - bofrankbo/LLM-Predict-Stock：50 檔個股新聞 2004～2013，只找到 2010-11、2011-11、2012-11 的標準指數名單。
- 2025-02 以後用 WebSearch 摘要補（ctee、udn、technews、newtalk 等）。

### 人工抄錄
- 檔案：lists_transcribed.tsv，每列附來源 ID 或網址。
- 涵蓋 **44 次 MSCI 審核**（2010-11～2026-08）。
- 2008～2010-08 的審核，以及 2012-05、2014-11 小型刪除 45 檔等名單找不到。2015-11 小型刪除只有 20／48 檔。
- **富時全球指數的台股增刪：** 台灣媒體與 GitHub 幾乎沒有報導，**0 筆**。
- 高股息 ETF：沿用 etf_rebalance 的 306 筆。另外新增 00713、00918、00929、00940 在 2025-11～2026-06 的 7 次審核（hd_etf_transcribed.tsv）；00939、00934、00936 查不到完整名單。

### 名稱對應
- 名稱→代碼：namemap.py，用 stock_list、tw_revenue 歷史簡稱，加上人工別名，並交叉驗證。
- 只有 1 個名稱對不上（ITH-KY）。

### 資料關卡（通過）
- 排除 src 3 後，有效 MSCI 刪除 **321 筆**；生效日都寫在來源裡。
- 另外用成交值查證：33 個審核生效日的事件股，E 日成交值都是平常的 4.9～28.9 倍（群組中位數）。生效日正確。

### 價格與已知限制
- 價格用面板 adj／close_raw，E 在 src 3 期間的事件全部排除。
- 2019～2023 面板只有存活股。
- 2024 年 MSCI 的 4 次審核全在 src 3 期間，沒有用到。

## 3. 探索（只用挑選期，230 筆可交易事件）

**CAR 對上市等權（%）**

| 類別 | n | E−10～−1 | E 日 | E+1 | E+1～+5 | E+1～+20 |
|---|---|---|---|---|---|---|
| MSCI 標準刪除 | 46 | −3.04 | **−2.31**（t −6.2） | +1.27 | +0.03 | +1.21 |
| MSCI 小型刪除 | 69 | +1.92 | −0.72（t −2.0） | +1.32 | +2.29 | +2.40 |
| MSCI 標準納入 | 30 | +4.45 | +1.53（t 2.4） | −1.45 | **−4.70** | −5.38 |
| MSCI 小型納入 | 73 | +1.54 | +0.18 | — | −2.07 | −1.53 |
| 臺灣50 刪除 | 6 | −0.71 | −0.55 | — | +1.58 | +5.65 |

**賣壓強度的定義**
- pressure ＝ 類別典型被動賣出額 K ÷ 20 日均成交值。
- 理由：刪除與否由流通市值排名決定，同類刪除股的市值集中在門檻附近，所以同一類裡被動賣出額大致固定。
- K 是粗估：MSCI 標準 5 億、小型 0.5 億、臺灣50／高股息 3 億。
- 我們沒有歷史 AUM 和權重，做不出逐筆的被動金額，所以用「類別 × 流動性」近似。

**劑量反應（挑選期刪除事件三分位：低／中／高）**
- E 日成交倍數中位數：4.5／7.9／13.2 倍。
- E 日異常：−0.65／−0.89／−2.42%。
- 20 日毛超額：+0.84／+2.82／+2.45%。
- Spearman 相關：pressure 對成交倍數 0.48，對 E 日異常 −0.26。
- E 日下跌的事件之後有回補：10 日 +1.86%，t 3.28；E 日沒跌的只有 +0.30%。

## 4. 模型迭代（只用挑選期，8 版，全部記在 iterations.json）

選擇規則（事前寫死）：高賣壓組淨 t ≥ 1.5，而且高減低 > 0；多版通過時取 t 最大的。

| 版本 | 進場 | 持有 | 切點 | 高組淨% | 中位% | t | 低組淨% | 高減低% | 通過 |
|---|---|---|---|---|---|---|---|---|---|
| V1 | E 收盤 | 5 | 三分 | 1.02 | 0.88 | 1.70 | −0.11 | 1.13 | ✓ |
| V2 | E 收盤 | 10 | 三分 | 0.99 | 0.55 | 1.37 | −0.59 | 1.58 | |
| V3 | E 收盤 | 20 | 三分 | 1.88 | 0.13 | 1.94 | 0.01 | 1.87 | ✓ |
| V4 | E 收盤 | 5 | 二分 | 1.58 | 0.98 | 2.05 | 0.17 | 1.41 | ✓ |
| V5 | E 收盤 | 10 | 二分 | 1.63 | 0.63 | 2.18 | 0.24 | 1.40 | ✓ |
| **V6** | **E 收盤** | **20** | **二分** | **2.96** | 0.38 | **2.71** | 0.03 | 2.93 | ✓ 選中 |
| V7 | 隔日開盤 | 5 | 三分 | 0.57 | 0.42 | 1.21 | −1.01 | 1.57 | |
| V8 | 隔日開盤 | 20 | 三分 | 1.42 | −0.02 | 1.67 | −0.65 | 2.07 | ✓ |

- **多重檢定：** 共 8 版，6 版通過。探索階段另外看過 6 組「持有期 × 進場方式」的描述統計。
- **讀法：** V6 的中位數只有 +0.38%，平均值靠右尾撐著。

## 5. 凍結規格

見 frozen_spec 欄位。PREREG sha256 9dd08c24…，2026-10-27T18:01Z，寫在樣本外計算之前。oos.py 會先斷言 PREREG 已存在才計算。

## 6. 樣本外結果（V6，淨 0.5%）

| 期間 | 高組 n | 淨平均 | 中位數 | t | 剔除前 5% | 低組淨 | 高減低 |
|---|---|---|---|---|---|---|---|
| 挑選 2010～2018 | 61 | +2.96% | +0.38% | 2.71 | +1.77% | +0.03% | +2.93%（t 2.08） |
| 驗證 2019～2023 | 8 | **−2.54%** | −2.24% | −1.14 | −3.31% | −0.01% | −2.53% |
| 保留 2025-09～2026-10 | 8 | **−3.25%** | −2.97% | — | −4.08% | −0.17% | −3.08% |
| 合併 2019～2026 | 16 | **−2.89%** | −2.24% | −1.62 | −3.27% | −0.12% | −2.78%（t −1.58） |

- **其他口徑：**
  - 壓力成本 1%：−3.39%。
  - 對 0050：−5.90%。
  - 分兩段：2019～2020 −0.73%（n 2），2021 以後 −3.20%（n 14）。
- **判定條件：** (i)(ii) 不成立；(iii) 兩段都是負的；(iv)(v) 見第 7 節。**判定：不可用。**
- **部位比較：**

| 期間 | 刪除承接 | 0050 | 主策略 EXFIN n10 | 同池等權 | 最大回撤 | 對 0050 t | 對主策略 t |
|---|---|---|---|---|---|---|---|
| 2010-11～2018 | 8.63% | 6.50% | 18.56% | 1.57% | −24.9% | 1.60 | −1.73 |
| 2019～2026-10 | 29.09% | 30.62% | 46.73% | 23.16% | −34.5% | −1.67 | −1.38 |

- 2019～2026 壓力成本下年化 28.96%。
- 平均只用 0.36 槽，所以部位大多時間等於 0050。

## 7. 穩健性與配置

- **分年（高組淨）：** 2013 年 +7.45%，2018 年 +7.64%；2014 年 −1.12%。2019 年後每一年都是負的：2020 −0.73、2021 −3.09、2022 −3.20、2025 −3.25。
- **合併窗的其他切法：**
  - 去掉貢獻最大的 1 檔（1515）：−3.27%，t −2.11；去掉前 3 檔：−4.15%。
  - 只看 MSCI：−2.31%；只看 ETF：−3.48%。
  - 改隔日開盤進場：−2.87%；持有 5 日：−0.66%；持有 10 日：−1.39%。
  - 分產業：沒有任何產業有 5 筆以上，無法拆。
- **P1（全部刪除事件）：**
  - 挑選期：20 日淨 +1.53%（t 2.73，中位數 +0.18%）；5 日 +0.89%（中位數 +0.58%）。
  - 樣本外：20 日 −0.29%（中位數 −1.99%）；5 日 +0.18%（中位數 −0.26%）。
- **P3（樣本外反轉）：**
  - 20 日淨：小型刪除 −1.59%（t −2.49），標準刪除 −2.97%（t −2.2）。
  - 臺灣50 刪除：20 日 +1.23%；5 日 +1.37%（t 2.58）。
  - 低成交值組 −1.54%（t −2.29），高成交值組 +0.98%。
  - 結果和「流動性越差，效果越大」的預測相反。
- **(iv) 相關性：** 月報酬相關對主策略 0.48、對處置部位 0.80，主要是共同的 0050 曝險；超額報酬的相關 −0.004。
- **(v) 配置：**
  - 2021-10～2026-10：主策略 70%＋處置 20%＋刪除 10% 年化 38.81%、t 0.86、月頻回撤 −28.6%；現行 80／20 是 39.43%、t 0.84、−28.8%。報酬變低，差異可忽略。
  - 2010-12～2026-10：主策略 90%＋刪除 10% 年化 30.28%、t 2.22；主策略 100% 是 31.44%、t 2.21。

## 8. P4：主策略持股遇到指數納入（2006～2026 全期，規則事前寫死）

- **事件數：** 2010～2026 只有 20 次（主檢定排除 src 3 後 14 次），平均每年約 1 次，不是原本估計的 4～8 次。
- **配對差（E 收盤 → 下個換股日開盤，減去其他持股）：**
  - 月分群 **−4.67%**，t **−1.71**；合併平均 −1.82%，中位數 −3.20%。
  - 2006～2015：−11.26%（2 個月）；2016～2026：−3.35%（t −1.08）。
  - 三個條件中 t 不夠 → **不採用**。
- **改版：** 「E 收盤賣出、資金放 0050」的年化是 26.94%，原版 27.08%，最大回撤同為 −60.4%（月頻）。
- **事後觀察（不能用來判定）：**
  - MSCI 標準納入 −7.39%（t −3.36，5 個月），MSCI 小型納入 −8.18%（t −2.97，3 個月），臺灣50 納入 +1.38%。
  - 主策略之外的全部 MSCI 標準納入股，在 E 後 5 日的回吐，挑選期是 −4.70%，樣本外是 −4.19%（n 28），兩段一致。

## 9. 機制是否被支持：部分支持

- **成立的部分：**
  - 被動資金確實在 E 日收盤集中成交：所有審核日的成交都是平常的 5～29 倍。
  - 2010～2018 的收盤價確實被壓低，而且有劑量反應，隔日回補約 +1.3%。
- **不成立的部分：**
  - 2019 年後，E 日的價格讓步消失：標準刪除 +0.17%，小型刪除 +0.36%。價格反應提前到公告到生效之間（刪除股前 10 日 −2.5%，納入股 +4～6%）。
  - 對手（套利者）變多、變快以後，收盤承接已經沒有補償。
  - 剩下的 E+1 回補約 +0.9%，扣 0.5% 成本後，被之後 20 日的弱勢延續抵消。
- **固定 K 的設計瑕疵：** 2019 年後成交值中位數變成約 8 倍，用固定 K 算的 pressure 系統性變小，高賣壓組在樣本外只剩 16 筆。
  - 事後改用每年內的相對三分位，最高組 20 日淨 −1.30%（t −2.83），結論不變。

## 10. 散戶怎麼執行

- 不建議執行刪除承接。
- 如果要紙上追蹤「MSCI 標準／小型納入股，主策略持有時在 E 收盤賣出」，需要另開事前登記，從 2026-11 審核開始累積。
  - 流程：MSCI 約在 2、5、8、11 月中旬公告，月底最後交易日收盤生效；台股遇 228、端午等假日會提前。
  - 下單：當天 13:25～13:30 的收盤集合競價掛市價或限價賣出。

## 11. 限制

- 2008～2010-08 沒有名單；富時全球指數沒有資料。
- 部分小型指數名單不完整：2014-11 缺、2015-11 只有部分。
- 2019～2023 面板只有存活股。
- 2024 年的事件全部在 src 3 期間，被排除。
- K_class 是粗估常數。
- 高股息 ETF 新增的審核只有 2025-11 以後的 7 次，而且有 5 日或 8 日的過渡期，被動賣壓會分散。
- 樣本外高賣壓組只有 16 筆、8 個事件月，檢定力低；但點估計方向和挑選期相反，不是只差一點不顯著。
- raw/ 有約 978MB 的新聞 clone，已經放了 raw/.gitignore，請不要提交。

## 12. 程式與產出

- **程式：** build_events.py、namemap.py、freeze_events.py、iflib.py、explore.py、iterate.py、prereg.py、oos.py、posthoc.py、sleeve.py、p4.py、tables.py、compare.py。
- **產出：**
  - events_v2.csv（含來源網址與查證欄）、lists_transcribed.tsv、hd_etf_transcribed.tsv。
  - explore.json、iterations.json、results.json（DATA_FREEZE、PREREG、OOS_events、POSTHOC、SLEEVE、P4、TABLES、COMPARE）。
  - 逐筆事件表：event_returns_all.csv（每筆附來源網址，在 events_v2.csv 的 src 欄）、p4_events.csv。
  - 部位淨值：sleeve_nav.parquet。

## 凍結規格

【凍結規格 V6（PREREG sha256 9dd08c248726fb39340c229566022be91a0340d90216765c67f0fbe7dbce35c9，2026-10-27T18:01:12Z；事件表凍結 events_v2.csv sha256 b19b78f2d36d2a7de29e78d99aa8ced1fed24fc5eae7c309b72cf6a6aa705759，2026-10-27T17:57:11Z，凍結時還沒算任何報酬）】
資料夾：/home/user/-/research/tw_stock_selection/agents/o_index_deletion_flow/
1. 事件：events_v2.csv 裡的四類刪除事件。
   - MSCI 全球標準指數刪除，含轉入小型指數。
   - MSCI 全球小型指數刪除，不含升級到標準指數的。
   - 臺灣50 刪除。
   - 高股息 ETF 刪除：0056、00878、00919、00713、00918、00929、00940。
   - 富時全球指數：0 筆，蒐集不到。
2. 篩選：
   - 必須是上市股，而且在面板內。
   - E−1 的 60 日均成交值 ≥ NT$1,000 萬，收盤 ≥ 10 元；E 日要有成交。
   - E 日收盤漲停（≥ 前收 ×(1+6.5%)；2015-06-01 起為 9.5%）不買。
   - E 落在 2024-02-15～2025-09-23（src 3 是 13:20 近似收盤）的事件整筆排除。
3. 賣壓強度：pressure ＝ K_class ÷ E−1 的 20 日均成交值。
   - K_class（NT$）：MSCI 標準刪除 5 億、臺灣50 刪除 3 億、高股息 ETF 刪除 3 億、MSCI 小型刪除 0.5 億。
   - K 是事前粗估的常數，不隨年份調整。
4. 只交易高賣壓組：pressure ≥ 2.7478，也就是挑選期刪除事件的中位數，固定不變。低賣壓組只用來檢定劑量反應。
5. 進出場：生效日 E 收盤集合競價買進，價格等於官方收盤；持有 20 個交易日，E+20 收盤賣出。
6. 基準與成本：
   - 基準是上市等權，也就是同一個窗口全體上市股買進持有報酬的等權平均。
   - 一般成本來回 0.5%（買 0.1%、賣 0.4%），壓力成本來回 1%。
7. 統計：淨平均、中位數、按事件月分群的 t、剔除前 5% 後的平均、勝率；高減低用同事件月配對差的 t。
8. 部位：
   - 10 槽固定金額，每筆＝部位淨值 ÷ 10；滿槽就跳過；同一天有多筆時，pressure 高的優先。
   - 單筆 ≤ E−1 的 20 日均成交值 × 5%。
   - 閒置資金放 0050，手續費 0.0855%，賣出另加證交稅 0.1%。
9. 期間：
   - 挑選期 2008～2018（實際有名單的是 2010-11 起）。
   - 驗證期 2019～2023。
   - 保留期 2025-09-24～2026-10。
   - 判定窗：驗證＋保留合併 2019～2026。
10. 判定：
   - 可用要 (i)～(v) 全部成立：(i) 高賣壓組淨平均 > 0、t ≥ 2、中位數 > 0、剔除前 5% 後 > 0；(ii) 高減低 > 0 且 t ≥ 1.5；(iii) 2019～2020 和 2021 以後兩段都 > 0；(iv) 和主策略、處置部位的月相關 < 0.3；(v) 主策略 70%＋處置 20%＋刪除 10% 的 t 不低於主策略 80%＋處置 20%，回撤也不加深。
   - 只過 (i)：待觀察。
   - (i) 不成立，或驗證期淨報酬 ≤ 0：不可用。
11. P4（事前寫死，不調參數）：
   - 名單：主策略 EXFIN n10 gapfix 的正式名單，即 v2_core.V2('gapfix', exfin=True).plan(10,'official')。
   - 事件：持有期內遇到 MSCI 標準／小型或臺灣50 的納入生效日 E（e_k ≤ E < e_{k+1}−1）。
   - 配對差：E 收盤到 e_{k+1} 開盤的報酬，減去同窗其他持股的平均。
   - 改版：E 收盤賣出，資金放 0050，額外成本＝權重 × 0.5%。
   - 採用條件：平均 < −1%、t ≤ −2，而且 2006～2015 和 2016～2026 兩段都 < 0。
重做順序：python3 -I -B build_events.py → freeze_events.py → explore.py → iterate.py → prereg.py（只能跑一次）→ oos.py → posthoc.py → sleeve.py → p4.py → tables.py → compare.py。

## 反方審查

[
 {
  "final": "不可用",
  "issues": [
   "判定「不可用」站得住，而且證據比研究員寫的更強。不靠 K_class，直接看樣本外全部可交易的 MSCI 刪除（n 105，15 個事件月），扣 0.5% 後 20 日淨超額的月平均是 −3.52%，月分群 t −2.78。研究員用的高賣壓組只有 16 筆，其中 8 筆是高股息 ETF、只有 2 筆是 MSCI 標準刪除，代表性不足，但結論方向相同。",
   "樣本外高賣壓組不是挑選期的同一群股票。K 是名目固定值，而成交值中位數從 0.53 億漲到 4.43 億（約 8 倍），所以樣本外只有 6.4% 的事件落進高組，挑選期是 51%。合併窗的 −2.89%（t −1.62）裡，2022-05 一個月就貢獻 −11.45%；拿掉這個月，其餘 7 個月平均只剩 −1.18%。FINDINGS 說「不是只差一點不顯著」，說得太滿；P5 仍然觸發，靠的是點估計 ≤ 0，不是檢定力。",
   "挑選期的 +2.96% 很脆弱。這個數字是混合平均，月平均是 +4.07%。拿掉 2013 年，t 從 2.71 掉到 1.82；拿掉 2010、2013、2018 三年，剩 +1.75%（t 1.23，10 個月）。中位數只有 +0.38%，前 5 筆單筆超額有 +17～29%。8 版的 max-T 月區塊 bootstrap，家族 p 0.013，代表挑選期本身通過多重檢定，但它主要靠少數年份撐起來。",
   "機制「部分支持」被高估。E 日異常的 t −6.2 是把同一天的事件當成獨立樣本算出來的；改按審核日分群後，標準刪除是 −1.59%（t −2.74，14 期），小型刪除只有 −0.74%（t −1.21），不顯著。樣本外標準刪除按審核日加權是 −0.41%（t −0.77），不是 +0.17%。兩段的差粗估 t 約 1.5，「2019 年後被套利掉」在統計上沒有成立，只是一種說法。",
   "劑量反應和類別混在一起。K_class 讓 MSCI 標準刪除被機械地排進高賣壓組，挑選期高組有 36 筆標準刪除、23 筆小型刪除。成交倍數 4.5／7.9／13.2 倍，和 E 日異常 −0.65／−0.89／−2.42%，很可能只是「標準刪除對小型刪除」的類別差異，沒有做類別內的檢定。K 的敏感度方面：改成 K 全部相同、小型 0.2 或 1.5 億、標準 15 億，挑選期高組月平均 t 都在 2.05～3.0，所以挑選期的結果不是靠某個特定 K。",
   "生效日對齊只驗證了 MSCI。高股息 ETF 刪除的 E 日成交是平常的 2.1 倍，E 到 E+2 一樣高，成交高峰落在 E 日的比例只有 25%（MSCI 標準是 95%、臺灣50 是 87%）。這些事件的 E 日收盤不是被動資金集中賣出的時點，卻佔了樣本外高組的一半，會污染「E 日收盤承接」這個檢定。",
   "樣本外沒有完全保持「沒看過」。etf_rebalance 組之前已經看過臺灣50／高股息 ETF 增刪在 2019 年後的報酬，並得到「多數為負」的結論，本研究之後才把高股息 ETF 刪除放進名單。這只會讓樣本外偏負，不會推翻「不可用」，但要寫進限制。",
   "存活者偏誤：2019～2023 面板只有存活股；挑選期有 17 筆、樣本外有 4 筆的代碼落在 delisted_unknown 清單，面板外還有上櫃股。偏誤方向應該是讓樣本外報酬偏高，所以「不可用」偏保守、站得住。但這會影響「2019 年後 E 日價格讓步消失」的機制敘述。",
   "P4 不加入 SOP 是對的，但 FINDINGS 的寫法偏重月分群的 −4.67%。逐筆合併平均只有 −1.82%，逐筆 t −0.6。輪流拿掉一個事件月，t 在 −1.29 到 −2.90 之間跳；只有拿掉 2021-06 那個月（航運股以臺灣50 納入名義進場，那個月 +15%）才會過 −2。事後的 MSCI 標準 t −3.36 是 5 個月的子群、小型 t −2.97 是 3 個月的子群，都是從 3 類裡事後挑出來的，不能當證據。",
   "反過來的一個發現：P4 主策略以外的全部 MSCI 標準納入股，E 收盤後 5 日超額按審核日分群，挑選期 −4.97%（t −3.95，14 期），樣本外 −5.16%（t −2.71，15 期）。「納入股生效後回吐」在個股層級比 P4 本身穩，支持紙上追蹤；但不支持直接用到主策略，因為 P4 的窗口是到下個換股日，而且年均只有約 1 次事件。",
   "部位比較的資訊量很低。平均只用 0.36 個槽，淨值約 96% 是 0050。「2019～2026 年化 29.1%、0050 30.6%、對 0050 t −1.67」主要反映部位很小、閒置資金都在 0050，不能拿來佐證策略本身好或不好；對主策略的 t −1.38 也沒有意義。"
  ],
  "note": "我寫了驗證程式 /home/user/-/research/tw_stock_selection/agents/o_index_deletion_flow/review_stat_o1.py（用 python3 -I -B 執行），完整輸出在 /tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/review_stat_o1_out.json，研究員的檔案都沒動。\n\n重現結果：V6 挑選期高組 61 筆、混合平均 +2.96%、月分群 t 2.71；樣本外 16 筆、−2.89%、t −1.62，和研究員的數字一致。\n\n前視偏誤：pressure 用的是 E−1 的 20 日均成交值，切點固定用挑選期的中位數；PREREG 的時間戳早於樣本外計算。我沒有找到前視偏誤。\n\n結論：維持「不可用」，而且因為樣本外全部 MSCI 刪除的 t −2.78，證據更強。但有兩點要下修：\n- **機制**：建議從「部分支持」改成「挑選期弱支持；2019 年後被套利掉，統計上沒有成立」。主要原因是 E 日的 t 值把同一天的事件當成獨立樣本，按審核日分群後明顯變小；劑量反應也和類別混在一起。\n- **挑選期的正報酬**：主要靠 2010、2013、2018 這三年。\n\nP4 維持不採用。月分群 t 沒有過 −2，逐筆 t 只有 −0.6；事後的子群結果不能當證據。"
 },
 {
  "final": "不可用",
  "issues": [
   "成本不是失敗的主因，失敗在成本之前就發生了。樣本外 2019～2026 高賣壓組的毛超額已經是 -2.39%（中位數 -1.74%，月分群 t -1.29，8 個事件月），零成本也是負的；成本 1%／1.5%／2% 時，t 分別是 -1.95／-2.28／-2.60。挑選期毛超額 +3.46%，但中位數只有 +0.88%，損益兩平成本靠右尾撐著。不可用的結論與成本假設無關，研究員的判定站得住。",
   "漲跌停與可成交性對結論沒有影響。高賣壓組在 E 日收盤漲停而被排除的，樣本外 1 筆、挑選期 0 筆。E 日跌停 2 筆（挑選期），跌停時收盤買得到。E+20 出場日跌停或無成交 0 筆。2021-09 以後有處置記錄的期間，刪除股在 E 日處置中的有 0／231 筆，持有期內碰到處置的 1 筆（不在高組），預收款券不是問題。",
   "容量會進一步稀釋這個部位。高賣壓組的 20 日均成交值中位數，樣本外是 4,250 萬、第 10 百分位 1,370 萬。1,000 萬帳戶每槽 100 萬時，樣本外有 41% 的交易被『單筆 ≤ 5%×v20』限額卡住（挑選期 39%）。sleeve.py 預設資本 100 萬，所以資本越大，部位越接近 0050，原本就很小的超額會更少。",
   "FINDINGS 說『所有審核日的成交都是平常的 5～29 倍』，這句話只適用於 MSCI。高股息 ETF 刪除（樣本外 113 筆）E 日成交倍數的中位數只有 2.08 倍。這些 ETF 有 5～8 日的過渡期，被動賣壓不集中在一個收盤，E 收盤承接的前提對這類事件不成立。高組裡 ETF 部分樣本外 -3.48%，混入不同機制的事件。",
   "挑選期正報酬集中在少數月份，『機制部分支持』裡『2010～2018 收盤承接可賺』這一半偏樂觀。V6 高組 16 個事件月中，2010-11 +14.96%、2013-08 +9.58%、2013-11 +10.87%、2016-05 +11.01%、2018-11 +11.32% 撐起平均。去掉 2013 年，t 1.82；只看 2014～2018，+2.62%，t 1.48，未達門檻。E 日價格壓力（MSCI 標準刪除 -2.31%，t -6.2）是穩健的，但它轉成 20 日扣成本報酬並不穩健。",
   "是否需要盤中盯盤：E 收盤承接，以及 P4 的事後版本『E 收盤賣出』，都必須在 13:25～13:30 收盤集合競價時段下單，或靠券商條件單。早上掛限價單可能在盤中先成交，拿不到收盤價。MSCI 一年 4 次，加上臺灣50 與各家高股息 ETF 的生效日，散戶需要固定在該時段操作，FINDINGS 第 10 節沒有講清楚。",
   "P4 事後子群組（只看 MSCI 納入）不應被當成接近可用。2016 年以後只看 MSCI 納入，-6.5%，t -3.6，但只有 6 個事件月。整體 t 只有 -1.71，是因為 2021-06 臺灣50 航運三檔納入（2603、2609、2615，各 +13～17%）把結果拉回來。去掉最差的一個月，t 變 -1.29。改版年化 26.94% 對 27.08%，配對期差 t -0.46，執行面沒有改善。若該股下期仍在名單，『E 收盤賣、換股日買回』會多一次 0.3% 證交稅加手續費；0.5% 的假設大致合理，但沒有空間。",
   "數字小差異：SLEEVE 欄 2019～2026 為 27.78%／0050 29.29%／t -1.83，COMPARE 欄（窗從 2018-12-31 起）為 29.09%／30.62%／t -1.67。FINDINGS 用的是後者，但沒有註明兩者差在起點，方向一致。",
   "2019～2023 面板只有存活股，會讓刪除股報酬偏高。樣本外結果是在偏樂觀的資料下仍然為負，這一點支持不可用。"
  ],
  "note": "我同意研究員的判定：刪除承接不可用，P4 不加入 SOP。從成本與可成交角度看，失敗發生在成本之前：樣本外 16 筆高賣壓組的毛超額已經是 -2.39%，不是被 0.3% 證交稅或滑價吃掉的。漲跌停、處置與預收款券、出場日跌停的影響都接近 0 筆。容量方面，1,000 萬帳戶約 4 成交易會被 5% 均量限額卡住，只會讓部位更接近 0050。執行上需要在 13:25～13:30 收盤競價時段下單或設條件單。機制『部分支持』應再保守一些：E 日價格壓力是真的（t -6.2），但挑選期 20 日扣成本報酬集中在 2013、2010-11、2018-11 等少數月份；只看 2014～2018 是 t 1.48。『成交是平常的 5～29 倍』只對 MSCI 成立，高股息 ETF 只有 2.1 倍。P4 的 MSCI 子群組只有 6 個事件月，是事後挑出來的，只能事前登記後紙上追蹤。驗證程式：/home/user/-/research/tw_stock_selection/agents/o_index_deletion_flow/review_exec_o1.py（python3 -I -B，只讀研究員產出）。"
 }
]

## 獨立複製

{
 "reproduced": true,
 "my_numbers": "I rebuilt the whole pipeline in /home/user/-/research/tw_stock_selection/agents/o_index_deletion_flow/replicate_o1/ without reading any of the researcher's programs. Order: build_events → freeze_events → explore → iterate → prereg (run once) → oos → posthoc → sleeve → p4 → tables → compare, all with python3 -I -B. 140 of 143 key numbers fall within tolerance; the comparison is in compare.json and the flattened figures in results_replica.json.\n\n**Data gate (exact match)**\n- events_v2.csv sha256 matches the frozen value; 1,354 rows, 1,093 in the panel.\n- Recomputing in_panel, src3_period, period and src_E agrees on all 1,354 rows.\n- vol_ratio_E (E-day turnover ÷ median of E−25..E−6) agrees on 1,353 rows.\n- Valid MSCI deletions: 321 (2010-11-30 to 2026-08-31, src 3 excluded); effective dates 100% stated in the source; FTSE 0.\n- 33 review effective dates, group medians 4.9 to 28.9x.\n\n**Exploration (230 events; 46 / 69 / 30 by class, exact)**\n- MSCI standard deletion: E-day −2.31% (t −6.23), turnover 13.7x, E+1 +1.27%, E+1 to E+20 +1.21%.\n- MSCI small-cap deletion: E-day −0.72% (t −2.01), next 5 days +2.26%, 20 days +2.37%.\n- MSCI standard inclusion: 10 days before +4.45%, E-day +1.53%, next 5 days −4.70%.\n- Selling-pressure tertiles: turnover 4.5 / 7.9 / 13.2x; E-day −0.65 / −0.89 / −2.42%; 20-day gross excess +0.82 / +2.96 / +2.45%.\n- Median pressure cut 2.7483.\n\n**V6 selection period**\n- High-pressure group n 61 (16 months), net +2.96%, median +0.37%, t 2.71, trimmed +1.77%.\n- High minus low (month-paired) +4.78% (t 2.08); stress cost +2.46%.\n\n**Out of sample**\n- Validation: n 8, −2.54%, median −2.24%, t −1.14. Holdout: n 8, −3.25%.\n- Combined: n 16 (8 months), −2.90%, median −2.24%, t −1.62, trimmed −3.28%; stress −3.40%; vs 0050 −5.90%.\n- Low group: n 235, −0.12%. High minus low −2.78%, month-paired −3.62% (t −1.58).\n- Two segments: 2019–2020 −0.72% (n 2), 2021 on −3.21% (n 14). Conditions (i), (ii), (iii) all fail, so not usable.\n\n**Sleeve (10 slots, idle cash in 0050)**\n\n| Period | Sleeve | 0050 | Main strategy | Same-pool EW | Max drawdown |\n|---|---|---|---|---|---|\n| 2010-11～2018 | 8.51% | 6.50% | 18.56% | 1.57% | −24.9% |\n| 2019～2026-10-12 | 29.09% | 30.62% | 46.73% | 23.16% | −34.5% |\n\n- 2010-11～2018: t vs 0050 1.48, monthly excess +0.163%.\n- 2019～2026: t vs 0050 −1.82, monthly excess −0.105%, t vs main −1.42; stress-cost CAGR 28.96%.\n- Slots used 0.368 on average; 12.3% of days hold a position.\n- Monthly correlation: with main 0.483, with the disposal sleeve 0.799; excess-return correlation −0.0048.\n- Allocation, 2021-10～2026-10: 70/20/10 39.35%, t 0.81, drawdown −28.6%; 80/20 39.82%, t 0.79, drawdown −28.8%.\n- Allocation, 2010-12～2026-10: 90/10 30.35%, t 2.16; 100% main 31.47%, t 2.16.\n\n**P1 / P3**\n- All deletions, selection period +1.54% (t 2.72, median +0.18%); out of sample −0.30% (median −2.01%).\n- Out of sample by class: MSCI small-cap −1.60% (t −2.50), MSCI standard −2.97% (t −2.19), Taiwan 50 +1.22%, high-dividend ETFs +0.88%.\n- P3 (out-of-sample turnover tertiles): low −1.51% (t −2.20), high +0.93%.\n\n**Post-hoc**\n- After 2019, E-day: standard deletion +0.17%, small-cap deletion +0.36%; turnover still 17.5x and 5.0x.\n- Standard inclusion 10 days before: +6.26%. MSCI deletion 10 days before: −2.17%.\n- 20-day median turnover: 0.53 → 4.43 億.\n\n**P4**\n- 20 events, 14 after excluding src 3. Month-clustered −4.67% (t −1.71); pooled mean −1.82%, median −3.20%.\n- 2006～2015: −11.26% (2 months). 2016～2026: −3.35% (t −1.08).\n- By class: MSCI standard −7.39% (t −3.36), MSCI small-cap −8.18% (t −2.97), Taiwan 50 +1.38%.\n- The 20-event list matches the researcher's p4_events.csv one for one; paired differences agree to 6e-16.\n- Sell-on-inclusion version: CAGR 28.15% → 28.02% (−0.13pp), max drawdown −65.0% unchanged. Not adopted.",
 "diff_explained": "**The three numbers outside tolerance**\n1. **P4 base level.** The researcher reports a main-strategy base of 27.08% CAGR with −60.4% drawdown. Mine is 28.15% / −65.0%, using the shared ia_core.simulate on the EXFIN n10 gapfix official list from 2006-04-11 to 2026-10-12. My result matches main_v2/navs_gapfix.parquet 'EXFIN|n10|normal' with zero difference. I tried other windows, the latest end date, monthly and weekly drawdown, and period-compounding with flat costs, and none reproduces 27.08 / −60.4. The researcher probably used a different engine or window. The number that drives the decision does agree: selling on inclusion changes CAGR by −0.13pp (researcher −0.14pp) and leaves drawdown unchanged, so it is still not adopted.\n2. **Pre-effective 10-day return for deletions after 2019.** Researcher −2.5%, mine −2.17% for MSCI deletions. The frozen spec doesn't define which group this is. The panel-traded standard-deletion group gives exactly −2.50%, and tradable MSCI deletions give −2.42%. This is a definitional difference in a post-hoc descriptive figure and doesn't affect the verdict.\n\n**Conventions I inferred and then confirmed against the numbers**\n- **Liquidity filter:** the 60-day average turnover allows 40 observations, and the exploration sample keeps limit-up events; only trading drops them. That reproduces 230 / 46 / 69 / 30 exactly. With 60 required observations and limit-up removed, I got only 222.\n- **Event-window figures:** \"next 5 days\", \"next 20 days\" and \"10 days before\" are sums of daily abnormal returns. The V6 trade returns are buy-and-hold minus the equal-weight buy-and-hold.\n- **5% trim:** values above the 95th percentile are dropped; this reproduces −3.27/−3.28.\n- **Disposal sleeve:** it is strategy_ensemble DISP_REAL from 2021-10. That gives correlation 0.799 and allocation drawdowns −28.6 / −28.8 exactly. With disposal_v2 TWSE_K10_0050 instead, the correlation is 0.763.\n- **P3 low/high turnover groups:** these are out-of-sample tertiles, not medians.\n\n**Remaining small gaps (all within tolerance)**\n- **2010–2018 sleeve:** 8.51% vs 8.63%, and t 1.48 vs 1.60; probably small differences in slot or cash handling.\n- **Allocation:** CAGR differs by 0.03 to 0.54pp and t by about 0.05, likely from how monthly compounding and the final partial month are annualised. Main 100% is 31.47 vs 31.44.\n- **Middle tertile gross excess:** 2.96 vs 2.82, because events near the tertile boundary land in different groups. The recomputed median is 2.7483 vs 2.7478, so pressure differs very slightly.\n\n**What I could not replicate**\n- **Iteration step:** the frozen spec defines only V6, so I couldn't rebuild the researcher's 8 versions. iterate.py reproduces V6 exactly, plus a small grid of my own as a sensitivity check.\n- **Event table:** it comes from news transcriptions, so I didn't rebuild it from raw sources. I recomputed all derived columns and spot-checked the transcription lists: 1,012 of 1,059 names are found in the table, and the 47 misses are mostly F- (KY) name-mapping issues.\n- **Prereg timing:** event_frame computed the post-2019 returns as one batch before prereg.py ran, but I didn't look at them until after the prereg was written."
}

**最終判定（審查後）：不可用**
