# FINDINGS：論壇大神貼文績效驗證（forum_gurus）

## 一句話結論
**不可用。** 在 PTT Stock 板 2023-01 到 2025-04 的 1,005 篇上市個股 [標的] 文裡，跟著做多文買，扣 0.5% 成本後對上市等權沒有超額報酬（20 日 +0.45%，t_month 0.30；中位數 −1.42%，命中率 42%）。推文熱度最高的文和事前挑出的「大神」也都沒有穩定優勢。「反向指標」說法兩個半期方向相反。唯一值得留意的訊號是**空方文的標的之後偏弱**（20 日 −2.0%，t 2.05），但樣本只有 125 篇，兩個半期各自都不顯著，列為待觀察。Mobile01、Dcard 和財經 YouTuber 找不到可用的歷史資料，無法測試。

## 判定總表
| 規則 | 判定 | 關鍵數字（扣 0.5% 成本，t 依月份分群） |
|---|---|---|
| R1 跟單做多所有標的文 | **不可用** | 5 日 +0.07%（t 0.40）；20 日 +0.45%（t 0.30）。前半 20 日 −0.21%（t −0.88），後半 +1.26%（t 0.79）。成本 1% 時 20 日 −0.05%（t −0.49） |
| R1 空方文（做多的反向，不實際放空） | **待觀察** | 個股 20 日異常 −2.01%（t −2.05）；對發文前 20 日漲幅同十分位的股票是 −2.40%（t −2.52）。前半 t −0.52、後半 t −1.09，各自不顯著；n=125 |
| R2 熱度前 10% 做多文 | **不可用** | 5 日 +0.79%（t 1.80）；20 日 −1.19%（t 0.42）。前半 20 日 −2.62%，後半 +1.05%（t 0.95） |
| R3「大神」（前半挑、後半驗） | **無法測試** | 前半發文 ≥ 10 篇的作者只有 3 位；選出的作者後半只有 4 篇（做多 3 篇） |
| R4 反向指標（熱度前 5% 做多文） | **不可用** | 20 日 −0.81%（t 0.27）。前半 −4.33%（t −0.40），後半反而 +2.71%（t 1.48）、5 日 +3.39%（t 2.58），不支持「越熱越弱」 |
| 和主策略（52 週 × 營收）重疊 | 描述 | 11.3% 的做多文標的在主策略 M2 的 30 檔持股裡。重疊的 20 日 +1.43%（t 0.23），不重疊的 +0.33%（t 0.36），沒有差別 |
| Mobile01／Dcard／YouTuber | **無法測試** | 沒有公開歷史資料；股癌整理站只有 18 集、29 檔，且全是非台股代號 |

## 資料：找過的地方
- **用到的資料：GitHub `a607ernie/Ptt-Stock-Data`。** GitHub Actions 每 12 小時爬一次 PTT Stock 板的 [標的]／Re:[標的] 文，並 commit 一個快照。我把 git 歷史裡 2,211 個快照（2023-01-09 ～ 2025-05-03）依文章 ID 合併，得到 3,772 篇 [標的] 原文。欄位有作者、發文時間、內文和含時間戳的推文。只讀 git 物件，沒有執行該 repo 的程式。
  - 2023-02 ～ 2024-11 每月 100～250 篇。2024-12 ～ 2025-04 快照變稀疏，每月只剩 21～94 篇，有漏抓。
- **找過但不能用：**
  - `jameschen108/tw-stock-sentiment`：有完整的預先登記研究，但原始資料放在作者本機，repo 內沒有。
  - `soniccol/PTTData`：資料在外部 MySQL，這個環境連不到。
  - `kuo23/PTT-Stock`：只有 notebook，沒有資料。
  - `Arthur0234/taiwan-limitup-research`：只有規劃 issue，沒有資料。
  - `ryvn-dev/gooaye-notes`：股癌只有 18 集（2026-07～09），29 檔全是非台股代號。
  - ptt.cc 本身 DNS 解析失敗。WebSearch 在 Kaggle 和 HuggingFace 上沒找到 PTT Stock 資料集。
  - GitHub 的搜尋 API 被 session 限制擋下，只能靠 WebSearch 找 repo。

## 事前固定的規則（先寫進 run_forum.py 開頭的說明，再跑結果）
- **事件**：
  - 只取 [標的] 原文，不含 Re:。
  - 方向看內文「分類」欄：只有「多」或只有「空」才算；範本原樣、討論、心得、請益、多空並列都排除。
  - 標的必須恰好一檔上市股票。判定順序：標題代號 → 標題公司簡稱 → 內文「標的：」那一行。標題含 ETF、美股、指數就排除；先刪掉範本裡的「(例 2330.TW 台積電)」。
  - 同作者 5 個交易日內同檔、同方向只留第一篇。
- **時間**：PTT 時間就是台灣時間。
  - 交易日 09:00 前發文 → 當日開盤進場。
  - 盤中、盤後、假日發文 → 下一交易日開盤進場（主規則）。
  - 敏感度：盤中發文改用當日收盤進場，結果相同（20 日 +0.37%，t 0.14）。
  - 進場日距發文超過 7 天的剔除（面板缺 2024-01-02～02-14）。
- **報酬**：還原開盤到 h 日後還原開盤，h = 5、20。
  - 對手是同日流動性池（60 日均值 ≥ 1,000 萬、股價 ≥ 10）的等權報酬。
  - 開盤漲停視為買不到（8 篇）；股票本身沒過流動性門檻的剔除（110 篇）。
- **挑選期／驗證期**：發文早於 2024-03-01 為前半（挑選），之後為後半（驗證）。
- **熱度**：用內文推文的時間戳，算到進場時點為止的「推」數，避免偷看進場後的推文。門檻用前半的分位數固定：前 10% 為 65 推，前 5% 為 78 推。
- 事件數一路下來：3,772 篇 → 方向明確的單一台股 1,650 篇（其中上櫃 442 篇，面板沒有上櫃日線，只能排除）→ 上市 1,208 篇 → 最終 1,005 篇（做多 880、做空 125）。
- 共 518 位作者，前 10 位占 15.1%。抽查 41 篇靠簡稱或內文判定標的的文章，約 1 篇錯（「美國科技過熱-空」被對到 2406）；股票期貨文對到現股。

## R1 細節
- **做多文分布**：
  - 平均略正、中位數為負，屬於彩券型分布：少數 AI 題材股 20 日超額 +40～80%，例如 3231、2382、1519、3535。
  - 1% 縮尾後 20 日 +0.35%（t 0.17）。
- **逐年（20 日淨額）**：2023 −0.29%（t −0.46）；2024 +1.36%（t 1.68）；2025 年 1～4 月 +0.52%（t −0.61）。
- **作者在追漲**：做多文發文前 20 日原始漲幅平均 +8.5%（中位數 +5.5%），空方文 +11.0%。
  - 對發文前漲幅同十分位的股票比，做多文 20 日 −0.03%（t −0.33）：做多文的表現和「同樣漲過一段的股票」一樣，沒有額外資訊。
- **空方文**：20 日個股異常 −2.01%，命中率（股價真的相對偏弱）69%。
  - 但前半、後半各自不顯著，而且這項是報告項目，不是事先登記的假設。
  - 可以當成「別買 PTT 有人發空方文的股票」的濾網，繼續觀察，不建議放空。
- **熱度五分位（做多，20 日淨額）**：由低到高 −1.19%、+1.12%、−1.12%、+2.62%、+0.82%，沒有單調關係。

## 外部實證（文獻，附證據等級）
1. **jameschen108/tw-stock-sentiment（GitHub 畢業專題，有預先登記；證據等級中高）**
   - 資料：PTT [標的] 文 2016–2023 共 10,705 篇，含上櫃與已刪文。
   - 跟著看多文買，扣成本後 5 日 −0.83%、20 日 −1.51%（對大盤）；和流動性、過去報酬相近的股票比只落後 0.11%。
   - 看多和看空的差距（5 日 +0.41%）幾乎全部來自收盤鎖漲跌停、實際買不到的 4.7% 文章；排除後 t 只有 0.47。
   - 「作者過去準不準」、推噓比等 14 個特徵，FDR 校正後都不顯著。
   - 2024/01–2025/01 樣本外檢定 H1、H2 都未通過。
   - 和本研究結論一致；本研究用等權對手，所以水準略高。
   - https://github.com/jameschen108/tw-stock-sentiment
2. **Kakhbod, Kazempour, Livdan & Schürhoff《Finfluencers》（SFI 工作論文 23-30、AEA 2025；證據等級高）**
   - 樣本：StockTwits 約 2.9 萬人。28% 有技巧、16% 無技巧、56% 是「反技巧」。
   - 跟單平均每月落後市場 0.35%。越不準的人追蹤者越多。反向操作在樣本外每月 +1.2%（美股，2013–2017）。
   - https://www.sfi.ch/fr/publications/n-23-30-finfluencers
3. **Bradley, Hanousek, Jame & Xiao, RFS 37(5) 2024（WallStreetBets；證據等級高）**
   - GameStop 軋空前，DD 分析文能預測報酬；之後板上風氣轉向炒作，預測力消失。
   - https://ideas.repec.org/a/oup/rfinst/v37y2024i5p1409-1459..html
4. **暨南大學碩士論文（2008）〈Web2.0 社群資訊品質之評估－以PTT股票版為例〉（證據等級低）**
   - PTT 推薦股 5、10 日報酬優於 24、50 日；作者的背景資料無法辨別建議好壞。只看摘要，未查成本處理。
   - https://www.airitilibrary.com/Article/Detail/U0020-1308200821542800
5. **政大林彥丞（2016）碩士論文（證據等級低中）**
   - 每週看多／看空發文數與回文數，對大盤和個股報酬沒有一致的預測力。
   - 據 WebSearch 摘要轉述（論文本身未打開），來源：https://thesis.lib.nccu.edu.tw/detail/a226b628d8b1f7e3bb3d5416c0183480/
6. **PTT 版友〈股板名人跟單報酬統計表〉（2021；證據等級很低）**
   - 每位作者只取最近 5 篇，有挑選偏誤，數字是圖片。
   - https://www.ptt.cc/bbs/Stock/M.1625931348.A.BE4.html
7. **財經 YouTuber、Podcast**：找不到台灣的系統性績效研究。
   - 股癌第三方整理站自己聲明不計算報酬：https://github.com/ryvn-dev/gooaye-notes
   - 「哲哲 2953 篇發文」的分析只見於 udn 報導（https://udn.com/news/story/12806/9573661），內容是大盤情緒，不是選股；這個環境打不開原文。

## 限制
- 樣本只有 27 個月，t 值的檢定力有限。只涵蓋上市股，上櫃 442 篇無法測。2024-12 之後有漏抓。
- 47% 事件的進場價落在面板來源 3（Yahoo 5 分 K 近似）期間。
- R3 需要多年、同一批作者的資料才能測；本資料只有 3 位作者符合門檻。
- 熱度是爬蟲最後一次快照時的推文，在進場前截斷；改用最終推數（有前視）結論相同。

## 使用者若要自己驗證（記錄方式）
- **每篇記錄**：發文時間到分鐘、作者 ID、平台（PTT/Mobile01/Dcard/YouTube）、文章或影片網址、股票代號、方向（多／空）、有沒有寫目標價與停損、發文前 20 日漲幅、當下推文數或觀看數。
- **只追蹤事先選定的作者**：例如 5 位，先寫下選人理由，之後不換人。每位都記全部的文，不只記賺錢的。
- **事後自動計算**：隔日開盤價、5 日與 20 日後開盤價、同期 0050 與上市等權報酬，扣 0.5% 成本。
- **判定門檻**：每位作者累積滿 30 篇、而且滿 12 個月後才下結論；要求方向調整後平均報酬 > 0，且 t_month ≥ 2。
- 刪文和改文要留截圖，避免存活者偏差。

## 檔案
- `/home/user/-/research/tw_stock_selection/agents/forum_gurus/build_posts.py`：合併 git 快照。
- `/home/user/-/research/tw_stock_selection/agents/forum_gurus/run_forum.py`：解析、回測、R1～R4、重疊分析；重用 exec_rebuild/exlib.py 的還原開盤、開盤漲停判定和主策略選股 `selection('M2', 30)`。
- `/home/user/-/research/tw_stock_selection/agents/forum_gurus/results.json`：所有數字，含前半、後半、敏感度。
- `/home/user/-/research/tw_stock_selection/agents/forum_gurus/events.parquet`：1,005 筆事件。
- `/home/user/-/research/tw_stock_selection/agents/forum_gurus/posts_raw.parquet`：22 MB，含貼文全文，可用 build_posts.py 重建，**建議不要 commit**。
- 唯讀 clone 位置：`/home/user/a607ernie/ptt-stock-data`、`/home/user/jameschen108/tw-stock-sentiment`（另有 soniccol、kuo23、arthur0234、ryvn-dev 四個）。
- 依指示沒有寫 FINDINGS.md、沒有 git commit。git status 顯示 forum_gurus 底下有檔案已被追蹤、狀態為 M，應該是主研究員提交過早期版本。

Sources: [a607ernie/Ptt-Stock-Data](https://github.com/a607ernie/Ptt-Stock-Data), [jameschen108/tw-stock-sentiment](https://github.com/jameschen108/tw-stock-sentiment), [SFI Finfluencers](https://www.sfi.ch/fr/publications/n-23-30-finfluencers), [Sherwood 報導](https://www.sherwood.news/markets/your-favorite-social-media-guru-is-probably-a-bad-stock-picker-study-shows/), [RFS WSB](https://ideas.repec.org/a/oup/rfinst/v37y2024i5p1409-1459..html), [SUFE 摘要](https://academicnewsletter.sufe.edu.cn/info/351572), [Airiti 2008](https://www.airitilibrary.com/Article/Detail/U0020-1308200821542800), [政大論文](https://thesis.lib.nccu.edu.tw/detail/a226b628d8b1f7e3bb3d5416c0183480/), [PTT 跟單統計](https://www.ptt.cc/bbs/Stock/M.1625931348.A.BE4.html), [udn 哲哲](https://udn.com/news/story/12806/9573661), [gooaye-notes](https://github.com/ryvn-dev/gooaye-notes), [soniccol/PTTData](https://github.com/soniccol/PTTData), [kuo23/PTT-Stock](https://github.com/kuo23/PTT-Stock), [Arthur0234 issue](https://github.com/Arthur0234/taiwan-limitup-research/issues/8)
