# FINDINGS：2006～2013 PIT 股數與證交所產業別快照盤點（只做資料重建與 D1／D2 閘門檢核，不回測任何報酬）

資料夾：/home/user/-/research/tw_stock_selection/agents/o_pit_shares_industry_2006_2013/
研究員：2026-11-03

## 判定框

- **最終判定：2006～2013 仍非 PIT（閘門不通過）。只存檔 raw/ 與 data/gate.json，不輸出 shares_pit_2006_2013.parquet。**
- 閘門逐項：
  - D1a-2006（留一錨點）：**通過**。誤差中位數 0.0%，前 100 重疊中位數 100、最小 99。
  - D1c：**不過**。2008～2010 年仍有月份的前 100 名裡有 1 檔要用 ADV 插補，門檻是 0。
  - D2：**不過**。dated 產業覆蓋只有 55.4%，門檻 98%。
- 和 G 門檻的關係：
  - 本題沒有策略報酬，G1～G4、G7、G8 不適用。
  - 本題只對 G5、G6 負責：G5 未證實（D1c、D2 不過）；G6 待獨立複製員依 GATE_SPEC 重建。
- **一句話：** 2006～2013 的「股數」大致補得起來：存活股幾乎完美，缺口只剩約 6 檔早期已下市股。補不起來的是「2011-07 以前的產業別」：GitHub 上找不到任何有日期的快照。
- 對主策略：在有日期快照的股月中，金融旗標不一致只有 2 檔（2841 台開、5854 合庫），未超過 10 檔。但 2006-01～2009-07 共 44.6% 的股月無法判定。依事前規則，**主策略 2006～2013 段的 G5 不能宣稱已證實，交使用者裁定是否降級**。本題不重算報酬。

## 1. 研究問題

- **問題 1（股數）**：能不能用公開來源，為 2006～2013 建出時點正確的股數（市值＝股數 × 原始收盤），讓 D1 閘門通過？
- **問題 2（產業）**：能不能為主策略前 300 流動性池建出有日期依據的產業別，並計數主策略金融旗標與當時分類不一致的股票？
- **為什麼要做**：這是 G2 全期與 G4 第一段（2006～2012）能否被評估的唯一前置作業。

## 2. 機制（資料機制，不是交易機制）

- **為什麼 o12 的 D1 失敗**：o12 只靠除權因子，從 2013 或 2021 錨點往回推。
- **事前假說**：誤差來自除權表抓不到的股本事件，包括現增、CB 轉換、2008 年以前的員工分紅配股、減資、合併，而且這些事件集中在名次 60～120 的邊界股。
- **本題檢驗結果：只部分成立。**
  - 事件遺漏確實讓 o12 舊回推誤差很大：2006 年中位數 16%，2408 南科、2357 華碩等誤差超過 100%（對數）。
  - 但誤差不集中在邊界股：前 20 名 6.8%，60～120 名 4.9%。
  - 而 D1 重疊數卡關的主因，是「錨點日前已下市、沒有任何錨點、只能 ADV 插補的股票」：
    - o12 原協定用官方事件後重疊最小 92；
    - 把這些股票換成真值後升到 94。
  - 錨點法（路徑 A）能把存活股補到幾乎零誤差，但下市股覆蓋只有 42%，產業別則完全缺 2006～2009 的快照。

## 3. 資料

**盤點與事前提前結束規則**（explore.json）：若覆蓋（≤ 1 年口徑）< 90%，且路徑 B 事件來源也不存在，就直接判「資料不足」。實際覆蓋 97.1% ≥ 90%，所以繼續。

### 3.1 新找到的股數來源

全部取自 GitHub 公開 repo，commit 與 sha256 記在 raw/manifest.json；PIT 日期證據等級都是 A（列內有期末日期）。

| 代碼 | 來源（repo） | 內容 | 頻率 | 涵蓋 | 含下市股 |
|---|---|---|---|---|---|
| tejm | JunJie-Chang/ff5_replication（TEJ Pro price.xlsx） | 月底流通在外股數 | 每月 | 2000～2025 | 否（只抽 6 欄，不讀報酬欄） |
| tejq | JackWang0102/The-Cross-Section-of-Expected-Stock-Returns-in-Taiwan（TEJ 期末市值） | mv÷同日收盤 | 2005～2007 半年；2008 起每季 | 2004-12～2020-12 | 否 |
| gq | yhzbambu/Lab_project（goodinfo 經營績效） | 季股本(億) | 2006～2007 只有 Q2／Q4；2008 起每季 | 460 檔 1xxx／2xxx | 否 |
| ga | henry753951/finance（goodinfo StockAssetsStatus） | 年底股本(億) | 每年 12-31 | 609 檔 | 是（部分） |

**其他使用的資料**
- twsec 季股本：2013 年起。
- o8 shares_pit：2013 年起。
- TWT49U：joeycchen72，**2014-01 起**。
- o12 產業表、上市主檔、ISIN 2023 快照、weihautin 2015／2016／2017 快照。

**更正一項前提**：本題敘述中的「icoolhome TWT49U 2003～2026 封存」實際上只是 3 個 2026-01 的單日檔（o8 manifest 已註明不用）。本中心**沒有 2014 年以前的 TWT49U**。

**找不到的來源**（搜尋紀錄在 raw/search_log.jsonl）
- (a) MOPS 2006～2012 實收資本額封存：找不到；改用 TEJ 與 goodinfo 二手封存代替。
- (b) 證交所逐檔市值表：找不到。
- (c) 2014 年以前的現增、CB 轉換、減資事件表：找不到。tw-cb-data 只有 CB 日行情與債息表，不含轉換股數。
- (e) 2006～2013 臺灣50 審核名單：找不到，P5 不做。

### 3.2 產業快照

- toomore/goristock 與 toomore/grs 全歷史中找到 6 個版本，有 5 個檔內 UPDATE 列可定出日期：
  - 2011-07-28、2011-09-26、2012-03-28、2012-07-31、2013-12-18；
  - 另 1 版沒有 UPDATE 列，以 commit 時間 2011-08-17 為準（B 級）。
- **2011-07 以前沒有任何 dated 快照。** FinMind 類的全清單裡 date 欄全部是 2023 年的抓取日，不能當 PIT。

### 3.3 紀律

- 只讀 close_raw、open、value、flag；不讀 adj、ret。
- 除權因子改用開盤跳空估計。
- s5_posthoc 的程式掃描結果：0 個違規。

## 4. 探索發現（非閘門診斷）

**覆蓋**（ADV 前 150 股月，2006～2013 共 14,400 股月）

| 群組 | 距錨點 ≤ 1 季 | ≤ 1 年 |
|---|---|---|
| 全部 | 93.0% | 97.1% |
| 存活股 | 99.8% | — |
| 2021 年前下市股（占 11.8% 股月） | 42.1% | 75.8% |

- 逐年 ≤ 1 季覆蓋：2006 88.6% → 2013 97.8%。
- 一年內都沒有錨點的股票：6286 立錡、2854 寶來證、2446 全懋、3009 奇美電、3534、2350 環電、2391、2473、3007、3214。

**D1b（真值本身是否可信）**
- TEJ vs goodinfo（兩家不同的整理者）：大型股 |對數差| 中位數 0.050%～0.058%，門檻 ≤ 1%。各期間都只有約 1.5% 的配對差超過 2%。
- 差異大的個案是定義或時點不同：
  - 3045 台灣大：TEJ 流通股數不含子公司持有的庫藏股；
  - 2412 中華電、2408 南科：減資前後的認列時點不同。
- tejm 與 tejq 完全相同（同一家 TEJ），不算兩個獨立來源。
- tejq 與 o8 的 2013～2018 真值對數差中位數 2e-6：TEJ 期末市值＝報表股本 × 收盤，與 o8 的真值口徑相同。

**產業 dated 覆蓋**（主策略前 300 池）
- 2006～2008：0%；2009：42%；2010：98.9%；2011～2013：100%。合計 55.4%。

## 5. 模型迭代紀錄（iterations.json；3 個版本、2 輪選版規則，照實揭露）

**三個版本**
- V1：事件日整段階梯（o8 規則）。
- V2：事件日先套估計乘數，剩下的變動做內插。
- V3：純對數內插。

**校準窗**：2014Q1～2020Q4，不與閘門窗重疊。診斷有兩種：(a) 全錨點留一；(b) 只留 12 月錨點的留一（模擬下市股只有年底股本的情況）。

| 版本 | (a) 中位數 | (a) p90 | (b) 中位數 | (b) p90 |
|---|---|---|---|---|
| V1 | 0 | 0.02% | 0 | 2.49% |
| V2 | 0 | 0.05% | 0 | 2.37% |
| V3 | 0 | 0.05% | 0 | 3.18% |

- 重疊：三版都是 ≥ 98。
- **第 1 輪**：選版規則（看中位數）退化，因為三版中位數全是 0，規則變成「取最簡單」，選出 V3。
- **第 2 輪**：看過第 1 輪結果後，改用 (b) 的 p90 選版，選出 **V2**。這屬於看過結果後的規則修改，但只在校準窗內，照實揭露。

## 6. 凍結規格

全文見 frozen_spec 欄位。GATE_SPEC sha256 56e36098…，凍結後才跑閘門。

## 7. 閘門結果（一次跑完，data/gate.json）

**D1a-2006**
- 32 個季末，誤差合併中位數 0.0000，p90 0.76%。
- 重疊中位數 100、最小 99。
- 依名次分組，誤差中位數都是 0。
- 最大誤差：2357 華碩 2010Q2 78.8%（對數），原因是 2010 年分割和碩造成的股數斷點；其次是 2475 華映、3481 群創 2010Q1。
- **限制：留一檢核只能驗證有真值的股票。** 真值前 100 中沒有錨點的股票平均 7.7 檔（2006Q1 有 19 檔），它們的誤差驗證不到。

**D1c**
- 平均 0.885 檔（≤ 1 ✔）。
- 但 2008 年 0.75、2009 年 0.917、2010 年 0.667，2008 年起最大值是 1 ✘。
- 插補股：2446 全懋、3534、3007、3214、3012、2394。
- 另有距最近錨點超過 1 年的外推股：6286、3009。

**D2**
- dated 55.4% ✘，無法判定 44.6%。

**金融旗標不一致**（只算 dated 股月）：2 檔、52 股月
- 2841 台開：2011～2013 快照屬建材營造，主策略卻因 28xx 代碼規則判為金融。
- 5854 合庫：當時屬金融保險業，但 2011 年下市後不在現行 stock_list，也不是 28xx 代碼，所以主策略**沒有**排除它。

**主策略金融旗標的疑點**
- 無法判定的 12,728 股月中，主策略判為金融的有 861 股月。
- 「28xx 一律算金融」這條規則把 2841 台開誤列為金融。

**附帶勘誤（描述用）**
- o12 回溯產業表和 dated 快照的不一致率是 2.3%（369 股月）。s4 第一次算成 10.3%，原因是名稱尾字「業」沒有統一。已修正一次並揭露；這只影響描述，不影響閘門。
- 實際有差異的例子：2474 其他電子 vs 電腦週邊、2601 貿易百貨 vs 航運、2402、8163、2473。

**D1-retro（路徑 B，o12 原協定 2014～2020）**

| 版本 | 誤差中位數 | 重疊中位數 | 重疊最小 |
|---|---|---|---|
| TWT49U 官方事件版 | 0.27% | 96 | 92 |
| 不讀 ret 的開盤估計版（重做 o12） | 1.98% | 95.5 | 91 |
| 官方版 + 錨點日前下市股換成真值（上限） | 0.09% | 96 | 94 |

- 重疊最小值低於 93 的季末，集中在 2015-09～2017-06。
- 主要漏掉：2847 大眾銀、2325 矽品、5264、1704（都是錨點日前已下市、只能插補的股票），以及 2201、2915、9945、2327。

**預測對照**

| 預測 | 結果 | 說明 |
|---|---|---|
| P1 | 否證 | 補上官方事件後只到 92。現增、CB、減資事件來源不存在，無法完整執行路徑 B。上限分析顯示瓶頸是沒有錨點的下市股。 |
| P2 | 否證 | o12 舊回推誤差：前 20 名 6.8%、60～120 名 4.9%。誤差隨年代越早越大，不是集中在邊界股；電子股（14.7%）大於非電子股（10.7%）這點方向正確。 |
| P3 | 部分成立 | 找到季／年股本封存，覆蓋 93%（≤ 1 季）／97%（≤ 1 年），D1a 通過；但下市股缺口讓 D1c 失敗。 |
| P4 | 金融不一致部分成立；dated 覆蓋否證 | 不一致 2 檔 ≤ 10；dated 覆蓋 55% < 98%。 |
| P5 | 未做 | 找不到 2006～2013 臺灣50 審核名單。 |

總監事前估計 D1 通過機率約 25%；實際上 D1a 通過、D1c 差一點、D2 遠遠不到。

## 8. 穩健性

- 分年：
  - 覆蓋與 D1c 的缺口集中在 2006～2010；2011 年起 D1c 為 0，產業 dated 覆蓋 100%。
  - D1a 誤差在各年都接近 0；最大誤差來自公司分割或合併（2357、2475、3481、2885）。
- 分產業：o12 舊回推誤差在 2006～2008 的電子股較大。
- 去掉最大貢獻股：不適用（沒有報酬）。改看「卡住的股票清單」，見下一段。

## 9. 卡住的股票、季別、缺的來源（不通過時必報）

**股數**
- 無錨點、需插補：2446 全懋（2006～2008）、3534、3007、3214、3012、2394。
- 只有遠端錨點、靠外推：6286 立錡（2013 年以前沒有錨點）、3009 奇美電（只有 2005 年底股本）、2854 寶來證、2350 環電、2418 雅新。
- 缺的來源：這些下市股 2006～2010 的季報股本。MOPS 原站被擋，GitHub 上也沒有封存。

**產業**
- 2006-01～2009-07 沒有任何 dated 上市清單或產業快照。
- 缺的來源：證交所 2007-07 產業別調整公告、2006～2010 的 ISIN 或上市公司清單封存。

## 10. 散戶怎麼用（研究中心內部）

- 不要把 2006～2013 段的結果當成已通過 PIT 的證據。G2 全期與 G4 第一段仍只能描述。
- 如果使用者願意把股數與產業分開裁定：股數部分 2011～2013 已無插補、留一誤差接近 0，可以考慮只把 2011～2013 標「近似 PIT」。這是使用者的決定，本題不做。
- 對主策略的具體建議（只建議，不改工具）：「28xx 一律算金融」會把 2841 台開誤列為金融；已下市金融股（例如 5854 合庫）也不在排除名單裡。

## 11. 限制

- tejm 與 tejq 同出 TEJ，gq 與 ga 同出 goodinfo，所以實際上只有兩個獨立來源。
- TEJ 的流通股數與報表股本，對庫藏股多的公司（例如 3045）定義不同。本題同日優先用 TEJ。
- 錨點之間的內插使用「未來錨點」，季內有前視，與 o8 相同，屬「近似 PIT」口徑。
- 留一檢核看不到沒有真值的股票（平均 7.7 檔）。
- 選版規則在第 2 輪做過修改（已揭露）。
- 凍結後 plib.py 只改了一行（scratchpad 目錄 mkdir），不影響任何數字；GATE_SPEC 記的是改前 sha。
- scratchpad 中間檔（git 物件、TEJ 43MB xlsx、henry 日檔）已刪除；重建只需要 raw/（32MB）。
- 沒有修改其他資料夾，也沒有 git commit。

**輸出檔**
- results.json：PREREG、判定、關鍵數字、預測、posthoc
- explore.json、iterations.json
- data/GATE_SPEC.json、data/gate.json
- raw/manifest.json、raw/search_log.jsonl
- 程式：s0_search.py、s0_collect.py、s1_inventory.py、s2_iterate.py、s3_freeze.py、s4_gate.py、s5_posthoc.py、plib.py、pbuild.py

---
# 審查與複製（o13 工作流原文；總監強制勘誤見 original_o13/DIRECTOR_REPORT.md）

## 統計審查

**final**

不可用

**note**

結論：同意「2006～2013 仍非 PIT、不可用」，而且實際情況比 FINDINGS 寫的更差。

研究員說對的部分：
- 存活股的股數確實補得起來；加嚴的留一測試在存活股上仍然通過。
- D1c、D2 不過，判定成立。

被高估或不成立的部分：
1. D1a 實際上只檢驗了存活股的月資料內插：真值 93% 是 tejm，間距中位數 61 天。
2. 下市股：前 100 名中平均 9.75 檔，有真值的只有 22%；靠年底股本重建的誤差中位數 3.1%（超過門檻 3%），p90 12.8%。「只剩約 6 檔」低估了缺口。
3. D2 的 55.4% 含倒填的前視；只用當時已有快照的嚴格口徑只有 30.1%，2009～2010 是 0%。
4. 「2011～2013 近似 PIT」的建議證據不足。
5. D1-retro 的「缺口不是事件漏抓」不成立：換成真值只多 2 檔，反覆漏掉的仍是有減資等事件的存活股。

我的審查腳本是 /home/user/-/research/tw_stock_selection/agents/o_pit_shares_industry_2006_2013/review_stat_o13.py，完整輸出在 /tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/rev_o13.txt。研究員的檔案都沒有改動。

**issues**

- 我同意研究員的總判定「不可用（閘門不通過）」，但 FINDINGS 有幾處說法比證據樂觀。我寫了 review_stat_o13.py 重跑（python3 -I -B），D1a 的數字可以重現：中位數 0、p90 0.76%、重疊最小 99。
- D1a 通過這件事，幾乎只驗證了存活股的月資料內插。真值裡有 93.2% 來自 TEJ 月資料（tejm，只收 2025 年仍存活的股票）。拿掉檢核日前後 20 天的錨點後，剩下前後錨點的間距中位數只有 61 天，等於只測了「2 個月內插」。
- 已下市股在 D1a 裡大多驗證不到。各季末市值前 100 名中，平均有 9.75 檔是 2021 年前下市的股票（2006 年有 17.25 檔），其中只有 21.8% 有真值。
- 真值只能來自 goodinfo 年底股本（ga）的股票（n=53，幾乎都是下市股）重建得比較差：留一誤差中位數 3.1%，已超過 D1a 門檻 3%，p90 是 12.8%。所以「存活股幾乎完美」可以成立，但不能延伸到下市股。
- 「缺口只剩約 6 檔早期已下市股」低估了缺口。D1c 只算完全沒股數、要用成交值（ADV）插補的股票，不算靠遠端錨點外推的股票。前 100 名裡沒有真值的下市股約有 25 檔，例如 2311、2325、2448、3474、6286、5854、3009、2475、2384、2315。D1c 本身也有 2007～2010 年平均每月 1.2～2.4 檔距最近錨點超過 1 年。
- D2 的 55.4% 裡有前視。規格允許用「24 個月內的後續快照」，所以 2009-08～2011-07 的產業別其實是拿未來的快照（2011-07-28 起）倒填的。只用當時已存在快照的嚴格口徑，dated 覆蓋只有 30.1%：2009～2010 都是 0%，2011 是 41.4%。2841 台開與 5854 合庫的金融旗標不一致，起點 2009-08 也是靠後續快照算出來的。
- 第 10 節建議 2011～2013 可標「近似 PIT」，證據不夠。2011 年嚴格口徑的產業覆蓋只有 41%；同年前 100 名平均仍有 6 檔下市股沒有真值；季內內插還會用到下一個錨點（季內前視）。
- D1-retro 的機制結論說過頭了。把錨點日前已下市的股票換成真值，重疊最小值只從 92 升到 94，只差 2 檔。2015-09～2017-06 一再被漏掉的 2201、2915、2327、9945 都是存活股，看起來是減資、庫藏股註銷這類 TWT49U 沒收錄的事件。所以「缺口主要來自沒有錨點的已下市股，不是事件漏抓」不成立，比較接近「兩者都有」。
- P2 結論的樣本有偏、也不獨立。P2 說誤差不集中在名次 60～120 的邊界股，但名次分組只用有真值的股票，也就是存活股。同一檔股票（例如 2357、2408）跨十幾個季末重複計數，中位數之間不獨立，所以「否證」的力道有限。
- 真值來源的獨立性有限。tejm 和 tejq 對數差中位數 2e-8，等於同一份資料；goodinfo 和 TEJ 差 0.06%，但無法排除 goodinfo 的股本也是從同一個上游轉來的。
- 多重檢定的影響很小。試了 3 個建構版本、選版規則在第 2 輪改過，但 D1c、D2 都不受版本影響。我另外做了加嚴測試，D1a 在存活股上還是通過，所以版本選擇不改變判定：拿掉檢核日前後 100 天的錨點時，中位數 0.04%、重疊最小 98；只留 12 月錨點時，中位數 0.57%、p90 4.2%、重疊最小 97。
- 腳本裡的 B3（重建時去掉 tejm）是我自己的設計錯誤，結果不採用。錨點表已經依優先序合併，同一天有 tejm 與 tejq 時只留 tejm，所以去掉 tejm 會連 tejq 一起去掉，造成假的大誤差。腳本內已加註。

## 可執行審查

**final**

不可用

**note**

我審查後維持研究員的「不可用」判定。主要證據：D2 不過（55.4%，門檻 98%），而且其中一部分是用事後快照補上的；D1c 不過，而且在真正 PIT 的公告時滯口徑下更差。審查程式在 /home/user/-/research/tw_stock_selection/agents/o_pit_shares_industry_2006_2013/review_exec_o13.py（用 python3 -I -B 執行，沒有改研究員的任何檔案）。它做了三件事：(X1) 錨點只能在公告後使用、不用任何未來錨點的因果重建；(X2) 因果口徑下的額外插補檔數；(X3) 5854 和 2841 在主策略池中出現的月數、名次與均成交值。D1a 在因果口徑下仍然過門檻（重疊最小 96～97），但尾端誤差和插補都高於研究員的數字。2011 年在公告時滯下仍有插補，所以「近似 PIT」最多只能談 2012～2013，研究員的「2011～2013」要縮小。另外，主策略的金融排除規則在 2006～2011 段沒有 PIT 依據，而且漏掉了流動性名次約 76 的 5854 合庫，這應該明確告訴主策略負責人。周轉、證交稅、滑價、漲跌停、處置股、容量、盯盤這幾項，因為本題沒有報酬，都不適用。

**issues**

- 我同意「不可用、2006～2013 仍非 PIT」這個結論。本題沒有策略報酬，所以周轉、0.3% 證交稅加滑價、漲跌停買不到、處置與預收款券、30 萬～1,000 萬的容量、要不要盯盤，這些都不能直接評。從成本與可成交的角度，能審的只剩「資料實際拿得到的時點」。在這一點上，研究員的閘門比實際情況寬鬆。
- D1a 的留一檢核允許用檢核日之後的錨點做內插（研究員在限制欄有揭露），也沒有考慮財報公告的時滯。所以「誤差中位數 0.0%、p90 0.76%、重疊最小 99」不代表當時真的能取得這些股數。我寫了 review_exec_o13.py 驗證：只准用公告日已過的錨點，之後只靠除權事件乘數往後推，完全不用未來錨點。公告時滯照 2012 年以前的法定期限：Q1、Q3 為 1 個月，Q2 為 2 個月，Q4 為 4 個月；2013 年起 Q1～Q3 為 45 天，Q4 為 3 個月；TEJ 月股數給 15 天。結果在 32 個季末：重疊中位數 99、最小 97；p90 誤差 1.43%；誤差超過 5% 的股票占 5.4%。如果 TEJ 月股數也改用季報時滯，重疊最小 96，p90 3.26%，誤差超過 5% 的占 7.3%。在主策略換股日（每月第一個 ≥11 日），重疊最小 96、p90 約 2.8～3.1%。D1a 的門檻（中位數 ≤ 3%、重疊最小 ≥ 93）在這個口徑下仍然通過，但尾端誤差是研究員報的 2～4 倍。
- D1c 在真正 PIT 的口徑下更差。「真值前 100 裡還沒有任何已公告錨點、只能用 ADV 插補」的額外檔數，在換股日的平均是：2006 年 5.42、2007 年 1.33、2008 年 2.0、2009 年 2.17、2010 年 1.17、2011 年 0.42，2012～2013 年才是 0。季末口徑下 2012 和 2013 年也各有 0.25。所以研究員建議「2011～2013 可考慮標近似 PIT」範圍太寬。在公告時滯下 2011 年仍有插補，最多只能談 2012～2013。
- D2 的「dated」包含往後 24 個月內的快照。最早的快照是 2011-07-28，所以 2009 年的 42%、2010 年的 98.9%，以及 2011-01～07 的覆蓋，其實都是用事後的快照回填。嚴格只往回看的話，2011-07 以前的覆蓋是 0。我推算主策略換股日中只有 2011-08～2013-12 共 29 個月、約 30% 的股月有真正 PIT 的產業依據，遠比 55.4% 低。這不改變判定，但「2011～2013 產業 dated 覆蓋 100%」應該改寫成「2011-08 起」。
- 研究員把金融旗標不一致寫成「2 檔，未超過 10 檔」，聽起來影響很小，但其中一檔對主策略實際影響不小。5854 合庫在 2006-01～2011-11 有 71 個月在主策略前 300 流動性池內，名次中位數 76，60 日均成交值中位數約 3.5 億。主策略沒有把它當金融股排除，所以它可能真的被選進持股。2841 台開被誤排除 44 個月，名次中位數 239，均成交值約 0.78 億，影響較小。另外，無法判定的股月中，主策略判為金融的有 861 股月。所以主策略 2006～2013 段的金融排除規則本身就沒有 PIT 依據，應該明說，而不是只寫「≤ 10 檔」。
- D1c 用「全期平均 ≤ 1」當通過條件之一，把 2006 年平均每月 3.75 檔插補平均掉了。2006 年第 1 季的真值前 100 中，有 19 檔沒有真值，留一檢核驗證不到。所以「股數大致可以重建」不適用於 2006～2007 年。
- 選版規則在第 2 輪才改成看 p90，是在校準窗內看過結果後改的（研究員已揭露）。另外，凍結後 plib.py 被改過，GATE_SPEC 裡記的是改之前的 sha。兩件事都不影響數字，但流程上有瑕疵。
- 研究員的程式本身沒有負索引的問題：最早的檢核日 2006-03-31 對應面板索引 554，大於 520。這個 bug 出在我自己的腳本：換股日 2006-01 的索引是 504，504−520 為負，第一次跑時這一期重疊被算成 8。已改成 max(·, 0) 後重跑，上面的數字都是修正後的。

## 獨立複製

**reproduced**

True

**my_numbers**

I rebuilt everything from raw/, the GATE_SPEC and the panel, without reading any of the researcher's code. Programs are in /home/user/-/research/tw_stock_selection/agents/o_pit_shares_industry_2006_2013/replicate_o13/ (rlib.py, r1_build.py, r2_d1a.py, r3_d1c.py, r4_d2.py, r5_retro.py, r6_misc.py, r6b.py, r6c.py; outputs in out/). All run with python3 -I -B.

D1a-2006 (V2, 32 checkpoints)
- Pooled |log error| median 0.0000, p90 0.0076, sample 2,954 (researcher 2,955).
- Top-100 overlap: median 100, minimum 99.
- Result: pass.
- Stocks in the truth top 100 with no truth anchor: average 7.69 (researcher 7.66). 2006Q1 has 19; from 2011Q4 on, 0 or 1.

D1c
- Overall mean 0.896 (researcher 0.885); maximum from 2008 on is 1, so it fails.
- By year: 2006 3.58 | 2007 1.08 | 2008 1.00 | 2009 0.83 | 2010 0.67 | 2011–2013 0.
- Imputed stocks and months: 2446 27, 3534 27, 3007 13, 3012 8, 3214 8, 2394 3. Same six stocks as the researcher.
- Extrapolated more than 1 year from the nearest anchor: 6286 59 months, 3009 38 months. Same as the researcher; I also found a few short cases: 3367 4, 6004 2, 2854 2.

D2
- Pool: 28,551 stock-months (researcher 28,722).
- dated 55.41%, so it fails. By year: 2006–2008 0 | 2009 42.37% | 2010 98.86% | 2011–2013 100%.
- Undeterminable: 12,731 stock-months, of which the main strategy flags 863 as financial (researcher 12,728 / 861).
- Financial-flag mismatches: 2 stocks, 52 stock-months, matching exactly. 2841 台開 (24 stock-months): PIT says 建材營造, but the main strategy calls it financial. 5854 合庫 (28 stock-months): PIT says 金融保險, but it is missing from the main strategy's financial list.

D1-retro, with stocks that have no anchor ADV-imputed
| Version | Error median | Overlap median | Overlap min | Researcher |
|---|---|---|---|---|
| TWT49U official | 0.29% | 96 | 92 | 0.27% / 96 / 92 |
| Open-gap estimate | 1.98% | 95 | 91 | 1.98% / 95.5 / 91 |
| Official + true values for delisted (upper bound) | 0.09% | 96 | 94 | same |
| Open-gap + true values for delisted | 1.40% | 96 | 93 | same |

The P1 threshold needs overlap min ≥ 93, so the official version fails it at 92.

D1b source consistency (tejm vs gq / ga)
- |log difference| medians are 0.04%–0.06% for every definition of "large cap" I tried. For example, mcap top 150: gq 0.050% (n 2,670), ga 0.059% (n 864).
- Conclusion matches the researcher: the truth values are reliable.

Coverage (ADV120 top 150 stock-months, 14,400)
- Within 1 quarter of an anchor: 93.0–93.3%. Within 1 year: 97.2%.
- Survivors within 1 quarter: 99.6%. Stocks delisted before 2021 within 1 quarter: 42.3–46.1%, depending on how "1 quarter" is defined.
- Delisted stocks' share of stock-months: 11.8%.

P2 (o12 old backcast, 2006–2012)
- By rank: top 20 6.77%, ranks 21–59 7.75%, ranks 60–120 4.96%.
- By year: 2006 16.2%, 2007 11.3%, 2008 8.5%, 2009 6.9%, 2010 5.0%, 2011 2.2%, 2012 0.0%.
- Largest errors: 2408 +222% log, 2357 −193% log (both 2009-09-30).

GATE_SPEC sha256 checks out: 56e36098ae9bf31357fdfbbc8cf0a254b9306ecee86a291e3ed24399ba13b2f8.

Final verdict is the same: D1a passes, D1c and D2 fail, so 2006–2013 is still not PIT.

**diff_explained**

1) D1a: 2,954 vs 2,955. One truth top-100 stock without a truth anchor is classified differently, which comes from small differences in survivor checks or ADV imputation. The median, p90 and overlap match exactly.

2) D1c: 0.896 vs 0.885 (86 vs 85 imputed stock-months; 2446 has 27 vs 26, 3534 has 27 vs 25, 3012/3214 have 8/8 vs 10/8). The spec does not say which ADV120 min_periods to use, how the survivor mask handles the 6-day window, or how to treat close as stale. That only shifts the marginal rank-100 borderline. The set of imputed stocks is identical, the 2008+ maximum is still 1, and the verdict is unchanged.

3) D2 pool: 28,551 vs 28,722. ADV60 rolling details are not fully specified: whether NaN counts as 0, min_periods, and whether listing status is filtered. Using NaN as 0 gives 28,562. The researcher's own numbers do not fit together: 1 − 12,728/28,722 = 55.7%, not the 55.4% they report. My 55.41% and the undeterminable counts (12,731 / 863) are almost the same as theirs. The researcher's 28,722 may include a different set of months or stocks. The mismatch counts (2 stocks / 52 stock-months) match exactly.

4) D1-retro: if stocks with no 2021-03-31 anchor are simply excluded, the official version comes out at 0.16% / 94 / 90. If they are ADV-imputed (the same rule as the main construction), it comes out at 0.29% / 96 / 92, matching the researcher. So the original protocol ADV-imputes those stocks. The remaining 0.27% vs 0.29% gap and the open-gap overlap median of 95 vs 95.5 come from two calls the spec leaves open. One is whether the 1.02 threshold also applies to 權 events and which year's market median is used. The other is how the imputation ratio is set. Both upper-bound versions match exactly.

5) D1b sample sizes (1,694 / 686) are not matched. The spec does not define "large cap"; I tried mcap and ADV top 100/150, which gave n = 1,567–2,670 / 552–864. The medians (0.04–0.06%) agree with the researcher's 0.050% / 0.058%, and so does the conclusion.

6) Coverage for stocks delisted before 2021 (within 1 quarter): 42–46% vs 42.1%, depending on whether "1 quarter" means 90 days, 91 days or 63 trading days. With 63 trading days I get 42.3%. Survivors come out at 99.6% vs 99.8%, which is how "delisted" is defined (I use the listing-master delist_date < 2021).

7) Choosing between V1/V2/V3 and the two rounds of the selection rule were not redone (only V2 was rebuilt); this replication covers the gate numbers.
