# FINDINGS：產業輪動＋強勢大型股（o_lc_sector_strong_stocks，題 2）

資料夾：/home/user/-/research/tw_stock_selection/agents/o_lc_sector_strong_stocks/
研究員判定：**不可用**｜機制「先產業、後龍頭」：**不支持**

## 0. 結論

- **本題對 0050 是負的**
  - 選定版：題 1 凍結的 H52 前 3 產業內，各取 2 檔強勢股；SAT 載體＝B 70%＋強勢股 30%。
  - 可執行 K5T、100 萬、GROW、13 個 j 中位數：
    - 2019 年起：年化 27.5%，0050 28.8%，t −0.72。
    - 驗證期 2017～2026：t −0.48。
    - PIT 段 2014～2026：t −0.66。
  - G1～G7 沒有一條通過。
- **alpha 不來自產業，也不來自「產業 × 個股」**
  - 不分產業的同分數對照（CTRL）在每個口徑都比本題好。
    - 本題減對照：驗證期 −1.02pp／月（t −1.50）。
    - 切換前的 COMP 版本也一樣：−0.24pp（t −2.08）。
  - 產業層：效果約 0（驗證期 t 0.36），而且 2019 年起為負。
  - 產業內個股層：效果約 0（t −0.07）。
- **大型股個股動能本身有一點東西，但不顯著**
  - 個股獨立效果（CTRL−B）：驗證期 +1.15pp／月（t 1.55）。
  - CTRL PURE 可執行版：2019 年起年化 39.7%（0050 28.8%），t 1.06；追蹤誤差 24%，全期回撤 −66%；壓力成本下 t 0.39。
  - 這和 o3、o8 的「大型股選股力不足以顯著贏 0050」一致。
- **主策略的右尾有產業成分，但不到預測的程度**
  - 主策略（PIT 段）貢獻前 20 大的股票-年裡，45% 多數時間屬於題 1 前 3 產業；隨機期望 19%，預測門檻 60%。
  - 主策略超額對產業因子的 R² 只有 0.04。
- **不能當第二條腿**
  - 與主策略超額的相關 0.52（< 0.6，這點成立）。
  - 但它本身 IR 為負；與主策略 50/50 組合的 IR −0.157，主策略單獨 −0.176，bootstrap p 0.67。
  - 對（2330−0050）的 β 為 −0.31（預測是 ≥ 0），所以補不了「權值股領漲時輸 0050」的缺口，反而更輸。

## 1. 研究問題

- 主策略與 C 候選的右尾集中在熱門產業的大中型強勢股：2615、2603、2383、2368、2059、3231。
- 本題要回答兩件事：
  1. 這些報酬來自產業、個股，還是產業 × 個股的交互作用。
  2. 能不能做成一條追蹤誤差可控、大型股的第二條腿，補主策略在權值股領漲時的缺口。

## 2. 機制（事前）

- **贏家**：外資（大型、高流動、順勢），以及 00981A、安聯台灣科技這類由下而上的基金（在熱門產業裡買營收與股價同時最強的公司）。
- **假說**：兩段式資訊擴散。產業需求衝擊出現後，產業內營收與股價最先轉強的龍頭，因為營運槓桿與市占提升，會分到大部分衝擊；主題資金只能進大型、流動性高的股票。
- **對手**：照權重的被動資金、受 10% 上限的國內基金、在大型強勢股上有處分效應的散戶。
- **為什麼沒被套利**：集中持有少數龍頭的追蹤誤差大、主題結束時會崩盤（2022 航運、2024-07 AI 回檔）、大型基金的容量不夠。
- **可證偽點**：Q1 差中差 > 0，而且本題 > 不分產業對照（CTRL）。

## 3. 資料（全部原地唯讀引用，raw/manifest.json 有 sha256）

- **價格**：V7 凍結面板加下市股價。52 週比用 tcore.S52（gapfix）。
- **營收**：V4 已公告值 y3_ab。
- **PIT 市值**
  - 2013 年起：o8 shares_pit，加 4763／2317 修正；2014 年起通過閘門。
  - 2006～2012：o7 shares_est，非 PIT。這個版本把 2408 南亞科、2317 排在 2330 前面，可見嚴重失真，只作描述。
  - 題 1 的 D1 閘門同樣沒過：前 100 重疊最小 91，門檻 93。
- **產業別**
  - 切換後：用題 1 的 raw/industry_pit.csv（多份快照逐段生效）。
  - 切換前：用本題自建表。2023-07-03 起用現行分類；2007-07～2023-07 用 ISIN 2023-01-16 快照；2007-07 以前把電子 8 類合併。2023-07-03 新增 4 類已用 GVM、商周報導核實。
- **引擎**
  - 紙上：lc_lib.vrun，與 tcore.run 逐值相同，最大差 0。
  - 可執行：lc_exec.py，移植 o8 s7_exec 的 K5T 權重版，另加壓力成本與處置期收盤賣出。
  - 與 V7X 的差異：未實作 f2 的 20% 量能上限與 f3 的盤後零股量上限。大型股處置事件極少，這個差異的影響小。
- **主策略**：V7X B10 可執行淨值（navs_oos_R4，100 萬 GROW）與 V7 plans_j7 名單。

## 4. 探索（只用挑選期 2014-01～2016-12；explore.json）

- 宇宙每月 100 檔，有 2 家以上成員的產業約 18～19 個；金融平均 16.6 檔。
- 2330 在宇宙中的權重平均 18%。B 本身對 0050 每月 −0.22～−0.27pp。原因是全市值加權而非自由流通，又含第 51～100 名。
- **Q1 差中差**
  - 切換前：+0.07pp，t 0.09。
  - 切換後：−0.15pp，t −0.13。
  - 依事前規則（t < 1），在挑選期「先產業、後龍頭」看不到。
- **2330 被選中的月份比例**：切換前 33%，切換後 0.2%。H52 前 3 產業很少是半導體，就算是，2330 在產業內的分數也排不到前 2。
- **只有 2 家成員的產業**：分數較高者減較低者，切換後 −0.83pp（重疊觀測，只作描述），等於隨機甚至更差。
- **挑選期沒有空頭可看**：2008、2011 不在挑選期。
- **周轉**：選定版單邊每月約 10%；成本約 0.06～0.07pp／月（紙上）。

## 5. 模型迭代（iterations.json；誠實揭露）

- **版本數**：兩種產業排名 × 4 版（v1 原規格、v2 個股緩衝 n+2、v3 溢出給同產業、v4 溢出給骨架）× 48 格，加上參照，共約 390 組設定。另有 8 個候選做了 Q7(b)，每個 1,000 次。
- **切換前（COMP、k=5）**：IND 24 格的挑選期 IR 全部為負，最佳 v4 n1 PURE 為 −0.36。
- **切換後（H52、k=3）**
  - 最佳是 n=1 PURE，IR 0.35，Q7(b) p95.3。
  - 但 n=2 鄰域是 −0.08，不同號，不成立。
- **結果**：兩種產業排名都回到事前預設版（SAT、n=2、市值加權、15%）。
- **S2（周轉與追蹤誤差用本題的值）**：IR 0.3／0.5／0.66 時 G1～G7 通過機率都是 0%；拿掉 DSR 後是 0.4%／5.3%／21.1%。依使用者指示，不因 S2 停題。

## 6. 凍結規格

- 見 frozen_spec 欄位：PREREG sha 3c329aa8…，2026-11-03T04:31:40Z；另有 AMEND v1。
- 題 1 在本題凍結前 33 分鐘凍結，所以凍結版用的是題 1 選定版的產業排名。

## 7. 樣本外與全期結果

### 7.1 選定版（可執行 K5T，100 萬，GROW，13 個 j 中位數，對 0050）

| 期間 | 月超額 | p10 | t | 年化 | 0050 |
|---|---|---|---|---|---|
| 驗證期 2017～2026 | −0.087pp | −0.275 | −0.48 | 22.8% | 23.8% |
| 2019 年起 | −0.158pp | −0.294 | −0.72 | 27.5% | 28.8% |
| PIT 段 2014～2026 | −0.103pp | −0.225 | −0.66 | 19.3% | 20.4% |
| 全期 2006～2026（2014 年前非 PIT） | −0.080pp | −0.123 | −0.71 | 13.5% | 14.3% |
| PIT 段，壓力成本 | −0.207pp | −0.334 | −1.33 | 17.9% | 20.4% |

- 追蹤誤差 6.7%～7.8%。
- 窗頻最大回撤：PIT 段 −23.4%，全期 −53.7%。
- 30 萬 FIX 的 PIT 段 t −1.05；300 萬 −0.75；1,000 萬 −0.84。

### 7.2 對照（紙上一般成本；驗證期，對 0050 的 t／對 B 的 t）

- 本題 SAT：−0.21／0.24。
- 本題 PURE：0.07／0.23。
- CTRL SAT：0.95／1.53。
- CTRL PURE：1.36／1.55。
- IND_ONLY PURE：0.16／0.37。
- IND_ONLY SAT：−0.22／0.35。
- B：−0.96。

### 7.3 2019 年起和 0050 並列（可執行 100 萬 GROW，各 j 中位數年報酬 %）

| 年 | 本題 SAT | 本題 PURE | CTRL PURE | 主策略 | 0050 |
|---|---|---|---|---|---|
| 2019 | 23.6 | 24.8 | 17.0 | 12.4 | 33.8 |
| 2020 | 41.0 | 31.8 | 49.6 | 32.9 | 46.8 |
| 2021 | 21.1 | 27.8 | 79.1 | 29.3 | 11.6 |
| 2022 | −12.5 | −10.0 | −21.5 | −19.1 | −15.4 |
| 2023 | 23.7 | 39.6 | 25.4 | 27.6 | 16.7 |
| 2024 | 23.4 | −8.2 | 21.8 | 12.0 | 51.7 |
| 2025 | 50.6 | 60.4 | 74.1 | 61.8 | 48.5 |
| 2026（至 10-22） | 50.7 | 52.5 | 68.3 | 31.8 | 51.1 |

### 7.4 Q1～Q7

- **Q1 差中差**
  - 驗證期 +0.89pp（t 1.20）；2019 年起 t 0.53；PIT 段 t 1.00。
  - 門檻 t ≥ 2.0，不成立。
- **Q2 前 k 產業內強勢股對該產業整體**
  - n=1：+0.68pp（t 1.52；PIT 段 1.95）。
  - n=2：+0.36pp（t 1.23）。
  - n=3：+0.32pp（t 1.36）。
  - 方向符合預測，但沒有達到 t 1.5（n=1 例外）。這份個股優勢也沒有轉成組合超額，因為選中的產業本身不贏 B。
- **Q3 拆解（PURE，驗證期／2019 年起）**
  - 產業效果：+0.155pp（t 0.36）／−0.017pp（t −0.03）。
  - 產業內個股效果：−0.025pp（t −0.07）／+0.044pp（t 0.09）。
  - 個股獨立效果：+1.15pp（t 1.55）／+1.21pp（t 1.41）。
  - **研究員點預測「本題 > 對照，差額 t 1.0～1.5」被否證**：實際 −1.02pp，t −1.50。
- **Brinson（PURE 對 B，驗證期）**：配置 +0.07pp（t 0.18）；選股 +0.08pp（t 0.36）；交互 +0.29pp（t 1.26）。三項都不顯著。
- **Q4 子樣本（SAT 對 0050，PIT 段，pp／月）**
  - 產業營收離散度高：−0.05；離散度低：−0.10。
  - j0～j4：−0.03；j8～j12：−0.16。
  - 主題擴散期（2020～2021、2023～2025）：−0.18；其他期間：−0.05。這一點與預測相反，主題期反而較差。
  - 空頭後反彈 14 窗：−0.68（PURE −1.93）。符合「較弱」的預測。
  - 主題結束期：+0.26，與預測相反。2022 年窗頻回撤 −24.2%，0050 −29.5%；2024-07～08 為 −5.2%，0050 −5.5%。
- **Q5 主策略右尾歸因**
  - 主策略（PIT 段）前 20 大股票-年，45% 多數時間屬於前 3 產業，隨機期望 19%，預測 ≥ 60%，不成立。
  - 全期 35%，隨機 22%。
  - R² 0.04。
  - 本題名單與主策略重疊 13%，CTRL 為 26%。
- **Q6 第二條腿**
  - SAT 超額與主策略的相關 0.52，< 0.6，成立。
  - 對（2330−0050）的 β −0.31，預測 ≥ 0，不成立。
  - CTRL SAT 的相關 0.65。
- **Q7 安慰劑（驗證期 p 值，扣成本後／扣成本前；門檻 < 0.05）**
  - (a) 隨機產業：0.20／0.18。
  - (b) 產業內隨機選股：0.073／0.063。
  - (c) 兩者都隨機：0.063／0.042。
  - 真產業排名對隨機產業幾乎沒有優勢；產業內選股的優勢接近 p95，但沒有過。

### 7.5 與主策略比（可執行，PIT 段）

- 配對 t（本題 − 主策略）+0.30。
- 超額相關 0.51。
- 主策略本身 PIT 段 IR −0.18，2019 年起年化 22.4%。
- 50/50 組合 IR −0.157，主策略單獨 −0.176，bootstrap p 0.67。不建議當第二條腿紙上追蹤（條件 p < 0.10 不成立）。

## 8. 穩健性

- **分年**：見 7.3。輸 0050 最多的是 2019 年（−10pp）與 2024 年（−28pp）。
- **右尾（PURE）**：PIT 段 t 0.23 → 拿掉 2021 航運與 2025 AI 後 −0.30 → 拿掉貢獻前 10 檔（8454、1314、2408、2615、9945、2345、1722、3231、2603、2609）後 −2.05。少數股票就撐起全部的點估計。
- **鄰域（PIT 段 IR）**
  - IND n=1：+0.25（PURE）、+0.10（SAT）。
  - IND n=2、3：多數為負。
  - CTRL 24 格全部為正（0.04～0.39），但也全部不顯著。
- **三段幾何超額（可執行，pp／年）**：−0.55、+0.12、−1.59。拿掉最好 3 年 t −2.02。
- **切換前版本（COMP、k=5）**
  - DEF 驗證期對 0050 t −0.13。
  - 但 COMP 的 IND_ONLY PURE 驗證期 t 1.39（對 B 1.77），比題 1 選定的 H52 好。這是產業層「事後看起來較好的版本」，只作記錄，不採用。

## 9. G1～G8 逐條（主欄：可執行 K5T，100 萬 GROW，13 個 j）

| 門檻 | 實際 | 結果 |
|---|---|---|
| G1 可執行 | 一般成本 PIT 段 j 中位數 −0.103pp、p10 −0.225；壓力成本 −0.207／−0.334；全期一般成本 −0.080／−0.123 | ✘ |
| G2 顯著 | 全期 t −0.71；2019 年起 t −0.72 | ✘ |
| G3 窺探校正 | 全族群 2,385：SPA p 1.0、DSR 1.8e−16；F_main 461：p 1.0、DSR 3.9e−4 | ✘ |
| G4 穩健 | 三段 −0.55／+0.12／−1.59；拿掉最好 3 年 t −2.02（PIT 段）；鄰域翻號 | ✘ |
| G5 資料 | PIT 市值只有 2014 年起；2006～2013 非 PIT；2007～2023 個別公司改類無法 PIT | ✘ |
| G6 複製 | 待複製員 | ✘ |
| G7 容量 | 100 萬／300 萬／1,000 萬 GROW 全期 t −0.71／−0.77／−0.91 | ✘ |
| G8 前瞻 | 未開始 | ✘ |

- 「待觀察」條件：驗證期 t −0.48（門檻 ≥ 2）、Q7(a) p 0.20、Q7(b) p 0.073、G1 不過，所以全部不成立，判定**不可用**。
- 機制判定：
  - Q1 驗證期 t 1.20 < 2，而且 IND−CTRL < 0，所以「產業 × 個股」不成立。
  - IND_ONLY 與 CTRL 都不顯著（t 0.16／1.36），所以不能寫「alpha 來自產業」，也不能寫「alpha 來自大型個股動能」。
  - 只能說：點估計指向個股動能，產業層是多餘的。

**差多少、怎麼補**
- 本題可執行 IR：PIT 段 −0.18、2019 年起 −0.26。
- 要過 G2：全期 t ≥ 3 需要 IR 約 0.66；PIT 12.8 年要 t ≥ 3，需要 IR 0.84。差距約 0.85～1.0 IR，沒有可補的方向。
- 最好的對照 CTRL PURE（可執行 IR 0.31～0.39）要達 t 3，需要 (3/0.31)² ≈ 94 年；2019 年起要 t ≥ 2，需要約 28 年。
- 真正能補的只有前瞻時間。如果研究中心想保留這條線，建議只把 CTRL PURE 與 n=1 PURE 列入前瞻「機制觀察」，不投入資金。

## 10. 散戶怎麼執行（規則書；本題判定不可用，僅供紙上追蹤）

- 每月營收期限日 D_m 之後（例如 2026-11 的官方日是 11-10、11-11），收盤後依序做：
  1. 看題 1 的產業排名：可輪動產業依「產業指數 ÷ 52 週最高」排序，取前 3。已持有的產業只要排名 ≤ 5 就續抱。
  2. 在這 3 個產業的市值前 100 成員中，算 3 個百分位：52 週比、6-1 月強度、3 月營收年增率。取平均分數，每個產業取前 2；已持有的股票只要產業內名次 ≤ 3 就續抱。
  3. 6 檔等權（每檔 15% 上限不可行，所以是 1/6）。
- 帳戶配置：
  - 30%：強勢股腿。
  - 70%：骨架。回測用的是 B（PIT 市值前 100，市值加權）；實務可以用 0050 代替（描述版 SAT50 對 0050 的 t 約 −0.06）。
- 最新訊號日 2026-10-12（J0）的名單：
  - 產業：光電業、航運業、塑膠工業。
  - 股票：2603、2615、1303、1326、2409、3008，每檔占強勢股腿的 1/6。
- 100 萬帳戶（每檔 5 萬，以 10-12 收盤價計）：
  - 2603：195 股（零股）@255.5。
  - 2615：413 股 @121。
  - 1303：165 股 @303。
  - 1326：651 股 @76.8。
  - 2409：1 張＋406 股 @35.55。
  - 3008：6 股 @7,325。
- 1,000 萬帳戶（每檔 50 萬）：
  - 2603：1 張＋956 股。
  - 2615：4 張＋132 股。
  - 1303：1 張＋650 股。
  - 1326：6 張＋510 股。
  - 2409：14 張＋64 股。
  - 3008：68 股。
  - 下單前請重新核對價格。
- 扣成本的實盤預期（2019 年起）：年化約 27.5%，同期 0050 28.8%；預期每年少約 1～2pp，追蹤誤差約 8%。

## 11. 限制與誠實揭露

- 挑選期只有 36 個月（D1 沒過），檢定力很低。
- 2017 年以後的大型強勢股報酬，在 o3、o7、o9 已經部分看過，驗證期不是乾淨樣本外。
- 2006～2013 用的是非 PIT 股數（o7 shares_est），名次嚴重失真。三段中的 SEG1 與全期 t 只能當描述。
- 可執行引擎是 o8 的 K5T 權重移植版，不是 V7X 本體（缺 f2／f3）。2020-10 以前的零股以收盤價計，屬回測上緣。
- 凍結後的修正：
  - AMEND v1：j=2 的 2018-01 一窗資料錯誤。
  - s5 彙總程式修了兩個 bug：Q5 日期對應、Q7 日期字串。這兩個修正不影響任何規格。
- 題 1 的產業與宇宙，和本題的宇宙定義略有不同：題 1 沒有股數的存活股用 ADV 插補，本題直接排除。
- 給總監的觀察：題 1 資料夾的 data/oos_windows.parquet 檔案時間（03:56）比其 PREREG.json（03:58）早 2 分鐘，建議審查員核對題 1 是否在凍結前讀過樣本外。
- 最新月的部分營收（2603、2615、2409）在 V4 口徑下缺值，分數改用另外兩項的平均。
- 大型中間檔（cs 快取、paper／exec／placebo 結果，約 200MB）已從 scratchpad 刪除。要重現時，依序執行 s0b、s1、s2、s3、s4 paper／paper_comp／exec／placebo a、b、c／analysis，全部用 python3 -I -B。
- out/ 留有關鍵窗報酬（window_returns_key.parquet）與名單樣本，供複製比對。

---
# 審查與複製（o12 工作流原文，總監已依此強制勘誤，見 original_o12/DIRECTOR_REPORT.md）

## 統計審查

**final**

不可用

**note**

我從 /home/user/-/research/tw_stock_selection/agents/o_lc_sector_strong_stocks/out/window_returns_key.parquet 重算，選定版可執行主欄（VAL、Y19、PIT、FULL 的 t 與年化）和 FINDINGS 完全一致，研究員判定「不可用」我確認。目前的程式和資料雜湊都對得上 PREREG_AMEND 與 PREREG 的 data_sha256。

要修正的是幾個偏樂觀的子結論：
- CTRL「大型個股動能有一點東西」：其實只靠 2021 一年。拿掉 2021 後 t 0.56，拿掉最好 3 年 −0.43。
- n=1 PURE 列入前瞻：它是挑選期挑出來的最佳格，有選擇偏誤，不應列入。
- IND−CTRL：可執行版只有 t −1.4 左右。只能說產業層沒有證據加值，不能說它有害。

另外發現兩處文件瑕疵：PREREG 的文字欄和實際執行的 T1 模式不一致；S2 模擬沒有用到本題的追蹤誤差與周轉。PIT 段沒有看到存活者偏誤；2014 年以前的股數和 2015 年以前的產業別都不是 PIT，只能當描述。題 1 是否先看過樣本外，現在無法用檔案時間核實；但那種污染只會往有利方向偏，不影響本題的負面結論。

驗證程式：/home/user/-/research/tw_stock_selection/agents/o_lc_sector_strong_stocks/review_stat_o12.py（python3 -I -B，只讀不改研究員檔案）。輸出：/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/review_o12.json

**issues**

- 我重算後確認研究員判定「不可用」，找不到能推翻成「待觀察」的證據。用 out/window_returns_key.parquet 加 lc_lib.bench_win 重算選定版（K5T、100 萬、GROW、13 個 j 中位數）對 0050 的 t：驗證期 −0.48、2019 年起 −0.72（年化 27.5% 對 28.8%）、PIT 段 −0.66、全期 −0.71，和 FINDINGS 逐字相同。改用 Newey-West（落後 3 期）的 t 是 −0.43／−0.71／−0.58／−0.67，結論不變。13 個 j 裡最好的一個，PIT 段 t 也只有 0.08。
- 「大型股個股動能本身有一點東西」這句應該刪掉或降級。CTRL PURE 可執行版的正 t 幾乎全部來自 2021 一年：PIT 段 2021 年月超額加總 +65pp，其他 12 年合計約 +44pp，其中 2016 是 −23.5、2024 是 −24.6。拿掉 2021 後 t 從 1.13 掉到 0.56；拿掉 2021 和 2025 後是 0.08；拿掉最好 3 年是 −0.43。2019 年起則是 1.06、0.25、−0.15、−0.69。CTRL SAT 拿掉 2021 後直接轉負（PIT 段 −0.20）。所以這不是「有一點但不顯著」，而是單一年度（航運）的事件。
- 研究員建議把 n=1 PURE 列入前瞻「機制觀察」，這有選擇偏誤。n=1 PURE 是約 390 組設定、兩種產業排名裡挑選期 IR 最高的一格（0.35），鄰域 n=2 卻是 −0.08，研究員自己也判它不成立。把挑出來的最佳格拿去前瞻追蹤，等於讓多重檢定溢出到 G8。CTRL PURE 的分數公式（52 週比、6-1 月強度、營收年增）和 2017 年以後的大型強勢股，在 o3、o7、o9 已經看過，也不能算乾淨的前瞻候選。
- 多重檢定的自由度沒有完全算進去。研究員在 COMP 版挑選期 24 格全部為負之後，改用題 1 的 H52（k=3、產業緩衝 2）再跑一輪；這次換版本身就是一個分叉。兩種產業排名的樣本外都是負的，所以不影響「不可用」；但任何正面的子結論，例如 Q2 n=1 PIT 段 t 1.95、Q7(b) p 0.073，都要再打折。另外 13 個 j 的超額彼此平均相關 0.64（最低 0.26），單看任何一個 j 都會高估；例如 CTRL PURE 驗證期最好的 j 有 t 2.13，只是 j 窺探。
- IND−CTRL 的差，可執行版比紙上版溫和，FINDINGS 只報了紙上版。紙上 PURE 驗證期是 −1.02pp（t −1.50）；可執行配對是 PURE −0.93pp（t −1.40）、SAT −0.30pp（t −1.46），2019 年起 t −1.14／−1.15。方向一致、都不顯著，所以只能寫「沒有證據顯示產業層有加值」，不能寫成產業層「明確有害」。
- 前視偏誤：凍結檔描述和實際執行的程式不一致。PREREG body 的 sector_rule 欄寫的是「COMP、k=5」，industry 欄寫的是本題自建表；實際 mode 是 T1，用的是題 1 的 industry_pit.csv 和 H52。凍結文字和程式不一致，算文件瑕疵，複製員要以程式雜湊為準。題 1 的 industry_pit.csv 在 2015 年以前，是用 2015 年快照往回沿用的（source 欄寫「回溯：沿用最早觀測」），2015 年以前的產業別有前視。SEG1 與全期 t 只能當描述，這一點研究員有揭露。
- 題 1 是否先看過樣本外，這個問題沒有釐清，而且會傳到本題。本題凍結版直接沿用題 1 選定的 H52 排名。目前檔案時間是題 1 PREREG.json 04:00:33、oos_windows.parquet 04:03:19，已經看不到 FINDINGS 說的「parquet 早於 PREREG 2 分鐘」，檔案應該被重寫過，現在無法用檔案時間判定。就算題 1 被污染，偏誤也是往有利方向，不會推翻本題的負面結論；但不能拿來替產業層辯護。
- 存活者偏誤：PIT 段我查不到實質問題。每年 1 月的訊號日、60 日均成交值前 100 名裡，沒有股數而被排除的，只有 2014-01 的 2 檔（3697、5280），兩檔後來都下市了；2015 年以後是 0 檔。PIT 段宇宙每月平均有 3.35 檔（最多 9 檔）日後會下市的股票，可見下市股有留在宇宙裡。2006～2013 用 shares_est 往回推，缺部分下市股、名次也失真，只能當描述。
- AMEND v1 是在 exec 階段出現 RuntimeWarning 之後才做的，那時已經跑過樣本外。它只把 j=2 的 2018-01 一窗併入下一窗，影響很小，不改結論；但嚴格說，這是看過樣本外輸出之後才改的程式。
- S2 的檢定力模擬裡，切換前後兩版的 TE_used（0.0348 對 0.0468）和周轉都不一樣，by_IR 的 G1～G4 通過機率卻逐位相同。看起來模擬沒有真正用到這兩個參數，「S2 用本題周轉與追蹤誤差」的說法不成立。
- 第 10 節給了一個判定為不可用的策略具體的股數和下單清單（100 萬、1,000 萬帳戶）。這可能被誤讀成可以實盤，建議只保留紙上追蹤的說明。

## 可執行審查

**final**

不可用

**note**

On cost and executability, the 'not usable' verdict stands. I reproduced all headline t values exactly, and my copy of the engine matches the original with zero error. Three extra checks found nothing that rescues the strategy or makes it look worse: blocking orders at limit-up/limit-down (effect ≤0.03pp/month), entries during disposal periods (about 1.3% of new entries) and the pre-2020-10 odd lots filled at the close (no valuation bias). Two reporting issues need fixing: (1) cost under GROW is divided by initial capital and overstates cost, so read FIX (CTRL PURE about 4%/year, SEL about 1%/year); (2) the rulebook says to use 0050 as the skeleton but quotes expectations from the B skeleton, and the SAT50 version is flat against 0050 (t about -0.05). The strategy itself is about zero and highly sensitive to the rebalance day (j). CTRL PURE turns negative under stress costs for J6 and J12, so it should not get capital. My verification script is /home/user/-/research/tw_stock_selection/agents/o_lc_sector_strong_stocks/review_exec_o12.py (python3 -I -B, about 1 minute); it is read-only and leaves the researcher's files unchanged.

**issues**

- Verdict stands and the headline numbers reproduce exactly. I recomputed the median t over the 13 j values for executable K5T at 1M GROW from out/window_returns_key.parquet: SEL is -0.48 for validation 2017-2026, -0.72 from 2019, -0.66 for the PIT segment 2014-2026 and -0.71 for the full period. Under stress costs SEL is -1.03 / -1.16 / -1.33 / -1.60, CTRL SAT validation is 0.67 and CTRL PURE validation is 1.21. All match FINDINGS. My line-by-line copy of run_exec on the J0/J6/J12 plans has max error 0 against the researcher's stored window returns.
- Limit-up and limit-down blocking does not change the result. A variant that leaves an order unfilled when the stock opens locked at limit-up (buy) or limit-down (sell) blocks at most 10 orders per path, and most are in 2006-2008. The monthly mean moves by at most ±0.03pp and t moves by about ±0.07. The verdict cannot be overturned on this point and does not get better either.
- Disposal and full pre-payment at entry are rare: 6 to 9 of the 509 to 640 new sleeve entries were in a disposal period on the trade day (about 1.3%). These buys still fill at the open in the engine, which ignores batched matching every few minutes and pre-payment. The 2026 cases are clustered (2383, 3443, 6442, 3037 and others) and need watching in live trading, but they are not material in the backtest.
- Pre-2020-10 odd lots filled at the close: I checked for a valuation bias and there is none. The value of the position at t2 does not depend on when it was filled, and the cost of buying at the close already goes through cash. The variable fcl in lc_exec.py is computed but never used, which is harmless. The researcher's 'upper bound' note applies only to after-hours odd-lot liquidity.
- Reporting issue 1: cost_pp_month under GROW is divided by initial capital × number of months, so it grows with NAV and overstates cost. CTRL PURE at 1M shows 1.40pp/month under GROW against 0.33pp/month under FIX, and SEL shows 0.19 against 0.087. Read FIX as the real cost: SEL about 1%/year, CTRL PURE about 4%/year. CTRL PURE's 6 names add about 2.5 new names a month, so the sleeve turns over about 41% one-way each month.
- Reporting issue 2: the rulebook says retail can use 0050 for the 70% skeleton, but its 'expected live return 27.5% vs 28.8%' comes from the B skeleton. B itself trails 0050 under executable costs (PIT t -1.95). The SAT50 version (0050 skeleton) in results.json gives 29.4% vs 28.8% from 2019 and t -0.05, roughly flat. So the 'negative against 0050' for SAT is mostly B-skeleton drag, and the sleeve itself is near zero (SEL_PURE executable validation t -0.10, PIT -0.06). Either way it is nowhere near the thresholds, so the verdict does not change, but the rulebook expectation should be rewritten in SAT50 terms.
- CTRL PURE should not be treated as a candidate even for 'mechanism observation only'. With limit blocking plus stress costs, PIT t is +0.86 for J0 but -0.11 for J6 and -0.21 for J12, and full-period stress t is 0.19. Tracking error is 24%, max drawdown -66% and cost about 4%/year. A point estimate this sensitive to j and to cost is no reason to commit capital.
- Data timing and j sensitivity: J0 conservatively treats the previous month's revenue as unavailable on the deadline day and uses the month before that, so there is no look-ahead. But the topic-1 sector order flips from day to day: 光電 is #1 on 2026-10-12, #3 on 10-13 and #5 on 10-21. Executable validation t ranges from -0.20 (J0) to -0.85 (J6). Live results depend heavily on which day you rebalance, so read p10 rather than the median. In the latest rulebook, 2603, 2615 and 2409 have missing revenue in V4, while a retail investor on 10-12 could see actual revenue. The published picks may therefore not match what the rule produces on complete data.
- Minor optimistic assumptions: the executable fee uses a 2.8 (72%) discount for the whole 2006-2026 period, deeper than discounts in the early years; and the engine omits V7X's 20% volume cap (f2) and the after-hours odd-lot volume cap (f3). For large caps these are small and only push costs up, which can only hurt the strategy further.

## 獨立複製

**reproduced**

True

**my_numbers**

程式在 /home/user/-/research/tw_stock_selection/agents/o_lc_sector_strong_stocks/replicate_o12/（rlib.py、r1_plans.py、r2_paper.py、r3_stats.py、r4_exec.py、r5_exec_stats.py、r6_q.py、r7_summary.py），全部用 python3 -I -B 執行。結果存在 out/paper_stats.json、out/exec_stats.json、out/q_stats.json。以下都是 13 個 j 的中位數，括號內是研究員的數字。

【主欄：IND_n2_cap_c15_SAT，可執行 100 萬 GROW，一般成本，對 0050】
- 驗證期 118 窗：−0.101pp（−0.087），t −0.53（−0.48），IR −0.17（−0.15），年化 22.6%，0050 23.8%（22.8%／23.8%）。
- 2019 年起 94 窗：−0.152pp（−0.158），t −0.70（−0.72），年化 27.5%，0050 28.8%（一致），追蹤誤差 7.6%（7.8%）。
- PIT 段 154 窗：−0.117pp（−0.103），t −0.76（−0.66），年化 19.1%，0050 20.4%（19.3%／20.4%），窗頻最大回撤 −24.1%（−23.4%）。
- 全期 247 窗：−0.094pp（−0.080），t −0.81（−0.71），年化 13.3%，0050 14.3%（13.5%／14.3%），最大回撤 −56.1%（−53.7%）。
- 壓力成本：PIT 段 −0.231pp（−0.207），t −1.50（−1.33）；全期 t −1.83（−1.60）。
- 成本 0.13pp／月（0.19）；零股占成交 69%（64%）。
- 其他資金級距：30 萬 FIX 的 PIT 段 t −1.25（−1.05）；300 萬 GROW t −0.66（−0.75）；1,000 萬 GROW t −0.61（−0.84）。
- 三段幾何年超額（pp／年）：2006～2012 −1.37（−0.55）；2013～2018 −0.41（+0.12）；2019～2026 −1.57（−1.59）。
- 拿掉最好 3 年後的 t：PIT 段 −2.10（−2.02），全期 −2.02（−1.89）。

【紙上，一般成本】
- SAT 驗證期：對 0050 −0.047pp，t −0.25（−0.042，t −0.21）；對 B +0.026pp，t 0.17（+0.035，t 0.24）。
- PURE 驗證期：對 0050 t 0.01（0.07）；對 B t 0.17（0.23）。
- CTRL SAT 驗證期：+0.246pp，t 0.90（+0.258，t 0.95）；對 B t 1.47（1.53）。
- CTRL PURE：驗證期 +1.011pp，t 1.28（+1.064，t 1.36）；2019 年起 t 1.04（1.22）；PIT 段 t 1.30（1.34）。
- IND_ONLY：PURE 驗證期對 0050 t 0.16（0.16），對 B t 0.36（0.37）；SAT 驗證期 t −0.21（−0.22）。
- B 對 0050：PIT 段 −0.109pp，t −1.73（−0.11，t −1.72）；驗證期 −0.077，t −0.96（−0.078，t −0.96）。
- 挑選期：預設版 IR −0.42，t −0.73（−0.44，t −0.77）；對 B t 0.23（0.20）。v1 格點最佳是 n=1 PURE，IR 0.43（研究員最佳 0.35 是 v3 版）；n=2 PURE 的 IR −0.05、−0.23，都是負的，鄰域不成立的判定相同。

【可執行對照（100 萬 GROW）】
- CTRL SAT：驗證期 +0.156pp，t 0.58（+0.185，t 0.67）；2019 年起年化 30.6%（31.4%）；PIT 段 t 0.42（0.52）；壓力成本 PIT 段 −0.056pp（−0.015）。
- CTRL PURE：
  - 驗證期 +0.86pp，t 1.10（+0.93，t 1.21）。
  - 2019 年起 37.4%，t 0.91（39.7%，t 1.06）。
  - PIT 段 27.2%，t 1.10（27.4%，t 1.13）。
  - 全期 16.6%，t 0.82（18.3%，t 1.11）。
  - 追蹤誤差 24.8%（24%），全期最大回撤 −67%（−66%），壓力成本 PIT 段 t 0.32（0.39）。

【Q3（PURE，紙上，驗證期／2019 年起）】
- 產業效果：+0.154pp，t 0.36／−0.017，t −0.04（+0.155，t 0.36／−0.017，t −0.03）。
- 產業內個股效果：−0.026，t −0.07／+0.017，t 0.04（−0.025，t −0.07／+0.044，t 0.09）。
- CTRL−B：+1.09，t 1.47／+1.05，t 1.24（+1.15，t 1.55／+1.21，t 1.41）。
- IND−CTRL：PURE −0.94，t −1.35（−1.02，t −1.50）；SAT −0.285，t −1.34（−0.30，t −1.50）；2019 年起 −0.87，t −1.11（−0.90，t −1.23）。

【Q2（自定口徑）】
- 強勢股等權對產業市值加權：n=1 驗證期 +0.58pp，t 1.32，PIT 段 t 1.74（+0.68，t 1.52／1.95）；n=2 +0.27，t 0.99（+0.36，t 1.23）；n=3 t 1.05（1.36）。

【Q1（自定口徑）】
- 驗證期 +0.45pp，t 0.55（+0.89，t 1.20）；挑選期 t 0.50（−0.09）。這一項沒有重現。

【逐窗比對研究員的 window_returns_key.parquet（PIT 段）】
- 紙上 B：平均差 ≤0.002pp／窗，絕對差中位數約 0。
- 紙上 SEL：中位數絕對差 0.0002pp，平均差 |≤0.016|pp。
- 可執行 SEL：比研究員低 0～0.014pp／月。
- 訊號日：13 個 j 共 3,207 個日期完全一致。
- 選股：PIT 段 j=6 前 3 產業 154 窗全部一致；個股名單 SEL 有 12 窗不同，CTRL 有 35 窗不同。

**diff_explained**

結論：主欄（選定版 SAT 可執行對 0050，各期間的超額、t、年化、追蹤誤差、回撤，加上壓力成本）和紙上主要數字，都在合理誤差內重現。方向與判定完全一致：沒有一段顯著為正，IND 輸 CTRL，B 輸 0050。差異來源如下。

一、我照規格自寫時，一開始和研究員不同、後來查出原因的地方（都是資料口徑，不是研究員的錯）
1. 面板在 flag==1 的除權息日把 ret 設成 0
   - 共 23,583 格，包括 2330 每季配息。直接用 V7 adj 會讓 B 每月高約 0.03pp。
   - 我改成「當日成長＝收盤÷開盤 ×（1＋當日非事件股開盤跳空中位數）」之後，B 逐窗與研究員幾乎完全相同。研究員應該也是用同樣的 exfix。
   - 訊號（52 週比、RS）若也改用 exfix 指數，選股不一致反而變多（SEL 28 窗、CTRL 45 窗），所以訊號維持用 V7 adj。
2. 下市股價
   - 2823 中壽在 2019～2021 年 V7 沒有價格，必須用 o_infra_v3_rebuild/delist_prices.parquet 補，否則 B 少一檔。
   - 可執行引擎也要把「之後再無成交」的持股轉成現金，否則零成本下還有 −0.04pp／月的死錢拖累。
3. 訊號日
   - 規格寫「面板交易日」，所以 2018-01-13 週六殘列要算進第 j 日。
   - 2023-04 的期限日研究員用 04-17。規格沒有寫這一點，是我從研究員的輸出日期反推的。

二、還剩下的小差異
1. 選股不一致
   - SEL 約 8% 的窗、CTRL 約 23% 的窗名單不同，多半是分數近乎同分。例如 2020-04 的 6669 與 2324 只差 7e−5，接著被個股緩衝放大成持續好幾個月的差異。
   - 另外 3037 在 2022-02-22 缺一天，我的 gapfix 讓它的 52 週比變成 NaN。
   - 我試過三種百分位定義（m1、min、nall），也試過金融股營收納入排名池，選股不一致都沒有變少。
   - 影響：紙上 SAT 差約 0.01～0.04pp／月，CTRL PURE 差約 0.1pp／月，t 差 0.05～0.2。
2. 可執行引擎是我自寫的近似版，沒有移植 o8 s7_exec
   - 我的版本：K5T 衝擊用 V60 代替 OA；零股只加一跳；沒有開盤跳空漂移，也沒有處置期規則。
   - 結果：可執行比紙上約低 0.06pp／月（研究員約 0.04～0.06）；壓力成本 t 比研究員負約 0.2；成本占比 0.13 對 0.19pp（兩邊計算口徑可能不同）。
   - 1,000 萬方向和研究員相反：我的 t −0.61 比 100 萬好，研究員是 −0.84 比較差。原因是我的模型裡每筆最低 20 元手續費的影響大於衝擊，所以規模越大越好。
3. 三段幾何年超額的前兩段不一致（2006～2012：−1.37 對 −0.55；2013～2018：−0.41 對 +0.12）
   - 我的可執行版在 2006～2018 每月比研究員低約 0.03pp。
   - 2014 年前是非 PIT 股數，選股不一致較多。
   - 年超額的定義（段內年化相減）可能也不同。
   - 第三段一致（−1.57 對 −1.59）。
4. Q1 差中差沒有重現（驗證期 t 0.55 對 1.20，挑選期 t 0.50 對 −0.09）
   - 規格沒有寫上下半怎麼分、用什麼加權、窗怎麼定，我用的是每個產業內依分數對半、等權、毛報酬。
   - Q2 用等權強勢股時方向一致，但量偏小（n=2 驗證期 t 0.99 對 1.23）。

三、沒有複製的部分
- Q7 安慰劑、G3（SPA_c、DSR）、S2、Q5、Q6、Brinson、右尾分析、COMP 切換前版本、v2～v4 迭代、30 萬以外的 FIX 級距。
- 題 1 產業排名本身，直接用 data/t1_sector_order.json 當輸入。

四、出處註記
- PREREG 的 topic1_prereg_sha256 欄位是 73efe56a…，這是題 1 PREREG.json 的檔案雜湊；規格與 t1_sector_order.json 寫的是 73f79257…，看起來是 body 雜湊。兩者不是同一種雜湊，不一定是錯。
- t1_sector_order.json 與 manifest 記的題 1 lclib 雜湊是 962bda…，但目前檔案（以及 PREREG 的 data_sha256）是 1f9da61…。也就是說，產業排名檔是用較舊版本的 lclib 產生的，之後 lclib 有改過，需要研究員說明兩版是否會改變排名。
