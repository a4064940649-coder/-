# FINDINGS：搶在價格之前的成長股（o_prepriced_growth）

資料夾：/home/user/-/research/tw_stock_selection/agents/o_prepriced_growth/
- 程式：olib.py、explore.py、iterate.py、prereg.py、oos.py、diag.py
- 結果：explore.json、iterations.json、results.json
- 子代理不寫 .md；這份全文請主研究員存成 FINDINGS.md。

## 結論：不可用（P5 失敗條件成立；驗證期、保留期依規則沒有打開）

- **沒過閘門。** 研究計畫定的 24 個版本全部只在挑選期（2006-07～2013-12）跑過。
  - 扣成本後對同池等權 t 最高的是 **AVG|B1|n30：t 1.40**，低於事前閘門 1.5。
  - 這個版本年化 9.7%（0050 6.0%、同池等權 6.4%、主策略 n10 18.6%），對 0050 t 0.93，最大回撤 −58.6%。
  - 24 版 max-t 校正後 p 0.39。依規則判「不可用」，不打開 2014～2020 和 2021～2026。
- **訊號是真的，但這個結構付不起成本。**
  - 同一版本不扣成本 t 3.60（每月 +0.70%），24 版 max-t 校正後 p 0.005。
  - 但它每月單邊周轉 63%，一般成本每月吃掉約 0.43%，只剩 +0.27%。壓力成本下 t 0.31。
  - 周轉高的原因是規則本身：股價一漲出區間（漲到 52 週百分位 0.8 以上），就被規則賣掉，而這正是「價格追上營收」、報酬兌現的時候。
- **和主策略確實低相關，但合成在挑選期反而變差。** 月超額相關 0.02（n30）／−0.01（n10）。各半合成年化 13.8%，主策略單獨 18.6%；對 0050 t 1.53 對 1.86；回撤只從 −65.0% 改善到 −61.5%。這是挑選期內的描述，不參與判定。
- **機制部分支持。**
  - 營收強、價格在中段的股票，之後確實比較常走到新高、進入主策略前 30 名（P2 成立，但有機械成分，見第 7 節）。
  - 但「還沒反映」的溢酬比「已經在高點」的小：營收前 1/5 裡，中段每月 +0.51%，高段 +1.00%。這和主策略吃第二階段的說法一致。
  - 資訊擴散很快：超額集中在形成後第 1 個月（+0.58%，t 2.51），之後就不顯著。沒有一段夠長、可以「搶在前面」的反應不足期。
- **給研究中心的建議：** 這條腿不納入，也不列紙上追蹤。如果要延伸，下一個方向是降低周轉的「畢業制」，見第 10 節。它必須另外事前登記；而且 2006～2013 已經被這一題看過，只能用 2014 年以後當挑選期。

## 1. 研究問題

主策略（52 週高點百分位 0.5＋近 3 月營收年增率百分位 0.5，前 300 流動性池，月換股 10 檔，排除金融）要求股價已經接近高點，所以系統性錯過「研究型資金先買、價格還沒反映」的第一階段。

本題要回答：營收已經加速、創 12 月新高、但 52 週比還在中段的股票，能不能做成扣成本後贏同池等權、而且和主策略低相關的第二條腿。

## 2. 機制與對手

- **誰在賺：** 頂尖主動基金在獲利上修被價格反映之前布局。
  - 安聯台灣科技 2014～2020 年，扣掉市場、電子、52 週高、營收因子後，殘差 α 每月 0.78%（t 3.3），52 週高點載荷 −0.24。
- **散戶的公開代理：** 月營收，每月公布，是盈餘上修最早的公開領先指標。
- **為什麼可能沒被套利掉：**
  - 散戶注意力集中在已經大漲的股票。
  - 大型機構受追蹤誤差和容量限制，不能太早重壓。
- **對手：** 同池等權（扣成本）、0050、主策略 n10。

## 3. 資料

- **價格：** 面板 adj、close_raw、open、value，經由 main_v2/v2_core.py 和 independent_audit/ia_core.py 只讀 import。value.parquet 的 md5 斷言為 8272e253（修正版）。
- **營收：**
  - 訊號日早於 2021-03：tw_revenue.csv。
  - 之後：rev_5y 優先，缺的用長期營收補。
  - 2013 年有 IFRS 口徑斷點；2013 年以後長期營收只剩存活公司。主策略也受同樣影響，兩邊口徑一致。
- **安聯：** scratchpad/winners_attr/allianz_monthly.csv（FUND、0050、EW、C1、C2、ELEC）。
- **沒有新下載資料。**

## 4. 方法與紀律

- **成交規則：** 換股日、成交、成本、股池、排除金融、開盤漲停遞補、停牌凍結，全部直接沿用主策略 v2 的程式，沒有另調。
- **期間：** 挑選期 2006-01～2013-12（營收加速度要 6 個月年增率，實際 2006-07 起）。iterate.py 斷言模擬的最後一天 ≤ 2013-12-31；explore.py 的所有前瞻窗也截在 2013-12-31。
- **凍結：** prereg.py 依事前選擇規則機械式選版，寫入 PREREG。
  - sha256 12017357a887893dcb9381b2fcfbdc3c1e1d3064ac8c3ecf90e38a85880f3f08，時間 2026-10-27T17:17:38Z。
  - 同時記下 iterations.json 的雜湊。
  - oos.py 會驗證兩個雜湊，閘門關閉時只寫入 OOS＝NOT_OPENED，不載入任何 2014 年以後的報酬。
- **引擎核對：**
  - 主策略排除金融 n10 重建：全期年化 28.15%，對 0050 t 2.643，和 main_v2 相同。
  - 加權引擎 simulate_w 用等權時，和 ia_core.simulate 的最大差 1.7e-14。
- **版本數：** 研究計畫訂的範圍是 3 種營收訊號 × 4 種區間 × 2 種檔數，最多 24 版。通用工作方式寫「最多 8 版」，兩者衝突；我照研究計畫跑滿 24 版、全部記錄，多重檢定也用 24。

## 5. 探索（只用挑選期；explore.json）

- **停止規則（先寫進程式）：** 營收前 1/5 按 52 週百分位分 10 段，同時滿足下面三條才停止：
  - 段次和超額的 Spearman ≥ 0.8；
  - 第 10 段最高；
  - 中段平均 ≤ 低段平均。
- **結果：** Spearman 0.54、第 10 段最高、中段 > 低段，**沒有觸發停止**。
- **形狀**（AVG 營收前 1/5；各段下一個換股窗對同池等權的超額，%／月）：

  | 段 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
  |---|---|---|---|---|---|---|---|---|---|---|
  | 超額 | −0.54 | −1.14 | 0.97 | 0.99 | −0.26 | 0.52 | 0.47 | 0.84 | 0.70 | 1.30 |

  - 低段（1～3）−0.22，中段（4～8）+0.51（對等權 t 2.19），高段（9～10）+1.00。
  - 中段減低段 t 1.35；中段減高段 t −0.92。
  - 用其他營收定義也一樣是高段最好：
    - 加速度：中段 +0.07、高段 +1.22，中段減高段 t −2.53。
    - 新高比：中段 +0.17、高段 +0.62。
    - 3 月年增率：中段 +0.39、高段 +1.08。
  - 全股池（不限營收）：只有第 9、10 段是正的，第 6 段 −0.61（t −3.1）。
  - **結論：** 中段是正的，但「還沒反映」的溢酬小於「已經反映」的溢酬。
- **規模與產業：**
  - G（AVG 前 1/5 且 52 週百分位 0.4～0.85）每月 24.3 檔（17～34）。
  - G 的電子股占 47%，股池 49%；產業分布和股池幾乎一樣，半導體、零組件、電腦周邊居前。
  - G 有 56% 在流動性第 101～300 名，股池是 60%。
- **轉移機率（之後 k 個月內收盤創 252 日新高）：**

  | 組別 | 1 個月 | 3 個月 | 6 個月 |
  |---|---|---|---|
  | G | 25% | 42% | 54% |
  | 同池其餘 | 18% | 30% | 42% |
  | 營收強、價格低段 | 2% | 10% | 23% |
  | 營收強、已在高點 | 62% | 74% | 81% |

  - 營收前 1/5 的 3 個月轉移機率，從第 1 段 6% 單調升到第 10 段 78%。
- **P2：** 之後 3 個月內進入主策略前 30 名（只算當下不在前 30 名的股票）：
  - G 28.6%，同池其餘 8.5%（G 減其餘 t 14.2），營收強低段 7.2%，營收中段弱 8.6%，營收強高段 35.4%。
- **報酬集中（事後分組，不可交易）：** 之後 3 個月真的創新高的 G 股，3 個月超額 +13.2%；沒創新高的 −8.1%；創新高的比例 42%。報酬確實來自「價格追上營收」的那批。
- **資訊擴散速度（G 形成後第 h 個月的超額）：**

  | 第 h 個月 | 0 | 1 | 2 | 3 | 4 | 5 |
  |---|---|---|---|---|---|---|
  | 超額 %（t） | +0.58（2.51） | +0.19（0.75） | +0.28（1.08） | +0.40（1.49） | −0.03 | +0.37 |

  - 對照：營收強高段第 0 個月 +1.18（2.55）；營收中段弱 −0.68（t −3.48），之後幾個月也一直是負的。
- **子樣本（G 下月超額，對同子樣本等權）：**
  - 流動性第 101～300 名 +0.76%（t 2.76），前 100 名 +0.26%（0.72）：P4 的規模預測成立。
  - 電子 +0.47%（1.42），非電子 +0.71%（2.38）：P4 的電子預測不成立。
  - 2013 年 +0.83%（1.95）：P4「IFRS 年沒有效果」不成立；但 2013 營收口徑本身有問題，只當作資料警訊。
  - 2006～2008 +1.23%（3.16），2009～2012 +0.12%（0.35）：效果集中在前段。

## 6. 模型迭代（只用挑選期；iterations.json；24 版全部列出）

- 欄位：年化%、最大回撤%、對等權 t（一般成本）、對 0050 t、不扣成本 t、壓力成本對等權 t、和主策略相關、剔除 2013 年 t、月單邊周轉。

| 版本 | 年化 | 回撤 | t 等權 | t 0050 | 毛 t | 壓力 t | 相關 | 去 2013 t | 周轉 |
|---|---|---|---|---|---|---|---|---|---|
| ACC\|B1\|n10 | 7.7 | −66.3 | 0.59 | 0.60 | 1.74 | 0.02 | 0.07 | 0.64 | 66% |
| ACC\|B1\|n30 | 6.1 | −62.1 | 0.02 | 0.28 | 2.01 | −0.96 | 0.08 | 0.32 | 60% |
| ACC\|B2\|n10 | 8.6 | −65.7 | 0.68 | 0.69 | 1.83 | 0.12 | 0.20 | 0.58 | 68% |
| ACC\|B2\|n30 | 5.5 | −62.4 | −0.27 | 0.13 | 1.58 | −1.18 | 0.27 | −0.13 | 61% |
| ACC\|B3\|n10 | 12.1 | −64.5 | 1.20 | 1.13 | 2.40 | 0.62 | 0.34 | 1.24 | 71% |
| ACC\|B3\|n30 | 8.0 | −61.4 | 0.43 | 0.52 | 2.26 | −0.48 | 0.39 | 0.56 | 64% |
| ACC\|B4\|n10 | 12.2 | −64.8 | 1.19 | 1.13 | 2.32 | 0.63 | 0.38 | 1.04 | 67% |
| ACC\|B4\|n30 | 9.2 | −62.5 | 0.84 | 0.76 | 2.52 | 0.01 | 0.50 | 0.74 | 61% |
| NH\|B1\|n10 | 2.0 | −60.6 | −0.89 | −0.42 | 0.66 | −1.64 | 0.11 | −0.78 | 88% |
| NH\|B1\|n30 | 1.7 | −65.5 | −1.99 | −0.68 | 0.34 | −3.13 | 0.09 | −2.03 | 67% |
| NH\|B2\|n10 | 3.9 | −60.8 | −0.45 | −0.11 | 1.04 | −1.18 | 0.13 | −0.57 | 85% |
| NH\|B2\|n30 | 2.5 | −64.1 | −1.52 | −0.52 | 0.59 | −2.54 | 0.24 | −1.50 | 66% |
| NH\|B3\|n10 | 5.1 | −59.7 | −0.18 | 0.09 | 1.29 | −0.90 | 0.29 | −0.08 | 85% |
| NH\|B3\|n30 | 2.9 | −64.4 | −1.46 | −0.55 | 0.58 | −2.44 | 0.34 | −1.44 | 66% |
| NH\|B4\|n10 | 4.3 | −63.8 | −0.33 | −0.04 | 1.09 | −1.02 | 0.39 | −0.12 | 85% |
| NH\|B4\|n30 | 4.6 | −64.3 | −0.74 | −0.18 | 1.04 | −1.60 | 0.49 | −0.82 | 64% |
| AVG\|B1\|n10 | 7.0 | −60.3 | 0.25 | 0.39 | 1.72 | −0.47 | −0.01 | −0.15 | 77% |
| **AVG\|B1\|n30** | **9.7** | −58.6 | **1.40** | 0.93 | **3.60** | 0.31 | 0.02 | 1.36 | 63% |
| AVG\|B2\|n10 | 8.5 | −56.4 | 0.54 | 0.61 | 1.97 | −0.16 | 0.08 | 0.24 | 76% |
| AVG\|B2\|n30 | 8.2 | −61.0 | 0.63 | 0.63 | 2.60 | −0.34 | 0.22 | 0.47 | 63% |
| AVG\|B3\|n10 | 7.9 | −59.0 | 0.36 | 0.51 | 1.75 | −0.33 | 0.26 | 0.28 | 78% |
| AVG\|B3\|n30 | 7.0 | −59.9 | 0.11 | 0.37 | 2.08 | −0.85 | 0.40 | 0.00 | 64% |
| AVG\|B4\|n10 | 9.1 | −60.4 | 0.58 | 0.69 | 1.83 | −0.03 | 0.34 | 0.41 | 77% |
| AVG\|B4\|n30 | 8.9 | −60.7 | 0.66 | 0.75 | 2.35 | −0.16 | 0.49 | 0.31 | 62% |
| 主策略 n10（同期） | 18.6 | −65.0 | 1.81 | 1.86 | 2.58 | 1.44 | — | 1.74 | 約 62% |

- **規律**
  - 單用營收新高比（NH）最差：周轉 85%，扣成本後全部為負。台股營收季節性強，12 月新高比每個月換一批股票。
  - 營收加速度（ACC）在偏高的區間（B3、B4）比較好，但和主策略的相關也升到 0.34～0.50。越靠近高點，越像主策略的弱化版。
  - 平均分數（AVG）的 30 檔最穩：毛 t 2.1～3.6，但扣成本後都 ≤ 1.40。
  - 沒有任何版本扣成本後 t ≥ 1.5。
- **多重檢定**
  - 扣成本：max t 1.396，校正 p 0.389，95% 臨界值 2.66。
  - 不扣成本：max t 3.599，校正 p 0.005，95% 臨界值 2.67。

## 7. 凍結與樣本外

- 選中 AVG|B1|n30（挑選期 t 1.40，和主策略相關 0.02）。合成方式事前固定為主策略 n10＋同訊號同區間 n10 各半。
- 閘門 t ≥ 1.5 不成立，所以 **驗證期與保留期沒有計算**（results.json 的 OOS.status＝NOT_OPENED）。
- P3 原本要用安聯 2014～2020 的資料，屬驗證期，所以也沒有做。

## 8. 穩健性與事後診斷（只用挑選期，diag.py，不影響判定）

- **逐年**（第二腿／主策略 n10／同池等權／0050，%）：

  | 年 | 第二腿 | 主策略 n10 | 同池等權 | 0050 |
  |---|---|---|---|---|
  | 2006（7 月起） | 32.3 | 57.5 | 28.8 | 18.2 |
  | 2007 | 27.3 | 28.4 | 7.7 | 10.3 |
  | 2008 | −42.0 | −53.1 | −48.1 | −43.3 |
  | 2009 | 101.6 | 86.3 | 112.4 | 75.0 |
  | 2010 | 17.8 | 22.0 | 9.6 | 13.1 |
  | 2011 | −35.6 | −1.6 | −29.6 | −15.8 |
  | 2012 | 12.2 | 32.9 | 14.1 | 11.9 |
  | 2013 | 19.6 | 26.6 | 17.9 | 11.7 |

  - 8 年裡有 4 年輸同池等權，只有 2008、2009 兩年贏主策略。
  - 2011 年比主策略差 34pp。中段股在空頭沒有防守性，回撤 −58.6% 和主策略同一量級。
- **剔除 2013（IFRS）：** t 1.36；只看 2013 年 t 0.30。
- **剔除最大貢獻的 1 檔（3041 揚智，貢獻占 3.9%）：** 毛年化 15.10% → 14.65%，近似毛 t 2.78 → 2.64。報酬不是靠單一股票。
- **持股分布：** 電子 48%，流動性排名中位數 121，58% 在第 101～300 名。產業依序是半導體 12%、零組件 9%、電腦周邊 8%、光電 7%。
- **進場時點拆解：** 第二腿用訊號日收盤進場（理想、不可成交）和隔日開盤進場，每月超額 0.66% 對 0.70%，幾乎沒差。主策略 n10 則是 1.45% 對 1.32%。所以第二腿的問題不是開盤跳空，而是周轉成本。
- **挑選期內的合成：**
  - 一般成本：合成年化 13.8%、回撤 −61.5%、對 0050 t 1.53；主策略單獨 18.6%、−65.0%、1.86。
  - 壓力成本：11.2%（t 1.06）對 15.9%（t 1.52）。
  - 合成在年化和 t 上都比主策略差，只換到 3.5pp 的回撤改善。
- **P3 挑選期版（2006-08～2013-11，88 個月，Newey-West 3）：**
  - 在 EW−0050、電子、52 週高、營收因子之外，加入第二腿因子：載荷 n30 0.16（t 1.09）、n10 0.11（t 0.98），方向為正但不顯著。
  - 這段安聯本來就沒有殘差 α（M3 的 α 每月 −0.08%，t −0.32），所以沒有判斷力。

## 9. 機制是否被支持：部分支持

- **支持的部分**
  - 營收強、價格在中段的股票，之後走到新高的機率比同池其餘高（3 個月 42% 對 30%），進入主策略前 30 名的機率高 3 倍以上（28.6% 對 8.5%）。
  - 報酬確實來自後來追上的那批。P2 的方向成立。
  - 中小型（流動性第 101～300 名）比較強，和「大型機構容量限制」的說法一致。
- **不支持的部分**
  - 「還沒反映」的溢酬（中段 +0.51%／月）小於「已經反映」的溢酬（高段 +1.00%／月）。市場對營收的反應不足，主要發生在創新高之後，也就是主策略吃到的第二階段，不在第一階段。
  - 資訊擴散快：超額集中在第 1 個月，沒有一段夠長、可以「搶在前面」的期間。
  - 電子股沒有比較強；2013 年沒有消失。
  - 用區間規則持有，等於在股價畢業（漲出區間）時被迫賣出，周轉 63%～88%。公開營收的領先資訊扣掉 0.3% 證交稅加滑價後所剩無幾。
- **P2 的機械成分要注意。** 主策略分數有一半是營收，G 本來就營收強，只要股價上去就會進前 30。所以 P2 只能證明「價格在追營收」，不能單獨證明可以獲利。
- **頂尖經理人的優勢在散戶這裡複製不到。** 安聯的殘差 α 很可能來自拜訪、共識 EPS 上修這類非公開或付費資訊，以及低交易成本的部位管理。光靠月營收代理不出來；挑選期 P3 載荷也不顯著。

## 10. 散戶怎麼執行

- **不建議執行這條腿。** 主策略照原規則；處置股仍待觀察，上限 20%。
- **如果之後要延伸，下一題（需另外事前登記）是「畢業制」：**
  - 中段＋營收強時買進；持有到進入主策略前 30 名、或營收轉弱才賣；漲出區間不賣。
  - 用意是把周轉降到 30% 以下，讓毛超額每月 0.7% 留下大部分。
  - 2006～2013 已被本題看過，只能用 2014～2020 挑選、2021～2026 驗證，再加 2026-11 起的真實紙上追蹤。

## 11. 限制

- 驗證期和保留期沒有打開，所以第二腿和合成在 2014 年以後的表現未知。這是紀律要求，不是遺漏。
- 挑選期只有 89 個月，2008、2011 兩個空頭年的權重很大；而且效果集中在 2006～2008。
- 2013 年以後長期營收只剩存活公司，主策略和第二腿都有存活偏誤；2013 年的 IFRS 斷點會扭曲加速度和新高比。
- 探索的報酬是收盤到收盤、不扣成本、沒有漲停遞補；正式版本的結果以引擎（隔日開盤、成本）為準。
- 剔除最大貢獻股的 t 是近似：持有期是開盤到開盤，等權是收盤到收盤，差一天。
- 主結果用 BASE 營收時點，和主策略正式數字同口徑。AB（只用已公告營收）事前列為敏感度，但閘門沒開，就沒有跑；依 rev_announce 的經驗，n10 的影響 ≤ 1pp，不會改變結論。
- 通用工作方式寫「最多 8 版」，研究計畫寫「最多 24 版」；本研究照研究計畫跑 24 版，多重檢定也誠實用 24。

## 凍結規格

PREREG（o_prepriced_growth）
- 檔案：/home/user/-/research/tw_stock_selection/agents/o_prepriced_growth/results.json 的 PREREG
- sha256：12017357a887893dcb9381b2fcfbdc3c1e1d3064ac8c3ecf90e38a85880f3f08（對 json.dumps(PREREG, ensure_ascii=False, sort_keys=True) 取雜湊）
- 時間：2026-10-27T17:17:38+00:00。iterations.json 的 sha256 也記在同一個檔裡。oos.py 會先斷言 PREREG 存在、雜湊相符，而且 iterations.json 沒有被改過。

選中版本：AVG|B1|n30
- 選擇規則：只看挑選期，選對同池等權 t 最高、而且和主策略月超額相關 < 0.6 的版本。
- 這個版本的挑選期數字：t 1.3957，和主策略的相關 0.02。

股池（全部和主策略 v2 相同）
- 滾動窗用 gapfix：先剔除有值檔數 < 300 的日子；252 日與 60 日窗內，只豁免 2017-08-03、2018-07-12、2024-02-15～03-15 的缺值。
- 依 60 日均成交值取前 300 名，條件是均值 ≥ NT$1,000 萬、收盤 ≥ 10。
- 排除金融保險業：stock_list 的金融保險業，另加 28xx、5876、5880。
- 排除處置股；要求隔日有開盤成交；52 週比與營收 3 月年增率都要有值。

訊號
- 52 週比＝含息還原收盤 ÷ 252 日最高。取它在股池內的百分位 p52，留下 0.30 ≤ p52 ≤ 0.80 的股票（含端點）。
- 營收分數 AVG＝0.5×ACC 的股池百分位＋0.5×NH 的股池百分位。
  - ACC＝yoy3(m) − yoy3(m−3)。yoy3＝訊號月 m 之前 3 個月的單月年增率平均，3 個月都要有值。
  - NH＝最近一月營收 ÷ 前 11 個月營收最大值，11 個月都要有值，而且都 > 0。
- 排序：區間內依 AVG 由高到低取 30 檔。同分先比營收 3 月年增率（高者優先），再比代碼。

營收來源
- 訊號日早於 2021-03-01：用 tw_revenue.csv。
- 2021-03-01 以後：
  - yoy3(m) 用 rev_5y 的 yoy 欄，和主策略相同。
  - yoy3(m−3) 優先用 rev_5y，缺的月份用長期營收補。
  - 營收水準優先用 rev_5y，單位同為千元。
- 主結果用 BASE，和主策略正式數字 2.64 同一口徑；AB（只用已公告營收）事前列為敏感度。

成交與成本
- 每月第一個「日 ≥ 11」的交易日收盤算訊號，隔日開盤等權成交。
- 開盤漲停（依跳動點算）而且原本沒持有的股票，跳過並往下遞補。
- 停牌的持股凍結，等恢復交易再處理。
- 一般成本：手續費 0.0855%＋滑價 0.1%，賣出另加證交稅 0.3%。壓力成本：買賣各 0.5%。

合成方式（事前固定，不搜尋權重）
- 主策略 n10（排除金融）和第二腿 n10 各半資金，每月換股日再平衡。
- 兩邊重疊的持股，權重相加；成本只算淨交易。
- 選中版本是 n30，所以第二腿單獨檢定用 n30；合成依計畫原文，用同訊號、同區間的 n10。

期間
- 挑選期：2006-01～2013-12。因為營收加速度要 6 個月年增率，實際從 2006-07 開始；模擬截在 2013-12-31 收盤。
- 驗證期：2014-01～2020-12。保留期：2021-01～2026-10-12。

閘門
- 挑選期對同池等權 t ≥ 1.5，才能打開驗證期與保留期。
- 實際 t 1.40，閘門關閉，判「不可用」。依規則不計算後兩段；P3（安聯 2014～2020）也屬驗證期資料，不檢定。

如果閘門打開，原本的最終判定條件
- (i) 驗證期與保留期的 t 都 ≥ 1.5，合併 2014～2026 的 t ≥ 2。
- (ii) 和主策略的相關 < 0.5。
- (iii) 合成後跨循環對 0050 的 t > 2.64，年化 ≥ 主策略 −1.5pp，最大回撤不深於主策略。
- (iv) 合成在 2006～2013、2021～2026 兩段的年化都 ≥ 主策略 −3pp。
- (v) 壓力成本下 (iii) 方向不變。
- (vi) 安聯對第二腿的載荷 > 0。

重做步驟（全部用 python3 -I -B，在 /home/user/-/research/tw_stock_selection/agents/o_prepriced_growth/ 執行）
1. explore.py：產生 explore.json，約 20 秒。
2. iterate.py：產生 iterations.json，約 17 秒。
3. prereg.py：把 PREREG 寫進 results.json；PREREG 已存在時拒絕覆寫。
4. oos.py：檢查閘門，寫入 OOS＝NOT_OPENED。
5. diag.py：寫入 DIAG_posthoc_selection_only，約 15 秒。這是事後描述，只用挑選期，不影響判定。

核對數字
- 主策略排除金融 n10 重建：全期年化 28.15%，對 0050 t 2.6426，和 main_v2 的 2.64 相同。
- simulate_w 用等權時，和 ia_core.simulate 的最大差是 1.7e-14。

## 反方審查

[
 {
  "final": "不可用",
  "issues": [
   "Conclusion confirmed: I rebuilt AVG|B1|n30 independently. Net t 1.3957, gross t 3.5993 and cost drag 0.43%/month all match the researcher. Newey-West(3) t is lower (net 1.30, gross 3.34). With the bootstrap restudentized on every draw and blocks of 3, 6 and 12, the 24-version net max-t corrected p is 0.33 to 0.42, so the gate stays shut under any of them. Only the selection period was used; no data from 2014 onward was touched.",
   "The claim that the gross signal is real (max-t p 0.005) depends almost entirely on 2006-07 to 2008-12 (29 months). That stretch has gross +1.34%/month, t 3.49, net t 2.39. In 2009 to 2013 (60 months) gross is only +0.39%/month (t 1.83) and net is -0.21%/month (t -0.21). Re-running the 24-version gross max-t on 2009-2013 alone gives p 0.135. The signal only clears multiple testing in the early years, which looks like decay, so it cannot be described as 'real but eaten by costs'.",
   "The gross t 3.60 is a lucky peak, not a plateau. Holding AVG|B1 fixed, gross t by holding count is n10 1.72, n15 1.49, n20 2.02, n25 2.98, n30 3.60, n40 2.42, n50 1.57. Net t is 0.25, -0.21, 0.13, 0.94, 1.40, 0.09, -0.93. With n30 fixed, gross t by band is B1 3.60, B2 2.60, B3 2.08, B4 2.35. The chosen version sits exactly on the peak, so the 3.60 and the 1.40 both carry winner's-curse upward bias. These neighbour checks are review-only and play no part in the verdict.",
   "The gate result is fragile across years. Leave-one-year-out net t: dropping 2011 gives 2.11 and dropping 2012 gives 1.56, both of which would pass 1.5. Dropping 2007 gives 0.63. Passing or failing depends on a single bear year (2011, -0.83%/month net), so the 1.40 does not mean it was 'almost there'. It means the result is driven by a few years.",
   "The 'graduation rule' follow-up is not credible on these numbers. It assumes the 0.7%/month gross excess can be kept, but after 2009 gross excess is only 0.39%/month. Even at 30% turnover the cost drag is about 0.2%/month, leaving roughly 0.19% net. It is also proposed to use 2014-2020 as its selection period, and my coverage check shows tw_revenue.csv is survivor-only from 2013 on: on 2013-07-11, 27 of the top 300 later delisted and 0 of them have revenue. Before 2013 most do (e.g. 32/37 in 2006-07). A 2014-2020 selection period would run on a pool that has lost about 9% of names to survivorship bias, and the FINDINGS does not flag this as a precondition.",
   "The FINDINGS line '2013 年以後長期營收只剩存活公司' understates the problem. Coverage drops sharply in 2013 itself: the top 300 has 38 names without revenue (18 in 2013-01) and none of the later-delisted names have revenue. So the survivorship bias already reaches into the last year of the selection period (2013 contributes +0.13% net). The 'drop 2013' test only partly covers it.",
   "The P2 statistic t 14.2 (G minus the rest) is inflated. The 3-month forward windows overlap from month to month, which makes it roughly sqrt(3) too large, and the researcher admits the result is partly mechanical because revenue is half of the main-strategy score. It should not count as mechanism evidence. The 'mechanism partly supported' label should be read with this in mind.",
   "Look-ahead risk cannot be ruled out. In the selection period the signal date is always the first trading day on or after the 11th, so deadline_case is D<S and the unavailability mask never fires. That leaves BASE fully dependent on tw_revenue.csv carrying the originally announced values, and the dataset does not support point-in-time checks for restatements or late filers. The AB sensitivity was not run, and the '≤1pp' figure is borrowed from n10 experience on the main strategy, not checked on this n30, mid-cap-heavy leg. This would only push the gross signal down, so it does not change the verdict.",
   "Forking paths are under-counted. The 24-version max-t ignores what was looked at on the same 2006-2013 data during exploration: four revenue definitions × 10 deciles, the whole-pool deciles, and the quick AVG|B2 top-10 check. The version space was fixed in the plan, but the gross p 0.005 should be treated as an optimistic lower bound.",
   "Survivorship check on delistings came back clean: with delisted frozen holdings set to zero the net t is still 1.3957 (no holding froze on delisting), and the later-delisted share of holdings is 6.1% versus 7.7% for the pool. Freezing delisted stocks did not inflate the result."
  ],
  "note": "I agree with the researcher's verdict: unusable (不可用), and the validation and holdout periods should stay closed. The researcher's numbers reproduce exactly. My verification script is /home/user/-/research/tw_stock_selection/agents/o_prepriced_growth/review_stat_o1.py (run with python3 -I -B; selection period only; it writes nothing back to the researcher's files). Two points in FINDINGS should be downgraded. (1) '訊號是真的' (the signal is real, gross max-t p 0.005) holds only because of 2006-2008. Over 2009-2013 the gross max-t p is 0.135 and net excess is -0.21%/month, so this should be called an early-period effect that has decayed. (2) The basis for the 'graduation rule' follow-up is weak. Post-2009 gross excess is only 0.39%/month, so even at 30% turnover little is left. Its proposed 2014-2020 selection period would also use tw_revenue.csv, which keeps only surviving companies, and the follow-up's pre-registration would have to fix that data first. The mechanism verdict should drop from 'partly supported' to 'weak support': the P2 t is inflated by overlapping windows and partly mechanical, and the gross signal does not survive multiple testing after 2009."
 },
 {
  "final": "不可用",
  "issues": [
   "我同意「不可用」，而且從成本角度看比研究員寫的更差。研究員的一般成本是每邊滑價 0.10% 加 0.0855% 手續費，這是最寬鬆的假設。AVG|B1|n30 要讓對等權 t 達到 1.5，每邊滑價必須 ≤ 約 0.08%（滑價 0% 時 t 2.06，0.10% 時 t 1.40，0.15% 時 1.06，0.20% 時 0.73，0.30% 時 0.07）。它的持股開盤價跳動點中位數是 18bp，p90 是 39bp，所以 0.08% 不現實。不扣滑價、只扣 0.3% 證交稅時 t 2.62，可見光是證交稅就吃掉毛 t 3.60 的大半。",
   "n30 在散戶 30 萬～100 萬的資金完全做不到。30 萬分 30 檔，每檔 1 萬，但成交股的一張市值中位數 3.43 萬（p10 1.39 萬），99% 的交易連一張都買不到。100 萬時仍有 51% 不到一張，整張持有的權重誤差平均 62%。挑選期 2006～2013 沒有盤中零股，只有盤後零股，用開盤價成交零股根本不可能。最低手續費 20 元對 1 萬的部位是每邊 0.20%；把它算進去，n30 的 t 降到 0.64，對同池等權每月只剩 +0.12%。",
   "用資金規模校正成本後，30 萬～1,000 萬沒有任何一級能用。校正方式：最低手續費 20 元，加 0.1% 價差，再加平方根衝擊（σ_d×√(部位/60 日均值)）。n30 的 t：30 萬 0.49、100 萬 1.13、300 萬 0.94、1000 萬 0.57、3000 萬 −0.04。n10 的 t：30 萬 0.08、100 萬 −0.06、300 萬 −0.28、1000 萬 −0.71，平均超額 ≤ 0。同樣條件下主策略 n10 是 1.73／1.66／1.56／1.34。散戶整張做得到的版本（n10）不扣滑價時 t 也只有 0.68，滑價 0.15% 時平均超額 ≈ 0。",
   "FINDINGS 寫錯了一個口徑：它說「對同池等權（扣成本）」，但 iterate.py 的等權基準是 simulate(...,0,0)，也就是不扣成本。研究中心 main_v2 也是這個口徑，所以閘門前後一致。但如果改成扣成本的等權，選中版本的 t 會從 1.40 變成 1.76，越過 1.5。所以「沒過閘門」這個結論取決於基準要不要扣成本；改成扣成本的基準，結論就不一定成立。不過滑價只要拉到 0.15%，t 就回到 1.43，再加上整張與最低手續費的限制，判定不變。建議把文字改成「同池等權（不扣成本）」，並把這個脆弱性寫出來，避免日後被拿去當作「其實有過」的依據。",
   "處置股排除在挑選期其實沒有生效。disposal_5y 最早的 start 是 2021-09-24，2006～2013 的股池完全沒有排除處置股。FINDINGS 和 PREREG 寫的「非處置、沿用主策略」，在挑選期只是名義上的規則。處置股要分盤撮合、第二次處置還要預收款券，這些成交限制沒有被模擬，影響方向不利，但無法量化。",
   "開盤漲停遞補需要開盤時盯盤。引擎假設遞補股用同一天的開盤價成交，但實際上 9:00 開盤後才知道原股漲停，遞補只能在盤中、用不同於開盤的價格成交。n30 在挑選期遞補了 37 次，影響小，但「不需要盯盤」並不完全成立。",
   "影響小、已查過的項目：",
   "（1）賣出日開盤即跌停、實際賣不掉的比例：n30 0.24%（4 筆），n10 0.44%，可以忽略。",
   "（2）股池用了「隔日有開盤成交」這個前視條件，平均每月只剃掉 0.04 檔，可以忽略。",
   "（3）容量面，即使 1000 萬分 30 檔，占 60 日均值的中位數也只有 0.23%（p90 0.82%），流動性本身不是瓶頸；瓶頸是周轉乘以證交稅和整張限制。",
   "營收時點沒有驗證。AB（只用已公告營收）事前列為敏感度但沒有跑，研究員用 rev_announce 對主策略 n10 的經驗（≤ 1pp）推論影響不大。但 NH 和 AVG 用的是最新一個月營收的 12 月新高比，比主策略的 3 月年增率更依賴最新一個月，延後公告的影響可能更大。另外，tw_revenue.csv 是否為首次公告值（還是事後更正值）也沒有確認。影響方向都不利。",
   "第 10 節建議的「畢業制」在成本上沒有算過。超額集中在形成後第 0 個月（+0.58%），第 1～5 個月平均約 +0.24%。拉長持有雖然能降周轉，但每月毛超額會被稀釋；周轉 30% 時，一般成本仍約每月 0.20%，可能剩不到多少。第 10 節說「保留每月 0.7% 毛超額的大部分」，這是沒有依據的樂觀說法。另外，事前登記時應把最低手續費和整張可行性列為硬性條件，並且只用 n ≤ 10，或資金 ≥ 300 萬。",
   "研究員事前固定的合成（主策略 n10＋第二腿 n10 各半）在 30 萬時每檔約 1.5 萬，同樣大多買不到整張。所以挑選期合成的 13.8% 對小資金散戶不可複製。"
  ],
  "note": "我同意不可用，而且從成本和可成交性看，結論比研究員寫的更確定。\n\n- **成本：**研究員的一般成本（每邊滑價 0.10%）已經是最寬鬆的假設。n30 要在挑選期過 t 1.5，每邊滑價必須 ≤ 約 0.08%。\n- **整張與最低手續費：**n30 在 30 萬～100 萬幾乎買不到整張，而 2006～2013 沒有盤中零股。\n- **按資金規模校正成本後：**校正項目是最低手續費 20 元、價差 0.1% 加平方根衝擊。n30 在 30 萬～1000 萬各級的 t 是 0.49～1.13，n10 是 0.08～−0.71，沒有一級能用。\n\n另外有三點要改或補：\n- **等權基準口徑：**FINDINGS 寫等權基準「扣成本」，實際上程式用的是不扣成本的等權。這是研究中心的一致口徑，但如果改用扣成本的基準，t 會變成 1.76，閘門判定取決於這個選擇。需要更正文字，並註明這個脆弱性。\n- **處置股：**處置資料從 2021-09 才開始，挑選期其實沒有排除處置股。\n- **畢業制：**第 10 節的「畢業制」對成本的估計過於樂觀。\n\n驗證程式是 /home/user/-/research/tw_stock_selection/agents/o_prepriced_growth/review_exec_o1.py，輸出在 /tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/rev_exec.json。全部只用挑選期，沒有改動研究員的檔案。"
 }
]

## 獨立複製

{
 "reproduced": true,
 "my_numbers": "I rewrote everything from scratch (python3 -I -B) under /home/user/-/research/tw_stock_selection/agents/o_prepriced_growth/replicate_o1/: rlib.py (data, gapfix, pool, signals, tick-based limit-up, weight-based daily engine), rep_core.py, rep_extra.py, engine_check.py, rep_explore.py, rep_p3.py. Outputs are in rep_core_both.json, rep_extra.json, engine_check.json, rep_explore_ceil.json and rep_p3.json. I did not read the researcher's programs and did not import ia_core or v2_core. After my numbers were done I opened iterations.json, explore.json and results.json, only to compare against, and the winners_attr/allianz_monthly.csv file to get the P3 factors.\n\nEngine check: main strategy ex-financials n10, full period 2006-04-11 to 2026-10-12, 245 signals: annualized 28.148%, t vs 0050 2.6426. Matches the expected 28.15% and 2.6426.\n\nSelected version AVG|B1|n30, normal cost, 89 windows:\n- Annualized 9.747%, max drawdown -58.61%.\n- vs same-pool equal weight: t 1.3957, +0.268% per month.\n- vs 0050: t 0.933, +0.382% per month.\n- vs main n10: paired difference -0.656% per month, t -1.211.\n- Correlation of monthly excess with main: 0.0154.\n- Stress cost: 7.07%, t vs EW 0.308, t vs 0050 0.423.\n- No cost: t 3.599, +0.698% per month.\n- One-way turnover 63.3%; 2,700 holding-months over 90 rebalances.\n- Excluding 2013: t 1.363. 2013 only: t 0.302.\n- Largest contributor is 3041 揚智. Approximate gross annualized 15.104% → 14.650% without it (-0.454pp); approximate gross t 2.778 → 2.637.\n\nSame-period comparison:\n- Main n10: 18.561%, max drawdown -64.97%, t vs 0050 1.858, t vs EW 1.814.\n- 0050: 5.951%. Same-pool EW: 6.382%.\n- Yearly figures match the table to 0.1 for all four series in every year. Second leg: 32.3, 27.3, -42.0, 101.6, 17.8, -35.6, 12.2, 19.6.\n\n24-version grid: all 24 versions match iterations.json exactly at 4 decimals, both annualized return and t vs EW.\n\nMax-t bootstrap (my own demeaned bootstrap; the researcher's method is not known):\n- With cost: observed max t 1.396. iid: p 0.40-0.41, 95% critical value 2.55-2.59. Block length 3-12: p 0.36-0.43, critical value 2.63-2.75. Researcher: 0.389 and 2.66.\n- No cost: observed max t 3.599, p 0.001-0.008, critical value 2.54-2.78. Researcher: 0.005 and 2.67.\n\nHalf-and-half combination:\n- Normal cost: 13.82%, max drawdown -61.50%, t vs 0050 1.525. Main alone 18.56% (t 1.858).\n- Stress cost: combination 11.15% (t 1.064), main 15.92% (t 1.516).\n- AVG|B1|n10 correlation with main: -0.0098.\n\nExploration (G):\n- G averages 24.29 names per month (17-34).\n- Ten-segment excess returns: -0.54, -1.14, 0.97, 0.99, -0.26, 0.52, 0.47, 0.83, 0.70, 1.30. Spearman 0.539.\n- Low / mid / high: -0.24 / +0.51 / +1.00. Mid minus low t 1.345, mid minus high t -0.922.\n- New 52-week high within 3 months: G 41.8%, rest of pool 31.9%, strong-revenue low segment 9.8%, strong-revenue high segment 74.5%.\n- P2 (counted only for stocks not already in the main top 30 that month): G 28.5%, rest 8.9% (t 13.8), low segment 7.2% (t G minus low 12.74), high segment 34.1%.\n- 3-month excess: did make a new high +13.23%, did not -7.95%.\n- Event time months 0-5: +0.58 (t 2.51), +0.19, +0.27, +0.40, -0.03, +0.37.\n- Subsamples, each measured against its own sub-pool EW: liquidity rank 101-300 +0.76 (t 2.76), top 100 +0.26 (0.72); electronics +0.47 (1.41), non-electronics +0.71 (2.39); 2006-08 +1.23 (3.16), 2009-12 +0.12 (0.35), 2013 +0.83 (1.95).\n\nP3, Newey-West with 3 lags, 88 months: n30 loading 0.164 (t 1.087), n10 loading 0.111 (t 0.981), M3 alpha -0.082% per month (t -0.322).\n\nThe PREREG sha256 I computed equals 12017357...0f08.",
 "diff_explained": "The core numbers reproduce: engine check, the selected version, the same-period comparisons, yearly figures, the 24-version grid, the combination, the top-contributor drop and P3. The verdict, gate closed and not usable, still holds.\n\nTo get exact matches I had to settle these definitions. Most are details the spec does not state:\n\n1. The NH code does not follow the spec text. The spec says all 11 previous months must have values and all be > 0. The researcher's numbers match only if all 11 must have values, the largest of them must be > 0, and the latest month must be > 0. With this rule all 24 versions match to 4 decimals. With the literal spec rule, AVG|B1|n30 becomes t 1.378 and 9.68% annualized. The selected version and the closed gate do not change. This should be written into the spec or the code.\n\n2. The ACC and NH percentiles are each ranked over the pool members where that value exists. AVG is missing if either part is missing. Ranking only over stocks that have both, or within the 52-week band, does not match.\n\n3. Return measurement starts at the close of the first trade day, 2006-07-12, not the signal-day close. So the t windows run from the 2006-08 signal to 2013-12-31 (89 windows), and the July 2006 window is left out. The spec's \"first signal 2006-07-11, 89 windows\" fits this, but it is not spelled out. This choice favours the leg slightly. With signal-to-signal windows that include July and drop the December stub, t is 1.27 instead of 1.40, and annualized return from the signal close is 9.53%. Either way t is below 1.5 and the gate stays closed. The main n10 18.56% is also measured from the trade-day close, so the comparison is on the same basis. The full-period engine check of 28.15% uses the signal-day close.\n\n4. Max-t bootstrap: the resampling scheme is not documented. My iid and block-bootstrap results fall around the researcher's p 0.389 / critical value 2.66 and p 0.005 / 2.67. These are Monte Carlo and method differences, and the conclusion is the same.\n\n5. Exploration definitions:\n   - \"Revenue top 1/5\" means AVG percentile ≥ 0.8, inclusive. Using > 0.8 changes the ten-segment values slightly.\n   - The ten segments are ceil(p52 × 10).\n   - Low, mid and high are the averages of segments 1-3, 4-8 and 9-10 within each month. Pooling the stocks instead gives mid minus low t 1.48 rather than 1.35.\n   - Strong-revenue low segment is p52 ≤ 0.3 (10.4 names per month); high segment is p52 > 0.85.\n   - P2 only matches if stocks already in the main top 30 that month are left out of the count. Without that, G is 38%.\n   - The liquidity and electronics subsamples only match against their own sub-pool EW, not the whole pool.\n\n6. Small differences left, all in descriptive items with no effect on the verdict:\n   - Low segment -0.24 vs -0.22.\n   - Rest-of-pool new-high rate 32% vs 30%.\n   - P2 rest of pool 8.9% vs 8.5%, high segment 34.1% vs 35.4%, t 13.8 vs 14.2.\n   - 3-month excess for no new high -7.95% vs -8.1%.\n   These probably come from how \"rest of pool\" is defined or how the last few months of the 3-month window are handled.\n\n7. P3 matches only with Newey-West standard errors at 3 lags; plain OLS gives n10 t 1.33."
}

**最終判定（審查後）：不可用**
