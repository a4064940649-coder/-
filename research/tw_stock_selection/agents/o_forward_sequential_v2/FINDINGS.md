# 可證偽的前瞻序貫規則 2.0：以真實資料 CV 選版、依校準斜率放寬先驗、檢定力事前寫死，並建好 2026-11 起的前瞻追蹤產生腳本


輪次 o8｜研究員判定：不可用｜最終：不可用


## headline
判定規則不可用：在三種 DGP 下，沒有一條規則同時通過兩個凍結條件。D2（2019 年後殘差區塊 bootstrap）下 R2／R4 的 α＝0 誤升級為 6.2%／6.8%（> 5%）；把期限延到 42～120 個月也只到 5.7～7.2%。下修檢定力夠：α＝−1、36 個月內下修 52～59%，代價是誤下修 20～26%。CV 選版重現 V3，s＝0.29、k＝1.1。主決策線先驗 P(α>0)＝0.47，16 格配置的 x* 全部是 0%（平坦區間 0～30%），o7 的「90%」沒有重現。偽前瞻（2019-05～2026-09，89 個月）：P30 政策年化 29.7%，0050 30.4%（對 0050 t −0.19，對主策略 t +0.19）；主策略 j 中位數可執行 26.2%（對 0050 t −0.19）。保留期（2024-01 起，33 個月）在 2024-12 觸發下修：P30 56.6%，0050 61.4%（t −1.44）。前瞻腳本已乾跑通過，逐檔、同順序重現 2026-10-12 的 SOP 名單，並登錄 sha256。

## mechanism
o_tier（o7）失敗有三個原因：先驗用 V5，但真實資料 CV 最好的是 V3；先驗 sd 0.394 等於 179 個月的資料量，而且 no_refit 條款禁止修正；α＝−1pp／月時，24 個月內下修機率只有 0.25%，等於沒有退出機制。本題的作法：(1) 選版依真實資料時間切半 CV 的預測對數似然；(2) 先驗均值乘上校準斜率 s（0.27～0.32 取不利端），先驗 sd 放大到「事後估計對實現的預測區間覆蓋率 = 名目值」為止；(3) 決策規則不對稱：升級（加碼主策略或新腿）要求嚴格的誤升級控制，下修（退回 0050 核心）要求足夠的檢定力，因為下修的代價只是追蹤 0050；(4) 用模擬寫死操作特性（OC 曲線），規則先過 OC 才凍結。與已測研究的差別：o_tier 只報誤升級率、檢定力不足、先驗過窄；本題把檢定力列為凍結的硬條件，配置格只用 K5T（o7 方向）。

## mechanism_supported
部分支持

## predictions
(1) 可行性計算（事前寫下）：主策略 j 中位數可執行對 0050 的月超額 sd 約 5～6pp；α＝−1pp／月時，36 個月累積 z 約 −1.1～−1.2。所以要達到「36 個月內下修機率 ≥ 50%」，下修邊界大約落在累積 z ≈ −1.1，代價是 α＝0 時 36 個月內誤下修約 15～25%。預測這組限制「可行，但誤下修率不低」，要照實報告並由總監接受；如果誤下修超過 30%，就改報最短可行期限（預測約 48 個月）。(2) 升級邊界：在 α＝0 時，36 個月內誤升級 ≤ 5%（含所有中間檢視）；預測 α＝+1pp／月時，36 個月內升級的機率約 30～45%。(3) 用 V3 與校準放寬後的先驗，主策略 B10 K5T 在起點的 P(α>0) 約 0.45～0.6，多數級距的 x*（主策略占比）是 0～30%；預測 o7 的「30～100 萬 FIX 90%」不會重現。(4) 對應產生器：前瞻腳本在 2026-10-12 歷史訊號日乾跑，必須逐檔重現 SOP 上的 10 檔名單；j0／j1／j6 名單與 V7 j 掃描定義一致。(5) 否證：在三種資料生成過程下，OC 有任一項不符合限制，規則就不能凍結；凍結後若前瞻資料的月超額 sd 超過先驗假設 1.5 倍，就依事前寫死的條款重算邊界，這是唯一允許的 refit，而且只能放寬，不能收緊升級。

## frozen_spec
o_forward_sequential_v2 凍結規格（全文；複製員只看這段即可重做）
資料夾：/home/user/-/research/tw_stock_selection/agents/o_forward_sequential_v2/
執行順序（一律 python3 -I -B）：s0_copy_raw → s1_explore → s2a_prereg_design → s2b_cv_sel → s2c_iterate → s3_prereg → s4_cv_final → s5_oc_final → s6_pseudo_forward → s7_alloc → s8_robust → s11_check → s9_dryrun → s10_freeze_forward；s12_posthoc 屬事後診斷。

〇、登記鏈（results.json）
PREREG_DESIGN 9f5a41d767a5…5260（2026-11-02T12:49:59Z，早於任何 CV、先驗、OC 輸出）
PREREG 42dc6f4e9933…d8e（13:03:44Z；登記 13 個程式檔 sha256；s4 以後的程式都會先斷言它存在）
PREREG_FORWARD 80a0791ca03a…c0b1（13:14:27Z）
PREREG_FORWARD_AMEND1 c9f5231f850e…c302（13:16:40Z；修兩個空資料路徑的錯誤，make_forward.py 新 sha256 ec95c20507f2ddba585f9b5ece840f462c466480078683edae89796faa829ccd）

一、資料（raw/manifest.json 記錄 sha256；vendor_o7 是 o_tier 程式與資料的副本，只 import、不修改）
- 族群：o_main_spa_breaks F_long，2,319 條 × V4 格點 245 窗（pp／窗），161 群（clusters_k161），m_c ≥ 5 的有 31 群。主策略在第 114 群。
- 決策線：V7 navs_exec7 的 B10|J{j}|K5T|{normal,stress}|{級距}|{FIX,GROW}。
  - 每個 j 的窗：訊號日收盤 → 下一個訊號日收盤（訊號日取 plans_j7 AB_V7_B10 seed 0；終點取 jscan_end_days）。超額為對 0050 含息（finlab etf_0050.csv）的差，單位 pp。
  - 錨定月：D_m＝m 月 10 日起的第一個交易日；某天的錨定月＝滿足 D_m ≤ 該日的最大 m。
  - 每月取 13 個 j 的中位數，至少要有 7 個 j。主格為 100 萬 FIX、一般成本。
- 期間：SEL 為錨定月 ≤ 2019-04；VAL 為 2019-05～2023-12；HOLD 為 2024-01～2026-10。

二、方法版本 V1～V7（hmethods.py）
- 共同設定：
  - G2＝Efron log-spline 先驗，df 5、c0 0.1，格點 [−8,4] 步長 0.02，擬合 medoid 的 α̂。
  - 誤差拆成兩部分：族群共同 vP（medoid 平均序列的 HAC L6），加上個別誤差（成員減族群平均的 HAC L6）。
  - σ_w 用動差法：只取 m_c ≥ 3 的群，Σ(m_c−1)[Var(d_k) − 平均 HACVar(D_k − D̄_c)·m/(m−1)] ÷ Σ(m_c−1)。
- 各版本：
  - V1：焦點直接用 G2 收縮。
  - V2：medoid 事後 ⊛ N(B·d, B·s_d²)，其中 B＝2σ_w²/(2σ_w²＋s_d²)。
  - V3：V1 的 x̂ 減去群內最大值雜訊偏誤 b，b 取 200 個分位點做混合。b 的算法：成員先扣掉自己的均值，平穩 bootstrap L6、B＝400，同列重抽後取最大值。
  - V4：V2 的 d 減去 b（b 改用差序列計算）。
  - V5：群內 t3 先驗，sd＝√2σ_w。
  - V6：G3 NPMLE 加群內常態。
  - V7：大群（m ≥ 10）用自己的 σ_w。

三、CV（cvlib.py）
- 對每群 c：在估計窗上用拿掉 c 的族群擬合。tag=sel 為估計期群內 α̂ 最大的成員；medoid 只報告。
- 預測分布：事後 ⊛ N(0, s_V²)，s_V² 為驗證窗平均的 HAC L6 變異。記錄 logpd 與 PIT。
- 切分方式：
  - 挑選期：窗 0～156 時間切半（78／79）兩個方向，加滾動起點 {60,84,108}，各驗證 48 窗。
  - 最終：245 窗時間切半（122／123）兩個方向，加滾動起點 {84,120,157,192}，各驗證 48 窗。
  - 保留期偽前瞻：錨定月 < 2024-01 的 212 窗，時間切半加滾動起點 {84,120,157}。
- 選版：sel 的 logpd 合計最大者。差距 < 1 nat 視為同分，同分時取主策略 B10 估計期事後均值較低者。
- 校準斜率 s：half 與 roll 兩方案各自合併後做 OLS（y 對事後均值），取兩者中較小者，截在 [0,1]。
- 放大倍數 k：從 1.0 起以 0.1 遞增，取第一個讓 sel 的 80% 涵蓋率 ≥ 0.80 的值（只放大，不縮小）。
- 轉換：θ'＝s·μ＋k(θ−μ)，μ 為事後均值，再線性內插回原格點。
- 最終結果：V3、s 0.2939、k 1.1。

四、先驗（prior.py）：V*（V3）在全期族群上擬合。焦點序列為決策線對齊到族群窗（缺值補 0），群 114，再做 (s,k) 轉換。結果均值 −0.037、sd 0.449，存於 data/forward_prior_main.npz。

五、σ_design＝max(HAC_L6 長期 sd 全期, HAC_L6 長期 sd 2019-05 起) × 1.00＝6.1414。

六、規則族（seq2.py）。共同規定：第 12 個月之前不採取任何行動；升級與下修以先觸發者為準（吸收）。z_n＝S_n/(σ√n)。
- R1：每 6 個月檢視一次（12、18、…）。後驗以格點先驗乘常態概似（σ_design）。升級條件 P(α>0.25) ≥ 0.975；下修條件 P(α<0) ≥ 0.80。
- R2：每 6 個月檢視一次。升級 z_n ≥ c_u·√(36/n)；下修 z_n ≤ −c_d。
- R3：CUSUM，C_t＝min(0, C_{t−1}＋(x_t＋0.5)/σ)；t ≥ 12 且 C_t ≤ −h 時下修。升級條件同 R1。
- R4：在第 12、24、36、48、60 個月檢視。升級 z ≥ c_u4；下修 z ≤ −c_d4。
- 校準：用 iid 常態（σ_design）模擬 200,000 條路徑、120 個月，seed 20261102，以分位數精確解出門檻，使 α＝0 時 36 個月內的邊際誤升級＝4%，α＝−1 時 36 個月內的邊際下修＝55%。
- 結果：c_u 1.8668、c_d 1.2162、c_u4 2.0961、c_d4 1.1413、h 6.6923。

七、OC
- 三種 DGP：
  - D1：iid 常態，sd 為 σ_design。
  - D2：決策線 2019-05 起殘差（共 89 個月，已去均值），6 個月環狀區塊 bootstrap，不縮放。
  - D3：t4 除以 √2，乘上波動狀態乘數（0.7／1.4；高波動穩態機率 0.347；高→低 0.06、低→高 0.0319），再乘 σ_design。
- 參數：α ∈ {−1,−0.5,0,0.5,1}；每個 DGP×α 跑 40,000 條、60 個月，seed [20261105, round((α+2)·100), ord(DGP 末字元)]。
- 凍結條件：三種 DGP 下都要同時滿足「α＝−1 時 36 個月內下修 ≥ 50%」與「α＝0 時 36 個月內誤升級 ≤ 5%」。
- 選規則：通過者中取三種 DGP 誤下修最大值最低者。
- 後備程序：若沒有規則通過，或選中規則誤下修 > 30%，對 H_c ∈ {42,48,54,60,72,84,96,120} 重新校準 R2 與 R4（20,000 條、120 個月），找三種 DGP 下都滿足「檢定力 ≥ 50%、誤升級 ≤ 5%、誤下修 ≤ 30%」的最短 H_c。
- 結果：無規則通過，最短可行期限不存在（到 120 個月仍不行）。

八、refit 條款：前瞻滿 12 個月起，若前瞻 sd > 1.5σ_design，就把 σ 換成前瞻 sd 重算門檻；升級在 S_n 尺度上的邊界只能變嚴。這是唯一允許的 refit。

九、偽前瞻（s6）
- 驗證期起點（2019-05）：用 CV_SEL（V4、s 0.44、k 1.0）加 SEL 窗；σ 為 SEL HAC 4.11。
- 保留期起點（2024-01）：用 CV_PRE2024（V4、s 0.63、k 1.0）加 < 2024 的窗；σ＝max(<2024 HAC, 2019-05～2023-12 HAC)＝6.30。
- 規則：OC_FINAL 沒有選出規則，所以依程式預設使用 R2。門檻沿用最終校準值；限制：c_d 依 σ 6.14 校準。
- 決策從第 n＋2 個月起生效。主策略月報酬＝0050 官方窗報酬＋j 中位數超額。
- 兩種政策：P30 起始 30%，升級後 100%、下修後 0%；P100 起始 100%，下修後 0%。

十、配置（s7）
- 用 K5T（V7X 交件前用 V7）跑 16 格，先驗為 CV_FINAL 套用到各格的 j 中位數序列。
- 模擬：對（0050 官方窗報酬、超額殘差）成對做平穩 bootstrap L6，240 個月、20,000 條路徑，seed 20261106。α 從先驗抽樣。用 tlib.simulate，每月再平衡並扣成本。
- x 取 0～100%，每 10% 一格。x*＝argmax E[logW]／年；平坦區間為與 x* 相差 ≤ 0.25%／年的範圍。

十一、前瞻產生器 make_forward.py（PREREG_FORWARD＋AMEND1）
- 呼叫方式：python3 -I -B make_forward.py --data <tw-stock-screener>/data --e50 <finlab>/data/raw/etf_0050.csv，輸出預設寫到 agents/live_tracking/main_forward/。
- 日曆：J_j＝D_m 之後第 j 個交易日，以 history.csv.gz 的日曆為準。官方日為該月第一個 ≥ 11 日的交易日。
- 官方名單：直接呼叫 main_picks_today.score_on／execute（ex_fin），資料模式與該工具在訊號日執行時相同。
- J 名單：m−1 月營收只有在 ann_date ≤ 訊號日時才使用；ann_date 缺值時視為在 D_m 已公告。未公告的股票改用 m−2 月的 3 個月年增率。其餘條件同上。
- 成交規則（M2）：隔日開盤，開盤漲停且尚未持有的不買，停牌的不買。
- 紙上報酬引擎：逐日計算。換股日拆成「前收→開盤（舊持股）」、開盤成交付成本、「開盤→收盤（新持股）」三段。
- 成本：每邊手續費 0.1425%×0.28，加上 max(0.1%, 一跳/開盤)；賣出另加稅 0.3%。壓力情境每邊再加 0.4%，手續費改為 6 折。
- JMED：13 個 j 的窗全部完成後取中位數定稿。
- 名單快照：第一次寫入後凍結，之後的差異寫入 list_diffs.csv。
- 外部線：CAPM、REGIME、R_ROLL 從各自的輸出檔讀取；R2、T1、T2 標為 pending_external。
- trades.csv：由 fills_input.csv 補齊 16 個欄位，與 V7X s8_forward_calib.FIELDS 一致。
- decision_log：因規則沒有凍結，state 一律為 DESCRIPTIVE_ONLY。

十二、複製容許差：OC 各點 ±2pp；CV 選版要相同；s、k 各 ±0.05；x* 落在同一個平坦區間；乾跑時 2026-10-12 的名單要逐檔一致。

## key_numbers
【CV】挑選期 CV（SEL 157 窗，5 個切分）：選 V4_HNMAX（logpd −136.0），s＝0.44，k＝1.0（涵蓋率 0.877）。最終 CV（245 窗，6 個切分，186 次）：選 V3_G2MAX，logpd −227.3；V4 −233.3、V1 −257.7、V5 −247.3。s＝0.294（half；roll 為 1.34），k＝1.1，涵蓋率 0.806。保留期前 CV（<2024，212 窗）：選 V4（與 V3 同分），s＝0.63，k＝1.0。
【σ】決策線（B10 K5T 100 萬 FIX j 中位數）HAC sd：SEL 4.11、全期 5.03、2019-05 起 6.14（iid 6.51）。σ_design＝6.14。α＝−1 時 36 個月的 z 期望值為 −0.98。
【先驗（主決策線）】V3＋校準：均值 −0.037pp／月、sd 0.449、P(α>0) 0.474、P(α>0.25) 0.267。未校準 V3：均值 −0.124、P 0.387。
【門檻】c_u 1.867（OBF，於 36 個月）、c_d 1.216、c_u4 2.096、c_d4 1.141、CUSUM h 6.69（k_ref −0.5）。
【OC，36 個月，D1／D2／D3，各 40,000 條】
R2：下修檢定力 0.547／0.591／0.521；誤升級 0.038／**0.062**／0.040；誤下修 0.232／0.256／0.196；α＝+1 升級 0.207／0.220／0.187。
R4：0.546／0.587／0.526；誤升級 0.040／**0.068**／0.043；誤下修 0.230／0.253／0.196；升級 0.181／0.213／0.159。
R1：檢定力 0.18／0.19／0.16，60 個月內從未升級。
R3：檢定力 0.54／0.60／**0.47**，從未升級。
沒有規則通過。期限延長（R2）：H＝60 誤升級 0.060、誤下修 0.177；H＝120 誤升級 0.057、誤下修 0.091。最短可行期限：120 個月內不存在。
MVN 精確值對模擬：R2 誤升級 0.0394 對 0.038，檢定力 0.5476 對 0.5466。
【事後診斷（不改判定）】target_up 改 3%：R2 通過（誤升級 0.050、誤下修 0.258、檢定力 0.523）。σ 改 max(HAC, iid) 再加 3%：R2 誤升級 0.041、R4 0.047，兩者都通過，誤下修 0.25，α＝+1 升級 0.13～0.15。
【偽前瞻 2019-05 起，89 個月】j 中位數月超額 −0.13pp（sd 6.51），z_end −0.30，沒有觸發。年化：P30 29.73%（MDD −26.5%）、P100＝主策略 j 中位數可執行 26.24%（−35.0%）、0050 30.38%（−29.5%）、同池等權 21.40%、主策略官方日零衝擊 41.83%。壓力成本：P30 26.2%、主策略 15.1%。去掉 2021 後 z_end −1.05；去掉最大月（2021-05 +26pp）後 −0.97。
【偽前瞻 2024-01 起，33 個月】月超額 −0.81pp，於 2024-12（第 12 月）下修。P30 56.6%、P100 45.3%、主策略 44.8%、0050 61.4%、等權 26.5%、官方日零衝擊 62.5%。對 0050 t −1.44。
【配置，K5T V7（V7X 未交件），一般成本】P(α>0)：30 萬 FIX 0.480、100 萬 FIX 0.474、300 萬 FIX 0.441、1,000 萬 FIX 0.422；GROW 0.394～0.463。x* 16 格全部 0%。平坦區間：30 萬／100 萬 FIX 為 0～30%，1,000 萬為 0～10%。E[logW] 在 x＝100% 時比 x* 少 1.5～2.6%／年。10 年贏 0050 機率（x＝100%）0.36～0.42。壓力成本：P 0.29～0.33，平坦區間 0～10%。
【乾跑】2026-10-12 官方名單與 SOP 逐檔、同順序一致（10／10）。2026-09 官方名單與 V7 B10 一致。2026-08／09 共 25 個 J 名單：17 個與 20 個 AB seed 全部相同，平均重疊 9.64／10。2026-09 窗報酬對 V7：零衝擊平均絕對差 0.50pp，K5T 0.72pp，最大 2.15pp。冪等：兩次執行所有輸出 sha256 相同。V7 曆法缺陷：finlab taiex 多了 2026-09-20（週日）一列，使 V7 該月 J7 被跳過、J8～J12 往後錯一天。
【登記】PREREG_DESIGN 9f5a41d7…（12:49:59Z）；PREREG 42dc6f4e…（13:03:44Z）；PREREG_FORWARD 80a0791c…（13:14:27Z）；PREREG_FORWARD_AMEND1 c9f5231f…（13:16:40Z），make_forward.py sha256 ec95c205…a829ccd。

## findings
# FINDINGS：可證偽的前瞻序貫規則 2.0（o_forward_sequential_v2，方法題）

資料夾：/home/user/-/research/tw_stock_selection/agents/o_forward_sequential_v2/
前瞻輸出：/home/user/-/research/tw_stock_selection/agents/live_tracking/main_forward/（已初始化）
（子代理不能寫 .md。請主研究員把這份全文存成 FINDINGS.md。）

## 結論
**判定：不可用（判定規則）；前瞻產生器可用（乾跑通過，已登錄 sha256）。機制：部分支持。**

1. **依事前寫死的條件，規則不能凍結。** 4 條規則（R1～R4）沒有一條在三種 DGP 下同時通過「α＝−1 時 36 個月內下修 ≥ 50%」與「α＝0 時誤升級 ≤ 5%」。
   - R2 與 R4 都卡在 D2（2019 年後殘差的 6 個月區塊 bootstrap）。在 D2 下，α＝0 的誤升級分別是 **6.2%／6.8%**。原因是 2019 年後殘差的 iid sd（6.51）大於 HAC 長期 sd（6.14），而且右尾較厚。
   - 下修這一側其實可行：α＝−1 時 36 個月內下修 **52～59%**，α＝0 時誤下修 **20～26%**。這和預測的 15～25% 同一量級，但誤升級不受控。
   - 把期限延到 42～120 個月也救不回來：誤升級始終是 5.7～7.2%。問題出在升級門檻的校準緩衝（4%）不夠，不是期限不夠。所以「最短可行期限」在 120 個月內不存在。
2. **o7 的三個失敗原因，本題修好了兩個。**
   - (a) 選版改用真實資料 CV：最終 CV 選 **V3**（logpd −227.3，次佳 V4 −233.3），與 o7 相同；校準斜率 **s＝0.29**，與 o7 的 0.27～0.32 一致。
   - (b) 有了真正的退出機制：R2 在 α＝−1 時 36 個月內下修 55%；o7 的 e-process 在 24 個月內只有 0.25%，本題挑選期重跑為 0.7%。
   - 沒修好的是誤升級控制，它對非常態雜訊很敏感。
3. **只用事後 CV 的涵蓋率，無法把先驗放寬到夠寬。** 涵蓋率在 k＝1.1 時就達到 0.806，因為驗證期平均數的抽樣雜訊主導了預測區間。結果先驗 sd 只有 0.449pp，相當於約 **187 個月**的資料量，仍然壓過 36 個月的前瞻資料。
   - 依賴後驗的規則（R1，以及 R3 的升級）因此在 60 個月內**從未升級**，R1 的下修檢定力也只有 16～19%。
   - 有檢定力的只有頻率學派的邊界（R2、R4）。這是本題最重要的方法教訓。
4. **配置：16 格的 x* 全部是 0%。**
   - 主決策線（B10 K5T 100 萬 FIX，j 中位數）在起點的 **P(α>0)＝0.474**，落在預測的 0.45～0.6 之內。
   - 一般成本下，平坦區間是 30 萬／100 萬 FIX **0～30%**，到 1,000 萬為 0～10%。壓力成本下全部是 0～10% 或 0%。
   - o7 的「30～100 萬 FIX 90%」**沒有重現**，預測 3 成立。
5. **偽前瞻（樣本外）：主策略可執行版 2019 年後輸 0050，但規則抓不到，除非最近一段夠糟。**
   - 從 2019-05 起 89 個月，月超額 −0.13pp，z＝−0.30，沒有觸發。
   - 從 2024-01 起 33 個月，月超額 −0.81pp，在 **2024-12（第 12 月）觸發下修**。
6. **前瞻產生器 make_forward.py 乾跑全部通過。**
   - 2026-10-12 的官方名單與 SOP **逐檔、同順序一致**（10／10）。
   - 2026-09 官方名單與 V7 B10 一致。
   - J 名單 25 個中，17 個與 V7 的 20 個 AB seed 完全相同，平均重疊 9.64／10。
   - 2026-09 窗報酬對 V7：零衝擊平均絕對差 0.50pp，K5T 0.72pp。
   - 冪等：兩次執行的輸出 sha256 相同。
7. **對使用者的含義**
   - 研究中心目前**沒有**一條通過自己 OC 標準的前瞻判定規則。2026-11 起的前瞻資料照常收集，但依登記，decision_log 只記描述統計（z、後驗機率），不做升級或下修。
   - 事後診斷（第 9 節）顯示，把升級校準緩衝改成 3%，或 σ 改用 max(HAC, iid) 再加 3% 緩衝，R2 就能通過（誤升級 4.1～5.0%、誤下修 25%、檢定力 52%）。這要在下一輪重新登記後才能使用，而且它不看任何報酬均值，只看殘差形狀。

## 1. 研究問題
「2019 年後主策略能否打敗 0050」這個問題，歷史資料已經答不出來：被看過太多次，2019 年後的落後又和零無法區分。前瞻資料是唯一乾淨的證據。本題要交出四樣東西：
- 一條**事前寫死、可被證偽**的序貫規則：檢定力列為凍結的硬條件，誤升級要受控。
- 依真實資料 CV 校準的先驗。
- 只用 K5T 模型的配置平坦區間。
- 2026-11 起的前瞻產生腳本。

## 2. 機制（誰在賺錢、對手是誰、為何沒被套利）
- **對手：研究中心自己的資料窺探。** 2,319 條候選、主策略改版 ≥ 10 次、CV 校準斜率 0.27～0.32。只要沒有退出機制，雜訊就會被當成優勢；優勢消失時也不會撤退。
- **贏家特性：長期頂尖基金。** 扣費後每年多 3～6pp，但有 10 年級的空窗。能把它們和運氣分開的，只有長時間、事前約定的檢驗。
- **散戶的結構優勢。** 可以零成本做前瞻盲測，沒有贖回壓力，也沒有職涯風險，所以能依規則退回 0050。下修的代價只是「追蹤 0050」，因此規則應該不對稱：升級要嚴格控制誤升級，下修要有檢定力。
- **本題要檢驗的程序假說**
  - (i) 選版改用 CV、先驗乘 s 並放大 sd、改用頻率學派邊界之後，「36 個月內 α＝−1 下修 ≥ 50%、α＝0 誤升級 ≤ 5%」在三種 DGP 下都可行。
  - (ii) 代價是誤下修 15～25%。
  - **結果**：(ii) 成立；(i) 在 D2 下的誤升級不成立。所以判「部分支持」。

## 3. 資料
- 所有輸入都複製到 raw/ 或 vendor_o7/（manifest 記錄 sha256），原檔只讀。
- **族群**：o_tier 的 family_excess（2,319 條 × 245 窗）與 clusters_k161。
- **決策線**：V7 navs_exec7 的 B10 K5T，取 13 個 j 的錨定月窗超額，再取中位數。V7X（第 3 題）在本題執行時還沒有正式淨值，所以配置與先驗都用 V7 K5T，並加註。V7X 交件後依 s7 自動換輸入。
- **0050**：finlab-txd etf_0050.csv，raw/finlab 的 clone，HEAD 05a9cc35。
- **前瞻資料**：tw-stock-screener 的 sparse clone，放在 raw/screener，HEAD 353bc362，資料到 2026-11-02。
- **raw/ 共 282MB**，含兩個 git clone，主研究員提交時應排除。

## 4. 探索發現（只用 SEL，錨定月 ≤ 2019-04，explore.json）
- **決策線（100 萬 FIX K5T j 中位數）**：月超額均值 +0.58pp，iid sd 3.93，HAC sd 4.11，AR1 0.10，超額峰態 1.0。12 個月滾動 sd 在 1.8～6.2 之間，波動狀態切換明顯，所以 D3 是必要的。
- 8 格 sd 幾乎相同（3.92～3.97），均值隨資金規模遞減（30 萬 FIX +0.64 → 1,000 萬 GROW +0.25）。
- 官方日（零衝擊 B10）HAC 4.18；J1 可執行 3.75。j 中位數比官方日少 +0.20pp／月，相關 0.74。
- **可行性**：σ＝4.1 時，α＝−1 的 36 個月 z 期望值為 −1.46。後來看到的 2019 年後 σ＝6.14，同一數字只剩 −0.98，比預測的 −1.1～−1.2 更差。

## 5. 模型迭代紀錄（iterations.json；挑選期；誠實揭露）
- **挑選期 CV**（5 個切分 × 31 群 × 7 版）：選 V4_HNMAX，logpd −136.0（V2 −137.3、V3 −138.1、V1 −155.2）；s 0.44（roll 為 1.05）；k 1.0，涵蓋率已達 0.877。
- 5 個版本（OC 都在 SEL DGP 下跑，每個 DGP×α 20,000 條）：

| 版本 | 設定 | 結果 |
|---|---|---|
| I1 | o7 原規則 E7 | 下修檢定力 **0.7%**，重現 o7 的失敗 |
| I2 | 設計目標＝限制本身（5%／50%） | R2 檢定力 0.473、誤升級 0.053，不過 |
| I3 | 緩衝 4%／55%，j 中位數線 | **R2 通過**：檢定力 0.528、誤升級 0.044、誤下修 0.122；R4 不過（誤升級 0.051） |
| I4 | 緩衝，σ×1.15 | R2、R4 通過，誤下修 0.111，但檢定力只剩 0.50 |
| I5 | 緩衝，官方日代理線 J1 | R2、R4 通過，誤下修 **0.102**（R4） |

- **機械選版會選 I5**（誤下修最低）。我依 PREREG_DESIGN 的決策線定義（j 中位數）和 o7 教訓 1，凍結 I3。理由：兩者誤下修只差 0.017；這次覆寫已記在 iterations.json。
- **有效檢定數**：迭代 5 版；CV 7 個方法 × 3 套切分；最終 OC 4 條規則 × 3 DGP；後備程序 8 個 H；敏感度 15 組；事後診斷 12 組。

## 6. 凍結規格（摘要；全文見 frozen_spec）
- **登記鏈**：PREREG_DESIGN 9f5a41d7…（12:49:59Z）→ PREREG 42dc6f4e…（13:03:44Z，13 個程式檔 sha256）→ PREREG_FORWARD 80a0791c…（13:14:27Z）→ AMEND1 c9f5231f…（13:16:40Z，修兩個空資料路徑錯誤）。
- **決策線**：j 中位數，100 萬 FIX K5T，一般成本。
- **選版與先驗**：CV 選 logpd 最大者；s 取 half／roll 的較小者；k 是讓涵蓋率 ≥ 0.80 的最小值；σ_design＝max(HAC 全期, HAC 2019+)。
- **門檻校準**：iid 常態下設計目標為誤升級 4%、檢定力 55%。
- **OC**：3 種 DGP × 5 個 α × 40,000 條路徑。
- **後備程序**：H_c 42～120 個月。

## 7. 樣本外結果

### 7.1 真實資料 CV（全 245 窗）
| 版本 | sel logpd | 80% 涵蓋 | 斜率 |
|---|---|---|---|
| V1 | −257.7 | 0.73 | 0.67 |
| V2／V7 | −243.0 | 0.75 | 0.69 |
| **V3** | **−227.3** | 0.79 | 0.66 |
| V4 | −233.3 | 0.77 | 0.66 |
| V5 | −247.3 | 0.76 | 0.59 |
| V6 | −247.2 | 0.75 | 0.66 |
- 選 V3。s＝min(half 0.294, roll 1.34)＝**0.294**，k＝**1.1**（涵蓋 0.796 → 0.806）。
- 保留期前 CV（< 2024）選 V4（與 V3 同分），s 0.63。可見 CV 的選版與 s 都隨樣本而變：只看 half 選 V3，只看 roll 選 V4。

### 7.2 最終先驗與 σ
- σ：全期 HAC 5.03；2019+ HAC **6.14**、iid 6.51。σ_design＝6.14。
- 先驗：V3 加校準後均值 −0.037pp、sd 0.449、P(α>0)＝**0.474**（未校準 V3 為 0.387）。

### 7.3 OC（36 個月；每格 40,000 條路徑）
| 規則 | 下修檢定力 D1／D2／D3 | 誤升級 | 誤下修 | α＝+1 升級 | 通過 |
|---|---|---|---|---|---|
| R1 | 0.18／0.19／0.16 | 0／0／0 | 0.03 | 0 | 否 |
| R2 | 0.55／0.59／0.52 | 0.038／**0.062**／0.040 | 0.23／0.26／0.20 | 0.19～0.22 | 否 |
| R3 | 0.54／0.60／**0.47** | 0 | 0.20～0.24 | 0 | 否 |
| R4 | 0.55／0.59／0.53 | 0.040／**0.068**／0.043 | 0.20～0.25 | 0.16～0.21 | 否 |

- **R2 的完整曲線（D1）**
  - α＝−1 時，12／24／36／48／60 個月內下修：0.25／0.44／0.55／0.62／0.68。
  - α＝0 時，誤升級：0.001／0.012／0.038／0.070／0.103（36 個月以後 OBF 邊界繼續下降，60 個月達 10%）。
  - α＝+1 時，升級：0.004／0.07／0.21／0.36／0.48。
- **期限延長**（R2，三 DGP 取最差值）
  - H＝48：檢定力 0.52、誤升級 0.061、誤下修 0.21。
  - H＝60：誤下修 0.18。
  - H＝120：誤下修 0.09，誤升級仍有 0.057。
  - 結論：沒有可行期限。
- **模擬精度**：與多元常態精確解比較（D1 邊際），R2 誤升級 0.0394 對 0.038，檢定力 0.5476 對 0.5466，R4 也一致（s11）。
- **預測對照**
  - 預測 1：誤下修 15～25%。實測 20～26%，**大致成立**。
  - 預測 2：α＝+1 升級 30～45%。實測 **16～22%，不成立**：升級比預期更難。
  - 預測 5：OC 有任一項不符就不能凍結。**觸發。**

### 7.4 偽前瞻（凍結規則套到實際序列；先驗只用起點以前的資料）
| 起點 | 月數 | 月超額均值 | 觸發 | P30 年化 | P100（＝主策略 j 中位數可執行） | 0050 | 同池等權 | 官方日零衝擊 | P30 對 0050 t | P30 對主策略 t |
|---|---|---|---|---|---|---|---|---|---|---|
| 2019-05 | 89 | −0.13pp | 無（z −0.30） | **29.73%**（MDD −26.5%） | 26.24%（−35.0%） | **30.38%**（−29.5%） | 21.40% | 41.83% | −0.19 | +0.19 |
| 2024-01 | 33 | −0.81pp | **2024-12 下修** | 56.6%（−20.4%） | 45.3% | 61.4% | 26.5% | 62.5% | −1.44 | +0.55 |
- **壓力成本**：P30 26.2%／54.6%；主策略 15.1%／31.4%。
- **官方日零衝擊與 j 中位數可執行的落差**：2019 年後約 15pp，重現 o7 的拆解。
- **限制**：偽前瞻沿用最終校準的 c_d（1.216，依 σ 6.14 校準）。若 2019-05 起點依當時的 σ 4.11 重新校準，c_d 約 1.62，下修會更難。驗證期本來就沒有觸發，所以不影響結論。

### 7.5 配置格（K5T，V7；V7X 未交件）
| 級距 | P(α>0) 一般／壓力 | x* | 平坦區間 一般 | E[logW] x＝100% 比 x* 少 | 10 年贏 0050（x＝100%） |
|---|---|---|---|---|---|
| 30 萬 FIX | 0.480／0.330 | 0% | 0～30% | 1.49%／年 | 0.42 |
| 100 萬 FIX | 0.474／0.331 | 0% | 0～30% | 1.54 | 0.42 |
| 300 萬 FIX | 0.441／0.310 | 0% | 0～20% | 1.99 | 0.40 |
| 1,000 萬 FIX | 0.422／0.303 | 0% | 0～10% | 2.26 | 0.38 |
| 30 萬 GROW | 0.463 | 0% | 0～20% | 1.72 | 0.41 |
| 1,000 萬 GROW | 0.394／0.294 | 0% | 0～10% | 2.62 | 0.36 |
- 壓力成本下，平坦區間全部是 0～10% 或 0%。

### 7.6 前瞻產生器乾跑（DRYRUN，pass＝true）
- **SOP**：2026-10-12 官方名單為 2305、2347、2466、6834、2408、6213、2383、3653、1312、3231，逐檔、同順序一致。
- **V7 名單**：2026-09-11 官方名單與 V7 B10 逐檔一致。
- **J 名單**（2026-08／09，共 25 個）：17 個與 20 個 seed 全同。不一致的是 J0、J2、J5、J8、J10 等，各差 1 檔，原因是 V7 用 AB seed 模擬公告時點，本題用實際 ann_date。
- **窗報酬**（2026-09 錨定月，從 2026-08 起跑，持股已進入穩態）：對 V7 零衝擊平均差 0.50pp，對 V7 K5T 0.72pp，最大 2.15pp（J0，因名單不同）。
- **V7 曆法缺陷（新發現）**：finlab taiex_twii.csv 有 2026-09-20（週日）一列。V7 的 j 曆法因此跳過 2026-09 的 J7，J8～J12 也往後錯一天。2019 年後只有這一天，影響很小，但應通知 V7 維護者。
- **其他測試**：用假成交測過 trades.csv 的 16 欄補值（開盤價、前收、距開盤跳數、σ60、V60、oa_share_dec、占開盤量）；decision_log 的規則路徑也用假規則測過。

## 8. 穩健性
- **σ 設錯的影響**（R2，規則仍用 σ_design）
  - 真實 sd 為 1.2 倍：誤升級 7.8%（D1）、6.8%（D3），誤下修 30%。
  - 真實 sd 為 1.5 倍：誤升級 14%。
  - 真實 sd 為 0.8 倍：誤升級 1.3%，檢定力 0.51。
  - 結論：規則對 σ 很敏感。refit 條款（> 1.5σ 才重算）太寬鬆，應改成 > 1.2σ（下一輪）。
- **D2 的變形**：改用全期殘差，誤升級 2.0%；區塊長 3 個月，5.8%；12 個月，1.2%。D2 的失敗來自 2019 年後短區塊的雜訊結構。
- **CV 選版**：只看 half 選 V3；只看 roll 選 V4；以 medoid 為目標選 V3。主決策線的校準先驗 P(α>0) 依版本在 0.47（V3）到 0.63（V7）之間，選版決定了起點是悲觀還是中性。
- **年度與最大貢獻**
  - 偽前瞻逐年月超額：2019 −1.97、2020 −0.91、2021 **+2.21**、2022 −0.15、2023 +1.43、2024 **−2.85**、2025 +1.59、2026 −1.28（pp）。
  - 去掉 2021 年：z_end 從 −0.30 降到 −1.05，仍沒有觸發。
  - 去掉最大月（2021-05，+26pp）：z −0.97。
- **分產業**：不適用。決策線是組合層級的月超額，本題不改選股。

## 9. 事後診斷（看過失敗才做；不改判定；s12）
| σ | target_up | R2 通過 | 誤升級 | 誤下修 | 檢定力 | α＝+1 升級 |
|---|---|---|---|---|---|---|
| HAC 6.14 | 3% | 是（剛好） | 0.050 | 0.258 | 0.523 | 0.157 |
| HAC | 2.5% | 是 | 0.044 | 0.258 | 0.523 | 0.141 |
| max(HAC, iid 6.51) | 3% | 是（R4 也過） | 0.041 | 0.250 | 0.523 | 0.145 |
- **建議下一輪登記**：σ＝max(HAC, iid)、target_up 3%、規則 R2。這只用殘差形狀，沒有用到任何報酬均值，是乾淨的設計修正。
- **代價**：α＝+1pp／月時，36 個月內升級只有約 15%，60 個月約 35～40%。「證實優勢」本質上需要 5 年以上。

## 10. 機制是否被支持：部分支持
- **支持**
  - CV 選版、s 校準、有檢定力的退出機制都做得到：下修 50% 檢定力在 36 個月可達成，誤下修 20～26%。
  - o7 的「90%」被否定。
- **不支持**
  - 在 2019 年後真實雜訊結構下（厚尾、短期波動大於長期 sd），4% 的設計緩衝控制不住 5% 的誤升級。
  - 靠 CV 涵蓋率放寬先驗的力度不夠：先驗仍相當於約 187 個月的資料，後驗規則失去作用。

## 11. 散戶怎麼執行
1. **每月換股日**（2026-11-11）收盤後：先 `git pull` tw-stock-screener 與 finlab-txd，再執行：
   `python3 -I -B research/tw_stock_selection/agents/o_forward_sequential_v2/make_forward.py --data <tw-stock-screener>/data --e50 <finlab-txd>/data/raw/etf_0050.csv`
   - 11-10（J0）、11-11（J1＝官方日）、11-18（J6）收盤後各跑一次，名單才會在當時凍結。
   - 每月下旬 J12 收盤後再跑一次，讓 JMED 定稿。
2. **實際成交**填進 live_tracking/main_forward/fills_input.csv（date、code、side、order_time、limit_px、fill_px、shares、is_odd、list），腳本會補齊 trades.csv，給第 3 題校準 κ。
3. **目前沒有凍結的判定規則。** decision_log 只記 z 與後驗機率，不要依它改配置。配置仍照 SOP：使用者自選全押，或「0050 核心 70%＋主策略 30%」。
   - 依本題配置格，E[log W] 的平坦區間是 **0～30%**。以「長期對數財富」為目標的使用者，主策略比重不宜超過 30%。
   - 使用者可以接受大回撤、想押機制恢復的話，全押的期望代價約 1.5～2.6%／年（一般成本）。

## 12. 限制
- **V7X 未交件**：配置格與先驗用的是 V7 K5T（帶四項偏樂觀設定）。V7X 交件後 P(α>0) 只會更低。
- **J 線成本**：J 線紙上成本是 K5T 小帳戶近似，用的是 max(0.1%, 一跳)，沒有模擬大帳戶的參與率。
- **混合線**（M_E50_30、C_E50_70、M_2330_30）未計再平衡成本。
- **外部線**
  - R2、T1、T2 是 pending_external：題 1 與題 2 都判不可用，也沒有提供可接的報酬定義。
  - REGIME_V5 要等 o7_regime 的 ledger.csv。
  - R_ROLL 用交易紀錄重建，是等權近似。
- **偽前瞻**的主策略月報酬是近似值：0050 官方窗加 j 中位數超額。c_d 用最終 σ 校準。
- **本題沒有獨立複製。** 判定規則「可用」本來就需要複製員重現 OC 在 ±2pp 以內；本題規則已判不可用。不過 OC 的模擬精度已用多元常態精確解交叉檢查（差 < 0.2pp）。
- **CV 本身有雜訊**：選版隨切分方式變（V3 或 V4），s 介於 0.29 和 1.34 之間，取不利端是事前寫死的保守選擇。

## 13. 檔案
- **程式**：s0_copy_raw、s1_explore、s2a_prereg_design、s2b_cv_sel、s2c_iterate、s3_prereg、s4_cv_final、s5_oc_final、s6_pseudo_forward、s7_alloc、s8_robust、s9_dryrun、s10_freeze_forward、s11_check、s12_posthoc、fslib、cvlib、seq2、prior、make_forward。
- **結果**：results.json（PREREG 鏈、CV_SEL／FINAL／PRE2024、ITER、OC_FINAL、PSEUDO_FORWARD、ALLOC、ROBUST、CHECK_MVN、DRYRUN、POSTHOC）、explore.json、iterations.json。
- **資料**：data/oc_final.csv、data/forward_prior_main.npz、data/cv_*.parquet。
- **乾跑輸出**：forward_dry/。

## reviews
[
 {
  "final": "不可用",
  "issues": [
   "維持「不可用」，但失敗原因寫錯了。D2 的誤升級 6.2% 幾乎全由單一月份 2021-05（+26pp）造成，不是 FINDINGS 說的「2019 年後厚尾、短期 sd 大於 HAC」這種結構性雜訊。review_stat_o8.py 重跑 R2（40,000 條，36 個月）：基準 6.13%；只拿掉 2021-05 降到 2.62%；拿掉最大 3 個月降到 0.86%；縮尾 ±2.5sd 為 4.21%；只用 2024-01 起的殘差 2.42%，只用 2019-05～2023-12 的殘差 7.44%。殘差偏態 1.1、超額峰態 2.24，全來自少數幾個月。",
   "D2 只是一組 89 個月的實現值，它的誤升級本身抽樣誤差極大。對參考殘差做外層 6 月區塊 bootstrap（200 組，每組 8,000 條）：誤升級中位數 4.5%，q10～q90 為 1.2%～10.0%，P(>5%)＝0.44。所以 D2 下的通過或不通過都不是證據。前瞻期真實的誤升級率在 1%～10% 之間都說得通，規則的 OC 無法事前保證。照不利方向判，仍是不可用。",
   "第 9 節事後診斷「R2 通過（誤升級 0.050，剛好）」等於拿雜訊選版。這是看過失敗之後，從 2 種 σ × 3 個 target × 2 條規則共 12 組裡挑出來的，只跑 20,000 條，MC SE 約 0.0015。而且 σ＝max(HAC, iid 6.51) 的 iid sd 來自同一組 89 個月殘差（含 2021-05），再拿同一組殘差去做 D2 驗證，是循環論證。殘差縮放到 σ_design 只看形狀時，誤升級是 4.96%，正好卡在邊界。說它「乾淨的設計修正、只看殘差形狀」是過度宣稱，不能當成下一輪登記的有利預期。",
   "「下修檢定力夠、下修這一側可行」的說法過強。R2 在 36 個月內的下修率：α＝−0.5pp／月（年落後約 6%）只有 0.34～0.42；α＝0 有 0.20～0.26；α＝+0.5 仍有 0.11～0.13。所以下修觸發的概似比（α＝−0.5 對 α＝0）只有約 1.6～1.7，是很弱的證據。50% 檢定力只在 α＝−1pp／月（年落後約 12%，屬災難級）才達到。下修觸發時，有很大比例是誤下修。",
   "偽前瞻被標成「樣本外」不成立。B10／V7 主策略、2,319 條族群、決策線定義，都是看過 2019～2026 全期後迭代出來的。c_d＝1.216 與 σ_design 6.14 也用到 2019 年後含保留期的資料。所以 7.4 只能算 in-sample 的規則示範，不能當樣本外證據。用起點當時的 σ 重新校準後：VAL 的 c_d＝1.62，仍不觸發；HOLD 的 c_d＝1.187，2024-12 仍觸發。結論不變，但標籤要改。",
   "2019 年後「主策略輸 0050」幾乎就是一個月的事。全期月超額 −0.13pp，HAC t −0.20；去掉 2021 年為 −0.49pp（t −0.81）；只去掉 2021-05 為 −0.43pp（t −0.77）。主策略 2019 年後接近打平 0050，靠的是 2021-05 單月 +26pp，屬於少數月份依賴。這對主策略更不利，但 FINDINGS 沒有點出來。",
   "偽前瞻的 t 值是重複計數。VAL 期沒有觸發時，P30 超額＝0.3×主策略超額，所以 P30 對 0050 的 t（−0.1868）和主策略的 t 完全相同，「對主策略 t +0.19」只是它的相反數，不是獨立證據。另外表 7.4 把 P100 標為「＝主策略 j 中位數可執行」，但 HOLD 期 P100 年化 45.33%，主策略是 44.77%（P100 有下修切換），標示錯誤。",
   "CV 選版與 s 不穩定，「重現 o7」是選擇性陳述。CV_SEL 和 CV_PRE2024 都選 V4，只看 roll 也選 V4，只有 FINAL 合併與只看 half 選 V3。half 斜率在所有版本都是 0.27～0.33，roll 斜率都是 1.18～1.62，所以 s≈0.29 反映的是前後半期的 regime 斷點，不是版本校準。主線先驗 P(α>0) 依版本在 0.47～0.63 之間，x*＝0% 的結論取決於選 V3 和取 s 下限。方向對主策略不利，可以接受，但「CV 修好了選版」這一點的支持度應下調。",
   "R2 的 OBF 升級邊界在 36 個月以後繼續下降：α＝0 時 60 個月內誤升級 D1／D2／D3 為 10.3%／13.1%／9.5%，R4 為 5.2%／8.4%／5.6%。前瞻追蹤不會在 36 個月停下，36 個月的 OC 低估了實際的誤升級。refit 條款要到 1.5σ 才觸發；在 1.5σ 下誤升級是 14%，1.2σ 下 7～8%，所以這條條款保護不了誤升級。",
   "make_forward 的 J0 名單有輕微前視風險。程式碼中 ann_date 缺值時一律視為 D_m 當日已公告，若實際公告晚於 D_m，J0 會用到訊號當時還沒公開的營收。缺值比例沒有量化。乾跑只對了 2026-09／10 這兩個月，用的是 2026-11-02 的資料快照，不足以排除這個風險。"
  ],
  "note": "我自己寫了驗證程式 /home/user/-/research/tw_stock_selection/agents/o_forward_sequential_v2/review_stat_o8.py（python3 -I -B，只讀，沒有修改研究員的任何檔案），並重現了 D2 基準誤升級 6.1%。\n\n判定規則依登記的凍結條件確實不能凍結，所以「不可用」成立，我同意這個判定。但兩點應該修正：\n- 失敗原因應改寫為「D2 誤升級由單月 2021-05 主導，而且 D2 只是一組 89 個月的實現值，抽樣不確定性是 1%～10%」，而不是「2019 年後雜訊結構」。\n- 機制宜從「部分支持」下調到接近「不支持」：\n  - 下修的概似比只有約 1.6；\n  - CV 選版和 s 隨切分方式搖擺（V3／V4，s 0.29～1.34）；\n  - 第 9 節的事後修正是拿同一組樣本循環調出來的，看不出可行性；\n  - 偽前瞻不是真正的樣本外。\n\n前瞻產生器的乾跑結果可以接受，但 J0 的 ann_date 缺值處理要補量化。\n\n另外，主策略 2019 年後接近打平 0050，完全靠 2021-05 單月 +26pp；去掉這個月，月超額是 −0.43pp（HAC t −0.77）。這對主策略更不利。"
 },
 {
  "final": "不可用",
  "issues": [
   "主策略扣成本後本身就輸 0050，且周轉很高。2019-05 起的官方日 B10，單邊月周轉 67%，年 8 倍。光證交稅就是每月 0.20pp（每年 2.4%）；K5T-lite 全成本約每年 4.7%。可執行的 j 中位數年化 26.2%，0050 30.4%，每年少 4.1pp；壓力成本下 15.1%，每年少 15.3pp。16 格配置的 x* 全部是 0%，但第 11 節仍把「使用者自選全押」列為 SOP 選項，前後矛盾。預設應改為 0050，主策略最多 30%（最適是 0%）。",
   "決策線的 α=0 是算術月超額，不等於財富打平。2019-05 起算術月超額是 −0.13pp，換成對數是每月 −0.27pp（年化幾何落後 4.14pp）。成本級的真實落後約每月 −0.3～−0.5pp，在這個量級下，R2 36 個月內的下修率只有 34～42%，α=0 時的誤下修是 20～26%，兩者幾乎分不開。「下修這一側可行」只在 α=−1pp/月（約每年 −12%）這種不太可能的情形下成立。",
   "前瞻產生器在訊號日當晚不會產生名單。MP.execute 在隔日還沒有開盤價時回傳 None，build_lists 就 continue，不寫任何快照。我把 history 截到 2026-10-12 實跑 make_forward.py：paper_lists 是空的。所以第 11 節「J0／J1／J6 收盤後各跑一次，名單才會在當時凍結」是錯的：快照要到成交日開盤價已知後才寫，本檔也不會輸出可下單的名單（實際下單仍要用 main_picks_today.py）。另外，--asof 不會截斷價格資料（execute 用的是全日曆），所以無法用 asof 回放來驗證凍結。",
   "零股問題。2019 年後，每檔 10 萬（100 萬 FIX 決策線）時，一張的價格超過配置的占 49%，2024～2026 年是 64～78%；每檔 3 萬（30 萬）時占 89%。盤中零股不參加 09:00 的開盤集合競價，所以拿不到開盤價。make_forward 的 cost_side 沒有零股項，但 V7 K5T（先驗與 σ 的校準基礎）有零股、2% 現金、參與上限和處置賣出規則。前瞻決策線名為「100 萬 FIX K5T」，實際上是不分級距的 K5T-lite，口徑不一致。乾跑 2026-09 窗：ours 減 V7 K5T 的有號平均是 +0.59pp（9 窗），名單完全相同的 6 窗是 +0.33pp。只有一個月的資料，無法排除每月幾個 0.1pp 的樂觀偏誤，這已和 alpha_min 0.25 同一量級。",
   "容量。300 萬以下沒問題（開盤參與率中位數 ≤ 1.8%）。1,000 萬時，20.6% 的持股下單額超過開盤競價成交值的 20%，5% 超過 50%。make_forward 沒有參與上限，會高估 1,000 萬使用者的前瞻報酬。",
   "處置與預收款券。乾跑 42 份名單中，J 線有 5 檔次是在處置第一天買進（訊號日當晚已公告）。這些需要預收款券、分盤撮合，但引擎仍以開盤價加一般成本成交。2021-09 起，官方日持股有 7.2% 在持有期間碰到處置，賣出時要預收券；make_forward 一律以開盤價賣出，V7 則用收盤價減一跳。頻率不高，但偏誤方向都對策略有利。",
   "盯盤與決策線可持有性。開盤漲停時遞補下一名，要在 09:00 盯盤，而遞補股的開盤價實際拿不到（乾跑 0 次；V7 在 246 個月中 13 次）。JMED 是 13 個紙上組合超額的中位數，不是任何人能持有的組合。官方日零衝擊年化 41.8%，j 中位數可執行 26.2%，差約 15pp；這主要來自換股日運氣，不是成本。散戶照 SOP 在官方日交易，應預期 j 中位數的水準，而不是官方日的歷史數字。",
   "「P30 對主策略 t +0.19」是套套邏輯。P30 的超額等於 0.3 × 主策略超額，所以它對 0050 的 t 和主策略相同，對主策略的 t 只是加了負號，沒有提供獨立證據。P30 也沒計 0050 與主策略之間的再平衡成本。",
   "同意研究員的判定：規則不可用。從成本與可成交角度還要補兩點：前瞻線口徑不是 V7 K5T，而且對零股與處置偏樂觀；第 11 節的執行說明（何時凍結、全押選項）必須改寫，才能交給散戶使用。"
  ],
  "note": "我同意「判定規則不可用」，另外指出前瞻產生器有兩個需要改正的地方。第一，執行時點說錯：訊號日當晚實跑不會產生任何名單快照。第二，前瞻決策線的成本口徑與 V7 K5T 不同（沒有零股、現金、參與上限、處置賣出），而且偏誤方向全部對策略有利。主策略在 2019 年後扣成本輸 0050：一般成本每年少 4.1pp，壓力成本每年少 15pp；年周轉 8 倍。真實量級的落後（每月 −0.3～−0.5pp），規則 36 個月內分辨不出來。驗證程式：/home/user/-/research/tw_stock_selection/agents/o_forward_sequential_v2/review_exec_o8.py（python3 -I -B，只讀）。截斷資料的實跑放在 /tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/o8exec/（out/paper_lists 為空）。"
 }
]

## replication
{
 "reproduced": true,
 "my_numbers": "I rewrote everything from scratch under /home/user/-/research/tw_stock_selection/agents/o_forward_sequential_v2/replicate_o8/ (rlib, calib, oc, hier, prior, cv, cv_summary, cv_detail, pf, alloc, horizon, posthoc, mvn_check; all run with python3 -I -B). I did not read the researcher's s*/cvlib/seq2/prior/fslib code.\n\n[Decision line σ] HAC sd: SEL 4.110, full period 5.024, from 2019-05 6.1396 (iid 6.510, 89 months). Monthly excess mean -0.122. Expected z at α=-1 over 36 months: -0.98.\n\n[Thresholds] Exact match to 4 decimals: c_u 1.8668, c_d 1.2162, c_u4 2.0961, c_d4 1.1413, h 6.6923. This requires seed 20261102 with α=0 taking the first 200k×120 draw and α=-1 taking the second.\n\n[OC, 36 months, D1/D2/D3, 40k paths each]\n- R2: power 0.547/0.591/0.519; false upgrade 0.038/0.062/0.041; false downgrade 0.232/0.256/0.196; upgrade at α=+1 0.207/0.220/0.181.\n- R4: 0.546/0.586/0.524; 0.040/0.068/0.044; 0.230/0.253/0.196; 0.181/0.213/0.155.\n- R3: power 0.543/0.597/0.468, never upgrades.\n- R1 with my own prior (researcher's s=0.294, k=1.1): power 0.141/0.149/0.124. With my own CV values (s 0.308, k 1.2): 0.175/0.192/0.155. With the researcher's published prior grid (diagnostic only): 0.176/0.194/0.157. All three never upgrade (max upgrade within 60 months is 0.0003).\n- No rule passes.\n- Large-sample check of the MVN value (4M paths): R2 false upgrade 0.0393, power 0.5477.\n- Horizon extension for R2: H=60 false upgrade 0.060, false downgrade 0.175; H=120 0.056 and 0.094. Minimum power across DGPs stays about 0.52 but false upgrade is always above 5%, so no feasible horizon exists within 120 months.\n\n[Post-hoc]\n- target_up 3%: R2 max false upgrade 0.049, false downgrade 0.252, minimum power 0.522. Passes.\n- σ = 6.51 plus 3%: R2 false upgrade 0.038, R4 0.043, false downgrade 0.244, upgrade at α=+1 0.11–0.165. Passes.\n\n[Prior] V3 calibrated with s=0.2939, k=1.1: mean -0.032, sd 0.417, P(α>0) 0.467, P(α>0.25) 0.254. Uncalibrated V3: mean -0.107 (-0.121 in a run with a different bootstrap seed), P 0.385. With my own (s, k)=(0.308, 1.2): sd 0.455, P(α>0.25) 0.271.\n\n[CV]\n- Selection period (SEL, 155 fits): V4 is chosen with logpd -134.2 (V2 -135.6, V5 -135.9, V3 -136.3). s=0.490, k=1.0.\n- Final (186 fits): V3 is chosen with logpd -225.8; V4 -230.3, V2 -240.0, V5 -244.6, V6 -246.7, V1 -252.9, V7 -254.1. Slope half 0.308, roll 1.41, so s=0.308. k=1.2 with coverage 0.806; at k=1.1 coverage is 0.796, one observation short of 0.80.\n- Pre-2024: V3 -150.0 vs V4 -151.2, a 1.2 nat gap, which counts as no tie, so V3 is chosen. s=0.757 (V4's slope would be 0.705), k=1.0.\n\n[Pseudo-forward from 2019-05, σ=4.11] z_end -0.279, no trigger. Annualized:\n- P30 29.76% (MDD -26.5%), P100 26.34% (-35.0%), 0050 30.38% (-29.5%), equal weight 21.40%, official-day zero impact 42.33%.\n- Stress cost: P30 26.24%, main strategy 15.22%.\n- z_end without 2021: -1.03. Without the largest month (2021-05, +26.0pp): -0.955.\n\n[Pseudo-forward from 2024-01, σ=6.30] Monthly excess -0.786. Downgrade at 2024-12 (month 12).\n- With the decision taking effect from 0-based index n+2 (2025-03): P30 56.6%, P100 45.33%.\n- Main strategy 45.09%, 0050 61.37%, equal weight 26.45%, official-day zero impact 62.64%.\n- t vs 0050: -0.72.\n\n[Allocation, normal cost] P(α>0):\n- 30萬 FIX 0.473, 100萬 FIX 0.467, 300萬 FIX 0.433, 1000萬 FIX 0.413.\n- GROW 0.384–0.455.\n- x* is 0% in all 16 cells. Flat region: 0–30% for 30萬/100萬 FIX, 0–20% for 1000萬 FIX.\n- x=100% gives up 1.45–2.52%/yr. Probability of beating 0050 over 10 years: 0.36–0.42.\n- Stress cost: P 0.28–0.32, flat region 0–10% (0% from 300萬 up).\n\n[Dry run] main_picks_today run directly on 2026-10-12 with --ex-fin: 2305,2347,2466,6834,2408,6213,2383,3653,1312,3231. Matches the researcher's official_202610 file 10/10 in the same order. The 2026-09 official list matches the top 10 of V7 B10's last row.",
 "diff_explained": "Within tolerance:\n- σ and the thresholds match exactly.\n- R2, R4 and R3 OC are within 0.6pp at every point.\n- The horizon extension, MVN check, post-hoc diagnostics, pseudo-forward (z, annualized returns, MDD, trigger month) and allocation (x* is 0% everywhere; P(α>0) about 0.01 lower) all match.\n- The final CV picks V3 like the researcher, and the ranking V3 > V4 > V5 > V1 is the same. My logpd values are 1.5–5 nats higher because the Efron spline basis, bootstrap seed and handling of missing values differ.\n- The SEL CV picks V4 like the researcher, with k=1.0 the same. My s=0.49 vs their 0.44 sits exactly at the ±0.05 edge.\n\nOutside tolerance or not reproducible, with reasons:\n1) Final k: mine is 1.2, theirs 1.1. At k=1.1 my coverage is 0.796, only 1 of 186 observations short of 0.80. This is a single-observation boundary on a 0.1 step grid. The s values agree (0.308 vs 0.294; roll 1.41 vs 1.34).\n2) Pre-2024 CV: I pick V3 and the researcher picks V4 \"tied with V3\". My gap is 1.2 nats, just above the 1-nat tie line, so it is borderline. s=0.76 vs 0.63 is outside ±0.05. This choice only affects the prior in pseudo-forward; the R2 decisions and returns do not use the prior and are not affected.\n3) R1 OC with my own prior at s=0.294, k=1.1 has power 4pp lower (0.14 vs 0.18), because my prior sd is 0.417 vs 0.449. With the researcher's prior grid, or my own CV's (s, k)=(0.308, 1.2), I get 0.175/0.192/0.155, which matches. My R1 implementation is therefore correct, and the gap comes from small differences in prior shape (spline basis, b bootstrap).\n4) Effective timing in pseudo-forward: to reproduce P100 45.3% and P30 56.6%, the decision must take effect at 0-based index n+2, i.e. 2025-03, the 15th month. Reading \"takes effect from month n+2\" as the 1-based 14th month (2025-02) gives P100 43.6% and P30 56.0%. The spec wording is ambiguous; I recommend writing it out explicitly.\n5) The researcher's \"t vs 0050 -1.44\" for 2024–26 could not be reproduced. A plain t on the j-median monthly excess is -0.72 (mean -0.79, sd 6.23, n=33); official-day zero impact gives +0.23. Their definition is unclear.\n6) Official-day zero impact from 2019-05: 42.33% vs 41.83%, caused by the end-date definition of the last window.\n7) The full-period HAC is 5.024 vs their 5.03, and σ_design is 6.1396 vs 6.1414. These probably come from small calendar differences in how the 2026-09 month is built. They do not affect the thresholds.\n8) Not reproduced: the J-list overlap with AB seeds, the per-window return differences of 0.50/0.72pp, the idempotent sha256, the V7 calendar defect, and the registration chain (PREREG hashes). These require running the researcher's own make_forward/s9 or reading its outputs, which falls outside the scope of rebuilding from the spec. The 2026-10-12 official list was verified by running the SOP tool directly and matches 10/10.\n\nOverall: all key decision conclusions are reproduced. No rule passes; the shortest feasible horizon does not exist within 120 months; the pseudo-forward from 2019-05 does not trigger and the one from 2024-01 downgrades at 2024-12; x* is 0% everywhere; the final CV picks V3. The differences are confined to CV parameters sitting on boundaries: final k, and pre-2024 s and V3/V4."
}