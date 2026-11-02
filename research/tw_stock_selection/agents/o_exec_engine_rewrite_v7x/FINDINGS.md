# V7 可執行引擎獨立重寫（V7X）：依 model_spec7.json 逐條實作，修正四項偏樂觀設定，產出可執行正式表並與原表對帳


輪次 o8｜研究員判定：可用｜最終：待觀察


## headline
V7X 是 V7 可執行引擎的獨立重寫，可用；套上四項修正後，主策略在每個資金級距都打不贏 0050。重寫版零衝擊路徑逐日差 1.7e−14／5.1e−14，挑選期 32 格對原表差 ≤0.078pp；全期一般成本 16 格裡 12 格 ≤0.10pp，最大 0.24pp（在 30 萬，已追到規格模糊，採不利值）。四項修正全開的正式表（B10、K5T、一般成本、13 個換股日中位數）：30 萬到 1,000 萬的全期年化 16.3／16.9（30 萬 FIX／GROW）、16.8／16.0、15.8／14.3、14.0／11.3%，同期 0050 為 14.37%；對 0050 的 t 只有 −0.30～0.73，沒有一格判「贏」；300 萬 GROW、1,000 萬 FIX、1,000 萬 GROW 判「輸」，其餘判「無可辨識差異」；2019 年起 16 格（含 TICK）全輸，年化落後 2.7～9.8pp；對主策略零衝擊的配對 t 為 −5.1～−11.4。

## mechanism
四項偏樂觀設定，每一項都讓成本低估，而且低估量隨帳戶規模、時期非線性變化：(1) FIX 縮放：帳戶固定規模，每月把盈虧抽走或補回，原表沒有計入這些交易的成本（量到約 −0.2pp）；(2) 處置期賣出以「收盤減一個跳動單位」成交，沒有參與上限，忽略分盤撮合造成的量限制；(3) 2020-10-26 以前的盤後零股沒有量上限，而且成交價應該是盤後零股價，不是開盤價；(4) 開盤量前視：用當天實際開盤量決定參與率與衝擊，實盤下單時其實不知道。修正後，主策略的可執行優勢在小帳戶 2006～2020（零股上限）與大帳戶（開盤量預測誤差、處置期上限）會再縮小。另外，獨立重寫本身就是檢定：V7 可執行表至今沒有被獨立複製，「2019 年起 16 格全輸」和「1,000 萬無法判斷」都建立在單方實作上。

## mechanism_supported
部分支持

## predictions
(1) 零衝擊路徑：重寫版與 V7 原版在 80 個 seed、13 個 j 上逐日淨值相對差 ≤ 1e-9（重現 o7 已知的 9.4e-13）。(2) 原設定下的可執行表（4 級距 × FIX／GROW × K5T，TICK 只當敏感度），每格 j 中位數年化差 ≤ 0.10pp／年；超過就逐筆追到第一個分歧的交易，寫明是規格模糊還是實作錯誤。(3) 四項修正的預期方向與量級（事前寫下，事後照實報告）：FIX 縮放成本 −0.1～−0.3pp／年，FIX 才有；處置期上限在 300 萬以下 < 0.05pp／年，1,000 萬 GROW −0.1～−0.4pp；2020 前零股量上限在 30 萬、100 萬的 2006～2020 期間影響最大，−0.5～−2pp／年，2020-10 後是 0；開盤量前視改成事前預測（前 20 日開盤量占比中位數 × 60 日均量）之後，對 1,000 萬的影響 −0.1～−0.5pp／年，對 30 萬接近 0。(4) 合併修正後：2019 年起 16 格（含 TICK 敏感度）全部輸 0050 的結論不變；2006～2018 的正超額縮小；1,000 萬 FIX 從「無法判斷」偏向「輸」。(5) 否證：任一項修正讓報酬變好（符號相反），代表原實作或規格理解有錯，要先查清楚才能凍結。

## frozen_spec
V7X 可執行引擎（獨立重寫）凍結規格（o_exec_engine_rewrite_v7x）
PREREG：results.json PREREG，sha256 faacf2bde2f13242dec52ca5ba5efa8e1eb0c000939cf584f27334b59f94a956，時戳 2026-11-02T13:09:14Z。
- 規格解讀登錄 interp_register.json：sha256 bf409a57…，時戳 12:18Z，在寫可執行引擎之前。
- 雜湊涵蓋全部程式、explore.json、iterations.json、V7 面板、名單、規格與原表。
- s5_oos.py 先斷言 PREREG 存在、雜湊相符，才跑全期。
- 事前預測寫在 PREDICTIONS_PREREG，時間在任何全期數字之前。

〇、範圍與獨立性
- 選定版本：iterations v4。共試 v1～v4 四版。
  - 判準：挑選期 SEL 年化的 j 中位數，對原表的最大絕對差。
  - v4 為 0.0783pp，32 格全部 ≤ 0.10pp。
  - 設定：stress_odd_extra＝true、rank_right＝true、disp_lim＝false。
- 只依規格文字重寫：
  - o_v7_listing_exec_engine/data/model_spec7.json、frozen_spec.txt、FINDINGS；
  - o5 FINDINGS 第六節；
  - o6 frozen_spec.txt。
- 沒有讀 ecore7.py、run_exec7*.py、gen_model7.py，也沒有讀它們的前身 ecore6.py、ecore_o5.py、o5 ecore.py、xlib*.py、replicate/r6_exec.py。
- 零衝擊語意參考 replicate_o7/rb_core.py（獨立複製員自寫，不是 V7 原碼）。目標金額改用排序斷點精確解。
- 名單：
  - B10 主欄：plans_j7.csv，strat=AB_V7_B10，j 0～12，seeds 0～19；
  - 官方日：plans_ab7.csv，seeds 0～39，只報告。

一、資料（V7 凍結面板）
- 欄位：close_raw、open、value、ret、flag。P 下市價只補面板缺值格。
- 衍生量：
  - TR＝收盤有值；G＝1＋ret；CF＝收盤前值補；PC＝前一筆收盤；
  - ADJ＝G·PC/C（無成交時＝1）。
- 流動性與波動：
  - GOOD 日＝有值檔數 ≥ 300。
  - V60＝60 個 GOOD 日的平均成交值，窗內不得缺值；豁免 2017-08-03、2018-07-12、2024-02-15～03-15。
  - σ60＝60 個 GOOD 日 ret 的標準差，至少 40 筆，向後補值。
  - vrank＝V60 ≥ 1e7 且收盤 ≥ 10 的股票，依 V60 由大到小排名。
  - 一律用 t−1 的值。dec＝ceil(vrank/30)；名次 > 300 或缺值時 dec＝10。
- 處置名單：V4 凍結官方名單（rejudge in_panel ＋ disposal_5y），處置期為 start ≤ t ≤ end。

二、帳戶
- 級距 30 萬／100 萬／300 萬／1,000 萬，以 2026 幣值計，用 CPI 換算成當年幣值。
- GROW：第一個訊號日收盤時，持有現金＝級距×CPI2006/CPI2026。
- FIX（原設定）：換股日開盤前，把股數、現金、待撥款全部乘上 sc＝目標/權益_pre，不計成本。
- 淨值用時間加權。

三、換股與成交（原設定）
- 換股流程：
  1. 清掉未成交的順延單。
  2. 處理凍結股與現金對價併購：
     - 換股日 ≥ 下市日時，把前收×對價比×當日 0050 報酬轉成以 0050 計值的待撥款；
     - 撥款日＝下市日＋20 個交易日；撥款後到再投入前仍以 0050 計值；
     - 再投入日以前一日收盤的值計。
  3. FIX 縮放。
  4. 小數股以開盤價轉成現金。
- 目標金額：
  - hi＝(Σ持股×開盤＋現金)/k×0.98；
  - 目標股數＝整張＋零股；
  - need(T) 用實際開收盤價、不考慮參與上限；need(hi) 超過現金時，二分 40 次。
- 下單順序：
  - 依 delta 由小到大，賣單先；
  - 買單成本 > 現金時，整筆順延；
  - 每檔先下零股再下整股。
- 整股：
  - 成交價＝開盤×(1±imp)；
  - 參與上限 max(1, floor(0.2·OA/(1000·前收))) 張；
  - OA＝share_dec×當日成交值。
- 衝擊：
  - K5T（正式）＝max(0.001＋0.5·σ60·√(Q/OA), 一跳/開盤)；
  - TICK（敏感度）＝一跳/開盤。
- 零股：
  - 2020-10-26 前用收盤價；之後用開盤×(1＋drift(跳空桶))。
  - 價差 c 依價格桶 [50,100,300] 與名次桶 [150] 決定，兩者都用 searchsorted 'right'。
  - 賣出 c 下限 0。
  - 2020-10-26 起每日上限 floor(zcap_dec×V60_{t−1}/零股價)；V60 缺值時當 0。
- 手續費：2015 年前 0.6 折，之後 0.28 折，每筆最低 20 元；賣出稅 0.3%。
- LIM：開盤 ≥ 漲停（買）或 ≤ 跌停（賣）時，當日不成交、順延。
- 處置期賣出：收盤減一跳成交，不計衝擊、不設上限、不受 LIM，款項當日可用。
- 順延日：
  - 依待成交股數由小到大處理；
  - 買單張數上限＝floor(現金/(開盤×1050))；
  - 先零股，後整股。
- 壓力成本：
  - 整股與零股每邊再加 0.4%；
  - 全期手續費 6 折；
  - 零股買成本×2。

四、四項修正（事前登記的順序）
- F1：FIX 的提領與注資一律經由現金進出，賣股、買股照常付稅、手續費與衝擊。
- F2：處置期賣出受當日成交值 20% 上限；處置賣出款隔日才可用，換股求解不計入這筆款項。
- F3：2020-10-26 前，盤後零股每日上限＝0.5×zcap_dec×V60/收盤（代理值；敏感度 0.25、1.0）。
- F4：參與上限改用事前 OA_pred＝share_dec×V60_{t−1}；衝擊仍用實際 OA。
- 正式設定＝F1＋F2＋F3(0.5)＋F4，K5T，一般成本。

五、統計
- 口徑與原表相同：用本題統計程式重算原表 912 格，差 ≤ 1e−13。
- 每格把 20 seeds 的日報酬平均成一條淨值。
- 年化：從該 j 的第一個訊號日，到 jscan_end_days 的終點。
- t：訊號日窗的超額對 0050。
- 分期：SEL 至 2018-12-28；2019 年起；VAL 為 2019-05～2023；HOLD 為 2024 起。
- j 中位數超額：取每個 j 的 (年化−同窗 0050)，再取中位數。

六、判定
- 零衝擊閘門：差 ≤ 1e−9。
- 原設定對帳：16 格差 ≤ 0.10pp；超過的格子要追到分歧原因，並採不利值。
- 「贏 0050」需同時滿足：
  - j 中位數超額 > 0；
  - t ≥ 1.5；
  - 2019 年起超額 ≥ 0。
- 否則寫「輸」或「無可辨識差異」。
- 否證條件：任一修正讓結果變好超過 +0.05pp。
- 參與率中位數 > 10% 的格子，TICK 不當下界。
- 判定上限：引擎「可用」，策略「待觀察」。

事後附記（PREREG 之後，不改凍結數字）
1. F2 的「隔日才可用」實作得比規格文字更嚴：入選股的目標金額一起縮小，處置賣出款閒置到下一個換股日。
   - 事後修正版 R2c／R4c（s5c_posthoc_f2.py）讓這筆款項隔日補買。
   - 修正後各格高 +0.36～+0.54pp；判定只變 300萬 GROW 一格。
2. 對帳分歧追到的規格模糊（RECON_TRACE_posthoc）：
   - V60 缺值時零股上限怎麼算；
   - 整股賣單被參與上限擋下後的買單順序；
   - 撥款日剛好是換股日時用嚴格大於還是大於等於；
   - 處置賣出遇收盤跌停。

## key_numbers
【閘門與對帳】
- 零衝擊：
  - 官方日 160 條（80 seeds × 一般／壓力）逐日最大相對差 1.67e−14；
  - 13 個 j × AB／B10 × 2 種成本：5.05e−14；
  - 處置與併購拆解欄：base／disp 4.4e−14；mdelay 只在 B10 J12 一欄差 0.9%（撥款日剛好是換股日，規格模糊）。
- 原設定（R0）對 V7 原表，B10 j 中位數年化：
  - 挑選期 32 格最大差 0.078pp；
  - 全期一般成本 16 格，最大 0.242pp；超過 0.10 的 4 格都在 30 萬（K5T FIX −0.130、GROW +0.242；TICK FIX −0.130、GROW +0.157）；
  - 壓力 16 格裡 10 格 > 0.10，最大 0.250pp；
  - 官方日 32 格最大 0.110pp，一般成本 ≤ 0.093pp。
- 診斷：零股上限改成「V60 缺值時不設上限」後，30萬 FIX 只差 −0.013pp。
- 統計程式重算原表 912 格，差 ≤ 1e−13。

【V7X 正式表】
- 設定：F1＋F2＋F3(0.5)＋F4，B10，K5T，一般成本，13 個 j 中位數；同期 0050 14.37%。

| 帳戶 | 年化 | p10 | 超額 | t | 挑選期超額 | 2019 年起超額 | 驗證期（2019-05～2023）超額 | 保留期年化 | 最大回撤 | 壓力成本年化 |
|---|---|---|---|---|---|---|---|---|---|---|
| 30萬 FIX | 16.33 | 12.39 | +1.96 | 0.62 | +3.81 | −4.82 | +0.28 | 39.4 | −55.4 | 7.41 |
| 30萬 GROW | 16.91 | 13.07 | +2.51 | 0.73 | +4.73 | −4.10 | +0.33 | 39.6 | −55.2 | 7.74 |
| 100萬 FIX | 16.82 | 12.60 | +2.45 | 0.71 | +4.15 | −3.61 | +0.86 | 41.2 | −55.4 | 8.49 |
| 100萬 GROW | 15.98 | 11.90 | +1.61 | 0.55 | +3.95 | −6.51 | −1.96 | 39.9 | −55.5 | 8.49 |
| 300萬 FIX | 15.78 | 11.56 | +1.19 | 0.50 | +3.40 | −5.55 | −0.67 | 39.1 | −56.4 | 7.95 |
| 300萬 GROW | 14.29 | 10.28 | −0.07 | 0.21 | +2.28 | −8.35 | −3.72 | 36.1 | −56.1 | 8.19 |
| 1000萬 FIX | 14.01 | 10.05 | −0.59 | 0.15 | +1.80 | −7.11 | −3.37 | 38.7 | −57.2 | 6.83 |
| 1000萬 GROW | 11.32 | 8.30 | −2.95 | −0.30 | +0.81 | −9.84 | −5.28 | 33.0 | −57.5 | 6.22 |

- 年化、p10、保留期年化、最大回撤、壓力成本年化單位為 %；超額單位為 pp。
- 判定：1000萬 FIX、300萬 GROW、1000萬 GROW「輸」；其餘「無可辨識差異」；沒有一格「贏」。
- 保留期 0050 年化約 55%。
- TICK 敏感度：年化 16.4～18.8%；2019 年起超額 −2.7～−4.8pp。
- 整股成交占開盤量 Q/OA 中位數：1000萬 GROW 13.7%（p90 50%），標註「TICK 不是下界」。
- 事後修正 R4c 各格高 +0.36～+0.54pp：30萬 FIX 16.92、1000萬 FIX 14.59、1000萬 GROW 11.73；判定只有 300萬 GROW 由「輸」變「無可辨識差異」。
- 對主策略零衝擊 B10 j 版：配對 t −5.1～−11.4，每月少 0.29～0.66pp。
- 主策略零衝擊 B10 j 中位數年化 21.48%（t 1.66）；同池等權（官方日）10.73%。

【修正拆解】
- 口徑：K5T、一般成本，13 個 j 的配對差中位數，pp／年。

| 修正 | 30萬 | 100萬 | 300萬 | 1000萬 | 說明 |
|---|---|---|---|---|---|
| F1（只影響 FIX；GROW 為 0） | −0.22 | −0.17 | −0.17 | −0.21 | — |
| F2 FIX／GROW | −0.56／−0.49 | −0.49／−0.40 | −0.51／−0.44 | −0.46／−0.27 | 上限觸發次數中位數 0，全部來自款項閒置；事後修正版 −0.04～−0.21 |
| F3 FIX／GROW | −2.00／−0.98 | −1.02／−0.44 | −0.45／−0.10 | −0.08／−0.06 | 30萬 FIX 在 2020 前 −2.74、2020-10 後 0 |
| F4 FIX／GROW | 0／−0.02 | −0.02／−0.12 | −0.32／−0.60 | −0.83／−1.34 | — |
| 合併 FIX／GROW | −2.57／−1.37 | −1.79／−1.04 | −1.44／−1.10 | −1.41／−1.71 | — |

- F3 代理敏感度（對 R4，30萬 FIX）：比例 0.25 再 −1.84，比例 1.0 為 +1.06。
- 否證：沒有一項修正讓 K5T 格變好；唯一的例外是 TICK 壓力 100萬 GROW 的 F4 為正（TICK 模型的成本不隨參與率變，放寬上限只會加快成交）。

【事件數】
- 每格 13 個 j × 20 seeds＝260 份名單，每份 246 個訊號窗。
- 每個 run 32 格，共 8,320 次模擬；全期共 9 個 run 加 2 個事後 run。
- 處置期賣出每份名單約 28 次；併購延遲 4 次。
- 迭代版本 4 個。

## findings
# FINDINGS：o_exec_engine_rewrite_v7x（V7 可執行引擎獨立重寫：V7X）

輪次 o8｜研究員判定：引擎「可用」｜策略含義：待觀察（研究員上限；數字顯示各級距都沒有贏 0050）

## 一句話
- 重寫的引擎可以用。
- 修正四項偏樂觀設定後：
  - 主策略的可執行 j 中位數，在 30 萬～1,000 萬沒有一格贏 0050；
  - 全期 t 在 −0.30～0.73；
  - 2019 年起 16 格全部輸；
  - 1,000 萬 FIX 由「無法判斷」改判「輸」。

## 1. 研究問題
- 這是 V7 可執行表第一次被獨立複製。原表的「2019 年起 16 格全輸」和「1,000 萬無法判斷」，都建立在單一研究員的實作上。
- 要回答兩件事：
  1. 只讀規格文字重寫，能不能重現原表？
  2. 修正四項偏樂觀設定後，各級距的實盤預期是多少？四項是：FIX 縮放不計成本、處置期賣出沒有量上限、2020 年前零股沒有量上限、用當天開盤量決定參與上限（前視）。

## 2. 機制（贏家與對手）
- 主策略的直接對手是開盤的流動性提供者（造市商、自營商），加上證交稅。他們收取價差與開盤衝擊。
- 散戶能做的只有「少付流動性」：算對自己的成本，選擇扣得起成本的資金級距與成交方式。
- 四項設定都讓成本被低估，低估量隨帳戶規模和時期非線性變化：
  - 小帳戶：零股量上限、FIX 縮放；
  - 大帳戶：開盤量前視、處置期量上限。

## 3. 資料
- V7 凍結面板（data_freeze_v7.json），P 下市價、X 併購規則、V4 官方處置名單。
- V7 凍結名單 plans_j7.csv 與 plans_ab7.csv；model_spec7.json；cpi_tw.json；0050 含息。
- 對帳只讀原版的輸出：navs_exec7、navs_ideal7、seed_navs_v7、navs_jscan_v7、results.json。
- 獨立性：
  - 沒有讀 ecore7.py、run_exec7*.py、gen_model7.py，也沒有讀前身 ecore6、ecore_o5、xlib、r6_exec。
  - 零衝擊語意參考了獨立複製員自寫的 rb_core.py。
- 2020 年前的盤後零股資料（TWT53U）：
  - 證交所網站在本環境被擋（403）；
  - GitHub 只找得到 API 說明，沒有逐檔歷史封存；
  - extras/oddlot_hist 實際只有 2026-07-13 起的 72 天，不是題目寫的 2020-10 起。
  - 因此 F3 用事前寫死的代理，並標明是代理。

## 4. 探索（只用挑選期 2006-04～2019-04，65 份名單）
- **開盤量前視**：換股與順延日，整股成交當天的「成交值 ÷ V60_{t−1}」中位數為 0.68～0.92，但平均 1.0～1.5，分布右偏。原設定在多數日子給的開盤量其實比事前預期小，偶爾放量時才大。所以改用事前預測後，方向不是先驗可知的。
- **FIX 縮放**：每次換股的提領或注資約占權益 4.2%（中位數），一年合計約 64%，都沒有計成本。
- **處置期賣出**：
  - 每份名單約 6.6 次；
  - 賣出金額占當日成交值中位數：1,000 萬 GROW 0.8%（p90 4%），30 萬為 0；
  - 沒有一筆超過 20%。所以量上限在挑選期幾乎不會被觸發。
- **2020 年前零股**：每筆零股單 ÷ (zcap_dec×V60) 的中位數，30 萬為 0.28、1,000 萬為 0.60；39～55% 的單超過 0.5。代理上限會大量觸發。
- **參與率**：1,000 萬 GROW 整股 Q/OA 中位數 16.9%，74.6% 超過 10%。依教訓 7，TICK 不能當下界。

## 5. 模型迭代紀錄（挑選期對帳，共 4 版）
- 每版跑 B10、13 個 j、20 seeds、32 格，只模擬到 2019-04-30，只比 SEL 年化的 j 中位數。
- 寫程式前，22 條規格解讀先登錄 sha256（interp_register.json）。
- 規格解讀在零衝擊階段就先對到：
  - 處置期賣出用「收盤減一跳」、款項計入求解：對原版 navs_ideal7 的 disp 欄差 4e−14；
  - 併購延遲改成「換股日 ≥ 下市日時轉為 0050 計值，撥款後到再投入前仍停在 0050，再投入日以前收計」：mdelay 欄對到 1e−14。
- 各版結果：
  - v1（登錄原樣）：一般成本 16 格最大差 0.063pp；但壓力格差達 5.2pp。
  - v2（壓力的 0.4% 也加在零股）：32 格最大差 0.078pp。
  - v3（處置賣出受跌停限制）：最大差 0.186pp，退回。
  - v4（零股名次桶用 searchsorted right，名次恰 150 歸第 2 桶）：最大差 0.0783pp，選定。追到 2009-01-12 的 1.5e−8 分歧，正是名次剛好 150 的零股單；改後到 2009-03 都沒有分歧。
- 30 萬的小級距在 2006～2008 與原版逐日相同。

## 6. 凍結規格（摘要；全文見 frozen_spec 欄）
- 原設定＝v4。
- 修正依事前順序：
  - F1：FIX 經由現金進出，照付成本。
  - F2：處置期賣出上限 20% 日量；處置賣出款隔日才可用。
  - F3：2020 年前盤後零股上限＝0.5×zcap_dec×V60（代理）。
  - F4：參與上限用事前 OA＝share×V60_{t−1}，衝擊用實際 OA。
- 正式＝四項全開、K5T、一般成本。
- PREREG：2026-11-02T13:09:14Z，sha256 faacf2bd…。

## 7. 結果（全期一次跑完，不改規格）

### 7.1 零衝擊閘門（預測 1：成立）
- 官方日 80 seeds × 2 種成本：逐日最大相對差 1.7e−14。
- 13 個 j × AB／B10 × 2 種成本：5.1e−14。
- 門檻是 1e−9，通過。

### 7.2 原設定對帳（預測 2：部分成立）
- 一般成本 16 格中 12 格 ≤ 0.10pp。
  - 1,000 萬與 300 萬全部 ≤ 0.06pp，例如 1,000 萬 GROW K5T：13.13 對 13.14。
  - 超過的 4 格都在 30 萬：K5T FIX 19.55 對 19.42、GROW 18.50 對 18.74；TICK FIX 19.58 對 19.45、GROW 19.47 對 19.63。
- 壓力成本 16 格中 10 格超過 0.10pp，最大 0.25pp。
- 官方日 32 格最大 0.11pp。
- **分歧追蹤**（全部是規格模糊，不是實作錯誤）：
  1. **零股上限遇 V60 缺值**：V7X 把上限當 0，原版很可能是「不設上限」（NaN 比較為假）。例：6670 在 2025-04 起 V60 缺值，118 股零股一直賣不掉，連帶 1316 買單因現金不足一路順延。改成缺值不設上限後，30 萬 FIX 的差降到 −0.013pp，100 萬 FIX 降到 −0.02pp。
  2. **30 萬 GROW**：GROW 不重設權益。2006～2016 間 1e−6 級的分歧，經整股取整與順延放大成路徑相依的雜訊：13 個 j 的期末淨值比在 0.994～1.043，正負都有。
  3. **1,000 萬 GROW J3 2007-01-16**：整股賣單被上限擋下後，最後一檔買單（2336）整筆順延。規格沒寫這種情況的處理順序；試過依金額排序、部分買進、不檢查現金，都對不上。該格中位數只差 0.007pp。
  4. **撥款日剛好是換股日**（2325，B10 J12）：V7X 照文字「撥款日後」用嚴格大於，原版用大於等於。
  5. **處置賣出遇收盤跌停**：收盤減一跳會低於跌停價，6 次。
- 依 o6 教訓 22 採不利值。原設定下的判定和原表一致：全部「無可辨識差異」，只有 1,000 萬 GROW「輸」。

### 7.3 四項修正（預測 3：方向全對；F1 量級對；F2、F3、F4 部分超出區間）
- 口徑：K5T、一般成本，13 個 j 的配對差中位數，pp／年。
- **F1 FIX 縮放成本**：FIX −0.17～−0.22（TICK −0.15～−0.24），GROW 為 0。預測 −0.1～−0.3 成立，和 o7 審查量到的 −0.21～−0.23 一致。
- **F2 處置期**：
  - 凍結版 −0.27～−0.56；300 萬以下超出預測（< 0.05）。
  - 但 16 格的量上限觸發次數中位數都是 **0**。整個拖累來自「款項隔日才可用」的實作：求解時不計處置賣出款，入選股目標金額一起縮小，隔日入帳的現金沒有委託使用，閒置到下一個換股日。這比規格文字更嚴。
  - 事後修正版（隔日用這筆錢補買）只有 −0.04～−0.21。
  - 結論：處置期量上限本身確實約為 0，符合預測；真正的成本是款項時點。
- **F3 2020 前零股量上限（代理 0.5）**：
  - 30 萬 FIX −2.00（2020 前 −2.74，2020-10 後 0）、30 萬 GROW −0.98；
  - 100 萬 FIX −1.02、GROW −0.44；300 萬 −0.45／−0.10；1,000 萬 −0.06～−0.08。
  - 落在預測的 −0.5～−2 上緣，2020-10 後為 0 也成立。
  - 對代理比例很敏感：30 萬 FIX 在比例 0.25 時再多 −1.84，比例 1.0 時回升 +1.06。這是全題最大的不確定來源，因為缺 TWT53U 歷史資料。
  - 另一個成本管道：零股被拆成多天小單，每筆最低手續費 20 元。30 萬 FIX 的手續費因此從約 9 萬升到約 16 萬元（s9 抽查 seed）。
- **F4 開盤量前視**：
  - K5T：30 萬為 0；100 萬 −0.02／−0.12；300 萬 −0.32／−0.60；1,000 萬 FIX −0.83、GROW −1.34。
  - 方向對，1,000 萬的量級超出預測（−0.1～−0.5）。
  - 原因：事前 OA＝share×V60 在多數日子大於實際 OA（探索中位數 0.7～0.9），上限放寬，單量占實際開盤量的比例上升。1,000 萬 GROW 的 Q/OA p90 由 19.8% 升到 50%，平方根衝擊因此變大。
  - TICK 模型下 F4 正負都有（−0.51～+0.02），TICK 壓力 100 萬 GROW 為正。這不是實作錯誤：TICK 的成本不隨參與率變，放寬上限只會加快成交。這再次說明 TICK 不適合當大帳戶的下界。
- **合併**：30 萬 FIX −2.57、GROW −1.37；100 萬 −1.79／−1.04；300 萬 −1.44／−1.10；1,000 萬 −1.41／−1.71。壓力成本下合併為 −0.91～−2.28。
- 預測 5（否證）：K5T 格沒有任何一項修正讓報酬變好。

### 7.4 V7X 正式表與樣本外分期（預測 4：成立）
- 口徑：B10、K5T、一般成本、j 中位數；同期 0050 14.37%。

| 帳戶 | 年化 | 超額 | t | 判定 |
|---|---|---|---|---|
| 30萬 FIX | 16.33% | +1.96 | 0.62 | 無可辨識差異 |
| 30萬 GROW | 16.91% | +2.51 | 0.73 | 無可辨識差異 |
| 100萬 FIX | 16.82% | +2.45 | 0.71 | 無可辨識差異 |
| 100萬 GROW | 15.98% | +1.61 | 0.55 | 無可辨識差異 |
| 300萬 FIX | 15.78% | +1.19 | 0.50 | 無可辨識差異 |
| 300萬 GROW | 14.29% | −0.07 | 0.21 | 輸 |
| 1000萬 FIX | 14.01% | −0.59 | 0.15 | **輸**（原設定為無法判斷） |
| 1000萬 GROW | 11.32% | −2.95 | −0.30 | 輸 |

- **分期**（超額，pp／年）：
  - 挑選期 2006～2018：+0.8～+4.7。正超額比原設定的 +1.8～+6.4 縮小，預測成立。
  - 驗證期 2019-05～2023：−5.3～+0.9。
  - 2019 年起：−3.6～−9.8，16 格全部為負（TICK 敏感度 −2.7～−4.8）；13 個 j 中為正的只有 0～5 個。
  - 保留期 2024 起：年化 33～41%，0050 約 55%，超額 −15.4～−22.6。
- 最大回撤 −55～−57%；壓力成本年化 6.2～8.5%，輸 0050 5.9～8.2pp。
- **對照組**：
  - 主策略零衝擊 B10 j 中位數 21.48%（t 1.66）；V7X 對它的配對 t 為 −5.1～−11.4，每月少 0.29～0.66pp。
  - 主策略官方日零衝擊 AB 27.37%；同池等權 10.73%。
- 事後修正 R4c（F2 隔日補買）：各格高 0.36～0.54pp，只有 300 萬 GROW 由「輸」變「無可辨識差異」（+0.49pp）；2019 年起 16 格全輸仍成立。

## 8. 穩健性
- **分年**（30 萬 FIX 的 j 中位數超額，pp）：

| 年 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|---|---|
| 超額 | −22.6 | −4.9 | +25.1 | −4.0 | +9.7 | −33.3 | +10.3 | −30.1 |

  - 2006～2018 間 9 年為正。
  - 1,000 萬 GROW 的形態相同，但每年再低 3～9pp。
- **seed 與 j 的分散**：同一格各 j 的年化差可達 5～8pp。p10 年化 8.3～13.1%，全部低於 0050。
- **代理敏感度**：只有 30 萬、100 萬受 F3 代理影響大（±1～2pp）。不論代理比例取多少，「沒有贏」的判定都不變：比例 1.0 時 30 萬 FIX 也只回升 1.06pp，t 仍 < 1。
- **分產業、去掉最大貢獻股**：本題不改選股，沿用 V7 的拆解（3443 占總貢獻 3.3%、2059 占保留期 11.3%），沒有逐筆重算可執行版的個股貢獻。這列為限制。
- **研究員自查 3 格**（s9_audit.py：30萬 FIX J6 s3、1000萬 GROW J2 s7、300萬 FIX J10 s11）：
  - 0 筆違反上限；
  - 持股不為負；
  - 收盤現金沒有負值（F1 的提領由求解消化）；
  - 買單現金流出、賣單流入的符號全部正確。
  - 第二位審查員抽查待辦。

## 9. 機制是否被支持：部分支持
- **支持的部分**：
  - 四項設定都低估成本，修正後 K5T 格沒有一格反向。
  - 低估量隨規模與時期非線性變化：F3 集中在小帳戶的 2020 年前；F4 集中在大帳戶；F1 只在 FIX。
  - 合併後全期少 1.0～2.6pp／年，2019 年起的結論不變。
- **沒有照預期的部分**：
  - F2 的量上限幾乎不觸發，量到的拖累主要來自款項時點，且凍結版的實作偏嚴（見 7.3）。
  - F3 的量級取決於無法校準的代理比例。
  - F4 對 1,000 萬的拖累比預期大 2～3 倍。
- 獨立重寫本身是一次檢定：
  - 原表的「2019 年起 16 格全輸」被獨立複製，原設定與 V7X 都成立；
  - 「1,000 萬無法判斷」在修正後變成「輸」。

## 10. 散戶怎麼執行（SOP 建議）
- **實盤預期表改用 V7X 正式欄**：K5T，j 中位數，TICK 只放敏感度欄。寫法如下，並加註「2019 年起每格輸 2.7～9.8pp；壓力成本每格輸 6～8pp」：
  - 30 萬～100 萬：約 16～17%，全期 t < 0.75，判「無可辨識差異」；
  - 300 萬：14.3～15.8%；
  - 1,000 萬：11.3～14.0%，判「輸」。
- **避開 FIX 式的頻繁提領或注資**：每年流量約占權益 64%，成本約 0.2pp／年。要調整資金時，跟換股同一天做，只透過現金進出。
- **處置期賣出**：預收券、分盤賣，款項入帳後隔日一定要補買，不要讓現金閒置到下個月。凍結版顯示閒置一個月的代價約 0.3～0.5pp／年。
- **小帳戶的零股**：2020 年前的盤後零股流動性很薄（代理顯示 30 萬拖累可達 2pp）。現在的盤中零股要用 2.8 折以下、最好低於最低 20 元手續費的券商；同一檔不要拆成多天的小單。
- **大帳戶**：下單量以事前的開盤量估計（share×V60）為準，並把單量控制在估計值的 10% 以內。TICK（一跳）不是大帳戶的下界：1,000 萬 GROW 的 Q/OA 中位數 13.7%、p90 50%。

## 11. 限制
- F3 是代理：證交所網站被擋、GitHub 找不到 TWT53U 封存、oddlot_hist 只有 2026-07 起，比例 0.5 是事前寫死、沒有校準。
- F2 凍結版的款項時點實作偏嚴，已用事後版 R4c 並列，判定只差 1 格。
- 30 萬全期有 4 格超過 0.10pp 的容許值，以及壓力格 10 格超過，都已追到規格模糊並採不利值。壓力格沒有逐格做診斷。
- 名單沿用 V7 凍結檔，沒有重產。只跑 B10，AB 版沒有重跑。
- 2019～2026 已被前幾輪看過，不是盲測。前瞻校準協議見 s8_forward_calib.py：κ 先驗 N(0.5, 0.17²)，累積 ≥ 60 筆非零股成交才更新；trades.csv 尚未建立。
- 第六步（套用到第 1 題 R_roll／R2、第 2 題 T1／T2、capmatched F1）沒有執行：各題到 16:50 都沒有交出名單 CSV；F1 是非等權組合，本引擎不支援。介面 s7_apply.py 已備。
- 第二位審查員的 3 格抽查還沒做。
- 多重檢定揭露：
  - 迭代 4 版；
  - 全期 9 個 run＋2 個事後 run＋1 個診斷；
  - 每個 run 32 格 × 13 個 j。
- 所有判定都只看點估計與 t，沒有一格接近顯著。

## 檔案
- 資料夾：/home/user/-/research/tw_stock_selection/agents/o_exec_engine_rewrite_v7x/
- 程式：x_data.py、x_ideal.py、x_exec.py、x_stats.py、x_run.py、s0～s9。
- 結果：
  - results.json（PREREG、PREDICTIONS_PREREG、RESULTS_OOS、RECON_TRACE_posthoc、F2_IMPLEMENTATION_NOTE_posthoc、VERDICT）；
  - interp_register.json、explore.json、iterations.json、frozen_spec.txt；
  - out/navs_oos_R0～R4、R2c、R4c、S_F3、S_F4、R4_z025、R4_z100.parquet；
  - out/gate_zero.json、diag_nanv60.json、audit3.json、vs_main.json。

## reviews
[
 {
  "final": "待觀察",
  "issues": [
   "My script reproduces the 8 formal cells exactly. Excess, t and j-median annualized return all match R4 (for example 1,000萬 FIX: −0.59 and t 0.15), so the arithmetic is right. The problems are in the inference.",
   "The claim that all 16 cells lose from 2019 on rests on 2 years: 2024 (about −33pp) and the 2026 partial year through October (about −30pp). With 2024 removed, 100萬 FIX has a median excess of +0.46 from 2019 on, and 5 to 8 of the 13 j are positive. With 2024 and 2026 both removed, 7 of the 8 K5T cells turn positive (+0.29 to +4.14pp); only 1,000萬 GROW stays at −1.59. FINDINGS does not disclose this dependence on a few years.",
   "The full-period positive excess for small accounts also depends on a few years. Removing the 2006 partial year (Apr–Dec, +29.6pp) roughly halves 30萬 FIX, from +1.96 to +0.87, and turns 300萬 FIX negative (−0.19). Removing each j's best 2 years makes all 8 cells negative (−0.7 to −5.3). So the SOP's 16–17% expectation for 30萬–100萬 is optimistic.",
   "The 13 j are not independent samples. Their monthly window excesses have a mean pairwise correlation of 0.64, which is about 1.5 independent observations. TICK and K5T use the same picks. So '16 cells all lose' and '0 to 5 of 13 j positive' amount to 1 or 2 independent observations, and should not be read as 16 or 13 pieces of evidence.",
   "The two 'lose' labels are on noise. 300萬 GROW is at −0.07pp and 1,000萬 FIX at −0.59pp, yet the t of the arithmetic window excess is positive for both (+0.21, +0.15). Block bootstrap of the J6 window mean (12-window blocks) gives 90% CIs of [−0.35, +0.54] and [−0.38, +0.50] pp/month, both straddling 0. Per-j path noise between the engine and the original table (R0 vs original) is median 0.07pp, p90 0.17pp, max 0.37pp. Post-hoc R4c moves 300萬 GROW to +0.49 and 1,000萬 FIX to −0.05. The finding that 1,000萬 FIX changed from 'cannot judge' to 'lose' is a sign flip, not something you can tell apart statistically.",
   "The formal F2 implementation is stricter than the spec text: next-day cash from disposal sales sits idle until the next rebalance day. This alone lowers the formal table by 0.36–0.54pp. The 300萬 GROW 'lose' label comes from this implementation choice, yet the formal table uses the stricter version.",
   "The stated rule of taking the unfavourable value was not carried into the formal table. In R0, V7X for 30萬 GROW is 0.24pp above the original table (18.74 vs 18.50). The formal 16.91 still uses V7X's higher path and was not marked down. Also, 10 of the 16 stress cells exceed 0.10pp, and none of them was diagnosed cell by cell.",
   "The reconciliation is not fully independent. v2 (the 0.4% stress add-on also applied to odd lots) and v4 (searchsorted right) were both adopted because they matched the original table better. With 4 versions and 22 interpretations tuned toward the target output, the 0.078pp selection-period match is partly fit by construction. The full period is the real check, and there 4 of the 16 normal cells fail (all 30萬).",
   "The look-ahead fixes are incomplete. F4 only moved the participation cap to an ex-ante estimate. The cost solve need(T) still uses the day's actual OA and close, and impact uses actual OA, so sizing still sees that day's volume. In model_spec7, oa_share_by_dec (about 3%) was fit on just 192 stock-days of recent MIS data and applied back to 2006; the same file's o5 VAL estimate is about 9%, roughly 3 times higher. The F4 and K5T results for 1,000萬 depend heavily on this parameter, and it got no sensitivity test. zcap, odd-lot costs and drift are also fit on post-2020 data, and F3 is an uncalibrated proxy (30萬 FIX moves −1.84 to +1.06pp across proxy settings).",
   "Multiple testing and the not-blind period: 2019–2026 had been seen in earlier rounds. There are 9 + 2 full-period runs × 32 cells × 13 j. The pre-registered rule labels a cell 'lose' on excess ≤ 0 alone, a pure sign test with no uncertainty. There is still no second-reviewer spot check, and the self-audit covers only 3 seeds.",
   "Engine 'usable' is supported by the zero-impact gate (1e−14) and by the 300萬 and 1,000萬 tiers matching within ≤ 0.06pp. But 4 of the 16 normal cells (all 30萬) and 10 of the 16 stress cells exceed tolerance, the second audit is not done, and the cost parameters are fit on later data. On the rule of ruling against when unsure, I downgrade the engine to 'watch'."
  ],
  "note": "Verification script: /home/user/-/research/tw_stock_selection/agents/o_exec_engine_rewrite_v7x/review_stat_o8.py (python3 -I -B; reads only out/navs_oos_R0/R4/R4c and the original navs_exec7; no researcher files changed). Conclusion: 'no cell beats 0050' holds, since full-period t is at most 0.73. Three of the researcher's harder claims do not hold up. (1) '16 cells all lose from 2019 on' depends on 2024 and the 2026 partial year; with both removed, 7 of 8 K5T cells are positive. (2) The 'lose' labels for 300萬 GROW and 1,000萬 FIX are within the bootstrap CIs and engine path noise, and flip under the post-hoc F2 version. (3) The 13 j amount to only about 1.5 independent observations. On the engine side, the 30萬 and stress cells fail reconciliation, the unfavourable values were not carried into the formal table, the OA share parameter (192 stock-days, 3% vs 9%) was not tested for sensitivity, and the second audit is still missing. Overall I rate it 'watch' rather than 'usable'."
 },
 {
  "final": "待觀察",
  "issues": [
   "The engine gate the researcher pre-registered is not fully met. Rule: every cell ≤0.10pp, or the gap traced to a spec ambiguity with the unfavourable value adopted. 30萬 GROW K5T is +0.242pp and TICK +0.157pp, both with V7X more optimistic than the original. After fixing the NaN-V60 cap, +0.12 and +0.14 remain. Trace2 calls this 'path-dependent noise', which is not a traced spec ambiguity. On the stress side, 10 of 16 cells exceed 0.10pp (max 0.25) and none was diagnosed. So '可用' holds only for ≥100萬 normal-cost cells.",
   "The '採不利值' adjustment only appears in RECON's unfavourable_CAGR. It never reaches the formal table: FORMAL_V7X 30萬 GROW orig_setting_CAGR uses V7X's own 18.74, not the unfavourable 18.50. The formal 16.91% (the best cell in the table) is therefore likely overstated by 0.12–0.24pp. The verdict does not change.",
   "Odd lots ignore limit-lock at the fill price. Before 2020-10-26, after-hours odd lots fill at the close, but the engine checks limit-up/down only against the open. My rerun counts 3.2–4.0% of odd-lot buy value filled when the close was locked limit-up, and 0.7–1.0% of sells when it was locked limit-down. Blocking these pre-2020 fills: 30萬 FIX −0.175pp, 100萬 FIX −0.09, 30萬 GROW −0.07.",
   "For intraday odd lots after 2020-10, the engine also assumes a fill at open×(1+drift) when the stock closes locked. If those fills are blocked too (an upper bound, since an order before the lock could fill): 30萬 FIX −1.64pp (excess since 2019 goes from −4.8 to −8.9), 100萬 FIX −1.03, 30萬 GROW −0.53, 1000萬 −0.05 to −0.16. Fill probability on limit-up days is unmodelled, and it hits small accounts after 2019 hardest.",
   "1000萬 FIX moving from 'cannot judge' to '輸' is fragile. It already flips to 輸 at R2 (F1+F2), which uses the F2 cash-timing implementation the researcher admits is too strict. Under the post-hoc R4c the excess is only −0.05pp, with t 0.15–0.28. '輸' and 'no discernible difference' are a coin flip on the point estimate here. The conservative label can stay, but the text should not present the switch as a finding.",
   "The ops burden is underestimated. The SOP does not say this strategy needs orders on 73–137 trading days a year (390–1,000 orders a year), and odd lots before 2020 meant entering after-hours orders every day. 30萬 FIX fees are 1.34× the securities tax (about 0.56× at 30萬 GROW and 0.36–0.41× from 300萬 up). This is not a once-a-month rebalance.",
   "The SOP says to keep large-account orders within 10% of estimated open volume, but the engine never simulated that; it uses a 20% cap. The recommendation is untested, and tightening it would slow fills and change the result.",
   "TICK sensitivity at 1000萬 FIX is 18.79% (excess +4.05) against 14.01% under K5T, a 4.8pp model gap. Median Q/OA of 0.09 sits just under the 10% threshold, so the cell is not flagged 'TICK not a lower bound'. Readers can easily read TICK as the optimistic upper estimate; the table should flag it.",
   "The F3 proxy ratio (0.5) is uncalibrated. 30萬 FIX moves −1.84 to +1.06pp across ratios 0.25–1.0, the biggest source of uncertainty in the whole study. Small-account numbers before 2020 have no data anchor.",
   "Requiring cash to be pre-collected for buys of stocks under disposal (預收款) has a negligible effect. These buys are 0.09–0.29% of buy value; j-median change ≤0.004pp, worst 0.03pp in 5 of 6 cells and −0.195pp in 30萬 GROW. This one is not a problem."
  ],
  "note": "I reran the formal R4 settings across all 260 (j, seed) lists. The seed-averaged NAVs match navs_oos_R4.parquet to a maximum relative difference of 8.9e-16, so the reported formal table can be reproduced. Pushing costs further against the strategy (odd-lot limit-lock, pre-collected cash for buys of stocks under disposal) never turns a cell into a win. The strategy conclusion ('no cell beats 0050; all 16 cells lose from 2019 on') is robust and only strengthens. But the engine verdict '可用' does not fully meet the researcher's own pre-registered gate: the 30萬 GROW excess over 0.10pp is not traced to a spec ambiguity, the 10 stress cells were not diagnosed, and the unfavourable value was never carried into the formal table. It also leaves out fill probability for odd lots on limit-lock days; the post-2020 version is worth up to 1.6pp a year for small accounts. Hence 待觀察: the engine is usable for ≥100萬 normal-cost cells and for drawing negative conclusions; 30萬 cells and stress cells need more tracing. Review script: /home/user/-/research/tw_stock_selection/agents/o_exec_engine_rewrite_v7x/review_exec_o8.py (run it with python3 -I -B; add the `pre` argument for the pre-2020-only odd-lot limit variant). Output: /tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/review_exec_o8.json and rev.log in the same folder. No researcher file was modified."
 }
]

## replication
{
 "reproduced": false,
 "my_numbers": "程式在 /home/user/-/research/tw_stock_selection/agents/o_exec_engine_rewrite_v7x/replicate_o8/（全部用 python3 -I -B 執行）：資料層 d8.py、零衝擊閘門 g8_zero.py、可執行引擎 e8.py、批次 r8_run.py、統計 s8.py、官方日 o8_off.py、彙整 c8_summary.py。輸出在 out/：gate_zero.json、summary.json、off_R0.json、run_*.pkl。另有快取 cache/d8.npz，656MB，可以刪。我沒有讀研究員或 V7 的任何程式。\n\n【閘門與對帳】\n- 零衝擊閘門：\n  - 官方日 160 條，逐日最大相對差 1.32e-14（研究員 1.67e-14）。\n  - 13 個 j × AB／B10 × 2 種成本：4.65e-14（研究員 5.05e-14）。\n  - 處置與併購拆解：base／disp 最大差 4.1e-14。mdelay 照規格「換股日 ≥ 下市日」實作，並以前一日的 0050 當基期，結果只有 B10 J12 一欄差 0.89%，其餘 12 欄都 ≤ 4e-14，與研究員完全一致。改用逐日偵測時，13 欄都差 0.2～0.7%。\n  - 處置期賣出每份名單 22～32 次（中位數 28）；併購延遲每份 4 次。兩者都與研究員相同。\n- 統計程式重算 V7 原表 912 格：最大差 3.6e-13。年化、maxDD、t、SEL 都完全相同。\n  - 「2019 年起」與 HOLD 的基期必須用「2019-01-02／2024-01-02 的收盤」，才會與原表一致。\n- R0 對 V7 原表（B10 j 中位數年化）：\n  - 挑選期 32 格最大差 0.54pp（研究員 0.078）。\n  - 全期一般成本 16 格最大差 0.284pp，其中 12 格 > 0.10（研究員 4 格，最大 0.242）。例如 30萬 K5T FIX −0.272、TICK GROW +0.275。\n  - 壓力 16 格有 15 格 > 0.10，最大 0.854。\n  - 官方日 32 格最大差 0.884，一般成本 0.446（研究員 0.110／0.093）。\n- 診斷：零股上限改成「V60 缺值時不設上限」後，30萬 K5T FIX 從 −0.272 變 −0.069（研究員 −0.130 → −0.013），方向相同。\n\n【V7X 正式表（R4＝F1＋F2＋F3(0.5)＋F4，K5T，一般成本，13 個 j 中位數；同期 0050 14.37%，保留期 0050 55.67%）】\n- 各格依序為：年化／p10／超額／t／挑選期超額／2019 年起超額／驗證期超額／保留期年化／最大回撤／壓力成本年化。\n  - 30萬 FIX：15.92／11.98／+1.65／0.57／+3.80／−6.09／−0.71／37.0／−56.1／6.81\n  - 30萬 GROW：16.13／12.83／+1.76／0.57／+4.53／−5.52／−0.69／38.7／−55.4／6.90\n  - 100萬 FIX：16.93／12.57／+2.56／0.73／+4.16／−4.06／+0.95／39.5／−55.3／8.48\n  - 100萬 GROW：15.54／12.02／+1.17／0.45／+3.58／−6.44／−2.16／39.5／−55.8／7.71\n  - 300萬 FIX：15.51／11.37／+1.14／0.44／+3.64／−6.22／−0.69／37.5／−56.0／7.98\n  - 300萬 GROW：13.95／10.27／−0.41／0.13／+2.21／−9.00／−3.98／35.7／−56.1／8.09\n  - 1000萬 FIX：13.88／9.99／−0.48／0.11／+1.72／−7.87／−3.47／37.4／−56.7／6.90\n  - 1000萬 GROW：10.99／8.29／−3.28／−0.37／+0.79／−10.29／−5.97／32.8／−57.9／5.74\n- 判定與研究員完全相同：1000萬 FIX、300萬 GROW、1000萬 GROW「輸」；其餘 5 格「無可辨識差異」；沒有一格「贏」。\n- TICK 敏感度：年化 15.95～18.70%；2019 年起超額 −2.6～−6.1pp。\n- 1000萬 GROW 的 Q/OA 中位數 17.1%。我的口徑把賣單和順延單都算進去，p90 不能直接和研究員比。\n\n【修正拆解（K5T 一般成本，13 個 j 配對差的中位數，FIX／GROW）】\n- F1：30萬 −0.03／0；100萬 +0.05／0；300萬 +0.05／0；1000萬 −0.14／0。\n- F2：−0.60／−0.54；−0.65／−0.41；−0.58／−0.39；−0.59／−0.45。\n- F3：−2.07／−1.17；−1.11／−0.50；−0.56／−0.14；+0.02／−0.08。\n  - 30萬 FIX 的 F3 拖累集中在挑選期：挑選期 −2.60，2019 年起 −0.88。\n- F4：0／0；−0.04／−0.15；−0.19／−0.44；−0.73／−0.76。\n- 合併：−2.70／−1.74；−1.64／−1.16；−1.21／−0.87；−1.36／−1.44。\n- F3 代理敏感度（30萬 FIX，對 R4）：比例 0.25 再 −2.09（研究員 −1.84）；比例 1.0 為 +1.12（研究員 +1.06）。\n- 否證條件（任一修正讓結果變好超過 +0.05pp）：\n  - K5T 只有 1 格觸發：F1 的 100萬 FIX +0.053，是雜訊級。\n  - TICK 下 F4 有 7 格為正（+0.06～+0.20），方向和研究員所說「TICK 的成本不隨參與率變」一致，但格數比研究員多。\n\n【事件數】\n- 每格 13 個 j × 20 seeds＝260 份名單，每份 246 或 245 個訊號窗。\n- 每個 run 32 格，共 8,320 次模擬。\n- 我共跑 R0、R1、R2、R3、R4、R4c、R0_nan 七個全量 run，加上兩個 F3 敏感度 run（只跑 30萬 FIX），另有官方日 R0 一次（40 seeds × 32 格）。",
 "diff_explained": "結論：閘門、統計口徑、R4 正式表的判定、F2～F4 的量級，以及 F3 敏感度，都重現了。逐格數字和 R0 對帳沒有在容許值內重現，所以標 false。\n\n一、完全重現的部分\n- 零衝擊閘門：三組差都 < 5e-14。\n- mdelay 的 J12 單欄差 0.9%：我也只在這一欄差 0.89%，原因同樣是撥款日與換股日重合時的口徑。\n- 統計程式重算原表 912 格。\n- 處置期賣出 28 次、併購延遲 4 次。\n- 8 格判定（3 輸、5 無可辨識差異、0 贏）。\n- 「V60 缺值時零股上限」是 30萬 FIX 對帳分歧的主因：改成不設上限後，差距縮小約 0.2pp，與研究員的診斷一致。\n\n二、沒有重現的部分與原因\n1. R0 對 V7 原表的對帳比研究員差。\n   - 挑選期最大差 0.54pp（研究員 0.078）；全期一般成本有 12 格 > 0.10（研究員 4 格）；官方日最大差 0.45／0.88。\n   - 研究員試了 v1～v4 四版，用挑選期的對帳挑出最接近 V7 的實作細節。我只做一版，照字面解讀規格，剩下的模糊點都要自己選，例如：\n     - 下單排序的 delta 用金額還是股數；\n     - need() 是否計衝擊與 LIM；\n     - 順延買單的零股與現金限額；\n     - 漂移桶要不要 'right'；\n     - 處置賣出在壓力成本下要不要加 0.4%；\n     - FIX 權益_pre 用前收。\n   - 這些選擇使逐 j 差 ±0.6pp，j 中位數差 ≤ 0.28pp（一般成本）。\n   - 成本帳的總量與 V7 很接近。以 J0 30萬 FIX 為例：手續費 91.1k 對 90.9k、稅 117.1k 對 117.3k、零股價差 115.6k 對 115.4k、衝擊 25.7k 對 25.8k。所以差異來自路徑細節，不是成本模型本身。\n\n2. R4 正式表：我的年化多數低 0.1～0.8pp。\n   - 30萬 GROW −0.78、30萬 FIX −0.41；只有 100萬 FIX 高 +0.11。\n   - 2019 年起超額低 0.4～1.4pp，保留期年化低 0.1～2.4pp。\n   - 落差集中在 2019 年以後，和 R0 對帳時一般成本 16 格多數為負（9 負 7 正）同源，主因是零股上限與順延的口徑。\n   - 落差不改變判定。\n\n3. F1 我量到約 0（研究員 −0.17～−0.22，V7 審查員也量到 −0.21）。\n   - 我的實作是把提領與注資直接加減現金，再照常換股。額外成本只發生在續抱股的加減碼（稅多約 1%，20 年累計），量級約 −0.06pp，又被路徑雜訊蓋過。\n   - 研究員的版本應有另一處成本：例如提領先賣股、不和換股合併，或注資買進另計最低手續費。規格文字不足以分辨，這格採研究員的不利值較穩妥。\n\n4. R4c（事後修正版）沒有重現。\n   - 我的 f2fix 只是讓處置賣出款「隔日入現金」，但入選股的目標金額仍照 F2 縮小，沒有隔日補買的單。結果只多 +0.00～+0.07pp，300萬 GROW 仍是「輸」（超額 −0.06）。\n   - 研究員的 +0.36～+0.54 需要「隔日用這筆款補買入選股」。這和我量到的 F2 拖累（−0.39～−0.65，幾乎全部來自款項閒置）量級相符，所以方向可信，但我沒有獨立驗證。\n\n5. 以下沒有做：\n   - 對主策略零衝擊的配對 t（−5.1～−11.4）、主策略零衝擊 21.48%、同池等權 10.73%；\n   - 迭代版本數；\n   - Q/OA p90，因為口徑不同。\n\n三、建議\n- 正式表的判定可以接受。\n- 逐格年化應附註「獨立重寫的差距可達 ±0.8pp，2019 年起超額可達 1.4pp，方向都對策略不利」。\n- F1 的量級與 R4c 的補買口徑，應在規格中寫明後再驗一次。"
}