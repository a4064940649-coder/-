# bugfix_log：o_forward_infra_live（PREREG_FORWARD 凍結後的修改）

規則（PREREG_FORWARD.bugfix_rule）：凍結後只准修 bug，不能改選股規則、線的定義、成本參數或 G8。每次修改都要記錄：
- 修改前後的 sha256；
- diff 摘要；
- 重算前後 forward_returns 的差異。

機器可讀的版本寫在 results.json 的 `bugfix_log`。make_forward.py 比對 sha 時，會用「PREREG 登記值＋bugfix_log 的新 sha」，結果寫在 runs.jsonl 的 sha_matches_prereg。

這次的修改依據是 original_o9/DIRECTOR_REPORT.md 第 4 節第 3 點 (a)～(f)，以及第 3.3 節審查推翻的第 1、2、4 項。期限是 2026-11-10 收盤前。

---

## 2026-11-03（台北）BF-1～BF-4

### sha256

| 檔案 | 修改前（PREREG 登記） | 修改後 |
|---|---|---|
| make_forward.py | ffcaabfbe04b4af2d1056db4f3ec8ebc7416184dd68574c83dddd9c8a735e0cd | c53457abd91991ae5d28334d052d294c2789735f0663b455470be22191b63cd6 |
| fwd_data.py | 807a6f824da7c885b262d701091601919dbb575f92d10b2b41b036ac75f6e020 | 1d92540f13f9374d68e775740f27db160151b6009000d2cee88528e9b53a1b5f |
| daily_step0.sh | 92f30c9657ad445507ac6ff41b6ced0f3f3fa48ac728630140b2029382c2a59f | bcd8b7b2c6c4de07bc5c05a1a7265023439cd11d8b48a03f68b14e7d49e6fdef |
| fallback_manual.sh | 8e13565ef212058669fd1550791fc6d37e4eddc46b185ffdead05c3fd930edf3 | bf3d03ba3c4c921c38c647855cb75ec86f763a58330d8e5874df1a46a642d4dd |

- 沒有修改：xlive.py、disp_sat.py、main_picks_today.py、x_exec.py、frozen_spec.txt。這些檔案的 sha 仍和 PREREG 登記值一致。
- 新增兩支工具，不在 PREREG 登記，只讀資料：
  - a2_pit.py（sha256 522b3c76e5c773c8a809ed6853baf892021ad2a79b9ea058a47f4a7a4c035774）：用真實 git 快照重做 A2。
  - check_official.py（sha256 8ffa53f223e6ac53095c8b8d0953ac8d80367c184ca65b4dc0dac0c5d8ad65ce）：訊號日後人工核對；另提供 `--make-union`，給 fallback_manual.sh 建立「處置聯集」資料夾。
- 修改前的檔案可用 `git show HEAD:research/tw_stock_selection/agents/o_forward_infra_live/<檔名>` 取回。

### BF-1｜官方日線的營收截斷日（make_forward.py、fwd_data.py；對應 (a)）

- **問題**
  - rev_hist 的 `ann_date` 是上游「第一次抓到」的日期，不是 PIT 欄位，而且會被事後覆寫。
  - 2026-10-12 晚上公告、10-13 03:48 才抓到的 71 筆 9 月營收被記成 10-13。
  - 舊版官方日線只保留 `ann_eff ≤ 訊號日` 的營收，把這 71 筆剔除了。但成交日開盤前就看得到這些營收，main_picks_today 也會用到。
  - 結果是在真實快照 50bda15 上，前 10 名只有 9/10 重疊。
- **為什麼不改用 m−2 補值**：main_picks_today 遇到 m−1 月營收缺值時，是把該檔剔除，不是改用 m−2。如果官方日線照 J 線的做法用 m−2 補值，反而會和 main_picks_today 不一致。所以採用 (a) 的另一個選項：「只用成交日開盤前可得的資料」。
- **修正後的規則**
  - 官方日線的營收截斷日 `rev_cutoff` 規則：
    - 凍結時鐘早於成交日 09:00 → `rev_cutoff`＝成交日；
    - 凍結時鐘晚於成交日 09:00 → `rev_cutoff`＝訊號日（保守，同時 ts_ok＝false）。
  - 價格、處置等其他資料一樣截到訊號日。
  - 重算比對（--recheck）沿用快照 meta 登記的 `rev_cutoff`。
  - meta 新增三個欄位：`rev_cutoff`、`rev_loaded`、`rev_cutoff_rule`。
  - J 線（J0～J12）規則完全不變。
- **正式模式為什麼可行**：快照本身就是凍結當下的資料，不可能含有「凍結之後才抓到」的營收。所以在正式模式下，這個規則等於「凍結時資料夾裡的營收」。
- **重播模式的限制**：重播時如果用事後的資料，`ann_date`＝成交日的列可能是開盤後才抓到的，這是只存在於重播的前視。用真實 git 快照時沒有這個問題。
- **這不是規格變更**：frozen_spec.txt 開頭就寫「選股規格不改：以 main_picks_today.py --ex-fin 為準」，BF-1 只是讓程式回到這個口徑。

### BF-2｜資料過期偵測接上正式排程路徑（make_forward.py、fwd_data.py、daily_step0.sh；對應 (b)）

- **問題**：daily_step0.sh 不帶 `--asof`，程式就用 `want = hcal[-1]`，DATA_STALE 分支永遠進不去。上游延遲時，程式回傳 0，而且默默不寫名單。
- **修正**
  - 不帶 `--asof` 而且不是重播時，`fwd_data.expected_last_day(台北時鐘)` 會推算應有的最後交易日：
    - 當天是交易日而且已過 18:00 → 當天；
    - 否則 → 前一個交易日。
    - 交易日依 history 日曆加 tw_holidays.json 判斷。
  - 18:00 的依據：上游「每日資料」大約 15:25、17:40 提交。
  - 下列任一情況，狀態設為 DATA_STALE、回傳碼 2，不寫任何名單，並在 stderr 用中文提示改跑 fallback_manual.sh：
    - history 最後一天早於應有日；
    - 處置來源 warnings_hist 最後一天早於應有日，而且 updated.json 的日期也早於應有日。
  - asof 改成「應有日」，不再是資料最後一天。例如 15:30 執行時，就算上游已經有當天資料，也要等 18:00 後的那次執行才凍結當天名單。
  - 新增兩個參數：`--ready-hour`；`--expect-now`（只供測試，不影響凍結時戳）。
  - daily_step0.sh 原本最後一行是 `tail`，回傳碼永遠是 0。現在改為傳出 make_forward 的回傳碼。
- **測試**
  - `--expect-now 2026-11-11T22:30` 而資料只到 11-02：回傳碼 2，狀態 DATA_STALE，沒有寫入名單。
  - 11-03 06:19 實際時鐘：應有日＝11-02，回傳碼 0。
  - daily_step0.sh 透過 `FWD_EXTRA_ARGS`（只供測試）跑過以上兩種情況，回傳碼分別為 2 和 0。

### BF-3｜磁碟檢查（daily_step0.sh；對應 (d)）

- 執行前先檢查剩餘空間：
  - < 1GB：先清本題可重建的快取（work/asof_*、work/check_official、work/mp_union）；仍不足就中止，回傳碼 3，提示先跑 fallback_manual.sh；
  - < 2GB：只警告。
- 不會刪除其他人的資料。
- 記錄檔名改成精確到秒，避免同一分鐘內的多次執行混在一起。
- 測試：`FWD_MIN_KB` 設得很大時，回傳碼 3。
- 2026-11-03 06:20 時，磁碟剩約 2.4GB（o9 審查時只剩 40～147MB，後來已有人清出空間）。work/ 只有 10MB、replay/ 只有 3.4MB，不需要清。
- replicate_o9/work（297MB）是複製員的產物，不在可清除範圍，沒有動。

### BF-4｜備援也排除處置股（fallback_manual.sh）

- 原本的備援只跑 main_picks_today 讀原始資料，處置只看 disposal_5y，而 disposal_5y 從 2026-10-05 起停更。
- 現在另外產生 `main_picks_union_<時戳>.txt`：同一份資料，但 extras/disposal_5y 換成 disposal_5y ∪ warnings_hist（處置），和產生器同口徑。
- 兩份都寫入 SHA256SUMS；下單以聯集版為準。
- 測試：輸出導到 scratch 跑一次，聯集版前 10 名就是更正後的 10-12 名單。

### 重算前後 forward_returns 的差異

- **正式資料夾**：live_tracking/main_forward/forward_returns.csv 修改前後都沒有資料列。第一個錨定月 2026-11 的訊號日 11-10 還沒到，所以差異為 0。
- **重播**：replay/seq 只是驗收產物，沒有重產。新規則下，重播的官方 2026-10 名單尾端由「6691、6196」變成「2395、6691」。對應的重播窗超額只是驗收描述，不是證據。
- **A4 冪等**：預設路徑（J 線與主視圖）的截斷資料夾沒變。asof 2026-11-02 的 MANIFEST sha 修改前後都是 20bff401a7a8…。

### 驗收 A2 重做（對應 (e)；out/accept_a2_pit.json）

用 raw/screener 稀疏 clone 的完整歷史，以 git show 取出真實快照。共用 clone 是 shallow，最早只有 10-26 的 commit。

| 訊號日 | 上游 commit（台北時間） | 產生器 vs MP --ex-fin（處置聯集） | 舊規則 vs MP（處置聯集） | 產生器 vs MP（原始資料） |
|---|---|---|---|---|
| 2026-10-12 | 83f11fc 10-12 17:40 | 10/10 同順序 | 10/10 | 8/10 |
| 2026-10-12 | 4531dc6 10-12 22:38 | 10/10 同順序 | 10/10 | 8/10 |
| 2026-10-12 | 4686735 10-13 03:18 | 10/10 同順序 | 10/10 | 8/10 |
| 2026-10-12 | 50bda15 10-13 03:48 | 10/10 同順序 | **9/10** | 8/10 |
| 2026-10-12 | 84c7943 10-13 08:47 | 10/10 同順序 | **9/10** | 8/10 |
| 2026-09-11 | b44e5a7 10-08 00:05（非 PIT） | 10/10 同順序 | 10/10 | 10/10 |

- 「原始資料」欄的 8/10，差的就是 2466、6213：main_picks_today 只讀已停更的 disposal_5y，漏排了這兩檔處置股。產生器是對的。
- 2026-09 錨定月沒有真實 PIT 快照：
  - 上游 repo 在 2026-09-29 才建立；
  - main_picks_today 需要的檔案要到 10-08 00:05 的 b44e5a7 才全部到齊；
  - 所以 09-10、09-11 兩晚無法重現，表中 09-11 那列只是事後回填資料上的一致性檢查。
  - 09-10 是 J0，沒有 main_picks_today 對照。

### 相關紀錄

- live_tracking/main_forward/CORRECTIONS.md C-1：2026-10-12 名單更正（刪 2466、6213；尾端是 3231、2395、6691）。
- BACKLOG.md 每日流程新增第 0 步（對應 (c)）。排程由使用者設定。
- live_tracking/main_forward/CHECK_1110_1111.md：11-10、11-11 兩晚人工核對清單（對應 (f)）。

---

## 2026-11-03（台北）BF-5｜月營收覆蓋率：官方補值＋覆蓋率閘門

> 編號說明：交辦時寫「BF-2」，但 BF-2 已用於「資料過期偵測」（見上），所以本次記為 BF-5。交辦的「回傳碼 3」也已被 BF-3（磁碟不足）使用，本次新回傳碼為 **4**。

### sha256

| 檔案 | 修改前（BF-1～BF-4 後） | 修改後 |
|---|---|---|
| make_forward.py | c53457abd91991ae5d28334d052d294c2789735f0663b455470be22191b63cd6 | 6d7e5d20102674b5935511e56337302680996a21868e727dcb13c8d8ca6e39b6 |
| fwd_data.py | 1d92540f13f9374d68e775740f27db160151b6009000d2cee88528e9b53a1b5f | 903591b9f8640f29c4609db42eff54c8d606c17a75253009a2d44128291f84b7 |
| daily_step0.sh | bcd8b7b2c6c4de07bc5c05a1a7265023439cd11d8b48a03f68b14e7d49e6fdef | fb9b442044be955915a98ae7b068c55d0944fc32503ecf3b7cba6b942aeafe46 |
| fallback_manual.sh | bf3d03ba3c4c921c38c647855cb75ec86f763a58330d8e5874df1a46a642d4dd | 9b47df7bcb59cfd2a947db0bc3ece1047154f81779ecbeb4a615eceea2ac4f30 |

- 新增 rev_official_fetch.py（不在 PREREG，sha256 0a0adc00db422bba28e277816b39a2f5de4da37bc8f8cd8ea17366efd7fd44f3）。
- 工具 check_official.py 改版（不在 PREREG）：8ffa53f2… → ae0597aeff628a4a2708727442dcef1bcc1e16bf6c7cb014af5baab1a477ec7f。
- results.json 的 `bugfix_log` 已加 BF-5 的新 sha，make_forward 比對 PREREG 時會用到。
- 沒有修改：xlive.py、disp_sat.py、main_picks_today.py、x_exec.py、frozen_spec.txt、a2_pit.py。

### 問題（量化見 out/accept_bf5_rev_coverage.json）

- **交辦前提要修正**：rev_5y 的 2026-09 確實只有 319 檔，但產生器和 main_picks_today 都會用 rev_hist 補缺月（rev_hist 9 月有 1,864 檔），所以「只看 rev_5y」的缺口不是實際缺口。另外，「m−1 缺就沿用 m−2」只適用於 J 線；官方日線照 main_picks_today 的規則，m−1 缺值的股票會被**剔除**。
- **2026-10-12 訊號日的實際缺口**：
  - 股池 258 檔（前 300 大、收盤 ≥ 10、52 週有值、非處置、非金融）。
  - 其中 14 檔上游從來沒有營收（1590、2637、3661、3673、4763、4927、4958、5871、6415、6451、6456、6525、6781、6933，幾乎都是 KY 公司）。回測用的是同一份上游資料，所以這 14 檔本來就不在策略股池，不算缺口，分母剩 244 檔。
  - 10-12 17:40、22:38 和 10-13 03:18 的快照：244 檔中缺 14 檔的 9 月營收（覆蓋率 94.3%）：1402、1519、1795、2105、2312、2332、2371、2404、2412、3033、3045、4904、9904、9914。
  - 10-13 03:48（50bda15）以後：0 檔缺，覆蓋率 100%。
- **風險**：照 CHECK_1110_1111.md，官方日是晚上 22:30 凍結。如果上游那時還沒抓齊 m−1 營收，官方日線會凍結一份「缺營收股票被剔除」的名單，而且快照不會覆寫。10-12 就是這樣：22:30 的名單尾端是 3231、6691、6196，不是 C-1 更正後的 3231、2395、6691。

### 修正

1. **rev_official_fetch.py（新增）**
   - 抓證交所 OpenAPI t187ap05_L 和櫃買 OpenAPI mopsfin_t187ap05_O。某個市場的 OpenAPI 失敗時，改抓公開資訊觀測站 t21sc03 的靜態彙總頁（上個月；國內＋外國公司）。
   - 每次執行開一個新目錄 `raw/official_rev/<台北 YYYYMMDDTHHMMSS+0800>/`：原始回應原樣存檔，另有 FETCH.json（URL、HTTP 狀態、開始／完成時間、sha256、錯誤）和 SHA256SUMS。
   - 每列的可得時點＝該請求的**完成時間**，不往回套成公告日。
   - 下載內容視為不可信資料：只用 json 和 html.parser 解析，逐欄驗證；讀取時 sha256 不符的整個目錄不用。
   - 程式放在 raw/ 之外，用 `python3 -I -B` 執行，並以參數傳路徑。
2. **fwd_data.py**
   - `Source(official=…)` 讀所有抓取目錄。`build_asof` 在「上游截斷後仍缺」的 (code, ym) 才用官方值補進 rev_hist：ann_date＝最早抓到的日期，數值＝期限前最後一次抓到的值。
   - 使用期限 t_off：
     - 官方日線（rev_cutoff＝成交日）：抓取完成須早於成交日 09:00；
     - J 線與主視圖：抓取日期 ≤ 截斷日；
     - 兩者都不晚於執行時鐘（重播＝sim-clock）。
   - 只補「上游在 ym 前 3 個月內有營收」的股票，不會把 KY 等上游從來沒有的公司帶進股池。
   - 補到列時 MANIFEST 的 key 加上 `official_fill`，並在 build 和 View 斷言 max_fetched_ts < t_off。沒補到任何列時，資料夾與 MANIFEST 和 BF-5 前完全相同（asof 2026-11-02 仍是 20bff401…）。
   - 新增 `rev_present`（覆蓋率用）和 `fill_dir_official`（fallback／check_official 用）。
3. **make_forward.py**
   - 新增參數 `--official-rev`、`--rev-cov-min`（預設 0.99）、`--no-defer-official`、`--rev-gate-replay`。
   - 閘門 `rev_gate`：m*＝法定期限日 ≤ asof 的最新營收月份。
     - J≥1，或當天是官方日：股池（扣掉上游沒有 m*−1、m*−2 營收的股票）中，有 m* 營收的比例 < 99% → 回傳碼 4。
     - J0（不是官方日）：只檢查「官方已公告」的部分。沒有官方資料時記為 J0_UNVERIFIED，不回傳 4。
   - 覆蓋率不足，而且時鐘還在官方日當天時，官方日線本次不凍結（`official_deferred`），留給成交日 09:00 前的下一次執行（06:49 排程）。成交日早上那次不論覆蓋率都會凍結。
   - 重播只記錄覆蓋率，不延後也不回傳 4，所以 a2_pit 等既有驗收不受影響。
   - 快照 meta 新增 `official_fill`、`rev_gate` 兩個欄位；runs.jsonl 新增 `rev_gate`、`official_deferred`、`official_rev`。
   - 選股規則、排名公式、J 線定義都沒有改：補值只填資料缺口，score_on／rank_j 沒有動。
4. **daily_step0.sh**
   - git fetch 之後，每月 9～20 日（台北）先跑 rev_official_fetch.py。抓不到不中止。`FWD_OFFICIAL_FETCH=1` 強制抓，`=0` 不抓。
   - make_forward 帶 `--official-rev raw/official_rev`。
   - 回傳碼 4 時，用中文提示改跑 fallback_manual.sh。
5. **fallback_manual.sh**
   - 第一步先跑 rev_official_fetch.py，輸出記錄寫入 fallback/official_rev_fetch_<時戳>.txt，並加進 SHA256SUMS。
   - 聯集版的 rev_hist 補上「現在之前抓到」的官方月營收。檔頭會寫抓取回傳碼和補值列數。
6. **check_official.py**
   - `--make-union D U [官方根目錄]`。
   - 快照 meta 有 official_fill 時，main_picks_today 對照也用同一個 t_off 補值。另新增 `--official-rev` 參數。

### 官方來源連線結果（照實記錄）

- 2026-11-03 14:33（台北）從本容器執行：openapi.twse.com.tw、www.tpex.org.tw、mopsov.twse.com.tw 都被代理以 CONNECT 403 拒絕（政策擋下）。另外試過 mops.twse.com.tw、www.twse.com.tw、data.gov.tw、web.archive.org，也都不通；只有 GitHub 可以連線。
- 失敗紀錄保留在 raw/official_rev/20261103T143314+0800/FETCH.json（以及 14:49、14:51 兩次測試）。
- 要讓補值實際生效，需要在環境的 Network access 設定把上述主機加進 Allowed domains，或者在能連官方網站的機器上跑 rev_official_fetch.py，再把抓取目錄整個複製進 raw/official_rev/。
- **MOPS HTML 解析器**沒有用真實回應實測過。**OpenAPI 解析**是依公開欄名（公司代號、資料年月、營業收入-當月營收、營業收入-去年同月增減(%)…）寫的，同樣沒有用真實回應實測。
- raw/ 在根 .gitignore 裡（`agents/*/raw/*/`），所以 raw/official_rev 不會進 git。若要保存抓取證據，請另外加例外規則。

### 2026-10-12 名單重算（對應交辦第 5 點）

- 用目前上游資料（aad3fc2），營收截到成交日 10-13：股池缺 9 月營收 0 檔，前 10 名＝C-1 紀錄（2305、2347、6834、2408、2383、3653、1312、3231、2395、6691）。
- 事後把所有 9 月營收（不論抓取時間）都算進去：名單也相同。
- 官方值是 11-03 才抓（而且沒有抓到），依時點規則本來就不能用於 10-12 訊號。
- 結論：**名單不變，CORRECTIONS.md 不新增更正**。

### 測試（scratch；輸出不寫入正式資料夾）

| 測試 | 結果 |
|---|---|
| 83f11fc（10-12 17:40 快照），模擬時鐘 10-12 22:30 | rev_gate LOW 94.3%（230/244）；`--rev-gate-replay` 時 rc 4，official_202610 延後、J0 照常凍結 |
| 同上＋測試夾具（上游 9 月營收排成 OpenAPI 格式、標記 22:00 抓到；**不是官方資料**，只測程式路徑） | 補 58 列（股池 14 檔），PASS，官方名單＝C-1 紀錄；check_official.py PASS（10/10 同順序） |
| 83f11fc，時鐘 10-13 08:00，夾具標記 10-13 09:30 抓到 | 抓取晚於時鐘不用；rc 4；成交日早上不延後，照常凍結 |
| 50bda15，時鐘 10-13 03:48 | PASS 100%，名單＝C-1 紀錄；check_official.py PASS |
| 夾具竄改 1 個位元組 | sha256 不符，整個目錄不用 |
| 抓取時間＝t_off／t_off＋1 秒 | 0 列／995 列（嚴格「早於」） |
| a2_pit 6 個案例用新程式重跑 | 產生器名單與 out/accept_a2_pit.json 完全相同；與 main_picks_today --ex-fin（處置聯集）6/6 同順序 |
| A1 斷言（loaded_max_dates、dir_check、rev_loaded ≤ rev_cutoff、官方 max_fetched_ts < t_off ≤ 開盤／凍結時鐘） | 全部成立 |
| asof 2026-11-02 截斷資料夾（開／不開 --official-rev） | MANIFEST sha 都是 20bff401…（A4 不變） |
| daily_step0.sh 實跑（`FWD_EXTRA_ARGS=--out <scratch>`） | rc 0（9 月營收 J13 覆蓋率 100%，256/256）；加 `--rev-cov-min 1.01` → rc 4，提示正確 |
| fallback_manual.sh（OUT 改到 scratch） | 官方抓取 rc 1 照常產生名單；聯集版 10-12 名單＝C-1 紀錄 |

### 重算前後 forward_returns 的差異

- live_tracking/main_forward/forward_returns.csv 修改前後都沒有資料列，差異為 0。
- 2026-10 的 SOP 名單重算後不變。
