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
