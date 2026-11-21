# TWT49U 除權除息計算結果表（封存）

- 內容：證交所「除權除息計算結果表」（TWT49U）逐筆事件，欄位 `date,code,prev_close,ref_price,value,type`
  （除權息日、代號、除權息前收盤、除權息參考價、權值＋息值、種類：息／權／權息）。只有上市，沒有上櫃。
- 來源：公開鏡像 https://github.com/joeycchen72/taiwan-market-history 的 `data/twse-exright/`，
  commit `fb838a00719a31b5e7591b17070bd5a3b616c6c7`（2026-11-02 11:29:55 UTC）。
  - 從本環境連不到證交所網站（回「安全性考量」頁面），所以用這份第三方鏡像（2026-11-21 更新：環境網路放寬後，證交所 TWT49U 低頻率請求可以連到，但同一天第 4 次請求就被轉到安全頁，一天最多抓 1～3 次；封存來源不變，官方只當核對用）；可信度檢查見
    `agents/data_exright_fix/AUDIT.md` 第 1 節（前收和官方日 K 一致 99.99%；除權息日開盤落在參考價 ±10% 內 99.92%）。
- 抓取日：2026-11-02（o_0050_tilt_pit 下載）；2026-11-16 原樣複製到這裡封存（檔案逐位元相同，見 `SHA256SUMS`）。
- 涵蓋：2014-01-16～**2026-11-20**（含 ETF 等非四碼代號；引擎只用四碼 `[1-9]ddd`）。155 個月檔。
  - **2026-11 月檔來源不同（2026-11-21 補）**：由官方日 K 倉庫的事件檔 `data/universe/exright` 轉成同格式（程式 `agents/eng_audit_fixes_2026-11/convert_exright.py`；重疊期 2015-07-02～2026-10-29 共 11,781 筆逐欄相同、134 個月檔解壓後逐位元相同）。是「到產生當天為止」的部分月份，每次出名單前要重產覆蓋並重算 SHA256SUMS：先 `convert_exright.py verify <倉庫> twse-exright <輸出.json>`（rows_different 必須 0），再 `convert_exright.py build <倉庫> <暫存> 2026-10-30`，只複製 `.csv.gz`。11 月以後封存只有這一個來源；**2026-11-21 起有官方第二來源可核對**：證交所 TWT49U（`agents/eng_official_second_source/fetch_official.py exright`，原始回應存 `raw/official_snapshots/exright/`）。已核對 2026-11-01～11-20 共 55 筆、2026-07 整月 425 筆，封存、GitHub 倉庫、官方三方逐欄相同（`eng_official_second_source/crosscheck_2026-11-21.json`）。每日例行會比近 30 日並記在 `agents/audit_daily/source_check_log.csv`。
- 相關封存（2026-11-21 起正式採用）：`raw/twse_reduce/`（上市減資／面額變更恢復買賣參考價）、`raw/e50_tr_official/`（0050 含息序列，除息事件取自本封存與官方日 K 倉庫的聯集）。
- 誰在讀：
  - `agents/o_ext_canslim_xs/cslib.py` 的 `Panel(exright_fix=True)`（預設 False＝不讀，舊結果逐位相同）。
  - `agents/s18_live_list/live_surge/make_list.py --exright-fix`（預設開啟）。
  - `xq_scripts/site/live_tr.py`（2026-11-18 起；網站 live.json 的含息口徑）：只當備援——官方日 K 倉庫 `data/universe/exright` 沒有那一天的檔時才用這裡的事件。
- 每月補資料：2026-10-29 之後的事件還沒有。請每月從證交所 TWT49U（或同一鏡像）補 `twse-exright/YYYY/YYYY-MM.csv.gz`，
  同格式，補完重算 `SHA256SUMS`（`find twse-exright -name '*.csv.gz' | sort | xargs sha256sum > SHA256SUMS`），並更新本檔「涵蓋」。
  沒補到的月份：除權息日的漲跌價差仍是 0（當天報酬記 0、少算股利），名單產生器會在 meta 列出這段期間名單股的可疑格子。
