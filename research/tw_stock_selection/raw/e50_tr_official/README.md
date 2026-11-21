# 0050 含息序列（自建，候選）

- 內容：`etf_0050_tr.csv`，欄位 `date,adj`（交易日、含息還原收盤）。2015-01-05～2026-11-20，2890 個交易日；只列 0050 有成交的日子（2025-06-11～17 分割換發停牌不列，引擎讀檔時沿用前一天）。
- 算法：含息還原收盤＝官方日 K 原始收盤 × 事件日累乘倍數。除息日倍數＝前收 ÷（前收 − 息值）；分割日倍數＝前收 ÷ 分割後參考價，取最接近的整數。等於股利在除息日以參考價再投入。
- 事件：`etf_0050_events.csv`（23 個：22 次除息＋2025-06-18 一拆四），每個事件的前收都和日 K 前一成交日收盤相符。除息事件是 TWT49U 封存（`raw/twt49u`）與官方日 K 倉庫 `data/universe/exright` 的聯集（兩邊 22 筆完全相同）；分割來自倉庫的 `data/universe/etfsplit`。
- 來源：日 K 以研究中心封存 `raw/jb7k6295r7-bot_tw-stock-data`（git HEAD `107f90739a3d`）為主，它沒有的日子用官方日 K 倉庫 tsd（git HEAD `df9a1360a2f8`）；兩邊重疊的日子收盤完全相同。不連網、不用 jonatw、不用倉庫現成的還原因子。
- 產生程式：`agents/eng_v2_candidate_and_site/build_raw_archives.py`（`python3 -I -B build_raw_archives.py [輸出根目錄] [tsd 路徑]`；加 `--check` 會和稽核對照檔 `agents/r_fix_e50_reduce/data/e50_tr_mine.csv` 逐筆比對）。重產兩次結果逐位元相同。`build_info.json` 記錄來源與筆數。
- 驗證：與稽核獨立重建的序列 2,890 天逐筆相同（差 0）；稽核另和工程第一版互相核對（`agents/r_fix_e50_reduce/FINDINGS.md`）。
- 為什麼要換：現用的 `/home/user/jonatw/finlab-txd/data/raw/etf_0050.csv` 在 2015-07～2025-07-18 與本檔逐日相同，但 2025-07-21、2026-07-21 兩次除息各少算約 0.02%，2026-08-12～10-14 有 20 天與官方收盤差 1～4 個跳動點，累積多出 +0.58%（根因查不到）。
- 誰在讀：`agents/o_ext_canslim_xs/cslib.py` 的 `E50_SOURCE`（**預設 None＝不讀，仍用現用檔，舊結果逐位相同**）；候選正式管線 `agents/official_nav/run_official_v2.py` 設成 `cslib.E50_OFFICIAL_CSV` 才讀。**產品擁有者尚未採用，正式數字還沒有換。**
- 限制：只從 2015-01-05 起（官方日 K 的起點）；2008～2014 沒有官方日 K 與事件檔，連續帳戶（S75 等）那一段仍是舊來源。回測引擎的最後一天仍是 `cslib.END＝2026-10-22`（加權指數檔只到那天），本檔雖然到 2026-11-20，引擎只用到 END。
- 更新：官方日 K 倉庫更新後重跑產生程式（會整檔重產並重寫 `SHA256SUMS`）；重跑前後 `etf_0050_tr.csv` 舊日期的值不應改變。
