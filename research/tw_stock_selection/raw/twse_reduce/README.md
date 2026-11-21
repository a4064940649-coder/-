# 上市減資／面額變更恢復買賣事件（封存；正式，2026-11-21 起）

- 內容：`twse_reduce_events.csv`，欄位 `date,code,prev_close,ref_price,reason`（恢復買賣日、代號、停止買賣前收盤、恢復買賣參考價、原因）。2015-08-24～2026-11-16，320 列：彌補虧損 132、退還股款 178、面額變更 10。只有上市，沒有上櫃。
- 來源：官方日 K 倉庫 tsd（git HEAD `df9a1360a2f8`）的 `data/universe/reduce`（減資恢復買賣）與 `data/universe/parvalue`（面額變更；原檔 reason 空白，這裡填「面額變更」）。原始 321 列，同一天同一代號重複的 1 列已去掉（留最後一筆）。
- 讀不到的事件檔：2015-01-23、2015-03-20、2015-04-10、2015-04-16 四天（本機沒有這幾個檔的內容），都在回測起點 2015-07-01 之前。
- 產生程式：`agents/eng_v2_candidate_and_site/build_raw_archives.py`（同 0050 含息序列；`--check` 會和稽核抽出的 `agents/r_fix_e50_reduce/data/ev_reduce_tsd.csv`、`ev_parvalue_tsd.csv` 逐筆比對：320 列逐欄相同）。
- 用途：官方日 K 在減資／面額變更恢復買賣日的漲跌價差是 0，引擎把當日報酬記成 0，或（前一天沒成交、差距在 10.5% 內時）把參考價調整當成一般漲跌。開啟修正後，上市、當天有成交、當天在本表的格子 → 漲跌價差＝收盤 − 本表參考價。2015-07～2026-10-22 會套用 311 格。
- 誰在讀：`cslib.py` 的 `REDUCE_FIX`（**預設 False＝不讀，舊結果逐位相同**；`cslib.load_reduce()`、`cslib.reduce_fix_chg()`）；正式管線 `agents/official_nav/run_official.py`（2026-11-21 起開啟）；名單產生器 `agents/s18_live_list/live_surge/make_list.py`（2026-11-21 起 `--reduce-fix` 預設開，`--no-reduce-fix` 可關；`make_list_v2.py` 是切換前的候選版）；`xq_scripts/site/monthly_rebalance.py` 結算時也開；網站溫度計 `xq_scripts/site/make_thermo.py --reduce-fix`（仍預設關，等 S83 用新口徑重產）。**2026-11-21 起正式採用。**
- 已知限制：退還股款視為在恢復買賣日以參考價再投入（近似；173 個事件平均差 −0.02 個百分點）；恢復買賣日當天沒成交的 3 個事件（1418 2017-11-27、1529 2018-12-22、1538 2024-12-30）與因颱風停市順延的 3 個事件（2314、3055 2016-09-28；2323 2019-09-30）不會套用；上櫃減資不在範圍。
- 更新：官方日 K 倉庫更新後重跑產生程式。第二來源（2026-11-21 起）：證交所 TWTAUU 可用 `agents/eng_official_second_source/fetch_official.py reduce --start … --end …` 抓來核對（不含面額變更）；`rebalance_readiness.py` 第 (d) 項會比。**尚未用完整的官方回應實測逐筆比對**（2026-11-21 只探查過一次、留了欄位與 2 筆樣本，之後主站當天被擋）。每月出名單前若有新的減資事件要先重產（和 TWT49U 月檔同一個時間點）。
