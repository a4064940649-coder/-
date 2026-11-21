# FINDINGS：工程——修正版候選（A＋B）與網站兩個新區塊（2026-11-21）

（工程師不能寫 .md，由總監依其最後回覆存檔；完整步驟見 plan.json，驗證見 verify/，頁面測試截圖見 page_test/shots/。）

## 修正版候選（未切換正式數字，等產品擁有者說「採用」）
- 新資料封存：`raw/e50_tr_official/`（0050 含息序列 2015-01-05～2026-11-20，2,890 天，與稽核自建序列逐筆相同）、`raw/twse_reduce/`（減資／面額變更事件 320 列）。
- `o_ext_canslim_xs/cslib.py` 新增 `REDUCE_FIX = False`、`E50_SOURCE = None` 兩個開關（預設關）。開關全關時正式結果逐位不變（48.76%／34.40%／381 筆／逐日差 0／46.40%；面板 22 個陣列雜湊相同）；總監套用後再跑每日回測檢查 PASS。
- 候選正式管線 `official_nav/run_official_v2.py` → `official_nav/out_v2/`：48.86%／34.46%／−31.61%／381 筆／46.42%，0050 2019 起 28.82%；與稽核 A＋B 淨值逐日差 0。
- `s18_live_list/live_surge/make_list_v2.py`（`--reduce-fix` 預設關）與 `make_list_reduce_fix.diff`：開關關時 2026-11 名單輸出逐位元相同；開關開時前 20 名、F1、換股計畫零差異。線上用的 `make_list.py` 沒動。
- 切換清單 11 步在 plan.json；風險表會變 7 列（回撤 −32.1→−31.6、落後 0050 最多 15.3→14.4、年化 48.76→48.86、不買最賺 5 檔 44.88→44.98 等）。

## 網站新區塊（總監已上傳，2026-11-21）
- `xq_scripts/site/make_thermo.py` → `data/thermo.json`（227 個月）：2015-05～2026-09 共 137 個月與 S83 存檔逐月相同（差 <1e-14）；2008-01～2015-04 共 88 個月沿用研究存檔值（官方日 K 2015 年才開始）。
- `data/rules.json`：7 條照做須知（由 README 第 32.1 節逐字抽出；第 6、7 條標「未獨立重現」）。
- `index.html` 加「照做須知」「環境溫度計」兩個區塊（線上版＋97 行，未刪任何一行）；假環境 Playwright 15 個情境通過（1280／400／360 寬不橫向捲動、既有區塊文字不變）。
- 已上傳：datasets/thermo（v1）、datasets/rules（v1）、files/index.html（v7）。**線上實際顯示未能由總監親眼確認**（本環境開不了網站），請產品擁有者開站看一眼。
- 溫度計讀數：2026-10 完整月（10-30）5.45%／3.14%／16.88% → 熱；2026-11 到 11-20（未結束）0.93%／4.24%／21.58% → 熱（飆股密度已落到冷，連續 17 個月含未結束月）。README 第 32.2 節的 3.91%／3.15%／21.0% 是 10-22 的讀數。

## 之後每天
`python3 -I -B xq_scripts/site/make_thermo.py xq_scripts/site/data/thermo.json <tsd> --check <暫存檔>`，上傳後更新 datasets/thermo 的 source.url 與 updated（網站資料集現為 11 個）。
