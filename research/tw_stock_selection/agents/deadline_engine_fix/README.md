# deadline_engine_fix：營收申報期限修正接進共用引擎（預設關閉）

依據 S40 建議 1、2（含複製員註記）與 S41 處理 A。做法同除權息開關（`cslib.EXRIGHT_FIX`）：加開關、預設 False、關閉時逐位相同。
**沒有把開關預設改成開啟，也沒有重產網站、改 README 或名單程式**（要不要改預設、重產網站，由總監決定）。

## 改了哪些檔
| 檔案 | 改動 |
|---|---|
| `agents/o_ext_canslim_xs/cslib.py` | 新增 `DEADLINE_FIX = False`、`INS15_MASK = False`、`REV_DEADLINE_DIR`；新函式 `deadline_exceptions()`、`ins15_list()`、`ins15_hidden_cols()`、`ins15_mask_row()`；`rev_avail_days()` 在 `DEADLINE_FIX` 且 pit 時讀例外表。 |
| `agents/s10_exit_details/s10lib.py` | `make_signal_s10()` 在算出訊號列 k（含位移）後加一行 `s = cslib.ins15_mask_row(P, m, k, s)`（關閉時回傳同一個物件）。 |
| `raw/rev_deadline_exceptions/` | 新增 `deadline_exceptions.csv`、`ins15_list.csv`、`README.md`（來源、每年依金管會公告更新的說明、SHA256）。 |
| `agents/deadline_engine_fix/` | 本資料夾：`run_engine.py`、`out/`、本檔。 |

SHA256（改前 → 改後）：cslib.py `7cb2c3fc…` → `c808e43e…`；s10lib.py `c7f90069…` → `bcefa250…`。

## 規則
- `DEADLINE_FIX=True`：`rev_avail_days(pit=True)` 對例外表 `apply=1` 的月份，期限改成延後後的日子，可得日＝延後期限（含）以後第一個交易日的下一個交易日（＝期限後下一個交易日換股）。2019-01：2019-02-12 → **2019-02-18**；2023-03：2023-04-11 → **2023-04-13**。2020-01 只註記（`apply=0`）。所有呼叫 `cslib.rev_avail_days` 的程式都會吃到（s2lib／stlib／s17lib／s19lib.a10_rows 等）；自己算換股日的（s19lib.b11_rows、s9lib conv、s42lib.av_family）不受影響。
- `INS15_MASK=True`：名單內公司自 `start_month`（2026-01）起，訊號列（含位移）早於 15 日期限日（次月 15 日（含）以後第一個交易日）→ 該股當月 SURGE＝缺值（不排名、不買；持有中依 E0 不在前 20 名而賣出＝實盤 make_list.py）。期限日＝訊號日視為當晚可見（同 S40／S41）。位移 1～3 的路徑因訊號日較晚，遮的格數較少（171／133／57／19 格）。目前只接在 `make_signal_s10`（現行 E0）；其他訊號函式（s4lib.make_signal、s19lib 門檻版等）不讀這個開關。

## 驗證（out/results.json）
鏈＝S40 第 3 步：s20lib→s17lib→s10lib、`EXRIGHT_FIX=True`、0050 停牌修正、上市非創新板、STD SURGE、E0、W10_70、嚴成本；標準路徑 100 萬位移 0；20 條路徑＝{60,80,100,150,200} 萬 × 位移 {0,1,2,3}。

| 組合 | 開關 | 2019 起年化 | 差（標準路徑） | 20 條中位數差 | S40 對照 | 與 S40 差距 |
|---|---|---|---|---|---|---|
| OFF | 都關 | 48.99%（全期 34.54%、最大回撤 −32.11%、已出場 381 筆） | — | — | CUR | 與改引擎前逐位相同（淨值、成本加倍淨值、X、sig、交易、av） |
| DL | 只開 DEADLINE_FIX | 48.85% | −0.14 個百分點 | −0.30 | DL −0.14／−0.30 | 0.00（20 條逐條 0.00） |
| INS | 只開 INS15_MASK | 48.90% | −0.09 個百分點 | 0.00 | PM −0.09／0.00 | 0.00（20 條逐條 0.00） |
| ALL | 都開 | 48.76% | −0.24 個百分點 | −0.35 | ALL −0.24／−0.35 | 0.00（20 條逐條 0.00） |

容差 0.02 個百分點：全部通過，而且數字與 S40 results.json 完全相同（浮點逐位）。
F1 改變的月份：2019-01（訊號改 02-18）、2023-03（改 04-13）、2026-05（少了 2885）；第一次淨值不同：DL／ALL 2019-02-13、INS 2026-06-12。
注意：S40 的 PM 用逐檔可見時間（Tasi），本開關用法定 15 日期限規則（同 S41）；兩者在這段期間的標準路徑與 20 條路徑結果都相同。

## 用法
```python
import cslib                      # 或經 s20lib／s17lib／s10lib 取得同一個模組
cslib.EXRIGHT_FIX = True
cslib.DEADLINE_FIX = True         # 期限例外月份延後換股（要在呼叫 rev_avail_days／rev_family 之前設）
cslib.INS15_MASK = True           # 15 日期限公司訊號日看不到當月營收（在 make_signal_s10 時讀）
P = s20lib.load(); F = s17lib.rev_family(P, defs=['STD'])
sg = s10lib.make_signal_s10(P, U, dict(S=F['S']['STD'], av=F['av']), shift=0)
```
重跑驗證：`python3 -I -B run_engine.py`（約 1 分鐘、峰值約 1.8 GB；暫存 scratchpad/dlfix/）。
可選 `--ref <檔>` 逐位比對改引擎前的參考（參考檔是改引擎前用 `run_engine.py OFF --save-ref` 存在 scratchpad、已刪除）。
改引擎前參考的逐位比對（2026-11-17 跑）：nav、成本加倍 nav、X、sig、交易清單、av 六項全部 True；現存 `out/results.json` 是之後不帶 `--ref` 的重跑（其餘數字相同），所以裡面沒有 `OFF_bitwise_vs_preedit_ref` 欄位。

## out/
- `results.json`：各組合標準路徑與 20 條路徑指標、對 OFF 的差、對 S40 的比較、驗收結果、改動的換股日、遮罩格數。
- `nav_std.csv.gz`：四個組合的標準路徑淨值。
- `trades_diff.csv`：各組合與 OFF 不同的交易。
- `changed_cells.csv`：改動的換股日與每個位移下被遮的（營收月份, 代碼）。
