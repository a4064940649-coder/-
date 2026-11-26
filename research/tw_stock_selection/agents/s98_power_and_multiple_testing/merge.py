#!/usr/bin/env python3
"""S98：把 run_a.py、run_b.py 的輸出併成 results.json。順序：python3 -I -B run_a.py → run_b.py → merge.py"""
import json
from pathlib import Path
H = Path(__file__).resolve().parent
a = json.loads((H / 'results_a.json').read_text()); b = json.loads((H / 'results_b.json').read_text())
dev = ["PREREG 未列但加做：正式淨值月切點敏感度（每月第 1/6/11/16/21/26 日後第一個交易日）、近親＋S78 合併族群 SPA、大贏家 since_2019 窗、V_BRIDGE 2008-01～2013-01 與 V_MASK_full 序列。",
       "S69 驗證用種子 20261118（PREREG 未列）；三組最大差 0.54 個百分點，通過 ≤1.5 門檻。",
       "廣族群 SPA 的焦點用 11 日切點窗（S52 作法）；v3 淨值在格點以 ffill 對齊。",
       "DSR 的 N_rho 以 N_all 帶入（ρ̄ 取自可取得族群，非 4,026 條全體）。",
       "README 文字版規則（alt）的『m≥9 通過』用固定 9 個百分點，TE 放大 25% 時誤判率會跟著上升；主版本用 PLAN_FROZEN 的 t 界線則不會。"]
(H / 'results.json').write_text(json.dumps(dict(topic='S98', prereg_sha256=a['prereg_sha256'], deviations=dev, task_A=a, task_B=b), ensure_ascii=False, indent=1))
(H / 'results_a.json').unlink(); (H / 'results_b.json').unlink()
