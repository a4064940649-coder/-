#!/usr/bin/env python3
"""步驟 0c：由 raw/exec/ecore.py（o_exec_capacity_tiers 的可執行模擬器，唯讀複製）產生本題的 ecore_tick.py。
只做三處文字替換（每處斷言恰好一次），其餘逐字相同：
 1) 資料路徑改為本題 raw/exec/。
 2) 整股衝擊新增 impact＝'tick'：買在開盤價＋1 個跳動單位、賣在開盤價−1 個跳動單位（不含 0.1% 基礎滑價、不隨下單量變化）；
    參與上限、零股、手續費、稅、順延規則全部不變。'sqrt'（o5 κ 模型：0.1%＋0.5σ√(Q/OA)）與 'flat' 照舊。
 3) 現金需求估計（_Ctx.cost）同步。
跳動單位（證交所股票，2005 年起）：<10 元 0.01；10～50 元 0.05；50～100 元 0.1；100～500 元 0.5；500～1000 元 1；≥1000 元 5。
用法：python3 -I -B s0_make_ecore_tick.py
"""
from pathlib import Path
HERE = Path(__file__).resolve().parent
src = (HERE / 'raw' / 'exec' / 'ecore.py').read_text(encoding='utf-8')

REPL = [
    ("DATA, RAW = HERE / 'data', HERE / 'raw'",
     "DATA, RAW = HERE / 'raw' / 'exec', HERE / 'raw' / 'exec'   # 本題：輸入全部在 raw/exec/\n"
     "def tick_size(p):\n"
     "    \"\"\"證交所股票升降單位（2005 年起六級）。\"\"\"\n"
     "    if p < 10: return 0.01\n"
     "    if p < 50: return 0.05\n"
     "    if p < 100: return 0.1\n"
     "    if p < 500: return 0.5\n"
     "    if p < 1000: return 1.0\n"
     "    return 5.0"),
    ("    if fl['impact'] == 'sqrt' and np.isfinite(oa) and oa > 0 and np.isfinite(sig):\n        imp = 0.001 + 0.5 * sig * math.sqrt(Q / oa)\n    else:\n        imp = 0.001",
     "    if fl['impact'] == 'tick':\n        imp = tick_size(D.O[t, j]) / D.O[t, j]\n    elif fl['impact'] == 'sqrt' and np.isfinite(oa) and oa > 0 and np.isfinite(sig):\n        imp = 0.001 + 0.5 * sig * math.sqrt(Q / oa)\n    else:\n        imp = 0.001"),
    ("            imp = 0.001 + (0.5 * sig * math.sqrt(lots * 1000.0 * pc / oa) if ok else 0.0)",
     "            imp = (tick_size(o) / o) if fl['impact'] == 'tick' else 0.001 + (0.5 * sig * math.sqrt(lots * 1000.0 * pc / oa) if ok else 0.0)"),
]
for a, b in REPL:
    assert src.count(a) == 1, a[:60]
    src = src.replace(a, b)
hdr = ('# 本檔由 s0_make_ecore_tick.py 自 raw/exec/ecore.py（o_exec_capacity_tiers/ecore.py 的複製）產生；'
       '只改資料路徑並新增 impact=\'tick\'（每邊一個跳動單位）。\n')
(HERE / 'ecore_tick.py').write_text(hdr + src, encoding='utf-8')
print('ecore_tick.py written')
