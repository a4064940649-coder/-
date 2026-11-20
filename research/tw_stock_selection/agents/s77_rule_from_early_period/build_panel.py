#!/usr/bin/env python3
"""S77 連續面板：逐字重用 S75 的 build_panel.py（不改原檔），只把暫存路徑與 panel_check.json 的輸出位置換成本題。
python3 -I -B build_panel.py → scratchpad/s77/cache_cont/、本資料夾 panel_check.json（應與 S75 的 panel_check.json 除執行時間外相同）"""
from pathlib import Path
HERE = Path(__file__).resolve().parent
src = (HERE.parent / 's75_continuous_2008_2026' / 'build_panel.py').read_text()
a = "HERE = Path(__file__).resolve().parent"; b = "scratchpad/s75')"
assert src.count(a) == 1 and src.count(b) == 1
src = src.replace(a, f"HERE = Path({str(HERE)!r})").replace(b, "scratchpad/s77')")
exec(compile(src, 's75/build_panel.py(S77 改道)', 'exec'), {'__name__': '__main__', '__file__': str(HERE / 'build_panel.py')})
