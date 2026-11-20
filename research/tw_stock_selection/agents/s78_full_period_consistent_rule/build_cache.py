#!/usr/bin/env python3
"""S78 第 0 步：重建 S75 的連續面板快取（S75 的 scratchpad 快取已不在）。
做法：讀 s75_continuous_2008_2026/build_panel.py 原始碼（原檔不動），只把 HERE（輸出 panel_check.json 的位置）與 TMP（快取位置）換成本題的，再 exec。
python3 -I -B build_cache.py → scratchpad/s78/cache_cont/、本資料夾 panel_check.json（應與 S75 的 panel_check.json 除執行時間外相同）"""
import json, hashlib
from pathlib import Path
HERE = Path(__file__).resolve().parent
SRC = HERE.parent / 's75_continuous_2008_2026' / 'build_panel.py'
TMP = '/tmp/claude-0/-home-user/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s78'
src = SRC.read_text()
a = "HERE = Path(__file__).resolve().parent"; b = "TMP = Path('/tmp/claude-0/-home-user/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s75')"
assert src.count(a) == 1 and src.count(b) == 1
src = src.replace(a, f"HERE = Path({str(HERE)!r})").replace(b, f"TMP = Path({TMP!r})")
exec(compile(src, str(SRC), 'exec'), {'__name__': '__main__', '__file__': str(SRC)})
mine = json.loads((HERE / 'panel_check.json').read_text()); ref = json.loads((SRC.parent / 'panel_check.json').read_text())
for d in (mine, ref):
    d.pop('runtime_s'); d.pop('maxrss_mb')
assert mine == ref, '面板檢查與 S75 不同'
print('panel_check 與 S75 相同；build_panel.py sha256', hashlib.sha256(SRC.read_bytes()).hexdigest())
