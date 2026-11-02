#!/usr/bin/env python3
"""複製員 o7：載入凍結引擎鏈（ia_core→v2_core→drj_core→o3_core→v4lib），只把 ia_core 面板路徑換成 V7 data/。
不 import 研究員 o7 的任何腳本（v7env/v7common/...）。先核對 V7 面板 sha256＝data_freeze_v7.json。"""
import sys, types, json, hashlib
from pathlib import Path
HERE = Path(__file__).resolve().parent
O7 = HERE.parent; AG = O7.parent
IA = AG / 'independent_audit' / 'ia_core.py'
PDIR = O7 / 'data'
def sha(f): return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def check():
    fz = json.loads((O7 / 'data_freeze_v7.json').read_text())
    bad = [k for k, v in fz['v7_panel'].items() if sha(v['path']) != v['sha256']]
    assert not bad, bad
    return fz
def load():
    check()
    if 'ia_core' not in sys.modules:
        src = IA.read_text(encoding='utf-8')
        old = "P = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel')"
        assert src.count(old) == 1
        src = src.replace(old, f"P = Path({str(PDIR)!r})")
        m = types.ModuleType('ia_core'); m.__file__ = str(IA); sys.modules['ia_core'] = m
        exec(compile(src, str(IA), 'exec'), m.__dict__)
    sys.path.insert(0, str(AG / 'o_v4_revenue_patch_official'))
    import v4lib as L
    sys.path.pop(0)
    assert str(L.K.P) == str(PDIR)
    return L
