#!/usr/bin/env python3
"""複製員 o6 的環境載入（自己寫，不用研究員 v6env.py）：
把 independent_audit/ia_core.py 的原始碼讀進來，只把面板路徑 P 換成指定資料夾，以模組名 ia_core 註冊到 sys.modules，
再 import v4lib（→ drj_core → v2_core → ia_core；o3_core）。引擎程式一字不改。
環境變數 O6R_PANEL＝compat（預設；與研究員面板逐格相同）或 spec（颱風停市事件依規格順延）或任一資料夾路徑。"""
import os, sys, types
from pathlib import Path
HERE = Path(__file__).resolve().parent
AG = HERE.parent.parent
IA = AG / 'independent_audit' / 'ia_core.py'

def load(panel=None):
    panel = panel or os.environ.get('O6R_PANEL', 'compat')
    pdir = Path(panel) if '/' in panel else HERE / 'data' / f'panel_{panel}'
    assert (pdir / 'close_raw.parquet').exists(), pdir
    if 'ia_core' not in sys.modules:
        src = IA.read_text(encoding='utf-8')
        old = "P = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel')"
        assert src.count(old) == 1
        src = src.replace(old, f"P = Path({str(pdir)!r})")
        m = types.ModuleType('ia_core'); m.__file__ = str(IA)
        sys.modules['ia_core'] = m
        exec(compile(src, str(IA), 'exec'), m.__dict__)
    sys.path.insert(0, str(AG / 'o_v4_revenue_patch_official'))
    import v4lib as L
    sys.path.pop(0)
    assert L.K is sys.modules['ia_core'] and str(L.K.P) == str(pdir)
    return L
