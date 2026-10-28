#!/usr/bin/env python3
"""引擎接線（不改任何既有檔案）：V7 版（由 o_v6 v6env.py 複製，只改面板路徑為本題 data/）：把 independent_audit/ia_core.py 的原始碼讀進來，只把面板路徑 P 換成指定資料夾，
以模組名 'ia_core' 註冊到 sys.modules，之後 import 的 main_v2/v2_core → o_disposal_rejudge/drj_core → o_infra_v3_rebuild/o3_core
→ o_v4_revenue_patch_official/v4lib 都會用到這份 ia_core。策略規格、成本、處置、營收、AB 全部沿用 V4 引擎，一字不改。
用法：import v6env; L = v7env.load('v7')   # 或 'v4'（原 scratchpad 面板，用於對照）"""
import sys, types, hashlib
from pathlib import Path
HERE = Path(__file__).resolve().parent
AG = HERE.parent
IA = AG / 'independent_audit' / 'ia_core.py'
OLD = "P = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel')"
ENGINE_FILES = [IA, AG / 'main_v2' / 'v2_core.py', AG / 'o_disposal_rejudge' / 'drj_core.py', AG / 'o_infra_v3_rebuild' / 'o3_core.py',
                AG / 'o_v4_revenue_patch_official' / 'v4lib.py']

def sha(f): return hashlib.sha256(Path(f).read_bytes()).hexdigest()

def load(which='v7'):
    assert 'ia_core' not in sys.modules, 'ia_core 已載入；一個行程只能用一種面板'
    src = IA.read_text(encoding='utf-8')
    assert src.count(OLD) == 1
    if which == 'v7':
        src = src.replace(OLD, f"P = Path({str(HERE / 'data')!r})")
    m = types.ModuleType('ia_core'); m.__file__ = str(IA)
    sys.modules['ia_core'] = m
    exec(compile(src, str(IA), 'exec'), m.__dict__)
    sys.path.insert(0, str(AG / 'o_v4_revenue_patch_official'))
    import v4lib as L
    sys.path.pop(0)
    assert L.K is m
    return L
