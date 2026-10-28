#!/usr/bin/env python3
"""獨立複製引擎：用 o4 獨立複製員自寫的 rcore.py（o_v4_revenue_patch_official/replicate_o4，不 import v4lib／ia_core）在「複製員自建的 V6 面板」上
跑 AB_V6 與 AB_V6_B10（40 seeds，一般成本）。只 import 規格點名的 main_v2/v2_core.make_unavail 產生 AB 抽樣（v2_core 會連帶載入原 scratchpad 面板，但只用它的抽樣函式）。
python3 -I -B rep_run.py [rep|mine] → replicate/out_<x>.parquet"""
import sys, types, time
from pathlib import Path
import numpy as np, pandas as pd
H = Path(__file__).resolve().parent; TOP = H.parent; AG = TOP.parent
which = sys.argv[1] if len(sys.argv) > 1 else 'rep'
RC = AG / 'o_v4_revenue_patch_official' / 'replicate_o4' / 'rcore.py'
src = RC.read_text(encoding='utf-8')
OLD = "PNL = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel')"
assert src.count(OLD) == 1
src = src.replace(OLD, f"PNL = Path({str((H / 'data') if which == 'rep' else (TOP / 'data'))!r})")
K = types.ModuleType('rcore_v6'); K.__file__ = str(RC); exec(compile(src, str(RC), 'exec'), K.__dict__)
sys.path.insert(0, str(AG / 'main_v2')); import v2_core as V2; sys.path.pop(0)
T0 = time.time()
out = {}
for tag, kw in (('AB_V6', {}), ('AB_V6_B10', {'bound': 'B10'})):
    rets = []
    for s in range(40):
        pl = K.build_plan(unavail=V2.make_unavail('mc', s), **kw)
        nav, st = K.run(pl, *K.rates('normal')); rets.append(nav.pct_change())
    out[tag] = (1 + pd.concat(rets, axis=1).mean(axis=1).fillna(0)).cumprod()
    print(tag, round(time.time() - T0), flush=True)
pl = K.build_plan()
out['V6'], _ = K.run(pl, *K.rates('normal'))
out['V6_signal_days'] = pd.Series(1.0, index=pd.DatetimeIndex([K.IDX[p['i_d']] for p in pl]))
pd.DataFrame(out).to_parquet(H / f'out_{which}.parquet')
print('done', round(time.time() - T0))
