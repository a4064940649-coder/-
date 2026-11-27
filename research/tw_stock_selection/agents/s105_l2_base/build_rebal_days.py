#!/usr/bin/env python3
"""S105 步驟 1：取出 S75 v3 口徑（V_BRIDGE）各位移的「每月換股日」。
換股訊號列 k（RQ 的鍵，用 k 日收盤資料），引擎在 k+1 日開盤交易；本研究底倉在 k+1 日收盤再平衡。
不改引擎；載入方式同 s103_futures_overlay/build_sleeves.py（漲跌幅覆寫只影響撮合，不影響訊號，故省略）。
用法：python3 -I -B build_rebal_days.py → rebal_days.json"""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
S75 = HERE.parent / 's75_continuous_2008_2026'; V2 = S75 / 'v2_2026-11-21'
sys.path.insert(0, str(S75))
import numpy as np
import s75lib as K
from s75lib import cslib
cslib.E50_SOURCE = V2 / 'e50_spliced.csv'; cslib.REDUCE_FIX = True
P, U = K.load_cont()
F, RT = K.surge(P, K.VERSIONS['V_BRIDGE'])
out = {}
for s in K.SHIFTS:
    sg = K.signal(P, U, RT, s)
    ks = sorted(sg['RQ'].keys())
    out[str(s)] = dict(signal=[str(P.IDX[k].date()) for k in ks], trade=[str(P.IDX[k + 1].date()) for k in ks if k + 1 < P.T])
(HERE / 'rebal_days.json').write_text(json.dumps(out, indent=0))
print({k: (len(v['trade']), v['trade'][:2], v['trade'][-2:]) for k, v in out.items()})
