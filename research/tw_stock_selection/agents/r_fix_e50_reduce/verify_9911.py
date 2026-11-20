#!/usr/bin/env python3
"""查 XQ 版在修正 B 之下 2016-04 名單多出 9911 的原因：腳本股池用「以最後一天為基準往回還原」的價格 × 成交量算 60 日均成交值門檻。
用法：python3 -I -B verify_9911.py base|B"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
mode = sys.argv[1]
AG = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AG / 's19_xq_threshold')); sys.path.insert(0, str(Path(__file__).resolve().parent))
import s19lib
from s19lib import s10lib, s9lib, s8lib, s7lib, s6lib, s5lib, s4lib, s3lib, s2lib, stlib, cslib
TMP = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/r_fix/cache')
for mod, attr in ((s19lib, 'S19'), (s10lib, 'S10'), (s9lib, 'S9'), (s8lib, 'S8'), (s7lib, 'S7'), (s6lib, 'S6'), (s5lib, 'S5'), (s4lib, 'S4'), (s3lib, 'S3'), (s2lib, 'S2'), (stlib, 'S1'), (cslib, 'TMP')):
    setattr(mod, attr, TMP)
cslib.EXRIGHT_FIX = True; cslib.DEADLINE_FIX = True; cslib.INS15_MASK = True
import rfix; rfix.pre(cslib, mode)
rfix.ensure_cache()
P = s19lib.load()
INNOV, _ = s4lib.innov_panel(P)
UX, Cadj, liqX = s19lib.script_universe(P, INNOV)
j = P.ci['9911']; t = int(np.searchsorted(P.IDX.values, np.datetime64('2016-05-11')))
print(mode, '9911 2016-05-11 原始收盤', P.C[t, j], '還原收盤', round(Cadj[t, j], 3), '60日均成交值(萬)', round(liqX[t, j] / 1e4, 1), '在腳本股池', bool(UX[t, j]))
for code, d in (('1418', '2017-11-28'), ('1529', '2018-12-24'), ('1538', '2024-12-31')):
    print(code, d, 'in UX', bool(UX[int(np.searchsorted(P.IDX.values, np.datetime64(d))), P.ci[code]]))
