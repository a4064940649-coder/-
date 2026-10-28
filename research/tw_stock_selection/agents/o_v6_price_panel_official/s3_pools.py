#!/usr/bin/env python3
"""閘門用：各換股訊號日的前 300 流動性股池與合格池（只輸出名單，不算任何報酬）。
python3 -I -B s3_pools.py v6|v4 → data/pools_<v>.json"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import v6env, v6common as C
which = sys.argv[1]
L = v6env.load(which)
K = L.K
L.setup_px()
YFX, Y5F = L.load_rfix(); DMT = L.disp_mask_fn('trade')
S = L.V4(YFX, Y5F, DMT)
out = {}
for d in S.sig_days:
    if d < pd.Timestamp('2023-11-01'): continue
    cs = S.cross_section(d)
    out[str(d.date())] = {'top300': [K.COLS[j] for j in np.where(cs['top'])[0]], 'eligible': [K.COLS[j] for j in np.where(cs['el'])[0]],
                          'exec': str(K.IDX[cs['e']].date())}
pl = S.plan(10, 'official')
out['_picks'] = {str(p[3].date()): [K.COLS[j] for j in p[1]] for p in pl if p[3] >= pd.Timestamp('2023-11-01')}
out['_nblk_total'] = S.nblk
C.jdump(out, C.DATA / f'pools_{which}.json')
print(which, len(out) - 2, 'signal days from 2023-11')
