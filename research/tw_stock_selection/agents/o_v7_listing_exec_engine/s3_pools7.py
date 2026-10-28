#!/usr/bin/env python3
"""閘門與描述用：各換股訊號日（2023-11 起）的前 300 流動性股池、合格池與官方名單（V7 面板、V4 引擎；只輸出名單，不算任何報酬）→ data/pools_v7.json。"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import v7env, v7common as C
L = v7env.load('v7'); K = L.K
L.setup_px()
YFX, Y5F = L.load_rfix(); DMT = L.disp_mask_fn('trade'); GAP = L.gap_dict()
out = {}
for tag, kw in (('', {}), ('B10', dict(gap=GAP, pfix=1.0))):
    S = L.V4(YFX, Y5F, DMT, **kw)
    for d in S.sig_days:
        if d < pd.Timestamp('2023-11-01'): continue
        cs = S.cross_section(d)
        if tag == '':
            out[str(d.date())] = {'top300': [K.COLS[j] for j in np.where(cs['top'])[0]], 'eligible': [K.COLS[j] for j in np.where(cs['el'])[0]],
                                  'exec': str(K.IDX[cs['e']].date())}
    pl = S.plan(10, 'official')
    out['_picks' + tag] = {str(p[3].date()): [K.COLS[j] for j in p[1]] for p in pl if p[3] >= pd.Timestamp('2023-11-01')}
C.jdump(out, C.DATA / 'pools_v7.json')
print(len([k for k in out if not k.startswith('_')]), 'signal days from 2023-11')
