#!/usr/bin/env python3
"""開盤漲停遞補次數（逐訊號日）：照 v2_core.V2.plan 的同一套邏輯重放，只記錄被跳過的候選（不算報酬）。
python3 -I -B s4_blk.py v6|v4 → data/blk_<v>.json（PREREG 後執行）"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import v6common as C
C.check_prereg()
import v6env
which = sys.argv[1]
L = v6env.load(which); K, V = L.K, L.V
L.setup_px()
YFX, Y5F = L.load_rfix(); DMT = L.disp_mask_fn('trade'); GAP = L.gap_dict()
def replay(S):
    out = {}; held = set()
    for d in S.sig_days:
        cs = S.cross_section(d); el = cs['el']
        if el.sum() < 50: continue
        gu = K.gap_up_open(cs['e']); cand = S.order(cs, 'official')[:int(el.sum())]
        pick = []; blk = []
        for j in cand:
            if gu[j] and j not in held: blk.append(K.COLS[j]); continue
            pick.append(j)
            if len(pick) == 10: break
        out[str(d.date())] = blk; held = set(pick)
    return out
res = {'V': replay(L.V4(YFX, Y5F, DMT)), 'V_B10': replay(L.V4(YFX, Y5F, DMT, gap=GAP, pfix=1.0)),
       'AB_s0': replay(L.V4(YFX, Y5F, DMT, unavail=V.make_unavail('mc', 0)))}
C.jdump(res, C.DATA / f'blk_{which}.json')
print(which, {k: sum(len(v) for d, v in r.items() if '2024-01-01' <= d <= '2025-09-30') for k, r in res.items()})
