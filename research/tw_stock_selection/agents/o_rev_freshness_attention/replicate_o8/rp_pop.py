import sys
from pathlib import Path
import numpy as np, pandas as pd
AG = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(AG / 'o_v7_listing_exec_engine')); import v7env; sys.path.pop(0)
L = v7env.load('v7'); K, V = L.K, L.V
L.setup_px(); YFX, Y5F = L.load_rfix(); DMT = L.disp_mask_fn('trade')
S = L.V4(YFX, Y5F, DMT)
o = {'el': 0, 'top': 0, 'ok': 0, 'n': 0}
for d in S.sig_days:
    if not (pd.Timestamp('2019-05-01') <= d <= pd.Timestamp('2026-09-30')): continue
    cs = S.cross_section(d); o['el'] += int(cs['el'].sum()); o['top'] += int(cs['top'].sum()); o['ok'] += int(cs['ok_other'].sum()); o['n'] += 1; o['tasi'] = o.get('tasi', 0) + (int(cs['ok_other'].sum()) if d >= pd.Timestamp('2024-06-01') else 0)
print(o)
