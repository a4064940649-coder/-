#!/usr/bin/env python3
"""診斷（PREREG 之後、規格不變）：o3 官方名單版 26.95% → V4 的逐項拆解（每次只改一項），以及名單差異。"""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import v4lib as L
K, V = L.K, L.V
res = json.loads((HERE / 'results.json').read_text(encoding='utf-8')); assert 'PREREG' in res
XR = L.setup_px()
YFX, Y5F = L.load_rfix(); YRD = L.load_rdel(); Y5 = L.y5_orig()
DMT, DMS = L.disp_mask_fn('trade'), L.disp_mask_fn('signal')
steps = [('o3: R_DEL, signal, fin2 off, KY implicit', dict(YL=YRD, Y5=Y5, dm=DMS, fin2=False, exclude_foreign=False)),
         ('+ explicit KY/DR/F-', dict(YL=YRD, Y5=Y5, dm=DMS, fin2=False, exclude_foreign=True)),
         ('+ fin2', dict(YL=YRD, Y5=Y5, dm=DMS, fin2=True, exclude_foreign=True)),
         ('+ trade-day disposal', dict(YL=YRD, Y5=Y5, dm=DMT, fin2=True, exclude_foreign=True)),
         ('+ R_FIX = V4', dict(YL=YFX, Y5=Y5F, dm=DMT, fin2=True, exclude_foreign=True))]
out = []; prev = None
for lab, kw in steps:
    YL_, Y5_, dm = kw.pop('YL'), kw.pop('Y5'), kw.pop('dm')
    S = L.V4(YL_, Y5_, dm, **kw)
    nav, ew, sd, pl, lg = L.run_nav(S, xr=XR)
    st = L.seg(nav, ew, sd, '2006-04-01', '2026-10-12')
    pk = {str(p[3].date()): set(K.COLS[j] for j in p[1]) for p in pl}
    d = {}
    if prev is not None:
        for s in pk:
            if pk[s] != prev[s]: d[s] = {'in': sorted(pk[s] - prev[s]), 'out': sorted(prev[s] - pk[s])}
    out.append({'step': lab, 'CAGR': st['CAGR'], 't': st['t_0050'], 'n_months_changed': len(d), 'changes': d})
    print(lab, round(st['CAGR'], 3), round(st['t_0050'], 3), len(d), list(d.items())[:6], flush=True)
    prev = pk
(HERE / 'diag_decomp.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
