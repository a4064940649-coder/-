#!/usr/bin/env python3
"""P1 變體：層基準＝層內全部有市值股（不套衛星合格條件）市值加權；前 30% 取自合格股，等權／市值。"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import rlib as L, rstrat as S
from r1_main import XS, DATES
out = {}
def run(lo, hi, top_w, bench_all, nofin):
    WT, WB = {}, {}
    for d in DATES[:-1]:
        x = XS[d]
        tier = (x.rank >= lo) & (x.rank <= hi)
        mb = tier & np.isfinite(L._On[x.ex]) if bench_all else (tier & x.el)
        if nofin: mb &= ~L.ISFIN
        mt = tier & x.el
        wb = np.zeros(L.NC); wt = np.zeros(L.NC)
        ib = np.where(mb)[0]; wb[ib] = x.cap[ib]; wb /= wb.sum()
        o = [j for j in x.order if mt[j]]; k = max(1, int(round(0.3 * len(o)))); top = np.array(o[:k])
        wt[top] = x.cap[top] if top_w == 'cap' else 1.0; wt /= wt.sum()
        WT[d] = wt; WB[d] = wb
    nt, _ = S.simulate(WT, 'normal'); nb, _ = S.simulate(WB, 'normal')
    d_ = (S.win_returns(nt, DATES) - S.win_returns(nb, DATES)).dropna()
    return {p: (round(S.sel(d_, p).mean() * 100, 3), round(S.tstat(S.sel(d_, p)), 2)) for p in ('SEL', 'VAL', 'PRE')}
for lo, hi in ((51, 300), (11, 50), (2, 10)):
    for tw in ('eq', 'cap'):
        for nofin in (False, True):
            k = f'{lo}-{hi}|top30_{tw}|bench_alltier{"_nofin" if nofin else ""}'
            out[k] = run(lo, hi, tw, True, nofin); print(k, out[k], flush=True)
json.dump(out, open(L.OUT / 'r3d_tier_alt.json', 'w'), indent=1, default=float)
