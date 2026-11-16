#!/usr/bin/env python3
"""S26：S8 複製員「回撤中位數 ≈ −23.1% − 1.58 × 每檔占總資產」那 17 格裡，S6／S7 的 12 格研究員結果沒有存 20 條路徑回撤，
這裡用 s8lib.simulate_s8（cap=None；與 S8 run.py 的路徑迴圈同一呼叫：s4lib.make_signal(shift=s)、5 種資金、嚴成本）補算，
修正前（fix 0）與修正後（fix 1）各跑一次，讓公式的修正前後用同一口徑。
python3 -I -B paths_dd.py <fix 0|1> → out/paths_dd_fix<0|1>.json"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common

fix = sys.argv[1] == '1'
common.set_fix(fix)
common.addpath('s8_drawdown_cap')
import s8lib                                                          # noqa: E402
from s8lib import s4lib, s2lib, stlib, COSTS                          # noqa: E402
common.redirect()
P = s8lib.load()
assert P.exright_fix == fix
U = stlib.universe(P, 'twse') & ~s4lib.innov_panel(P)[0]
RT = s2lib.rev_tables(P)
D0, DE = P.IDX[P.t0], P.IDX[-1]
D19 = P.IDX[P.IDX < pd.Timestamp('2019-01-01')][-1]
KW = dict(depth=10, mom_lo=0.10, exit='top20', keep_info=True)
CAPS = (6e5, 8e5, 1e6, 1.5e6, 2e6)
CELLS = {  # id: (sizing, sat_w, lock)  — S6、S7 的格（S8 的 5 格用 S8 run.py 本身的結果）
    'W10_70': ('W10', 0.7, False), 'W7_70': ('W7', 0.7, False), 'WN_70': ('WN', 0.7, False), 'W10_100': ('W10', 1.0, False),
    'W7_100': ('W7', 1.0, False), 'WN_100': ('WN', 1.0, False),
    'W10_70L': ('W10', 0.7, True), 'W7_70L': ('W7', 0.7, True), 'WN_70L': ('WN', 0.7, True), 'W10_80L': ('W10', 0.8, True),
    'W7_80L': ('W7', 0.8, True), 'WN_80L': ('WN', 0.8, True)}
out = {c: dict(paths={}, paths_dd={}) for c in CELLS}
for s in range(4):
    S = s4lib.make_signal(P, U, RT, shift=s, **KW)
    for cid, (sz, w, lk) in CELLS.items():
        for cap in CAPS:
            nav = s8lib.simulate_s8(P, S, U, sat_w=w, sizing=sz, lock=lk, cap=None, cost=COSTS['strict'], capital=cap)[0]
            k = f'{int(cap/1e4)}萬|+{s}'
            out[cid]['paths'][k] = stlib.cagr(nav, D19, DE); out[cid]['paths_dd'][k] = stlib.maxdd(nav, D0, DE)
    common.log('位移', s)
for cid in CELLS:
    out[cid]['median'] = common.med(out[cid]['paths'].values()); out[cid]['maxdd_median'] = common.med(out[cid]['paths_dd'].values())
    out[cid]['n_dd_within_36'] = int(sum(v >= -0.36 for v in out[cid]['paths_dd'].values()))
common.dump(f'paths_dd_fix{int(fix)}.json', dict(fix=fix, cells=out, peak_mb=common.rss()))
print({c: (round(out[c]['median'] * 100, 2), round(out[c]['maxdd_median'] * 100, 2)) for c in CELLS})
