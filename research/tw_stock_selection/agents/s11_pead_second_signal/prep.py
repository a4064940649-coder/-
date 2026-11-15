#!/usr/bin/env python3
"""S11 前置：建面板與創新板旗標快取（scratchpad/s11tmp/），檢查面板和 S6～S10 相同，不算任何報酬。python3 -I -B prep.py"""
import sys, time, resource
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s11lib
t = time.time()
P = s11lib.load()
A, names = s11lib.s4lib.innov_panel(P)
print('面板', P.T, P.N, str(P.IDX[-1].date()), '創新板', len(names), f'{time.time()-t:.0f}s', round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024), 'MB')
assert (P.T, P.N, str(P.IDX[-1].date())) == (2876, 2142, '2026-10-22'), 'P0 失敗：面板和 S6～S10 不同'
