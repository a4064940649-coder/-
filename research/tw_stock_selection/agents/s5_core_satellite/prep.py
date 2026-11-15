#!/usr/bin/env python3
"""S5 前置：建面板與創新板旗標快取（scratchpad/s5/），不算任何報酬。python3 -I -B prep.py"""
import sys, time, resource
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s5lib
t = time.time()
P = s5lib.load()
A, names = s5lib.s4lib.innov_panel(P)
print('面板', P.T, P.N, '創新板', len(names), f'{time.time()-t:.0f}s', round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024), 'MB')
