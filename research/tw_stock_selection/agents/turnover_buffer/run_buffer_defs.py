"""run_buffer／run_sens 共用：兩個策略的股池與分數（與 momentum_family/run_long.py、run_combo.py 相同）。"""
import sys
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import blib as B
M, L = B.M, B.L


def build_A():
    D = list(M.LDATES)
    el_univ, _, _ = M.universe(D)
    S = {k: M.at(v, D) for k, v in (('P1', M.P1), ('P2', M.P2), ('P3', M.P3), ('P6', M.P6))}
    el = el_univ & S['P1'].notna() & S['P2'].notna() & S['P3'].notna() & S['P6'].notna()
    W = {'select_2005-01_2019-04': ('2005-01-01', '2019-04-30'), 'valid_2019-05_2026-10': ('2019-05-01', '2026-10-31'),
         'full_2005-01_2026-10': ('2005-01-01', '2026-10-31'), 'valid_survivorOnly_2019-05_2023-12': ('2019-05-01', '2023-12-31'),
         'valid_2024-01_2026-10': ('2024-01-01', '2026-10-31'), 'valid_2024-01_2025-12': ('2024-01-01', '2025-12-31')}
    return D, S['P1'], el, W, 'select_2005-01_2019-04'


def build_B():
    D = list(L.DATES)
    base = L.t300_d & L.yoy3.notna() & ~L.inD_d
    p = L.s52_d; el = base & p.notna()
    sc = 0.5 * p.where(el).rank(axis=1, pct=True) + 0.5 * L.yoy3.where(el).rank(axis=1, pct=True)
    W = {'select_2021-03_2023-12': ('2021-03-01', '2023-12-31'), 'valid_2024-01_2026-10': ('2024-01-01', '2026-10-31'),
         'valid_2024-01_2025-12': ('2024-01-01', '2025-12-31'), 'full_2021-03_2026-10': ('2021-03-01', '2026-10-31')}
    return D, sc, el, W, 'select_2021-03_2023-12'


def seg(x, a, b): return x[(x.index >= pd.Timestamp(a)) & (x.index <= pd.Timestamp(b))]
