import sys; sys.path.insert(0, '.')
import numpy as np, pandas as pd
import rcore5 as R
sys.path.insert(0, '/home/user/-/research/tw_stock_selection/agents/main_v2'); import v2_core as V2; sys.path.pop(0)
assert list(V2.COLS)==list(R.COLS)
print('PCOL eq', np.array_equal(V2.PCOL, R.PCOL), 'IS15', np.array_equal(V2.IS15, R.IS15))
D = R.official_dates()
U1 = V2.make_unavail('mc', 0); U2 = R.unavail_mc(0)
for x in D:
    d = R.IDX[x['i_d']]
    a, b = U1(d), U2(d)
    if not np.array_equal(a, b): print('diff', d, a.sum(), b.sum(), V2.deadline_case(d), R._case(d))
