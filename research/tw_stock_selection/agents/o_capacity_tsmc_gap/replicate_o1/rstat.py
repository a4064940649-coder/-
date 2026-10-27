#!/usr/bin/env python3
"""共用統計：期間定義與分段統計。"""
import sys, json
import numpy as np, pandas as pd
sys.path.insert(0, '/home/user/-/research/tw_stock_selection/agents/o_capacity_tsmc_gap/replicate_o1')
import rcore as R

S = R.SIG[R.SIG <= R.EVAL_END]
B = R.E50
T = pd.Series(R.AC[:, R.CI['2330']], index=R.IDX)
PER = {'SEL': ('2006-04-12', '2019-04-11', '2006-04-01', '2019-03-31'),
       'VAL': ('2019-04-12', '2026-10-12', '2019-04-01', '2026-09-30'),
       'VAL_A': ('2019-04-12', '2023-12-31', '2019-04-01', '2023-12-31'),
       'HOLD': ('2024-01-01', '2026-10-12', '2024-01-01', '2026-09-30'),
       'ALL': ('2006-04-12', '2026-10-12', '2006-04-01', '2026-09-30')}
pb = R.win_returns(B, S)


def seg_stats(nav, per):
    a, b, ma, mb = PER[per]
    pr = R.win_returns(nav, S)
    m = (pr.index >= ma) & (pr.index <= mb)
    ex = (pr - pb)[m]
    te = ex.std(ddof=1) * np.sqrt(12)
    return {'cagr': round(R.cagr(nav, a, b) * 100, 2), 't': round(R.tstat(ex), 3), 'TE': round(te * 100, 1),
            'IR': round(ex.mean() * 12 / te, 3), 'mdd': round(R.maxdd(nav, a, b) * 100, 2), 'n': int(m.sum())}


