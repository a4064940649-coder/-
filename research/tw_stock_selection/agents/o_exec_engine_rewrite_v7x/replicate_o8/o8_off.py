#!/usr/bin/env python3
"""官方日（plans_ab7 AB_V7_B10 seeds 0～39）R0 32 格，對 V7 原表 B10|OFF。只報告。"""
import sys, json, pickle
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import d8, e8, s8, g8_zero as Z
from multiprocessing import Pool
CELLS = [(m, c, ti, a) for m in ('K5T', 'TICK') for c in ('normal', 'stress') for ti in ('30萬', '100萬', '300萬', '1000萬') for a in ('FIX', 'GROW')]
PA = {k: v for k, v in Z.load_plans('ab').items() if k[0] == 'AB_V7_B10'}
T1 = e8.TPOS[pd.Timestamp('2026-10-12')]
def job(k):
    plan = PA[k]; return {c: e8.Sim(plan, T1, dict(model=c[0], cost=c[1], tier=c[2], acct=c[3])).run().to_numpy() for c in CELLS}
if __name__ == '__main__':
    with Pool(4) as P: R = P.map(job, list(PA), chunksize=1)
    T = json.load(open(d8.O7 / 'results.json'))['RESULTS_EXEC']['table']
    plan = PA[list(PA)[0]]; t0 = plan[0][0] - 1; sd = [p[1] for p in plan]
    out = {}
    for c in CELLS:
        m = np.mean([r[c] for r in R], axis=0); m[0] = 0
        nav = pd.Series(np.cumprod(1 + m), index=e8.IDX[t0:T1 + 1])
        st = s8.cell_stats(nav, sd, '2026-10-12'); out['|'.join(c)] = st['CAGR'] - T['B10|OFF|' + '|'.join(c)]['CAGR']
    res = dict(diff=out, maxabs=max(abs(v) for v in out.values()), maxabs_normal=max(abs(v) for k, v in out.items() if '|normal|' in k))
    json.dump(res, open(HERE / 'out' / 'off_R0.json', 'w'), indent=1, ensure_ascii=False); print(json.dumps(res, indent=1, ensure_ascii=False))
