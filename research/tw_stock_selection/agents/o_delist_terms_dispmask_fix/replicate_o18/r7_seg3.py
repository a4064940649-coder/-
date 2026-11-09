#!/usr/bin/env python3
"""o18 題 2 獨立複製（步驟 7，補欄）：V7X 100 萬一般 orig 名單 × orig／v3_main 的三段超額（2006～2012／2013～2018／2019 起；j 中位數）。"""
import sys, json
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r5_v7x_disp as R5
sys.path.pop(0)
R2, S, D = R5.R2, R5.S, R5.D
if __name__ == '__main__':
    R5.ENG['orig'] = None; R5.ENG['v3_main'] = R2.v3_settle('main'); R5.PL['orig'] = R5.load_plans('orig')
    cell = R2.CELLS[0]
    jobs = [('orig', key, e, cell) for e in ('orig', 'v3_main') for key in R5.PL['orig']]
    navs = {}
    with Pool(4) as pool:
        for k, key, e, cl, nav in pool.imap_unordered(R5._job, jobs, chunksize=4): navs.setdefault((e, key[0]), []).append(nav)
    def cg(n, a, b):
        n = n.dropna(); ia = n.index[n.index >= a][0]; ib = n.index[n.index <= b][-1]
        yrs = (ib - ia).days / 365.25
        return ((n[ib] / n[ia]) ** (1 / yrs) - 1) * 100
    out = {}
    for e in ('orig', 'v3_main'):
        segs = {'S1': [], 'S2': [], 'S3': []}
        for j in range(13):
            nav = S.avg_nav(navs[(e, j)]); end = S.end_day('AB_V7_B10', j); nav = nav.loc[:end].dropna(); a0 = nav.index[0]
            E = S.E50
            for s, (a, b) in {'S1': (a0, pd.Timestamp('2012-12-31')), 'S2': (pd.Timestamp('2013-01-01'), pd.Timestamp('2018-12-31')), 'S3': (pd.Timestamp('2019-01-01'), end)}.items():
                segs[s].append(cg(nav, a, b) - cg(E, max(a, a0), b))
        out[e] = {s: float(np.median(v)) for s, v in segs.items()}
    (HERE / 'r7_seg3.json').write_text(json.dumps(out, indent=1)); print(out)
