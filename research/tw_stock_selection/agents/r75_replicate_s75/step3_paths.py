#!/usr/bin/env python3
"""R75 第 4 步：FINDINGS 的附帶數字——20 條路徑（本金 60／80／100／150／200 萬 × 換股日晚 0～3 日）與空頭段內跌幅（V_BRIDGE）。用只留上市欄的面板（cache_twse，已驗證與全欄逐日相同）。"""
import sys, json, time, resource, gc
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import r75lib as Q
from r75lib import R, L, stlib, s17lib, s10lib, s71lib
import numpy as np, pandas as pd
Q.setup('cache_twse', '2008-01-02', '2008-01')
P, U, fix = Q.load_all()
D0, DE = P.IDX[P.t0], P.IDX[-1]; D19 = P.IDX[P.IDX < pd.Timestamp('2019-01-01')][-1]
bench = pd.Series(P.E50, index=P.IDX).loc[D0:DE]
F = s71lib.rev_rule(P, 'B2', '2008-01')
paths = {}; std = None
for s in range(4):
    sg = s10lib.make_signal_s10(P, U, dict(S=F['S']['STD'], av=F['av']), shift=s)
    for cap in (6e5, 8e5, 1e6, 1.5e6, 2e6):
        nav, lg, st = L.sim(P, sg, U, cap=cap)
        paths[f'{int(cap/1e4)}萬|+{s}'] = dict(cagr=stlib.cagr(nav, D0, DE), maxdd=stlib.maxdd(nav, D0, DE), cagr19=stlib.cagr(nav, D19, DE))
        if s == 0 and cap == 1e6:
            std = nav
v = np.array([p['cagr'] for p in paths.values()])
BEARS = {'2008': ('2008-01-01', '2008-12-31'), '2011': ('2011-01-01', '2011-12-31'), '2015': ('2015-01-01', '2015-12-31'), '2018': ('2018-01-01', '2018-12-31'),
         '2020H1': ('2020-01-01', '2020-06-30'), '2022': ('2022-01-01', '2022-12-31'), '2024-07~2025-04': ('2024-07-01', '2025-04-30')}
seg = lambda s_, a, b: float((s_.loc[a:b] / s_.loc[a:b].cummax() - 1).min())
ys, yb = Q.yearly(std), Q.yearly(bench)
out = dict(paths=paths, paths_summary=dict(median=float(np.median(v)), min=float(v.min()), max=float(v.max())),
           bears={k: dict(strat=seg(std, a, b), e0050=seg(bench, a, b)) for k, (a, b) in BEARS.items()},
           beat_years=int(sum(ys[y] > yb[y] for y in ys)), n_years=len(ys), maxrss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))
(Q.TMP / 'step3.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
print(json.dumps({k: v_ for k, v_ in out.items() if k != 'paths'}, ensure_ascii=False, indent=1))
