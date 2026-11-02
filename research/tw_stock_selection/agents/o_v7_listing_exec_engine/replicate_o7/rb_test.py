import sys, time; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
t=time.time(); import rb_core as B, pandas as pd, numpy as np
print('load', round(time.time()-t,1), 'npatch', B.NPATCH)
R = pd.read_parquet(Path(__file__).resolve().parent.parent/'navs_v7.parquet')
SD = B.official_days(); t0 = B.TPOS[SD[0]]; t1 = B.TPOS[B.END]; sd = SD + [B.END]
print(len(SD), SD[0])
for k, b10, U in [('V7', False, None), ('V7_B10', True, None), ('AB_V7_s0', False, B.unavail_mc(0))]:
    pl, nb = B.plan(SD, B10=b10, unavail=U)
    for c in ('normal', 'stress'):
        nav, to = B.simulate(pl, *B.rates(c), t0, t1)
        s = B.seg(nav, sd, '2006-04-01', '2026-10-12')
        col = f'{k}|{c}'
        if col in R: x = R[col].dropna(); d = float((x - nav.reindex(x.index)).abs().max())
        else: d = None
        print(k, c, 'nblk', nb, 'turn', round(to,4), {kk: round(v,4) for kk, v in s.items()}, 'maxdiff', d)
