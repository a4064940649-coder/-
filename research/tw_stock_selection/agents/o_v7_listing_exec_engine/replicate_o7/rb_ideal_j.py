"""rb_ideal 的 j＝0～12 版（AB：disp 拖累；B10：mdelay 拖累；40 seeds），比對 navs_ideal7。"""
import sys, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import rb_core as B, rb_ideal as RI, rb_run as RR
from multiprocessing import Pool
def job(a):
    k, j, sd = a
    days = [(m, RR.jday(m, j)) for m in RR.ANCH]; days = [(m, d) for m, d in days if d is not None]
    de = RR.jday(pd.Period('2026-10', 'M'), j)
    if de is None or de > B.IDX[-1]: de = B.IDX[-1]
    days = [(m, d) for m, d in days if d < de]; mof = {d: m for m, d in days}; dl = [d for m, d in days]
    pl, _ = B.plan(dl, B10=(k == 'B10'), unavail=lambda d: RR.unav_j(mof[d], d, sd), mof=mof)
    bb, ss = B.rates('normal'); t0, t1 = B.TPOS[dl[0]], B.TPOS[de]
    out = {}
    for m in ('base', 'disp', 'mdelay'):
        out[m] = RI.simulate(pl, bb, ss, t0, t1, disp=(m == 'disp'), mdelay=(m == 'mdelay')).pct_change()
    return k, j, sd, out
if __name__ == '__main__':
    t = time.time()
    with Pool(4) as P: res = P.map(job, [(k, j, s) for k in ('AB', 'B10') for j in range(13) for s in range(40)], chunksize=4)
    I = pd.read_parquet(HERE.parent / 'navs_ideal7.parquet'); out = {}
    for k in ('AB', 'B10'):
        d = {m: [] for m in ('disp', 'mdelay')}; rd = {m: [] for m in ('disp', 'mdelay')}
        for j in range(13):
            nv = {m: (1 + pd.concat([r[3][m] for r in res if r[0] == k and r[1] == j], axis=1).mean(axis=1).fillna(0)).cumprod() for m in ('base', 'disp', 'mdelay')}
            cg = {m: B.cagr_mdd(n, '2006-04-01', '2026-10-12')[0] for m, n in nv.items()}
            rc = {m: B.cagr_mdd(I[f'{k}|J{j}|{m}'].dropna(), '2006-04-01', '2026-10-12')[0] for m in nv}
            for m in d: d[m].append(cg[m] - cg['base']); rd[m].append(rc[m] - rc['base'])
        out[k] = {m: dict(mine_median=round(float(np.median(d[m])), 4), mine_min=round(min(d[m]), 4), mine_max=round(max(d[m]), 4),
                          res_median=round(float(np.median(rd[m])), 4), res_min=round(min(rd[m]), 4), res_max=round(max(rd[m]), 4),
                          max_abs_diff=round(float(np.max(np.abs(np.array(d[m]) - np.array(rd[m])))), 5)) for m in d}
    json.dump(out, open(HERE / 'out' / 'ideal_j.json', 'w'), indent=1); print(json.dumps(out, indent=1), time.time() - t)
