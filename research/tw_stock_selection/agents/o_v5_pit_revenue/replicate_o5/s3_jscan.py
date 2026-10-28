#!/usr/bin/env python3
"""換股日分布 J0～J12（st_core 的 j 定義，依規格自行重寫）。AB 40 seed 平均。"""
import sys, json
sys.path.insert(0, '/home/user/-/research/tw_stock_selection/agents/o_v5_pit_revenue/replicate_o5')
import numpy as np, pandas as pd
from multiprocessing import Pool
import rcore5 as R
HERE = '/home/user/-/research/tw_stock_selection/agents/o_v5_pit_revenue/replicate_o5/'
TC = R._TX                                  # 加權指數日曆
SPECIAL = {'2019-02': pd.Timestamp('2019-02-15'), '2023-04': pd.Timestamp('2023-04-12')}
def first_td(ts):
    k = TC.searchsorted(ts); return TC[k] if k < len(TC) else None
def deadline(m):
    s = str(m); return first_td(SPECIAL[s]) if s in SPECIAL else first_td(pd.Timestamp(m.year, m.month, 10))
def deadline15(m): return first_td(pd.Timestamp(m.year, m.month, 15))
SKIP_NEXT = True   # 規格：次一面板日有值檔數 < 300 時該月跳過
def day_for(m, j):
    Dm = deadline(m)
    if Dm is None: return None
    k = TC.get_loc(Dm) + j
    if k >= len(TC): return None
    s = TC[k]
    if s not in R.TPOS: return None
    i = R.TPOS[s]
    if not R.GOOD[i] or i + 1 >= len(R.IDX): return None
    if SKIP_NEXT and not R.GOOD[i + 1]: return None
    return s
def rng(seed, d, tag=0): return np.random.default_rng([int(seed), int(d.strftime('%Y%m%d')), tag])
def unavail_for(m, s, seed):
    Dm = deadline(m); u = np.zeros(R.N, bool)
    if s < Dm: u[:] = True
    elif s == Dm or (str(m) == '2020-02' and s <= pd.Timestamp('2020-02-14')):
        u = rng(seed, s).random(R.N) < R.PCOL
    rm = str(m - 1)
    if '2026-01' <= rm <= '2026-08':
        D15 = deadline15(m)
        if s < D15: u |= R.IS15
        elif s == D15: u |= R.IS15 & (rng(seed, s, 15).random(R.N) < R.PCOL)
    return u
ANCH = [x['m'] for x in R.official_dates()]
def version(j):
    days = [(m, day_for(m, j)) for m in ANCH]; days = [(m, d) for m, d in days if d is not None]
    de = day_for(pd.Period('2026-10', 'M'), j)
    if de is None or de > R.IDX[-1]: de = R.IDX[-1]
    days = [(m, d) for m, d in days if d < de]
    return days, de
REV = R.Rev('fv')
def one(args):
    j, seed = args
    days, de = version(j)
    Dl = [dict(i_d=R.TPOS[d], i_e=R.TPOS[d] + 1, m=m, u=unavail_for(m, d, seed)) for m, d in days]
    pl = R.build_plan(Dl, REV, U=lambda d: None)
    return R.run(pl, *R.rates(), end=de), [R.IDX[p['i_d']] for p in pl], de
if __name__ == '__main__':
    out = {}
    with Pool(4) as P:
        for j in range(13):
            res = P.map(one, [(j, s) for s in range(40)])
            nav = R.avg_nav([r[0] for r in res]); ds = res[0][1]; de = res[0][2]
            W = list(zip(ds, ds[1:] + [de]))
            ex = R.excess(nav, W)
            out[j] = dict(t=R.tstat(ex), n=len(ex), cagr=R.cagr(nav, ds[0], de) * 100, first=str(ds[0].date()), end=str(de.date()))
            print(j, out[j], flush=True)
    t = np.array([out[j]['t'] for j in range(13)])
    out['summary'] = dict(median=float(np.median(t)), p10=float(np.percentile(t, 10)), min=float(t.min()), max=float(t.max()),
                          n_ge230=int((t >= 2.3).sum()), n_ge20=int((t >= 2.0).sum()))
    json.dump(out, open(HERE + 's3_jscan.json', 'w'), indent=1, default=str)
    print(out['summary'])
