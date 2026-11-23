#!/usr/bin/env python3
"""R97：20 條路徑（5 種本金 × 位移 0～3），V_BRIDGE、v2 口徑。python3 -I -B run_paths20.py base|lim7 → paths20_<mode>.json（一次一個情境）"""
import sys, json, gc, resource
from pathlib import Path
HERE = Path(__file__).resolve().parent
S75 = HERE.parent / 's75_continuous_2008_2026'
sys.path.insert(0, str(S75))
import numpy as np, pandas as pd
import s75lib as K
from s75lib import cslib, R
MODE = sys.argv[1]; assert MODE in ('base', 'lim7')
cslib.E50_SOURCE = S75 / 'v2_2026-11-21' / 'e50_spliced.csv'; cslib.REDUCE_FIX = True
P, U = K.load_cont()
cut = int(np.searchsorted(P.IDX.values, np.datetime64('2015-06-01'), side='left'))
if MODE == 'lim7':
    tk = cslib.tick                                           # 級距同引擎（已在 run_r97.py 用自寫版核對逐格相同）
    ref = P.REF[:cut]
    P.LIMUP = P.LIMUP.copy(); P.LIMDN = P.LIMDN.copy()
    t = tk(ref * 1.07); P.LIMUP[:cut] = np.floor(ref * 1.07 / t + 1e-9) * t
    t = tk(ref * 0.93); P.LIMDN[:cut] = np.ceil(ref * 0.93 / t - 1e-9) * t
D0, DE = P.IDX[P.t0], P.IDX[-1]; DH1 = K.last_before(P.IDX, '2015-07-01'); D19 = K.last_before(P.IDX, '2019-01-01')
F, RT = K.surge(P, 'B2'); paths = {}
for s in K.SHIFTS:
    sg = K.signal(P, U, RT, s)
    for cap in K.CAPS:
        nav, lg, st = K.sim(P, sg, U, cap)
        paths[f'{int(cap / 1e4)}萬|+{s}'] = dict(cagr_full=K.cagr(nav, D0, DE), maxdd=R.maxdd(nav), cagr_2008_2015H1=K.cagr(nav, D0, DH1), cagr_since2019=K.cagr(nav, D19, DE),
                                               n_closed=int((lg['why'] != 'open_end').sum()), blk_limup=int(st['blk_limup']), sell_defer_limdn=int(st['sell_defer_limdn']))
        del nav, lg, st
    del sg; gc.collect()
md = lambda k: float(np.median([p[k] for p in paths.values()]))
out = dict(mode=MODE, paths=paths, median={k: md(k) for k in ('cagr_full', 'maxdd', 'cagr_2008_2015H1', 'cagr_since2019')}, cagr_full_min=min(p['cagr_full'] for p in paths.values()), cagr_full_max=max(p['cagr_full'] for p in paths.values()),
           maxdd_worst=min(p['maxdd'] for p in paths.values()), maxrss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))
(HERE / f'paths20_{MODE}.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
print({k: v for k, v in out.items() if k != 'paths'})
