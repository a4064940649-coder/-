#!/usr/bin/env python3
"""R55 第 1 步：官方面板（2015-01-05～2026-10-22）逐位重現現行 48.76%／381 筆，存樣本內淨值供合併用。
python3 -I -B step1_is.py → scratchpad/r55/is_nav.csv、scratchpad/r55/step1.json"""
import sys, json, time, resource
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import r55lib as R
from r55lib import stlib, cslib, L
import pandas as pd

t0 = time.time()
R.point_cache(R.TMP / 'cache_is')
P, U, F, sg, nav, lg, st, fix = R.run_current()
assert (P.T, P.N, str(P.IDX[-1].date())) == (2876, 2142, '2026-10-22')
lastb = lambda d: P.IDX[P.IDX < pd.Timestamp(d)][-1]
D0, DE, D19 = P.IDX[P.t0], P.IDX[-1], lastb('2019-01-01')
x = lg[lg['why'] != 'open_end']
a0 = dict(since2019=stlib.cagr(nav, D19, DE), full=stlib.cagr(nav, D0, DE), maxdd=stlib.maxdd(nav, D0, DE), n_closed=int(len(x)))
a0['bitwise_vs_ACC'] = bool(a0['since2019'] == L.ACC['since2019'] and a0['full'] == L.ACC['full'] and a0['maxdd'] == L.ACC['maxdd'] and a0['n_closed'] == L.ACC['n_closed'])
bench = pd.Series(P.E50, index=P.IDX)
rs, rb = R.monthly_returns(nav, bench)
out = dict(A0=a0, ACC=L.ACC, is_monthly=R.stats_block(rs, rb), first_signal=sg['info'][0]['sig_date'], n_signal_months=len(sg['info']),
           runtime_s=round(time.time() - t0, 1), maxrss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))
pd.DataFrame({'nav': nav, 'e50': bench.reindex(nav.index)}).to_csv(R.TMP / 'is_nav.csv')
(R.TMP / 'step1.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
print(json.dumps(out, ensure_ascii=False, indent=1))
