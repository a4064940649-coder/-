#!/usr/bin/env python3
"""S71 驗收（只用現行凍結規則 MASK，不含任何候選規則）：
A1 官方面板（2015-01-05～2026-10-22）逐位重現現行 48.76%／381 筆（r55lib.run_current 原函式）；
A2 自建面板 2008-01～2013-01 重現 R55 主做法 +1.34 個百分點／NW t 0.52（r55lib.run_current 原函式）；
A3 s71lib.rev_rule('MASK') 的 SURGE 與 s17lib.rev_family 逐格相同（NaN 位置也相同）。
python3 -I -B accept.py → accept.json、scratchpad/s71/is_nav.csv、mask_nav.csv"""
import sys, json, time, resource
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s71lib as S
from s71lib import R, L, stlib, s17lib
import numpy as np, pandas as pd

t0 = time.time()
out = {}
# ---- A1
R.point_cache(S.TMP / 'cache_is')
P, U, F, sg, nav, lg, st, fix = R.run_current()
assert (P.T, P.N, str(P.IDX[-1].date())) == (2876, 2142, '2026-10-22')
lastb = lambda d: P.IDX[P.IDX < pd.Timestamp(d)][-1]
D0, DE, D19 = P.IDX[P.t0], P.IDX[-1], lastb('2019-01-01')
x = lg[lg['why'] != 'open_end']
a1 = dict(since2019=stlib.cagr(nav, D19, DE), full=stlib.cagr(nav, D0, DE), maxdd=stlib.maxdd(nav, D0, DE), n_closed=int(len(x)))
a1['bitwise_vs_ACC'] = bool(all(a1[k] == L.ACC[k] for k in ('since2019', 'full', 'maxdd', 'n_closed')))
bench = pd.Series(P.E50, index=P.IDX)
irs, irb = R.monthly_returns(nav, bench)
a1['is_monthly'] = R.stats_block(irs, irb)
pd.DataFrame({'nav': nav, 'e50': bench.reindex(nav.index)}).to_csv(S.TMP / 'is_nav.csv')
out['A1_official'] = a1
# ---- A2
R.point_cache(S.TMP / 'cache_pre')
P, U, F, sg, nav, lg, st, fix = R.run_current(start='2008-01-02', first_month='2008-01')
bench = pd.Series(P.E50, index=P.IDX).reindex(nav.index)
rs, rb = R.monthly_returns(nav, bench)
w = lambda a, b: R.stats_block(rs[(rs.index >= pd.Period(a, 'M')) & (rs.index <= pd.Period(b, 'M'))], rb[(rb.index >= pd.Period(a, 'M')) & (rb.index <= pd.Period(b, 'M'))])
a2 = dict(primary=w('2008-01', '2013-01'), stuck=w('2013-02', '2015-06'), secondary=w('2008-01', '2015-06'),
          n_signal_months=len(sg['info']), last_signal=sg['info'][-1]['sig_date'], n_buy=int(st['n_buy']))
ref = json.load(open(S.AG / 'r55_replicate_s55' / 'results.json'))['oos_main']['windows']
a2['ref_R55_main'] = {k: dict(ex_ann_pp=ref[k]['ex_ann_pp'], t_nw6=ref[k]['t_nw6']) for k in ('primary', 'stuck', 'secondary')}
a2['max_abs_diff_vs_R55'] = max(abs(a2[k][f] - ref[k][f]) for k in ('primary', 'stuck', 'secondary') for f in ('ex_ann_pp', 't_nw6', 'cagr_strat', 'cagr_0050'))
a2['pass'] = bool(a2['max_abs_diff_vs_R55'] < 1e-9)
pd.DataFrame({'nav': nav, 'e50': bench}).to_csv(S.TMP / 'mask_nav.csv')
out['A2_R55_main'] = a2
# ---- A3
G = S.rev_rule(P, 'MASK')
z0 = F['S']['STD'].to_numpy(); z1 = G['S']['STD'].to_numpy()
out['A3_mask_equals_engine'] = dict(same_nan=bool((np.isfinite(z0) == np.isfinite(z1)).all()), max_abs_diff=float(np.nanmax(np.abs(z0 - z1))),
                                    bitwise=bool(np.array_equal(z0, z1, equal_nan=True)), same_av=bool(F['av'] == G['av']))
out['runtime_s'] = round(time.time() - t0, 1); out['maxrss_mb'] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
(S.HERE / 'accept.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
