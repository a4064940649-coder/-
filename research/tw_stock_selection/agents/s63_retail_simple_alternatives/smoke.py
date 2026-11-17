#!/usr/bin/env python3
"""S63 事前檢查（寫 SPEC／PREREG 之前跑）：只算 (1) 現行格驗收（官方 48.76%／381 筆逐位）、(2) 各格訊號層的計數（訊號月數、每月檔數），不算任何非現行格的報酬。
python3 -I -B smoke.py is|oos → scratchpad/s63/smoke_<which>.json"""
import sys, json, time, resource
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s63lib as X
import numpy as np
which = sys.argv[1]
t0 = time.time()
P, U, RT, Rv, fix = X.build(which)
out = dict(which=which, T=P.T, N=P.N, t0=str(P.IDX[P.t0].date()), last=str(P.IDX[-1].date()))
sg = X.s10lib.make_signal_s10(P, U, RT, shift=0)
cnt = lambda s: dict(n_rows=len(s['info']), first=s['info'][0]['sig_date'], last=s['info'][-1]['sig_date'], mean_tgt=float(np.mean([x['n_tgt'] for x in s['info']])),
                     zero_months=int(sum(x['n_tgt'] == 0 for x in s['info'])))
out['C'] = cnt(sg)
out['D'] = cnt(X.s10lib.make_signal_s10(P, U, X.rt_quarterly(P, RT), shift=0))
out['E'] = cnt(X.s10lib.make_signal_s10(P, U, RT, shift=0, nsel=5))
sf = X.make_signal_lazy(P, U, RT, Rv, shift=0)
out['F'] = cnt(sf); out['F']['mean_qualified'] = float(np.mean([x['n_ma'] for x in sf['info']]))
assert [x['k'] for x in sf['info']] == [x['k'] for x in sg['info']]
if which == 'is':
    t1 = time.time()
    nav, lg, st = X.sim(P, sg, U)
    out['sim_secs'] = round(time.time() - t1, 1)
    D19 = P.IDX[P.IDX < '2019-01-01'][-1]
    acc = dict(since2019=X.stlib.cagr(nav, D19, P.IDX[-1]), full=X.stlib.cagr(nav, P.IDX[P.t0], P.IDX[-1]), maxdd=X.stlib.maxdd(nav, P.IDX[P.t0], P.IDX[-1]),
               n_closed=int((lg['why'] != 'open_end').sum()))
    out['acc'] = acc; out['acc_bitwise'] = bool(all(acc[k] == X.L.ACC[k] for k in X.L.ACC))
out['secs'] = round(time.time() - t0, 1); out['rss_mb'] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
(X.SCR / f'smoke_{which}.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
print(json.dumps(out, ensure_ascii=False, indent=1))
