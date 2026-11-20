#!/usr/bin/env python3
"""R75 第 1 步：正式引擎單獨跑官方面板（2015-07-01 起），逐位重現 48.76%／381 筆。"""
import sys, json, time, resource
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import r75lib as Q
from r75lib import R, L, stlib, s17lib, s10lib
import pandas as pd
t0 = time.time()
Q.setup('cache_is', '2015-07-01', '2016-01')
P, U, fix = Q.load_all()
F = s17lib.rev_family(P, defs=['STD'])
sg = s10lib.make_signal_s10(P, U, dict(S=F['S']['STD'], av=F['av']), shift=0)
nav, lg, st = L.sim(P, sg, U, cap=1e6)
lastb = lambda d: P.IDX[P.IDX < pd.Timestamp(d)][-1]
D0, DE, D19 = P.IDX[P.t0], P.IDX[-1], lastb('2019-01-01')
x = lg[lg['why'] != 'open_end']
a = dict(T=P.T, N=P.N, start=str(D0.date()), end=str(DE.date()), since2019=stlib.cagr(nav, D19, DE), full=stlib.cagr(nav, D0, DE), maxdd=stlib.maxdd(nav, D0, DE), n_closed=int(len(x)),
         final=float(nav.iloc[-1]))
a['bitwise_vs_ACC'] = bool(a['since2019'] == L.ACC['since2019'] and a['full'] == L.ACC['full'] and a['maxdd'] == L.ACC['maxdd'] and a['n_closed'] == L.ACC['n_closed'])
a['yearly'] = Q.yearly(nav)
a['runtime_s'] = round(time.time() - t0, 1); a['maxrss_mb'] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
pd.DataFrame({'nav': nav, 'e50': pd.Series(P.E50, index=P.IDX).reindex(nav.index)}).to_csv(Q.TMP / 'is_nav.csv')
pd.DataFrame(sg['info']).assign(targets=[' '.join(i['targets']) for i in sg['info']]).to_csv(Q.TMP / 'is_signals.csv', index=False)
lg.drop(columns=['j']).to_csv(Q.TMP / 'is_trades.csv', index=False)
(Q.TMP / 'step1.json').write_text(json.dumps(a, ensure_ascii=False, indent=1))
print(json.dumps(a, ensure_ascii=False, indent=1))
