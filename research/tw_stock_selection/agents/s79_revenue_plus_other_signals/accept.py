#!/usr/bin/env python3
"""S79 驗收（不含任何新規則的績效）：A1 spec={} 的訊號與 s10lib.make_signal_s10 逐格相同（4 個位移）；A2 標準路徑淨值與 S75 存檔 V_BRIDGE 逐日相同；
A3 特徵可用性（各特徵在 2008-01 第一個訊號日的股池內有值比例）。python3 -I -B accept.py → accept.json"""
import json, sys, hashlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import s79lib as Q
from s79lib import K
HERE = Path(__file__).resolve().parent
P, U, F, RT = Q.load()
FE = Q.features(P, U, F)
out = {'panel_sha256': {f: hashlib.sha256((Q.TMP / 'cache_cont' / f).read_bytes()).hexdigest() for f in ('panel_cs.npz', 'e50_ohlc.csv', 'innov.npz')}}
a1 = []
for s in Q.SHIFTS:
    a = K.signal(P, U, RT, s); b = Q.make_signal(P, U, RT, FE, {}, shift=s)
    a1.append(dict(shift=s, sig_equal=bool((a['sig'] == b['sig']).all()), X_equal=bool((a['X'] == b['X']).all()),
                   rank_equal=bool(np.array_equal(a['rank'], b['rank'], equal_nan=True)), n_months=len(b['info'])))
out['A1'] = a1
nav, lg, st = K.sim(P, Q.make_signal(P, U, RT, FE, {}), U, 1e6)
ref = pd.read_csv(Q.AG / 's75_continuous_2008_2026' / 'nav_2008_2026.csv.gz', index_col=0, parse_dates=True)['V_BRIDGE']
out['A2'] = dict(max_abs_diff_vs_S75_nav=float((nav.round(2) - ref).abs().max()), final=float(nav.iloc[-1]), cagr=K.cagr(nav))
D, M, IND = FE
k = RT['av'][pd.Period('2008-01', 'M')]; u = U[k]
out['A3'] = dict(first_signal_row=str(P.IDX[k].date()), univ=int(u.sum()),
                 daily={n: float(np.isfinite(np.asarray(x[k], float))[u].mean()) for n, x in D.items()},
                 monthly={n: float(x.loc[pd.Period('2008-01', 'M')].notna().to_numpy()[u].mean()) for n, x in M.items()},
                 industry_known_in_univ=float((IND[u] >= 0).mean()), n_industries=int(IND.max() + 1))
out['pass'] = all(x['sig_equal'] and x['X_equal'] and x['rank_equal'] for x in a1) and out['A2']['max_abs_diff_vs_S75_nav'] < 0.011
(HERE / 'accept.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
print(json.dumps(out, ensure_ascii=False, indent=1))
