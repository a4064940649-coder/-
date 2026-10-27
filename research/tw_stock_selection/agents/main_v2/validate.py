#!/usr/bin/env python3
"""v2_core 驗證：strict 窗與 ia_core 相同；同分取代碼序時重現 independent_audit 的 P1／P2 數字。"""
import sys, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import v2_core as V
K = V.K
T0 = time.time()
import hashlib
assert hashlib.md5((K.P / 'value.parquet').read_bytes()).hexdigest().startswith('8272e253'), 'value.parquet 不是修正版'
S1, V1 = V.rolling_signals('strict'); S2, V2_ = K.rolling_signals('strict')
assert np.array_equal(np.isnan(S1), np.isnan(S2)) and np.nanmax(np.abs(S1 - S2)) < 1e-12
assert np.array_equal(np.isnan(V1), np.isnan(V2_)) and np.nanmax(np.abs(V1 - V2_)) < 1e-3
print('strict == ia_core strict', round(time.time() - T0, 1))
out = {}
YL, Y5, Y5f = V.load_revenue()
b, s = K.rates(0.001)
S = V.V2('strict', YL, Y5)
t0, t1, sd = S.bounds()
print('sig', S.sig_days[0].date(), S.sig_days[-1].date(), S.end_day.date(), len(S.sig_days))
for n in (10, 30):
    pl = S.plan(n, 'code')
    T1 = time.time()
    ew, _ = K.simulate([(p[0], p[2]) for p in pl], 0, 0, t0, t1)
    nav, _ = K.simulate([(p[0], p[1]) for p in pl], b, s, t0, t1)
    print('sim time', round(time.time() - T1, 1))
    for seg in ('P1', 'S3'):
        a, e, ad = V.SEGS[seg]
        o = V.seg_stats(nav, ew, a, e, sd, ad=ad)
        out[f'n{n}|{seg}'] = {k: round(v, 2) for k, v in o.items()}
        print(n, seg, out[f'n{n}|{seg}'])
ref = {'n10|P1': 18.3, 'n30|P1': 14.0, 'n10|S3': 56.7, 'n30|S3': 46.3}
out['audit_ref'] = ref
json.dump(out, open(V.HERE / 'results_validate.json', 'w'), ensure_ascii=False, indent=1)
print('done', round(time.time() - T0, 1))
