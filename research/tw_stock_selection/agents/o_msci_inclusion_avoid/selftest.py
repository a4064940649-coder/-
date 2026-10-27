"""引擎自我檢查：無改版時與 main_v2 正式淨值逐日相同。不涉及任何事件報酬。"""
import time, json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
T0 = time.time()
import numpy as np, pandas as pd
import oa_lib as O
K, V = O.K, O.V
S = O.strategy(True)
t0, t1, sd = S.bounds()
out = {}
for n in (10, 30):
    pl, _ = O.plan_mod(S, n)
    ref = S.plan(n, 'official')
    assert all((a[0] == b[0]) and np.array_equal(a[1], b[1]) for a, b in zip(pl, ref)) and len(pl) == len(ref)
    nav, _ = O.simulate_mod(pl, *K.rates(0.001), t0, t1)
    navK, _ = K.simulate([(p[0], p[1]) for p in ref], *K.rates(0.001), t0, t1)
    off = pd.read_parquet(O.AG / 'main_v2' / 'navs_gapfix.parquet')[f'EXFIN|n{n}|normal']
    out[n] = {'max_abs_rel_diff_vs_Ksim': float(np.max(np.abs(nav / navK - 1))), 'max_abs_rel_diff_vs_main_v2': float(np.nanmax(np.abs(nav / off.reindex(nav.index) - 1)))}
    print(n, out[n], time.time() - T0)
json.dump(out, open(O.HERE / 'selftest.json', 'w'), indent=1)
