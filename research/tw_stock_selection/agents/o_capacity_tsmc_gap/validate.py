#!/usr/bin/env python3
"""檢查：V2R(1,300) 與 v2 EXFIN 的名單相同；simulate_w(等權) 與 ia_core.simulate 相同；淨值與 main_v2 navs_gapfix 相同。
只比對 main_v2 已公開的跨循環數字，不看任何新版本的結果。"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import ocore as O
K, V = O.K, O.V

S0 = V.V2('gapfix', O.YL, O.Y5, exfin=True)
S1 = O.V2R(1, 300)
out = {}
for n in (10, 30):
    p0 = S0.plan(n); p1 = S1.plan(n)
    same = len(p0) == len(p1) and all((a[0] == b[0]) and np.array_equal(a[1], b[1]) and np.array_equal(a[2], b[2]) for a, b in zip(p0, p1))
    out[f'plan_same_n{n}'] = bool(same)
    t0, t1, sd = S1.bounds()
    nav0, _ = K.simulate([(p[0], p[1]) for p in p1], *O.NORMAL, t0, t1)
    navw, _ = O.simulate_w(O.tsmc_plan(p1, 0.0), *O.NORMAL, t0, t1)
    out[f'simw_maxabs_n{n}'] = float((nav0 / navw - 1).abs().max())
    ref = pd.read_parquet(O.HERE.parent / 'main_v2' / 'navs_gapfix.parquet')[f'EXFIN|n{n}|normal'].dropna()
    c = nav0.reindex(ref.index).dropna()
    out[f'vs_main_v2_nav_maxabs_n{n}'] = float((c / ref.reindex(c.index) - 1).abs().max())
    ew, _ = K.simulate([(p[0], p[2]) for p in p1], 0, 0, t0, t1)
    st = O.segstat(nav0, ew, sd, *O.periods(sd)['ALL'])
    out[f'ALL_n{n}'] = {k: st[k] for k in ('CAGR', 't_0050', 't_EW', 'maxDD')}
print(json.dumps(out, ensure_ascii=False, indent=1))
assert out['plan_same_n10'] and out['plan_same_n30']
assert out['simw_maxabs_n10'] < 1e-9 and out['vs_main_v2_nav_maxabs_n10'] < 1e-9
(O.HERE / 'results_validate.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
