#!/usr/bin/env python3
"""S81 驗收（只跑現行格，不看任何新規則的成績）：
A1 make_signal(現行格) 與 s10lib.make_signal_s10(B2 SURGE) 的 sig／X(E0)／info 逐格相同。
A2 現行格淨值與 S75 存檔 nav_2008_2026.csv.gz 的 V_BRIDGE 逐日相同；年化 23.52%、期末 53,078,883.65 元。
A3 新掛鉤在現行格上不改結果：(a) simulate_s47(ndiv=10, slots=10) 逐日相同；(b) 「X＝不合格、RQ＝前 20 名、requal=1」的寫法逐日相同（＝最少持有期機制在 1 期時退化成現行出場）。
A4 搜尋空間格數與鄰近格。
python3 -I -B accept.py → accept.json"""
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import s81lib as Q
from s81lib import K, s10lib, L
HERE = Path(__file__).resolve().parent
t0 = time.time()
P, U = K.load_cont()
F, RT = K.surge(P, 'B2')
sg0 = s10lib.make_signal_s10(P, U, RT, shift=0)
cells = Q.grid(); cur = [c for c in cells if c['id'] == Q.CUR][0]
sg = Q.cell_signal(P, U, RT, cur)
a1 = dict(sig_same=bool((sg['sig'] == sg0['sig']).all()), X_same=bool((sg['X'] == sg0['XV']['E0']).all()),
          info_same=[(x['k'], x['targets']) for x in sg['info']] == [(x['k'], x['targets']) for x in sg0['info']], n_months=len(sg['info']))
t1 = time.time()
nav, lg, st = Q.simulate(P, sg, U, 10)
t2 = time.time()
ref = pd.read_csv(K.HERE / 'nav_2008_2026.csv.gz', index_col=0, parse_dates=True)['V_BRIDGE']
d = (nav.round(2) - ref.reindex(nav.index)).abs()
a2 = dict(max_abs_diff_twd=float(d.max()), final_nav=float(nav.iloc[-1]), cagr=K.cagr(nav), n_closed=int((lg['why'] != 'open_end').sum()),
          final_ok=bool(abs(nav.iloc[-1] - 53078883.65) < 0.01), cagr_ok=bool(abs(K.cagr(nav) - 0.2352) < 5e-5))
kw = dict(cost=s10lib.COSTS['strict'], capital=1e6, cap_pct=0.05, sat_w=0.7, sizing='W10', lock=False)
nav_b, _, _ = L.simulate_s47(P, sg, U, ndiv=10, slots=10, **kw)
sgr = Q.make_signal(P, U, RT, depth=10, mom_lo=0.10, keep=20, hold=2); sgr['hold'] = 1     # 借 hold≥2 的寫法（X＝不合格、RQ＝前 20 名），用 requal=1 跑
nav_c, lg_c, _ = Q.simulate(P, sgr, U, 10, force_requal=True)
a3 = dict(s47_ndiv10_max_abs_diff=float((nav_b - nav).abs().max()), requal1_max_abs_diff=float((nav_c - nav).abs().max()), requal1_n_closed=int((lg_c['why'] != 'open_end').sum()))
nb = Q.neighbors(cells)
a4 = dict(n_cells=len(cells), n_unique=len({c['id'] for c in cells}), cur_in_grid=True, min_neighbors=min(len(v) for v in nb.values()), max_neighbors=max(len(v) for v in nb.values()),
          symmetric=all(a in nb[b] for a, v in nb.items() for b in v))
ok = all(v for k, v in a1.items() if k.endswith('same')) and d.max() < 0.011 and a2['final_ok'] and a2['cagr_ok'] and a3['s47_ndiv10_max_abs_diff'] == 0 and a3['requal1_max_abs_diff'] == 0 and a4['n_cells'] <= 200 and a4['symmetric']
out = dict(A1=a1, A2=a2, A3=a3, A4=a4, sim_seconds=round(t2 - t1, 2), passed=bool(ok), runtime_s=round(time.time() - t0, 1))
(HERE / 'accept.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
print(json.dumps(out, ensure_ascii=False, indent=1))
assert ok, '驗收不過，不准往下做'
