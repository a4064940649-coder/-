#!/usr/bin/env python3
"""AB：只用已公告營收（V.make_unavail('mc', seed)），seed 0..19，各自跑原版與 A20，平均淨值後算統計。"""
import sys, json, pickle
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import r4_core as C
K, V = C.K, C.V
NORMAL = K.rates(0.001)
navs = {'BASE': [], 'A20': []}; sd0 = None
for seed in range(20):
    S = C.make_S(unavail=V.make_unavail('mc', seed))
    t0, t1, sd = S.bounds()
    if sd0 is None: sd0 = sd
    assert sd == sd0
    for v, kb in (('BASE', None), ('A20', 20)):
        pl = C.plan_buffer(S, 10, kb)
        nav, _ = C.sim([(p['e'], p['picks']) for p in pl], *NORMAL, t0, t1)
        navs[v].append(nav)
    print(seed, flush=True)
out = {}
for v in navs:
    nav = pd.concat(navs[v], axis=1).mean(axis=1)
    for per in ('ALL', 'HOLD', 'SEL'):
        a, b = C.PER[per]
        cg, dd = C.cagr_dd(nav, a, b)
        x = C.sel(C.winret(nav, sd0) - C.winret(C.E50.reindex(nav.index), sd0), a, b)
        out[f'{v}|{per}'] = dict(CAGR=cg, maxDD=dd, t=C.tstat(x))
    # 個別 seed 的 ALL 平均（對照）
    out[f'{v}|ALL_seedmean'] = float(np.mean([C.cagr_dd(n, *C.PER['ALL'])[0] for n in navs[v]]))
print(json.dumps(out, indent=1, default=float))
(HERE / 'r_ab.json').write_text(json.dumps(out, indent=1, default=float), encoding='utf-8')
