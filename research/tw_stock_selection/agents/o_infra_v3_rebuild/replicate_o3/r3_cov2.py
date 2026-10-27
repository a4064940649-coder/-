#!/usr/bin/env python3
"""B10 情境（同池等權、無 X）的永久停止交易事件：代碼數、事件加權的型態已知／對價可用／新聞核對比率。"""
import sys, json
sys.path.insert(0, '/home/user/-/research/tw_stock_selection/agents/o_infra_v3_rebuild/replicate_o3')
import r3_lib as L, r3_bounds_lib as B
sys.path.pop(0)
import numpy as np, pandas as pd
K = L.K
patch = sys.argv[1] == 'P1'
if patch: L.apply_patch()
YL, Y5 = L.load_rev(1)
mc = L.MC.set_index('code'); xr = L.XRules('main')
out = {}
for U in ('ALL', 'EXFIN'):
    for D in ('v2', 'hi'):
        sb = B.SB(YL, Y5, U == 'EXFIN', D, 1.0); t0, t1, sd = sb.bounds()
        for n in (10, 30, 'EW'):
            pl = sb.plan(n if n != 'EW' else 10)
            if n == 'EW': pl = L.ew_plan(pl)
            nav, lg = L.simulate(pl, 0.0, 0.0, t0, t1, None) if n == 'EW' else L.simulate(pl, *K.rates(), t0, t1, None)
            ev = []
            for d, cs, sh in lg['frozen']:
                t = K.IDX.get_loc(pd.Timestamp(d))
                for c in cs:
                    if xr.is_dead(t)[K.CI[c]]: ev.append(c)
            if not ev: out[f'{U}|{D}|{n}'] = {'dead_events': 0}; continue
            typ = [mc.at[c, 'type'] if c in mc.index else 'unknown' for c in ev]
            use = [(t in ('swap', 'failure')) or (t == 'cash' and np.isfinite(mc.at[c, 'cash'])) for c, t in zip(ev, typ)]
            news = [c in mc.index and mc.at[c, 'verified'] == 'news' for c in ev]
            out[f'{U}|{D}|{n}'] = {'dead_events': len(ev), 'dead_codes': len(set(ev)), 'type_known': float(np.mean([t != 'unknown' for t in typ])),
                                  'usable': float(np.mean(use)), 'news': float(np.mean(news))}
            print(U, D, n, out[f'{U}|{D}|{n}'], flush=True)
(L.HERE / f'my_cov2_{sys.argv[1]}.json').write_text(json.dumps(out, indent=1))
