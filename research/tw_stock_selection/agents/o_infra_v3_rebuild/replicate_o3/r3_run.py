#!/usr/bin/env python3
"""python3 -I -B r3_run.py P0|P1 ：全因子 R×X×D × 全市場/排除金融 × n10/n30/EW × 一般/壓力。"""
import sys, json, time
sys.path.insert(0, '/home/user/-/research/tw_stock_selection/agents/o_infra_v3_rebuild/replicate_o3')
import r3_lib as L
sys.path.pop(0)
import numpy as np, pandas as pd
K, V = L.K, L.V
P = sys.argv[1]
tie = sys.argv[2] if len(sys.argv) > 2 else 'first'
if P == 'P1': print('patched rows', L.apply_patch())
navs = {}; log = {}
t00 = time.time()
for R in (0, 1):
    YL, Y5 = L.load_rev(R)
    for D in ('v2', 'lo', 'hi', 'none'):
        for U in ('ALL', 'EXFIN'):
            s = L.S3(YL, Y5, U == 'EXFIN', D, hi_tie=tie)
            t0, t1, sd = s.bounds()
            pls = {n: s.plan(n) for n in (10, 30)}
            for X in (0, 1):
                xr = L.XRules('main') if X else None
                pre = f'{P}|R{R}|X{X}|D{D}|{U}'
                ew, lg = L.simulate(L.ew_plan(pls[10]), 0.0, 0.0, t0, t1, xr)
                navs[pre + '|EW'] = ew
                log[pre + '|EW'] = {'frozen_n': len(lg['frozen']), 'frozen_max': max([f[2] for f in lg['frozen']], default=0), 'conv': lg['conv'],
                                    'frozen_codes': sorted({c for f in lg['frozen'] for c in f[1]})}
                for n in (10, 30):
                    for cost in ('normal', 'stress'):
                        b, se = K.rates() if cost == 'normal' else K.rates('stress')
                        nav, lg = L.simulate(pls[n], b, se, t0, t1, xr)
                        navs[f'{pre}|n{n}|{cost}'] = nav
                        log[f'{pre}|n{n}|{cost}'] = {'frozen_n': len(lg['frozen']), 'frozen_max': max([f[2] for f in lg['frozen']], default=0),
                                                     'conv': lg['conv'], 'turnover': lg['turnover'], 'nblk': None,
                                                     'frozen_codes': sorted({c for f in lg['frozen'] for c in f[1]})}
            log[f'{P}|R{R}|D{D}|{U}|extra_removed'] = {str(k.date()): v for k, v in s.extra_removed.items()}
            log[f'{P}|R{R}|D{D}|{U}|picks10'] = {str(p[3].date()): [K.COLS[j] for j in p[1]] for p in pls[10]}
            log[f'{P}|R{R}|D{D}|{U}|picks30'] = {str(p[3].date()): [K.COLS[j] for j in p[1]] for p in pls[30]}
            print(R, D, U, round(time.time() - t00), flush=True)
df = pd.DataFrame(navs)
df.to_parquet(L.HERE / f'my_navs_{P}{"" if tie=="first" else "_"+tie}.parquet')
(L.HERE / f'my_log_{P}{"" if tie=="first" else "_"+tie}.json').write_text(json.dumps(log, ensure_ascii=False, default=str), encoding='utf-8')
ref = pd.read_parquet(L.O3 / f'navs_{P}.parquet')
bad = []
for c in df.columns:
    if c in ref.columns:
        dd = float((df[c] / ref[c].reindex(df.index) - 1).abs().max())
        if dd > 1e-9: bad.append((c, dd))
print('columns', len(df.columns), 'mismatch >1e-9:', len(bad))
for c, dd in bad: print('  ', c, dd)
