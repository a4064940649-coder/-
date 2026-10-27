#!/usr/bin/env python3
import sys, json, math
sys.path.insert(0, '/home/user/-/research/tw_stock_selection/agents/o_infra_v3_rebuild/replicate_o3')
import r3_lib as L
sys.path.pop(0)
import numpy as np, pandas as pd
K, V = L.K, L.V
A = pd.concat([pd.read_parquet(L.HERE / 'my_navs_P0.parquet'), pd.read_parquet(L.HERE / 'my_navs_P1.parquet')], axis=1)
LG = {**json.loads((L.HERE / 'my_log_P0.json').read_text()), **json.loads((L.HERE / 'my_log_P1.json').read_text())}
YL, Y5 = L.load_rev(0)
s = L.S3(YL, Y5, True, 'v2'); t0, t1, sd = s.bounds()
SD = sd
def st(key, n='n10', cost='normal'):
    nav = A[f'{key}|{n}|{cost}']; ew = A[f'{key}|EW']
    return L.stats_all(nav, ew, SD)
def fmt(o, segs=('ALL', 'S1', 'S2', 'S3', 'S4', 'P1')):
    return ' | '.join(f"{g} {o[g]['CAGR']:.2f} (t {o[g]['t_0050']:.2f}, tEW {o[g]['t_EW']:.2f}, DD {o[g]['maxDD']:.1f}, cal {o[g]['t_0050_cal']:.2f}, Calmar {o[g]['Calmar']:.2f})" for g in segs)
out = {}
keys = {'v2': 'P0|R0|X0|Dv2', '+R': 'P0|R1|X0|Dv2', '+X': 'P0|R1|X1|Dv2', '+P': 'P1|R1|X1|Dv2', 'V3': 'P1|R1|X1|Dhi',
        'V3_lo': 'P1|R1|X1|Dlo', 'V3-D': 'P1|R1|X1|Dnone'}
for nm, k in keys.items():
    o = st(k + '|EXFIN'); out[nm] = o
    print(nm, fmt(o))
o = st('P1|R1|X1|Dhi|EXFIN', cost='stress'); out['V3_stress'] = o; print('V3 stress', fmt(o, ('ALL',)))
o = st('P0|R0|X0|Dv2|EXFIN', cost='stress'); out['v2_stress'] = o; print('v2 stress', fmt(o, ('ALL',)))
print('V3-D S3 minus v2 S3', out['V3-D']['S3']['CAGR'] - out['v2']['S3']['CAGR'])
for U, n in (('ALL', 'n10'), ('EXFIN', 'n30'), ('ALL', 'n30')):
    a = st(f'P1|R1|X1|Dhi|{U}', n); b = st(f'P0|R0|X0|Dv2|{U}', n)
    out[f'V3|{U}|{n}'] = a; out[f'v2|{U}|{n}'] = b
    print(U, n, 'V3', fmt(a, ('ALL',)), ' v2', fmt(b, ('ALL',)))
# 全因子主效果
def cagr(k, n='n10', U='EXFIN'):
    nav = A[f'{k}|{U}|{n}|normal']; ew = A[f'{k}|{U}|EW']
    return V.seg_stats(nav, ew, *V.SEGS['ALL'][:2], SD, ad=V.SEGS['ALL'][2])['CAGR']
for n in ('n10', 'n30'):
    for U in ('EXFIN', 'ALL'):
        F = {}
        for P in (0, 1):
            for R in (0, 1):
                for X in (0, 1):
                    for D in ('v2', 'hi', 'lo', 'none'):
                        F[(P, R, X, D)] = cagr(f'P{P}|R{R}|X{X}|D{D}', n, U)
        fac = ['P', 'R', 'X', 'D']
        def lv(key, f, hi):
            i = fac.index(f)
            return (key[i] == 'hi') if f == 'D' else (key[i] == 1) if hi else None
        cells = {k: v for k, v in F.items() if k[3] in ('v2', 'hi')}
        def isH(k, f):
            i = fac.index(f); return (k[i] == 'hi') if f == 'D' else k[i] == 1
        me = {}
        for f in fac:
            me[f] = np.mean([v for k, v in cells.items() if isH(k, f)]) - np.mean([v for k, v in cells.items() if not isH(k, f)])
        inter = {}
        for i in range(4):
            for j in range(i + 1, 4):
                f, g = fac[i], fac[j]
                dHH = np.mean([v for k, v in cells.items() if isH(k, f) and isH(k, g)]); dHL = np.mean([v for k, v in cells.items() if isH(k, f) and not isH(k, g)])
                dLH = np.mean([v for k, v in cells.items() if not isH(k, f) and isH(k, g)]); dLL = np.mean([v for k, v in cells.items() if not isH(k, f) and not isH(k, g)])
                inter[f + 'x' + g] = ((dHH - dLH) - (dHL - dLL))   # 差中差
        out[f'factorial|{U}|{n}'] = {'main': me, 'inter_dd': inter, 'cells': {'|'.join(map(str, k)): v for k, v in F.items()}}
        print('factorial', U, n, {k: round(v, 3) for k, v in me.items()}, {k: round(v, 3) for k, v in inter.items()})
# 拆解：月均超額
def exm(k):
    nav = A[f'{k}|EXFIN|n10|normal']
    D = pd.DatetimeIndex([d for d in SD if d in nav.index]); nd = nav.reindex(D); pr = (nd.shift(-1) / nd - 1).iloc[:-1]
    bd = K.E50.reindex(D); bp = (bd.shift(-1) / bd - 1).iloc[:-1]
    x = (pr - bp); x = x[(x.index >= '2006-04-01') & (x.index <= '2026-10-12')].dropna() * 100
    return x
x0, x3 = exm(keys['v2']), exm(keys['V3'])
dec = {'mean_v2': x0.mean(), 'mean_V3': x3.mean(), 'sd_v2': x0.std(ddof=1), 'sd_V3': x3.std(ddof=1), 'n': len(x3),
       't_v2': x0.mean() / x0.std(ddof=1) * math.sqrt(len(x0)), 't_V3': x3.mean() / x3.std(ddof=1) * math.sqrt(len(x3)),
       't_mean_only': x3.mean() / x0.std(ddof=1) * math.sqrt(len(x3))}
dec['vol_contrib'] = dec['t_V3'] - dec['t_mean_only']
out['decomp'] = dec; print('decomp', {k: round(v, 4) for k, v in dec.items()})
# 凍結事件
fz = {}
for k in ('P0|R0|X0|Dv2', 'P0|R1|X0|Dv2', 'P1|R1|X1|Dhi', 'P0|R1|X1|Dv2', 'P1|R1|X0|Dhi'):
    for U in ('EXFIN', 'ALL'):
        for n in ('n10|normal', 'n30|normal', 'EW'):
            g = LG[f'{k}|{U}|{n}']
            fz[f'{k}|{U}|{n}'] = (g['frozen_n'], round(g['frozen_max'], 3), g['frozen_codes'], [c[:3] for c in g['conv']])
for k, v in fz.items(): print('frozen', k, v)
out['frozen'] = fz
json.dump(out, open(L.HERE / 'my_summary.json', 'w'), ensure_ascii=False, indent=1, default=float)
