#!/usr/bin/env python3
"""o16 反方審查（成本與可成交角度）獨立驗證：不 import 研究員程式，只讀 G8_PROTOCOL_v2a.json 參數。
重算 A1 合併規則（v2 與 A1 取較不利）之 24 個月誤升級：主口徑、嚴格口徑、網格外更強相依與更高成本缺口。"""
import json, math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
P_ = json.load(open(os.path.join(HERE, 'G8_PROTOCOL_v2a.json')))['params']
SD, C, CD, BV2, BA, DW = P_['sigma_design'], P_['c'], P_['c_d'], P_['b_v2'], P_['b_a'], P_['delta_w']

def gen(kind, par, mu, sig, P, rng, H=24):
    if kind == 'iid':
        e = rng.standard_normal((P, H))
    elif kind == 'MA':
        u = rng.standard_normal((P, H + 1)); e = (u[:, 1:] + par * u[:, :-1]) / math.sqrt(1 + par * par)
    elif kind == 'AR':
        u = rng.standard_t(5, (P, H)) / math.sqrt(5 / 3); e = np.empty((P, H)); e[:, 0] = u[:, 0]
        for t in range(1, H): e[:, t] = par * e[:, t - 1] + math.sqrt(1 - par * par) * u[:, t]
    return mu + sig * e

def states(X):
    P = X.shape[0]
    out = {}
    for name, den, b in (('v2', 'v2', BV2), ('A1', 'A', BA)):
        st = np.zeros(P, int)  # 0 validating, -1 failed, 1 pending, 2 S1
        lit_fail = X[:, :12].sum(1) <= 0
        st[lit_fail] = -1
        for n in (12, 18, 24):
            Xn = X[:, :n]; S = Xn.sum(1); sn = Xn.std(1, ddof=1)
            sg = np.maximum(C * SD, sn) if den == 'v2' else C * np.maximum(SD, sn)
            z = S / (sg * math.sqrt(n))
            act = st == 0
            st[act & (z <= -CD)] = -1
            if n in (12, 24):
                act = st == 0
                st[act & (z >= b * math.sqrt(24 / n))] = 1
            if n == 24:
                st[st == 1] = 2
        out[name] = st
    comb = np.minimum(out['v2'], out['A1'])
    return {k: float((v == 2).mean()) for k, v in (('v2', out['v2']), ('A1_only', out['A1']), ('comb', comb))}

rng = np.random.default_rng(20261107)
P = 200000
res = {}
cells = []
for sig in (8.0, 9.5, 10.4, 12.0):
    for kind, par in (('MA', 0.3), ('MA', 0.4), ('MA', 0.5), ('AR', 0.1), ('AR', 0.15), ('AR', 0.2), ('iid', 0)):
        for lab, mu in (('main_s0.78', 0.78 - DW), ('strict_s0.78', 0.78), ('gap0.5_over_stress', 0.5), ('gap0.25_over_stress', 0.25)):
            cells.append((sig, kind, par, lab, mu))
for sig, kind, par, lab, mu in cells:
    X = gen(kind, par, mu, sig, P, rng)
    r = states(X); key = f'{kind}{par}|{sig}|{lab}'
    res[key] = r
    print(key, {k: round(v, 4) for k, v in r.items()}, 'SE', round(math.sqrt(r['comb'] * (1 - r['comb']) / P), 4))
json.dump(res, open(os.path.join('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad', 'review_o16_out.json'), 'w'), indent=1)
