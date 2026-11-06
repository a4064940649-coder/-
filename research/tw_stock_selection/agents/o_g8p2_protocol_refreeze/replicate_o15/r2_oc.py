# OC 全格重做（自選 seed 集，P＝100,000／格），V1／V4 用協定凍結 b、c_d；V3 用我自己依同規則校準的 b、c_d
import json, sys, os
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rlib as L

BASE = int(sys.argv[1]) if len(sys.argv) > 1 else 15150711
P = int(sys.argv[2]) if len(sys.argv) > 2 else 100000
CAL = json.load(open(os.path.join(L.HERE, 'r1_calib.json')))
B3, CD3 = CAL['V3']['b_own'], CAL['V3']['cd_seeds'][0]
DG = {'D1': 1, 'D2': 2, 'D3': 3, 'D3x': 4, 'D4': 5}
ALPHAS = [-1.0, -0.5, 0.0, 0.5, 1.0, 2.0]


def job(a):
    kind, dgp, si, sval, sidx, ai, mode = a
    sigma = L.SIGMAS[si]
    alpha = ALPHAS[ai]
    rng = L.cell_rng(BASE, DG[dgp], si, sidx, ai, mode)
    mu = alpha + sval - (L.DW if mode == 0 else 0.0)
    X = L.gen(dgp, sigma, mu, P, rng)
    o = {}
    r1 = L.eval_v(X, L.B_P, L.CD_P, 0.0)
    o['V1_up'] = r1['up'].mean(); o['V1_fail'] = r1['fail'].mean(); o['V1_dn'] = r1['dn'].mean()
    o['V1_dnraw'] = r1['dn_raw'].mean(); o['V1_up12'] = r1['up12'].mean(); o['lit'] = r1['lit'].mean()
    if mode == 0:
        r3 = L.eval_v(X, B3, CD3, 0.2)
        o['V3_up'] = r3['up'].mean(); o['V3_dn'] = r3['dn'].mean()
        r4 = L.eval_v(X, L.B_P, L.CD_P, 0.0, upgrade=False)
        o['V4_up'] = r4['up'].mean(); o['V4_dn'] = r4['dn'].mean()
    else:
        g = L.eval_g8p1(X)
        o['G8P1_up'] = g['up'].mean(); o['G8P1_fail'] = g['fail'].mean()
    o = {k: float(v) for k, v in o.items()}
    key = f"{kind}|{dgp}|{L.SLAB[si]}|a{alpha}|s{sval}|m{mode}"
    return key, o


def jobs():
    J = []
    for dgp in ('D1', 'D2', 'D3', 'D3x', 'D4'):
        for si in range(9):
            for ki, s in enumerate(L.SGRID):
                J.append(('main', dgp, si, s, ki, 2, 0))
                J.append(('strict', dgp, si, s, ki, 2, 1))
    for dgp in ('D1', 'D2', 'D3'):
        for si in range(9):
            for ai in range(6):
                if ai == 2:
                    J.append(('dp', dgp, si, L.DW, 5, ai, 0))   # α＝0、s＝Δ_w
                    continue  # s＝0 α＝0 即 main 格
                for sval, sidx in ((0.0, 0), (L.DW, 5)):
                    J.append(('dp', dgp, si, sval, sidx, ai, 0))
    J.append(('strict', 'D1', 0, 0.3, 6, 2, 1))
    return J


if __name__ == '__main__':
    J = jobs()
    with Pool(4) as p:
        res = dict(p.map(job, J, chunksize=4))
    json.dump(dict(base=BASE, P=P, B3=B3, CD3=CD3, cells=res),
              open(os.path.join(L.HERE, f'r2_oc_{BASE}_{P}.json'), 'w'), indent=0)
    print(len(res), 'cells')
