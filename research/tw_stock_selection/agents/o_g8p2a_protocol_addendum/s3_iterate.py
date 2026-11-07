#!/usr/bin/env python3
"""步驟 3：集 A 校準（PREREG_DESIGN 之後）：495 格 × P＝200,000，各版本 b_a＝max(v2 b, max_cells q_0.9525(B*_comb)) → iterations.json。
A2 的「所在格 OC 最大誤升級 ≤ 0.05」在集 A 只用 σ_checked ≤ 10.4（格表要等集 B）；若集 B 有格 > 0.05，A2 本來就不合格。
用法：python3 -I -B s3_iterate.py"""
import sys, json, math, time
from pathlib import Path
from multiprocessing import Pool
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
import g8alib as L

SEED_A = 2026110701; P_A = 200_000; Q = 0.9525


def cell(args):
    dgp, si, ki = args
    rng = L.cell_rng(SEED_A, dgp, si, ki, 2, 0)
    X = L.gen(dgp, L.S_GRID[ki] - L.DW, L.SIG_GRID[si], P_A, 24, rng)
    Z = L.all_stats(X)
    out = dict(cell=f'{dgp}|{L.SIG_LABEL[si]}|{L.S_GRID[ki]}')
    o2 = L.outcomes_rule(Z['lit'], Z['v212'], Z['v218'], Z['v224'], L.B_V2, L.CD_V2)
    out['v2_up_setA'] = float(o2['up'].mean())
    for ver in ('A1', 'A2', 'A3'):
        out[f'q_{ver}'] = float(np.quantile(L.bstar_combined(Z, ver), Q))
    return out


if __name__ == '__main__':
    L.check_reg('PREREG_DESIGN')
    t0 = time.time()
    jobs = [(d, si, ki) for d in L.DGPS for si in range(9) for ki in range(5)]
    with Pool(4) as pool:
        rows = pool.map(cell, jobs, chunksize=4)
    by = {r['cell']: r for r in rows}
    it = dict(utc=L.now_utc(), seed_A=SEED_A, P_A=P_A, quantile=Q, n_cells=len(rows), runtime_s=round(time.time() - t0, 1), versions={})
    for ver in ('A1', 'A2', 'A3'):
        qs = {k: v[f'q_{ver}'] for k, v in by.items()}
        top = sorted(qs, key=qs.get, reverse=True)[:8]
        bmax = max(qs.values())
        by_dgp = {d: round(max(v for k, v in qs.items() if k.split('|')[0] == d), 4) for d in L.DGPS}
        it['versions'][ver] = dict(b_a=max(L.B_V2, bmax), max_cell_q=bmax, floor_binding=bool(bmax < L.B_V2), driving_cells=[(k, round(qs[k], 4)) for k in top],
                                   max_q_by_dgp=by_dgp, n_cells_q_above_v2b=int(sum(v > L.B_V2 for v in qs.values())))
    it['versions']['A4'] = dict(b_a=None, note='沒有升級路徑，不校準')
    it['cells'] = by
    it['v2_setA_max_up'] = max((v['v2_up_setA'], k) for k, v in by.items())
    (HERE / 'iterations.json').write_text(json.dumps(it, ensure_ascii=False, indent=1), encoding='utf-8')
    L.save_key('ITERATIONS', dict(file='iterations.json', sha256=L.sha(HERE / 'iterations.json'),
                                  b_a={v: it['versions'][v]['b_a'] for v in ('A1', 'A2', 'A3')}, utc=it['utc']), need=('PREREG_DESIGN',))
    print(json.dumps({k: v for k, v in it.items() if k != 'cells'}, ensure_ascii=False, indent=1))
