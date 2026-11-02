#!/usr/bin/env python3
"""閘門一：零衝擊路徑。V7X 自寫零衝擊引擎對 V7 原版：
(a) 官方日 80 條 seed 淨值（AB_V7、AB_V7_B10 各 40 seeds × 一般／壓力）→ seed_navs_v7.parquet；
(b) 13 個 j × {AB, B10} × {一般, 壓力} 的 40-seed 平均淨值 → navs_jscan_v7.parquet；
(c) 處置期出清與併購延遲（零衝擊拆解）13 個 j × {AB, B10} × {base, disp, mdelay, both} 的 40-seed 平均 → navs_ideal7.parquet（規格解讀核對，不是閘門）。
門檻：(a)(b) 逐日相對差 ≤ 1e-9。"""
import sys, json, time
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import x_data as D, x_ideal as I

OUT = Path(__file__).resolve().parent / 'out'


def rel(a, b):
    a = a.reindex(b.index).dropna(); b = b.reindex(a.index)
    a = a / a.iloc[0]; b = b / b.iloc[0]
    return float((a / b - 1).abs().max())


def job_off(args):
    strat, seed = args
    P = D.plans(strat, seeds=[seed], official=True)[('OFF', seed)]
    return strat, seed, {k: I.simulate(P, k) for k in ('normal', 'stress')}


def job_j(args):
    strat, j = args
    P = D.plans(strat, js=[j])
    res = {}
    for k in ('normal', 'stress'):
        res[k] = I.avg_nav([I.simulate(P[(j, s)], k) for s in range(40)])
    v = {}
    for nm, ds, md in (('base', False, False), ('disp', True, False), ('mdelay', False, True), ('both', True, True)):
        v[nm] = I.avg_nav([I.simulate(P[(j, s)], 'normal', disp=ds, mdelay=md) for s in range(40)])
    return strat, j, res, v


if __name__ == '__main__':
    t0 = time.time()
    S = pd.read_parquet(D.O7 / 'seed_navs_v7.parquet')
    J = pd.read_parquet(D.O7 / 'navs_jscan_v7.parquet')
    ID = pd.read_parquet(D.O7 / 'navs_ideal7.parquet')
    out = {'off': {}, 'j': {}, 'ideal_variants': {}}
    with Pool(4) as pool:
        for strat, seed, r in pool.imap_unordered(job_off, [(s, i) for s in ('AB_V7', 'AB_V7_B10') for i in range(40)]):
            for k, n in r.items():
                out['off'][f'{strat}|{k}|s{seed}'] = rel(n, S[f'{strat}|{k}|s{seed}'])
        for strat, j, r, v in pool.imap_unordered(job_j, [(s, j) for s in ('AB_V7', 'AB_V7_B10') for j in range(13)]):
            for k, n in r.items():
                out['j'][f'{strat}|{k}|J{j}'] = rel(n, J[f'{strat}|{k}|J{j}'])
            lab = 'AB' if strat == 'AB_V7' else 'B10'
            for nm, n in v.items():
                c = f'{lab}|J{j}|{nm}'
                if c in ID.columns:
                    out['ideal_variants'][c] = rel(n, ID[c])
    summ = {'off_max': max(out['off'].values()), 'off_n': len(out['off']),
            'j_max': max(out['j'].values()), 'j_n': len(out['j']),
            'ideal_variants_max_by_kind': {nm: max(v for c, v in out['ideal_variants'].items() if c.endswith('|' + nm))
                                          for nm in ('base', 'disp', 'mdelay', 'both')},
            'PASS': bool(max(out['off'].values()) <= 1e-9 and max(out['j'].values()) <= 1e-9),
            'runtime_s': time.time() - t0}
    out['summary'] = summ
    (OUT / 'gate_zero.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps(summ, ensure_ascii=False, indent=1))
    print(sorted(out['ideal_variants'].items(), key=lambda x: -x[1])[:12])
