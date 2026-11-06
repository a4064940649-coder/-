# c_d 與 b 依協定 c_d_rule / b_rule 重新校準（自選 seed）
import json, sys, os
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rlib as L

MY_A = 15150601   # 我的集 A 基底
MY_CD = [15150602, 15150612]


def calib_cd(delta, seed):
    rng = np.random.default_rng(seed)
    X = L.gen('D1', L.SIG_D, -1 - L.DW, 400000, rng)
    r = L.eval_v(X, 99.0, 99.0, delta)
    return float(-np.quantile(r['minz'], 0.55))


def bcell(args):
    delta, cd, di, si, ki = args
    dgp = ['D1', 'D2', 'D3'][di]
    rng = L.cell_rng(MY_A, di + 1, si, ki, 2, 0)
    X = L.gen(dgp, L.SIGMAS[si], L.SGRID[ki] - L.DW, 200000, rng)
    r = L.eval_v(X, 99.0, cd, delta)
    return (dgp, L.SLAB[si], L.SGRID[ki], float(np.quantile(r['Bstar'], 0.9525)))


if __name__ == '__main__':
    out = {}
    for name, delta in (('V1', 0.0), ('V3', 0.2)):
        cds = [calib_cd(delta, s) for s in MY_CD]
        out[name] = dict(cd_seeds=cds)
        print(name, 'c_d', cds, flush=True)
        for tag, cd in (('own', cds[0]), ('proto', L.CD_P if name == 'V1' else None)):
            if cd is None:
                continue
            jobs = [(delta, cd, di, si, ki) for di in range(3) for si in range(9) for ki in range(5)]
            with Pool(4) as p:
                res = p.map(bcell, jobs)
            res.sort(key=lambda t: -t[3])
            b = max(L.B1, res[0][3])
            out[name]['b_' + tag] = b
            out[name]['b_' + tag + '_worst'] = res[:5]
            print(name, tag, 'cd', cd, 'b', b, res[:3], flush=True)
    json.dump(out, open(os.path.join(L.HERE, 'r1_calib.json'), 'w'), ensure_ascii=False, indent=1)
