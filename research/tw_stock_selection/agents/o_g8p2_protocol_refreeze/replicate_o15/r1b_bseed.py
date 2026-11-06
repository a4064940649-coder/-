# b 的 seed 敏感度：另兩組集 A 基底，c_d 用協定值
import sys, os
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rlib as L
def bcell(a):
    base, di, si, ki = a
    rng = L.cell_rng(base, di + 1, si, ki, 2, 0)
    X = L.gen(['D1', 'D2', 'D3'][di], L.SIGMAS[si], L.SGRID[ki] - L.DW, 200000, rng)
    return float(np.quantile(L.eval_v(X, 99.0, L.CD_P)['Bstar'], 0.9525)), (di, si, ki)
if __name__ == '__main__':
    for base in (15150621, 15150631):
        with Pool(4) as p:
            r = p.map(bcell, [(base, di, si, ki) for di in range(3) for si in range(9) for ki in range(5)])
        print(base, max(r))
