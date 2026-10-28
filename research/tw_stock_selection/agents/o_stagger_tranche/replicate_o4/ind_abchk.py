import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import ind_core as C, numpy as np, pandas as pd
for seed in range(5):
    U = C.V.make_unavail('mc', seed); A = C.AB(seed)
    bad = [str(p) for p in C.ANCH if (U(C.OFF[p]) != A(p, C.OFF[p])).any()]
    print(seed, 'mismatch months', bad[:10], len(bad))
