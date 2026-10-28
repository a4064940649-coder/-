import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.argv = ['x', 'anchor', 'next']
import ind_jscan as S
C = S.C; import numpy as np, pandas as pd
def summ(ab, j=0):
    nav, lg, pl, sd, e_ = S.run_j(j, ab); ex = C.exw(nav, sd)
    P = np.array([C.per(S.run_j.am[d]) for d in ex.index])
    return round(C.cagr(nav)[0], 2), round(C.tst(ex), 3), round(C.tst(ex[P == 'HOLD']), 3), round(C.tst(ex[P == 'SEL']), 3)
print('J0 noAB', summ(None))
for s in range(5): print('J0 seed', s, summ(C.AB(s)))
def k2(p, s):
    k, ins = C.ab_kind(p, s)
    if str(p - 1) in ('2019-01', '2023-03'): k = 'all'
    return k, ins
for s in range(3): print('J0 v2-special seed', s, summ(C.AB(s, k2)))
class ABV:
    def __init__(self, seed): self.U = C.V.make_unavail('mc', seed); self.tag = ('V', seed)
    def __call__(self, p, s): return self.U(s)
for s in range(3): print('J0 V.make_unavail seed', s, summ(ABV(s)))
class ABall:
    def __init__(self, seed): self.rng = np.random.default_rng(seed); self.c = {}; self.tag = ('ALLDRAW', seed)
    def __call__(self, p, s):
        if (p, s) not in self.c:
            r = self.rng.random(C.N) < C.PCOL
            k, ins = C.ab_kind(p, s)
            u = np.zeros(C.N, bool)
            if k == 'all': u[:] = True
            elif k == 'mc': u = r
            if ins: u |= C.IS15
            self.c[(p, s)] = u
        return self.c[(p, s)]
for s in range(3): print('J0 draw-every-day seed', s, summ(ABall(s)))
