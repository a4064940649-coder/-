import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import rcore as C, numpy as np, pandas as pd
C.REVRULE['mode'] = 'anchor'
off = [(m, C.OFFICIAL[m]) for m in C.ANCHORS]
SD = [d for _, d in off] + [C.END]
class ABP(C.AB):
    def __init__(self, seed, how):
        super().__init__(seed); self.how = how
        if how == 'perm': self.pre = {m: self.rng.random(C.N) < C.PCOL for m in C.ANCHORS}
        if how == 'permU': self.pre = {m: self.rng.random(C.N) for m in C.ANCHORS}
    def __call__(self, m, s):
        k, ins = C.unavail_kind(m, s)
        if self.how == 'perm': u = np.zeros(C.N, bool) if k == 'none' else (np.ones(C.N, bool) if k == 'all' else self.pre[m])
        elif self.how == 'keyed':
            u = np.zeros(C.N, bool) if k == 'none' else (np.ones(C.N, bool) if k == 'all' else (np.random.default_rng([self.seed_, C.POS[s]]).random(C.N) < C.PCOL))
        return u | (C.IS15 if ins else False)
for how in ('perm',):
    for sd in range(5):
        ab = ABP(sd, how)
        pl = C.make_plan(off, 10, ab); nav, _ = C.tsim([pl], *C.NORMAL, C.POS[off[0][1]], C.TEND)
        print(how, sd, round(C.tstat(C.excess(nav, SD)), 4))
