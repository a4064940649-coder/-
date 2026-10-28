import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import rcore as C, numpy as np, pandas as pd
exec(open(pathlib.Path(__file__).parent / 'jvar.py').read().split('tgt =')[0].split('import rcore as C, numpy as np, pandas as pd')[1])
C.REVRULE['mode'] = 'anchor'
M20 = pd.Period('2020-02', 'M')
class ABX(C.AB):
    def __init__(self, var, seed=0):
        super().__init__(seed); self.var = var
        if var == 'perm': self.pre = {m: self.rng.random(C.N) < C.PCOL for m in C.ANCHORS}
    def __call__(self, m, s):
        if m == M20 and C.deadline(m) <= s <= C.D20:
            if self.var == 'none': return np.zeros(C.N, bool)
            if self.var == 'all': return np.ones(C.N, bool)
            if self.var == 'perm': return self.pre[m]
        if self.var == 'perm':
            k, ins = C.unavail_kind(m, s)
            u = np.zeros(C.N, bool) if k == 'none' else (np.ones(C.N, bool) if k == 'all' else self.pre[m])
            return u | (C.IS15 if ins else False)
        return super().__call__(m, s)
tgt = {0: (26.7782, 2.3663), 1: (24.6379, 2.1848), 2: (22.2232, 1.871), 3: (24.3927, 2.1856), 4: (22.1107, 1.7632)}
for var in ('mc', 'none', 'all', 'perm'):
    for j in tgt: print(var, j, tgt[j], run(j, ABX(var)))
