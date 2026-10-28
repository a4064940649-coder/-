import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.argv = ["x", "anchor", "next"]
import ind_jscan as S
C = S.C; import numpy as np, pandas as pd
TT = pd.read_parquet(pathlib.Path(__file__).resolve().parents[1] / 'navs_oos.parquet')
def cmp(nav, sd, col):
    a = C.wret(nav, sd); b = C.wret(TT[col].dropna(), sd); d = (a - b).abs(); return int((d > 1e-9).sum()), [str(x.date()) for x in d[d > 1e-9].index[:4]]
# official AB seed 0 vs OFF_AB0
days = [(p, C.OFF[p]) for p in C.ANCH]
class ABs(C.AB):
    def __call__(self, p, s):
        if s not in self.c:
            k, ins = self.kindf(p, s); u = np.zeros(C.N, bool)
            if k == 'all': u[:] = True
            elif k == 'mc': u = self.rng.random(C.N) < C.PCOL
            if ins: u = u | C.IS15
            self.c[s] = u
        return self.c[s]
for nm, ab in [('AB', C.AB(0)), ('shared', ABs(0))]:
    pl = C.plan(days, ab); t0 = C.POS[days[0][1]]
    nav, _ = C.tsim([pl], *C.NORMAL, t0, C.TEND)
    print(nm, 'OFF_AB0 diff', cmp(nav, C.OFFSD, 'OFF_AB0'), C.tst(C.exw(nav, C.OFFSD)))
    if nm == 'shared':
        nav, lg, pl, sd, e_ = S.run_j(0, ab)
        print('J0 after official shared', cmp(nav, sd, 'J+0'), C.cagr(nav)[0], C.tst(C.exw(nav, sd)))
nav = TT['OFF'].dropna(); print('OFF researcher t', C.tst(C.exw(nav, C.OFFSD)))
