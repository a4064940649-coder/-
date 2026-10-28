import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import ind_core as C, numpy as np, pandas as pd
col = sys.argv[1]; jj = None if sys.argv[2] == 'off' else int(sys.argv[2]); nmc = int(sys.argv[3])
TT = pd.read_parquet(pathlib.Path(__file__).resolve().parents[1] / 'navs_oos.parquet')[col].dropna()
days = [(p, C.OFF[p] if jj is None else C.J(p, jj)) for p in C.ANCH]
days = [(p, s) for p, s in days if s in C.POS]
rg = np.random.default_rng(0); STREAM = [rg.random(C.N) < C.PCOL for _ in range(700)]
fixed = {}   # (p,s) -> U
class ABF:
    tag = 'F'
    def __call__(self, p, s):
        k, ins = C.ab_kind(p, s); u = np.zeros(C.N, bool)
        if k == 'all': u[:] = True
        elif k == 'mc': u = fixed.get((p, s), np.zeros(C.N, bool))
        if ins: u = u | C.IS15
        return u
mcs = [(i, p, s) for i, (p, s) in enumerate(days) if C.ab_kind(p, s)[0] == 'mc']
found = []
for (i, p, s) in mcs[:nmc]:
    if i + 1 >= len(days): break
    s1 = days[i + 1][1]; tgt = TT.loc[s1] / TT.loc[s]
    hits = []
    for k, U in enumerate(STREAM):
        fixed[(p, s)] = U; C._RC.clear()
        pl = C.plan(days[:i + 1], ABF())
        nav, _ = C.tsim([pl], *C.NORMAL, C.POS[days[0][1]], C.POS[s1])
        if abs(nav.loc[s1] / nav.loc[s] - tgt) < 1e-10: hits.append(k)
    print(p, s.date(), 'hits', hits[:20], flush=True)
    fixed[(p, s)] = STREAM[hits[0]] if hits else np.zeros(C.N, bool)
