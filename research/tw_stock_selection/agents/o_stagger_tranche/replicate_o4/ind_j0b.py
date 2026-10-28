import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import ind_core as C, numpy as np, pandas as pd
T = pd.read_parquet(pathlib.Path(__file__).resolve().parents[1] / 'navs_oos.parquet')['J+0'].dropna()
p0, p1 = C.ANCH[0], C.ANCH[1]
s0, s1 = C.J(p0, 0), C.J(p1, 0)
tgt = T.loc[s1] / T.loc[s0]
print('first valid', T.index[0], s0, s1, 'target', tgt)
t2 = C.POS[s1]
def wr(U, tag):
    r = C.rank(p0, s0, U, tag); e = r['e']; gu = C.gapup(e)
    pk = [int(j) for j in r['order'] if not gu[j]][:10]
    g = C.Cn[e, pk] / C.On[e, pk] * np.prod(C.GR[e + 1:t2 + 1, pk], axis=0)
    return g.mean() / (1 + C.NORMAL[0]), pk
for nm, U in [('none', np.zeros(C.N, bool)), ('all', np.ones(C.N, bool))]:
    w, pk = wr(U, nm); print(nm, w, w - tgt, [C.COLS[j] for j in pk])
rg = np.random.default_rng(0)
U0 = rg.random(C.N) < C.PCOL
w, pk = wr(U0, 'u0'); print('seed0 first draw', w, w - tgt)
U0r = np.random.default_rng(0).random(C.N) >= C.PCOL
w, pk = wr(U0r, 'u0r'); print('seed0 reversed', w, w - tgt)
# how many eligible/top stocks get sampled?
print('mean PCOL', C.PCOL.mean())
hits = []
rg = np.random.default_rng(0)
for k in range(600):
    U = rg.random(C.N) < C.PCOL
    w, _ = wr(U, ('k', k)); C._RC.clear()
    if abs(w - tgt) < 1e-9: hits.append(k)
print('seed0 stream hits', hits)
for sd in range(200):
    U = np.random.default_rng(sd).random(C.N) < C.PCOL
    w, _ = wr(U, ('s', sd)); C._RC.clear()
    if abs(w - tgt) < 1e-9: print('fresh seed hit', sd)
