import sys, pathlib, time; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import rcore as C, numpy as np, pandas as pd
C.REVRULE['mode'] = 'anchor'
T = pd.read_parquet(pathlib.Path(__file__).resolve().parents[1] / 'navs_oos.parquet')['J+0']
m0, m1 = C.ANCHORS[0], C.ANCHORS[1]
s, s2 = C.jday(m0, 0), C.jday(m1, 0)
target = T.loc[s2] / T.loc[s]
t0, t2 = C.POS[s], C.POS[s2]
print(s, s2, target)
def wr(U, uid):
    r = C.ranking(m0, s, U, uid); e = r['e']
    gu = C.gap_up(e); pick = [int(j) for j in r['cand'] if not gu[j]][:10]
    g = C.Cn[e, pick] / C.On[e, pick] * np.prod(C.GROW[e + 1:t2 + 1, pick], axis=0)
    return g.mean() / (1 + C.NORMAL[0]), tuple(sorted(pick))
print('noAB', wr(None, None)[0], 'all', wr(np.ones(C.N, bool), -5)[0])
rg = np.random.default_rng(0); hits = []; seen = {}
t_ = time.time()
for k in range(3000):
    U = rg.random(C.N) < C.PCOL
    w, pk = wr(U, 20000 + k); C._CS.clear()
    if abs(w - target) < 1e-9: hits.append(k)
print('hits', hits[:50], len(hits), time.time() - t_)
for sd in range(1, 50):
    U = np.random.default_rng(sd).random(C.N) < C.PCOL; w, _ = wr(U, 90000 + sd)
    if abs(w - target) < 1e-9: print('seed first', sd)
