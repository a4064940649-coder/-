import sys, pathlib, time; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import rcore as C, numpy as np, pandas as pd
C.REVRULE['mode'] = 'anchor'
T = pd.read_parquet(pathlib.Path(__file__).resolve().parents[1] / 'navs_oos.parquet')
M20 = pd.Period('2020-02', 'M')
j = int(sys.argv[1])
s = C.jday(M20, j); t = C.POS[s]
tn = T[f'J{j:+d}']
nxt = C.jday(pd.Period('2020-03', 'M'), j)
target = tn.loc[nxt] / tn.loc[s]
# my previous holdings: use base plan to get held set and window return with candidate U, quick approx via full sim of window
days = [(m, C.jday(m, j)) for m in C.ANCHORS]
base = C.make_plan(days, 10, C.AB(0))
idx = [i for i, p in enumerate(base) if p[2] == M20][0]
prev = base[idx - 1]
def window_ret(U, uid):
    r = C.ranking(M20, s, U, uid)
    gu = C.gap_up(r['e']); held = set(prev[1].tolist()); pick = []
    for jj in r['cand']:
        if gu[jj] and jj not in held: continue
        pick.append(int(jj))
        if len(pick) == 10: break
    pl = [prev, (r['e'], np.array(pick), M20, s)]
    nav, _ = C.tsim([pl], *C.NORMAL, prev[0] - 1, C.POS[nxt])
    return nav.loc[nxt] / nav.loc[s], pick
print('target', target, 'mine', window_ret(C.AB(0)(M20, s), -1)[0], 'none', window_ret(np.zeros(C.N, bool), -2)[0], 'all', window_ret(np.ones(C.N, bool), -3)[0])
# Note window ret depends on prev holdings only through costs; compare
cands = {}
for sd in range(0, 30): cands[f'seed{sd}first'] = np.random.default_rng(sd).random(C.N) < C.PCOL
for name, g in [('s*100003+t', lambda: np.random.default_rng(0 * 100003 + t)), ('t', lambda: np.random.default_rng(t)), ('(0,t)', lambda: np.random.default_rng([0, t])),
                ('ord', lambda: np.random.default_rng(s.toordinal())), ('yyyymmdd', lambda: np.random.default_rng(int(s.strftime('%Y%m%d'))))]:
    cands[name] = g().random(C.N) < C.PCOL
# k-th draw of seed 0
rg = np.random.default_rng(0)
for k in range(260): cands[f'draw{k}'] = rg.random(C.N) < C.PCOL
res = []
for i, (k, U) in enumerate(cands.items()):
    w, pk = window_ret(U, 10000 + i)
    res.append((abs(w - target), k, w))
res.sort(); print(res[:10])
ok = sorted(int(k[4:]) for d_, k, w in res if d_ < 1e-12 and k.startswith('draw'))
print('matching draws', ok)
offmc = [m for m in C.ANCHORS if C.unavail_kind(m, C.OFFICIAL[m])[0] == 'mc']
print('official mc index of 2020-02', offmc.index(M20) if M20 in offmc else None, 'anchor idx', C.ANCHORS.index(M20))
