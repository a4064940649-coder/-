import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import ind_core as C, numpy as np, pandas as pd
T = pd.read_parquet(pathlib.Path(__file__).resolve().parents[1] / 'navs_oos.parquet')['J+0'].dropna()
ps = C.ANCH[:3]; ss = [C.J(p, 0) for p in ps]
tg = [T.loc[ss[i + 1]] / T.loc[ss[i]] for i in range(2)]
def wr(i, U, tag, held=()):
    p, s = ps[i], ss[i]; t2 = C.POS[ss[i + 1]]
    r = C.rank(p, s, U, tag); e = r['e']; gu = C.gapup(e)
    pk = [int(j) for j in r['order'] if not (gu[j] and j not in held)][:10]
    g = C.Cn[e, pk] / C.On[e, pk] * np.prod(C.GR[e + 1:t2 + 1, pk], axis=0)
    return g.mean(), pk
def gens(seed, p, s):
    t = C.POS[s]; ymd = int(s.strftime('%Y%m%d')); o = p.ordinal
    yield 't', np.random.default_rng(t)
    yield 's*100003+t', np.random.default_rng(seed * 100003 + t)
    yield '[s,t]', np.random.default_rng([seed, t])
    yield '[t,s]', np.random.default_rng([t, seed])
    yield 'ymd', np.random.default_rng(ymd)
    yield '[s,ymd]', np.random.default_rng([seed, ymd])
    yield 's+ymd', np.random.default_rng(seed + ymd)
    yield 'ord', np.random.default_rng(o)
    yield '[s,ord]', np.random.default_rng([seed, o])
    yield 's*1000+ord', np.random.default_rng(seed * 1000 + o)
    yield 'toord', np.random.default_rng(s.toordinal())
    yield '[s,toord]', np.random.default_rng([seed, s.toordinal()])
    yield '[s,0,ord]', np.random.default_rng([seed, 0, o])
    yield '[s,j0,t]', np.random.default_rng([seed, 0, t])
    yield 'yyyymm', np.random.default_rng(p.year * 100 + p.month)
    yield '[s,yyyymm]', np.random.default_rng([seed, p.year * 100 + p.month])
    yield 's*100003+yyyymm', np.random.default_rng(seed * 100003 + p.year * 100 + p.month)
for nm in [n for n, _ in gens(0, ps[0], ss[0])]:
    ok = []
    for i in range(2):
        g = dict(gens(0, ps[i], ss[i]))[nm]
        for mode in ('lt', 'ge'):
            pass
        U = g.random(C.N) < C.PCOL
        w, pk = wr(i, U, ('g', nm, i)); ok.append(abs(w / (1 + C.NORMAL[0]) - tg[i]) < 1e-7 if i == 0 else None)
        if i == 1: ok[-1] = w
    print(nm, ok)
print('targets', tg)
