import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.argv = ['x', 'anchor', 'next']
import ind_jscan as S
C = S.C; import numpy as np, pandas as pd
T = pd.read_parquet(pathlib.Path(__file__).resolve().parents[1] / 'navs_oos.parquet')
print(T.columns.tolist()[:40])
tn = T['J+0'].dropna()
for name, ab in [('seed0', C.AB(0)), ('noAB', None)]:
    nav, lg, pl, sd, e_ = S.run_j(0, ab)
    a = C.wret(nav, sd); b = C.wret(tn, sd)
    d = (a - b).abs(); bad = d[d > 1e-9]
    print(name, 'n diff windows', len(bad), [str(x.date()) for x in bad.index[:12]])
