import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
KS = sys.argv[1:]; sys.argv = ["x", "anchor", "next"]
import ind_jscan as S
C = S.C; import numpy as np, pandas as pd
T = pd.read_parquet(pathlib.Path(__file__).resolve().parents[1] / 'navs_oos.parquet')['J+0'].dropna()
for k0 in [int(x) for x in KS]:
    ab = C.AB(0)
    for _ in range(k0): ab.rng.random(C.N)
    nav, lg, pl, sd, e_ = S.run_j(0, ab)
    a = C.wret(nav, sd); b = C.wret(T, sd); d = (a - b).abs(); bad = d[d > 1e-9]
    ex = C.exw(nav, sd)
    print(k0, 'diff windows', len(bad), [str(x.date()) for x in bad.index[:6]], C.cagr(nav)[0], C.tst(ex))
