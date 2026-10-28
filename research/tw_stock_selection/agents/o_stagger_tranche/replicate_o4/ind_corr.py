import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.argv = ['x', 'anchor', 'next']
import ind_jscan as JS
C = JS.C; import numpy as np, pandas as pd
NV = {j: JS.run_j(j, C.AB(0))[0] for j in (1, 6, 7, 11)}
sd = lambda n: [d for d in C.OFFSD if d >= n.index[0]]
E = pd.DataFrame({j: C.exw(NV[j], sd(NV[j])) for j in NV}); W = pd.DataFrame({j: C.wret(NV[j], sd(NV[j])) for j in NV})
for nm, X in (('excess', E), ('ret', W)):
    X = X.dropna(); sel = X[X.index < pd.Timestamp('2019-01-01')]
    print(nm, 'ALL', round(X[1].corr(X[6]), 3), round(X[1].corr(X[11]), 3), 'SEL', round(sel[1].corr(sel[6]), 3), round(sel[1].corr(sel[11]), 3))
nav = NV[7]; ex = JS.C.exw(nav, [q for q in JS.run_j(7, C.AB(0))[3]]) 
