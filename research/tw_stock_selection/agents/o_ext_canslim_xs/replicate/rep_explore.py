# -*- coding: utf-8 -*-
import sys, os, time, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rep_lib as L
S = sys.argv[1]
t=time.time(); P = L.load_panel(S); print('panel', P.T, P.N, time.time()-t)
print('idx missing', P.idx_missing, 'e50 missing', P.e50_missing, 'bad', P.n_bad)
for b in P.bad_cells[:80]: print(b)
print('t0', P.t0, P.dstr[P.t0], P.dstr[P.t0+1], 't_odd', P.dstr[P.t_odd])
t=time.time(); F = L.fundamentals(P, True); print('fund', time.time()-t)
K = L.tech(P); print('tech', time.time()-t)
tw = (P.M==1)
sl = slice(P.t0, P.T)
for k,v in [('C',F[0]),('A',F[1]),('R',F[2]),('N',K['condN']),('S',K['condS']),('L',K['condL'])]:
    print(k, 'pass rate twse', (v[sl]&tw[sl]).sum()/ (tw[sl]&P.inbar[sl]).sum())
print('M', K['mok'][sl].mean())
i0=P.t0
vy=F[4][i0]; print('twse with valid CAGR calc (both>0) on', P.dstr[i0], int((np.isfinite(vy)&tw[i0]).sum()))
for d in ['2017-01-24','2017-03-30','2017-03-31']:
    j=P.dstr.index(d); print(d, int((np.isfinite(F[4][j])&tw[j]).sum()))
for u in ['twse','both']:
    sig = L.signals(P,F,K,u); print(u,'signal rows', int(sig.sum()), 'stocks', int(sig.any(axis=0).sum()))
