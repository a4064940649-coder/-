import sys
sys.path.insert(0, '/home/user/-/research/tw_stock_selection/agents/o_forward_infra_live/replicate_o9')
import numpy as np, pandas as pd
import rlib as L, r_engine as RE
Dm, M = RE.build_D(pd.Timestamp('2026-11-02'))
sys.path.insert(0, str(RE.V7X))
sys.modules.pop('x_data', None)
import x_data as DO
d0, d1 = pd.Timestamp('2025-10-01'), pd.Timestamp('2026-10-22')
codes = [c for c in DO.COLS if c in Dm.CI]
iF = [Dm.TPOS[d] for d in DO.IDX if d0 <= d <= d1 and d in Dm.TPOS]
iO = [DO.TPOS[d] for d in DO.IDX if d0 <= d <= d1 and d in Dm.TPOS]
print('dates', len(iF), 'codes common', len(codes), 'fwd cols', Dm.N, 'orig cols', DO.N)
jF = [Dm.CI[c] for c in codes]; jO = [DO.CI[c] for c in codes]
for k in ['C', 'O', 'V', 'R', 'PC', 'ADJ', 'V60p', 'SIGp', 'VRp', 'REF', 'E50', 'DISP']:
    A = getattr(Dm, k); B = getattr(DO, k)
    if A.ndim == 1:
        a, b = A[iF], B[iO]
    else:
        a, b = A[np.ix_(iF, jF)].astype(float), B[np.ix_(iO, jO)].astype(float)
    both = np.isfinite(a) & np.isfinite(b)
    nanmis = (np.isfinite(a) != np.isfinite(b)).sum()
    rel = np.abs(a[both] - b[both]) / np.maximum(np.abs(b[both]), 1e-12)
    print(f'{k:5s} nan-mismatch {nanmis:6d}  n>1e-9 {int((rel > 1e-9).sum()):6d}  maxrel {rel.max() if rel.size else 0:.3e}')
for k in ['R', 'PC', 'SIGp', 'C']:
    A = getattr(Dm, k)[np.ix_(iF, jF)]; B = getattr(DO, k)[np.ix_(iO, jO)]
    bad = (np.isfinite(A) & np.isfinite(B) & (np.abs(A - B) > 1e-9 * np.maximum(np.abs(B), 1))) | (np.isfinite(A) != np.isfinite(B))
    tt, jj = np.nonzero(bad)
    s = pd.DataFrame({'d': [Dm.IDX[iF[t]] for t in tt], 'c': [codes[j] for j in jj]})
    print(k, len(s)); print(s.groupby('c').d.agg(['min', 'max', 'count']).sort_values('count').tail(8))
