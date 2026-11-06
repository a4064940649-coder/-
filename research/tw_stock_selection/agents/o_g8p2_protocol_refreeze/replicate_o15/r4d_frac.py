# 壓力線較差比例：嚴格 < 與 ≤ 兩種計法
import sys, os, warnings
warnings.filterwarnings('ignore')
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r4_hist as H
for nm in ('V7_FIX100', 'V7X_GROW100'):
    N, S, W = H.lines(*{'V7_FIX100': (H.V7, 'FIX'), 'V7X_GROW100': (H.V7X, 'GROW')}[nm])
    sel = [m for m in N.index if '2006-04' <= m <= '2019-04']
    n, s = N.loc[sel].values, S.loc[sel].values
    ok = np.isfinite(n) & np.isfinite(s)
    print(nm, 'windows', ok.sum(), '<', (s < n)[ok].sum(), (s < n)[ok].mean(), '<=', (s <= n)[ok].mean(), 'ties', (s == n)[ok].sum(),
          'min(n-s)', np.nanmin(n - s))
