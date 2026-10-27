import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import mainlib as M
PL = M.plans()
t0, t1 = PL['t0'], PL['t1']
ref = pd.read_parquet(M.AG / 'main_v2' / 'navs_gapfix.parquet')
for n in (10, 30):
    nav, _ = M.simulate(PL[n], 'normal', t0, t1, mode='none')
    r = ref[f'EXFIN|n{n}|normal'].reindex(nav.index)
    print(n, 'maxabs diff vs navs_gapfix', float((nav - r).abs().max()), nav.iloc[-1], r.iloc[-1])
    navk, _ = M.K.simulate([(p[0], p[1]) for p in PL[n]], *M.K.rates(0.001), t0, t1)
    print(n, 'vs K.simulate', float((nav - navk).abs().max()))
print('EW diff', float((PL['ew'] - ref['EXFIN|EW'].reindex(PL['ew'].index)).abs().max()))
# 跌停價檢查
print(M.limit_down_price(np.array([100.0, 55.0, 9.99, 512.0]), 0.10))
