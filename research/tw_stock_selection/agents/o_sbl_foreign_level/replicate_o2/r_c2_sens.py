import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, math
import r_lib as L
K, IDX = L.K, L.IDX
D = L.build(); SD = D['sig_days']; END = D['end']
t0 = IDX.get_loc(SD[0]); t1 = IDX.get_loc(END)
Nn = pd.read_parquet(L.HERE / 'my_navs.parquet')
pts = pd.DatetimeIndex(list(SD) + [END])
def win(s): x = s.reindex(pts); return (x.shift(-1) / x - 1).iloc[:-1]
def tst(x): x = x.dropna(); return round(x.mean() / (x.std(ddof=1) / math.sqrt(len(x))), 2)
def cg(s, a, b):
    x = s.loc[a:b]; yrs = (x.index[-1] - x.index[0]).days / 365.25; return round((x.iloc[-1] / x.iloc[0]) ** (1 / yrs) * 100 - 100, 2)
orig = L.fo_at
for lab, f in (('lag0', lambda d, lagdays=0: L.FOn[L.FO.index.searchsorted(d, 'right') - 1 - lagdays]),
               ('lag1', orig), ('lag2', lambda d, lagdays=0: orig(d, lagdays + 1))):
    L.fo_at = f
    nav = L.sim(L.c_plan(D, 0.2), 'normal', t0, t1)
    w = win(nav) - win(Nn['main'])
    s = w[(w.index <= '2023-11-30')]; v = w[w.index >= '2024-03-01']
    print(lab, 'SEL', round(s.mean() * 100, 2), tst(s), 'VAL31', round(v.mean() * 100, 2), tst(v), 'CAGR sel', cg(nav, '2021-10-13', '2024-02-15'), 'B', cg(nav, '2025-10-13', END), 'ALL', cg(nav, SD[0], END))
