import numpy as np, pandas as pd, os
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
T0, T1 = pd.Timestamp('2006-04-11'), pd.Timestamp('2026-10-12')

def load_bench():
    b = pd.read_parquet(os.path.join(D, 'bench.parquet'))
    return b

def signals():
    p = pd.read_csv(os.path.join(D, 'plans_ab.csv'), dtype={'picks': str})
    return pd.DatetimeIndex(sorted(p.signal.unique()))

def cagr(nav):
    nav = nav.dropna()
    d = (nav.index[-1] - nav.index[0]).days
    return (nav.iloc[-1] / nav.iloc[0]) ** (365.25 / d) - 1

def mdd(nav):
    return (nav / nav.cummax() - 1).min()

def win_rets(nav, sig, end=T1):
    pts = [s for s in sig if s <= end] + ([end] if sig[-1] < end else [])
    pts = pd.DatetimeIndex(pts)
    v = nav.reindex(pts).values
    r = v[1:] / v[:-1] - 1
    return pd.Series(r, index=pts[:-1])

def tstat(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return x.mean() / x.std(ddof=1) * np.sqrt(len(x))

def tvs(nav, e50, sig, lo=None, hi=None, drop_best=0, drop_last=0):
    a = win_rets(nav, sig); b = win_rets(e50, sig)
    ex = a - b
    if lo is not None: ex = ex[ex.index >= pd.Timestamp(lo)]
    if hi is not None: ex = ex[ex.index <= pd.Timestamp(hi)]
    if drop_last: ex = ex.iloc[:-drop_last]
    if drop_best: ex = ex.drop(ex.sort_values().index[-drop_best:])
    return tstat(ex.values)
