import numpy as np, pandas as pd, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hier, rlib
def focus_aligned(idx):
    med = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'decision_line.csv'), index_col=0).iloc[:, 0]
    med.index = pd.PeriodIndex(med.index, freq='M')
    n = pd.read_parquet(os.path.join(rlib.RAW, 'v7/navs_exec7.parquet'), columns=['B10|J0|K5T|normal|100萬|FIX'])
    D = rlib.anchor_months(n.index); Ds = sorted(D.items(), key=lambda kv: kv[1])
    dd = pd.DatetimeIndex([d for _, d in Ds]); mm = [m for m, _ in Ds]
    am = [mm[dd.searchsorted(d, side='right') - 1] for d in idx]
    return np.array([med.get(m, 0.0) for m in am]), am
def summ(w):
    g = hier.GRID; mu = (g * w).sum(); sd = np.sqrt(((g - mu) ** 2 * w).sum())
    return dict(mean=mu, sd=sd, p0=w[g > 0].sum(), p25=w[g > 0.25].sum())
if __name__ == '__main__':
    idx, X, cl, med, ids = hier.load_family()
    foc, am = focus_aligned(idx)
    rng = np.random.default_rng(1)
    out = hier.fit_all(X, cl, med, -999, foc, rng=rng, Ymem=X[:, cl == 114])
    # note: c=-999 -> no cluster held out (full family); medoid/members of cluster 114
    for v, w in out.items():
        print(v, {k: round(x, 3) for k, x in summ(w).items()}, 'cal', {k: round(x, 3) for k, x in summ(hier.transform(w, 0.2939, 1.1)).items()})
    w = hier.transform(out['V3'], 0.2939, 1.1)
    np.savez('my_prior.npz', th=hier.GRID, p=w, raw=out['V3'])
