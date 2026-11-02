import numpy as np, pandas as pd, sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hier
from scipy.stats import norm
from prior import focus_aligned, summ
VERS = ('V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7')
def splits(name):
    if name == 'sel': T = 157; h = 78; roll = (60, 84, 108)
    elif name == 'final': T = 245; h = 122; roll = (84, 120, 157, 192)
    elif name == 'pre2024': T = 212; h = 106; roll = (84, 120, 157)
    a = np.arange(T)
    S = [('half', 'fwd', a[:h], a[h:]), ('half', 'bwd', a[h:], a[:h])]
    S += [('roll', str(o), a[:o], a[o:o + 48]) for o in roll]
    return S
def run(name):
    idx, X, cl, med, ids = hier.load_family()
    foc, am = focus_aligned(idx)
    if name == 'pre2024':
        # windows with anchor month < 2024-01, dropping the last one whose holding crosses the 2024-01/02 gap -> 212
        keep = [i for i, m in enumerate(am) if m < pd.Period('2024-01', 'M')][:212]
        X = X[keep]; foc = foc[keep]
    sizes = pd.Series(cl).value_counts(); C = sorted(sizes[sizes >= 5].index)
    rows = []; dens = []
    rng = np.random.default_rng(20261103)
    for sch, tag, E, V in splits(name):
        for c in C:
            cols = np.where(cl == c)[0]
            mE = np.nanmean(X[np.ix_(E, cols)], 0); k = cols[np.nanargmax(mE)]
            out = hier.fit_all(X[E], cl, med, c, X[E, k], VERS, rng=rng, Ymem=X[np.ix_(E, cols)])
            yv = X[V, k]; y = np.nanmean(yv); sV = np.sqrt(hier.hv(yv))
            for v, w in out.items():
                mu = (hier.GRID * w).sum()
                pdv = (w * norm.pdf((y - hier.GRID) / sV) / sV).sum()
                pit = (w * norm.cdf((y - hier.GRID) / sV)).sum()
                rows.append(dict(scheme=sch, split=tag, cluster=int(c), version=v, mu=mu, y=y, sV=sV, logpd=np.log(pdv), pit=pit))
                dens.append(w)
        print(name, sch, tag, flush=True)
    df = pd.DataFrame(rows); df.to_csv(f'cv_{name}.csv', index=False); np.save(f'cv_{name}_dens.npy', np.array(dens))
    # main-strategy posterior mean on the full estimation span (tie-break)
    T = X.shape[0] if name != 'sel' else 157
    tb = hier.fit_all(X[:T], cl, med, -999, foc[:T], VERS, rng=rng, Ymem=X[:T][:, cl == 114])
    tbm = {v: float(summ(w)['mean']) for v, w in tb.items()}
    json.dump(tbm, open(f'cv_{name}_tiebreak.json', 'w'))
if __name__ == '__main__':
    run(sys.argv[1])
