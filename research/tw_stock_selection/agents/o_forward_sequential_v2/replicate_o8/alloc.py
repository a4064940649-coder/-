import numpy as np, pandas as pd, sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rlib, hier, prior as pr, pf
S_CAL, K_CAL = float(sys.argv[1]) if len(sys.argv) > 1 else 0.2939, float(sys.argv[2]) if len(sys.argv) > 2 else 1.1
idx, X, cl, med, ids = hier.load_family()
def am_of(dates):
    n = pd.read_parquet(os.path.join(rlib.RAW, 'v7/navs_exec7.parquet'), columns=['B10|J0|K5T|normal|100萬|FIX'])
    D = rlib.anchor_months(n.index); Ds = sorted(D.items(), key=lambda kv: kv[1])
    dd = pd.DatetimeIndex([d for _, d in Ds]); mm = [m for m, _ in Ds]
    return [mm[dd.searchsorted(d, side='right') - 1] for d in dates]
AM = am_of(idx)
def sim(prior_w, exc, r50, x_grid, cost1, N=20000, H=240, seed=20261106):
    rng = np.random.default_rng(seed)
    res = exc - exc.mean(); T = len(res)
    a = rng.choice(hier.GRID, size=N, p=prior_w)
    bi = hier.stat_boot_idx(T, N, rng) if False else None
    # vectorised stationary bootstrap
    idx = np.empty((N, H), int); i = rng.integers(T, size=N); jump = rng.random((N, H)) < 1 / 6; new = rng.integers(T, size=(N, H))
    for t in range(H):
        if t > 0: i = np.where(jump[:, t], new[:, t], (i + 1) % T)
        idx[:, t] = i
    rb = r50[idx] / 100; rs = (r50[idx] + a[:, None] + res[idx]) / 100
    out = {}
    for x in x_grid:
        rp = x * rs + (1 - x) * rb
        # monthly rebalance back to x: traded fraction = |w_drift - x|
        wd = x * (1 + rs) / (1 + rp); tc = np.abs(wd - x) * cost1
        lw = np.log1p(rp) - tc
        out[round(x, 1)] = dict(elog=lw.sum(1).mean() / (H / 12) * 100, beat10=(lw[:, :120].sum(1) > np.log1p(rb[:, :120]).sum(1)).mean())
    return a, out
if __name__ == '__main__':
    r50 = pf.r50; rows = []
    for st in ('normal', 'stress'):
        for sz in ('30萬', '100萬', '300萬', '1000萬'):
            for fx in ('FIX', 'GROW'):
                cell = f'K5T|{st}|{sz}|{fx}'
                m = rlib.decision_line(cell=cell)
                foc = np.array([m.get(p, 0.0) for p in AM])
                w = hier.fit_all(X, cl, med, -999, foc, ('V3',), rng=np.random.default_rng(1), Ymem=X[:, cl == 114])['V3']
                w = hier.transform(w, S_CAL, K_CAL)
                common = [p for p in m.index if p in r50.index]
                exc = m.loc[common].values; rb = r50.loc[common].values
                cost1 = 0.0025 if st == 'normal' else 0.0065
                a, o = sim(w, exc, rb, np.round(np.arange(0, 1.01, 0.1), 1), cost1)
                el = {k: v['elog'] for k, v in o.items()}; xs = max(el, key=el.get)
                flat = [k for k in el if el[xs] - el[k] <= 0.25]
                rows.append(dict(cell=cell, P_pos=float(w[hier.GRID > 0].sum()), prior_mean=float((hier.GRID * w).sum()), xstar=xs, flat=(min(flat), max(flat)),
                                 loss100=el[xs] - el[1.0], beat10_100=o[1.0]['beat10']))
                print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv('alloc_%.3f_%.2f.csv' % (S_CAL, K_CAL), index=False)
