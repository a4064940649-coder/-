import numpy as np, pandas as pd, sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hier
from scipy.stats import norm
for name in sys.argv[1:]:
    df = pd.read_csv(f'cv_{name}.csv'); dens = np.load(f'cv_{name}_dens.npy'); tb = json.load(open(f'cv_{name}_tiebreak.json'))
    tot = df.groupby('version').logpd.sum().sort_values(ascending=False)
    best = tot.index[0]; ties = [v for v in tot.index if tot.iloc[0] - tot[v] < 1]
    chosen = min(ties, key=lambda v: tb[v])
    q = df[df.version == chosen]; W = dens[q.index.values]
    slopes = {}
    for sch in ('half', 'roll'):
        r = q[q.scheme == sch]; slopes[sch] = np.polyfit(r.mu, r.y, 1)[0]
    s = float(np.clip(min(slopes.values()), 0, 1))
    k = 1.0
    while True:
        cov = []
        for w, (_, row) in zip(W, q.iterrows()):
            w2 = hier.transform(w, s, k); pit = (w2 * norm.cdf((row.y - hier.GRID) / row.sV)).sum(); cov.append(0.1 <= pit <= 0.9)
        if np.mean(cov) >= 0.8 or k > 3: break
        k = round(k + 0.1, 1)
    print(name, 'n=%d' % len(q), 'logpd', {v: round(x, 1) for v, x in tot.items()}, 'ties', ties, 'tiebreak post-mean', {v: round(tb[v], 3) for v in ties},
          '-> chosen', chosen, 'slopes', {k_: round(v, 3) for k_, v in slopes.items()}, 's=%.3f k=%.1f cov=%.3f' % (s, k, np.mean(cov)))
