import numpy as np, pandas as pd, sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hier
from scipy.stats import norm
name = sys.argv[1]
df = pd.read_csv(f'cv_{name}.csv'); dens = np.load(f'cv_{name}_dens.npy')
for v in sys.argv[2:]:
    q = df[df.version == v]; W = dens[q.index.values]
    sl = {sch: np.polyfit(q[q.scheme == sch].mu, q[q.scheme == sch].y, 1)[0] for sch in ('half', 'roll')}
    s0 = float(np.clip(min(sl.values()), 0, 1))
    out = {}
    for s in (s0, 0.2939, 0.44, 0.63):
        for k in (1.0, 1.1, 1.2):
            cov = np.mean([0.1 <= (hier.transform(w, s, k) * norm.cdf((r.y - hier.GRID) / r.sV)).sum() <= 0.9 for w, (_, r) in zip(W, q.iterrows())])
            out[(round(s, 3), k)] = round(cov, 3)
    print(name, v, 'logpd %.1f' % q.logpd.sum(), 'slopes', {a: round(b, 3) for a, b in sl.items()}, out)
