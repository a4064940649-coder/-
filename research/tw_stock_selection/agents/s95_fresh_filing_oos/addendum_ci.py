#!/usr/bin/env python3
"""S95 補充（登記後追加、只描述）：H1 平均差的訊號日聚類自助法 90% 區間，用來說明檢定力。python3 -I -B addendum_ci.py → addendum_ci.json"""
import json
from pathlib import Path
import numpy as np, pandas as pd
H = Path(__file__).resolve().parent
t = pd.read_csv(H / 'trades_s95.csv.gz', parse_dates=['sig_date']); t['y'] = t.sig_date.dt.year
t['lab'] = np.where(t.y <= 2012, t.m.isin([5, 9, 11]), t.m.isin([3, 5, 8, 11])).astype(int)
def ci(x, seed=95, nb=10000):
    cl, c = np.unique(x.sig_date.to_numpy(), return_inverse=True); idx = [np.flatnonzero(c == i) for i in range(len(cl))]
    ex = x.ex.to_numpy(); lab = x.lab.to_numpy(); rng = np.random.default_rng(seed); D = []
    for _ in range(nb):
        s = np.concatenate([idx[i] for i in rng.integers(0, len(cl), len(cl))])
        if lab[s].min() == lab[s].max(): continue
        D.append(ex[s][lab[s] == 1].mean() - ex[s][lab[s] == 0].mean())
    return dict(diff=float(ex[lab == 1].mean() - ex[lab == 0].mean()), ci90=[float(np.quantile(D, .05)), float(np.quantile(D, .95))])
R = {k: ci(t[(t.y >= a) & (t.y <= b)]) for k, (a, b) in {'2008_2012': (2008, 2012), '2013_2015': (2013, 2015), '2008_2015': (2008, 2015), '2016_2021': (2016, 2021), '2022_2026': (2022, 2026), '2016_2026': (2016, 2026)}.items()}
(H / 'addendum_ci.json').write_text(json.dumps(R, indent=1)); print(json.dumps(R))
