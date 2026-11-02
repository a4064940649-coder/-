"""每個 seed（官方日 AB、B10）換股日賣出落在官方處置期間（start≤成交日≤end）的次數。"""
import sys, json; from pathlib import Path; HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd, rb_core as B
SD = B.official_days(); out = {}
def dispon(t):
    x = np.datetime64(B.IDX[t]); m = np.zeros(B.N, bool); m[B.DJ[(B.DS <= x) & (B.DE >= x)]] = True; return m
for k in ('V7', 'V7_B10'):
    cnt = []
    for s in range(40):
        pl, _ = B.plan(SD, B10=(k == 'V7_B10'), unavail=B.unavail_mc(s)); prev = set(); c = 0
        for p in pl:
            new = set(p['picks'].tolist()); dm = dispon(p['e'])
            c += sum(1 for j in prev - new if dm[j] and B.TR[p['e'], j]); prev = new
        cnt.append(c)
    out[k] = dict(min=min(cnt), max=max(cnt), median=float(np.median(cnt)))
print(out); json.dump(out, open(HERE / 'out' / 'dispsell.json', 'w'))
