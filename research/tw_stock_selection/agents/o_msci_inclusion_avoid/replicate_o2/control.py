import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json, warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import rlib as L
d = pd.read_csv(L.HERE / 'events_rep.csv', dtype={'code': str})
d['E'] = pd.to_datetime(d['E'])
std = d[(d.cls == 'STD') & d.liq_ok & ~d.src3]
msci_codes = d[d.family == 'MSCI'].groupby('review')['code'].apply(set).to_dict()
CS = np.cumsum(np.vstack([np.zeros((1, L.AR.shape[1])), L.AR]), axis=0)   # CS[i] = sum AR[0..i-1]
RC = np.cumsum(np.vstack([np.zeros((1, L.AR.shape[1])), np.log1p(L.R0)]), axis=0)
def window(Ai, Ei, kind):
    if kind == 'car_A-10_E': return CS[Ei + 1] - CS[Ai - 10]
    if kind == 'car_A-10_A-1': return CS[Ai] - CS[Ai - 10]
    if kind == 'ret20_E': return np.exp(RC[Ei + 1] - RC[Ei - 19]) - 1
    if kind == 'ret_A-10_E': return np.exp(RC[Ei + 1] - RC[Ai - 10]) - 1
    if kind == 'ret60_E': return np.exp(RC[Ei + 1] - RC[Ei - 59]) - 1
    if kind == 'car20_E': return CS[Ei + 1] - CS[Ei - 19]
    if kind == 'car_A-10_E-1': return CS[Ei] - CS[Ai - 10]
    if kind == 'ret_A-10_E-1': return np.exp(RC[Ei] - RC[Ai - 10]) - 1
    if kind == 'car_A_E': return CS[Ei + 1] - CS[Ai]
    if kind == 'ret20_E-1': return np.exp(RC[Ei] - RC[Ei - 20]) - 1
res = {}
UNIV = os.environ.get('UNIV', 'liq')
for kind in ('car_A-10_E', 'car_A-10_A-1', 'ret20_E', 'ret_A-10_E', 'ret60_E', 'car20_E', 'car_A-10_E-1', 'ret_A-10_E-1', 'car_A_E', 'ret20_E-1'):
    for per in ('select', 'oos'):
        g = []
        for (rv, Es), x in std[std.per == per].groupby(['review', 'E']):
            Ei = int(x.Ei.iloc[0]); Ai = int(x.Ai.iloc[0])
            liq = np.isfinite(L.V60[Ei - 1]) & (L.V60[Ei - 1] >= 1e7) & (L.CF[Ei - 1] >= 10) & L.TR[Ei]
            ex = np.array([c in msci_codes.get(rv, set()) for c in L.COLS])
            if UNIV == 'top300':
                v = np.where(liq, L.V60[Ei - 1], -np.inf); o = np.argsort(-v)[:300]; t3 = np.zeros_like(liq); t3[o] = True; liq = liq & t3
            el = liq & ~ex
            w = window(Ai, Ei, kind)
            el &= np.isfinite(w)
            thr = np.nanpercentile(w[el], 80)
            top = el & (w >= thr)
            c15 = CS[Ei + 6] - CS[Ei + 1]
            g.append(c15[top].mean())
        res[f'{kind}|{per}'] = (round(float(np.mean(g)) * 100, 2), round(L.tstat(g), 2), len(g))
for k, v in res.items(): print(k, v)
json.dump(res, open(L.HERE / 'control_rep.json', 'w'), indent=1)
