#!/usr/bin/env python3
"""紙上引擎：每個 j 的各組合窗報酬（一般／壓力／零成本），存 parquet。"""
import sys, pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import rlib as R

OUT = Path(__file__).resolve().parent / 'out'
FEE = 0.001425 * 0.6
COSTS = {'normal': (FEE + 0.001, FEE + 0.003 + 0.001), 'stress': (FEE + 0.001 + 0.005, FEE + 0.003 + 0.001 + 0.005), 'zero': (0.0, 0.0)}
E50 = pd.read_csv(R.O8 / 'raw' / 'bench' / 'etf_0050.csv', parse_dates=['date']).set_index('date')['adj']
def r50(t0, t1):
    a = E50.asof(R.IDX[t0]); b = E50.asof(R.IDX[t1]); return b / a - 1

def dense(P):
    w = np.zeros(R.N)
    for k, v in P.items(): w[k] = v
    return w

def run(days, W, cost, end_t=None):
    """W：list of dense 權重（與 days 對齊）。回傳窗報酬、換手。"""
    buy, sell = COSTS[cost]
    tend = len(R.IDX) - 1 if end_t is None else end_t
    h = np.zeros(R.N); cash = 1.0
    rets = []; tos = []
    for i, t in enumerate(days):
        t1 = days[i + 1] if i + 1 < len(days) else tend
        e = t + 1
        # d 收盤 → e 開盤（舊持股）
        with np.errstate(invalid='ignore', divide='ignore'):
            g = R.AOPEN[e] / R.AFF[t]
        g = np.where(np.isfinite(g), g, np.where(np.isfinite(R.AFF[e] / R.AFF[t]), R.AFF[e] / R.AFF[t], 1.0))
        hv = h * g; V1 = hv.sum() + cash
        hp = hv / V1
        w = W[i]
        if w.sum() <= 0:     # 權重和 0 → 持現金
            w = np.zeros(R.N); wc = 1.0
        else:
            wc = 0.0
        c = buy * np.clip(w - hp, 0, None).sum() + sell * np.clip(hp - w, 0, None).sum()
        tos.append(0.5 * np.abs(w - hp).sum() + 0.5 * abs(wc - (1 - hp.sum())))
        V2 = V1 * (1 - c)
        with np.errstate(invalid='ignore', divide='ignore'):
            base = np.where(np.isfinite(R.AOPEN[e]), R.AOPEN[e], R.AFF[e])
            G = R.AFF[t1] / base
        G = np.where(np.isfinite(G), G, 1.0)
        val = w * G; V3 = V2 * (val.sum() + wc)
        h = V2 * val / V3; cash = V2 * wc / V3
        rets.append(V3 - 1.0)
    return np.array(rets), np.array(tos)

CARR = lambda B, X: 0.7 * B + 0.3 * X

def main(j, keys=None):
    D = pickle.load(open(OUT / f'plans_J{j}.pkl', 'rb'))
    days = D['days']; plans = [p for _, p in D['plans']]
    B = [dense(p['B']) for p in plans]
    names = [k for k in plans[0] if k != 'B']
    rows = {}
    for cost in ('normal', 'stress', 'zero'):
        r, to = run(days, B, cost); rows[f'B|{cost}'] = r; rows[f'B|to'] = to
    for k in names:
        X = [dense(p.get(k, {})) for p in plans]
        for carr in ('PURE', 'SAT'):
            Wl = X if carr == 'PURE' else [CARR(b, x) if x.sum() > 0 else b for b, x in zip(B, X)]
            for cost in ('normal', 'stress', 'zero'):
                r, to = run(days, Wl, cost); rows[f'{k}_{carr}|{cost}'] = r
                if cost == 'normal': rows[f'{k}_{carr}|to'] = to
    tend = len(R.IDX) - 1
    rows['0050'] = np.array([r50(t, days[i + 1] if i + 1 < len(days) else tend) for i, t in enumerate(days)])
    df = pd.DataFrame(rows, index=pd.DatetimeIndex([R.IDX[t] for t in days], name='d'))
    df.to_parquet(OUT / f'paper_J{j}.parquet')
    return df

if __name__ == '__main__':
    js = [int(x) for x in sys.argv[1:]] or list(range(13))
    for j in js:
        df = main(j); print(j, df.shape, flush=True)
