# Independent replication library (replicate_o8). Written from the frozen spec only.
import numpy as np, pandas as pd, json, os
RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'raw')

def hac_var_mean(x, L=6):
    """Newey-West long-run variance of the series (Bartlett, L lags); returns long-run var (per obs)."""
    x = np.asarray(x, float); x = x[~np.isnan(x)]; n = len(x)
    d = x - x.mean(); g0 = d @ d / n; s = g0
    for l in range(1, L + 1):
        g = d[l:] @ d[:-l] / n; s += 2 * (1 - l / (L + 1)) * g
    return s

def hac_sd(x, L=6):
    return np.sqrt(hac_var_mean(x, L))

def anchor_months(cal):
    cal = pd.DatetimeIndex(sorted(cal))
    months = pd.period_range(cal[0].to_period('M') - 1, cal[-1].to_period('M'), freq='M')
    D = {}
    for m in months:
        t = pd.Timestamp(m.year, m.month, 10)
        i = cal.searchsorted(t)
        if i < len(cal): D[m] = cal[i]
    return D

def anchor_of(day, D):
    best = None
    for m, d in D.items():
        if d <= day and (best is None or m > best): best = m
    return best

def decision_line(cell='K5T|normal|100萬|FIX', strat='B10', plan='AB_V7_B10', return_j=False):
    n = pd.read_parquet(os.path.join(RAW, 'v7/navs_exec7.parquet'))
    p = pd.read_csv(os.path.join(RAW, 'v7/plans_j7.csv'))
    p = p[(p.strat == plan) & (p.seed == 0)]
    ends = json.load(open(os.path.join(RAW, 'v7/jscan_end_days.json')))
    e = pd.read_csv(os.path.join(RAW, 'finlab/data/raw/etf_0050.csv'), parse_dates=['date']).set_index('date')['adj']
    cal = n.index
    D = anchor_months(cal)
    # map day->anchor month quickly
    Ds = sorted(D.items(), key=lambda kv: kv[1])
    dd = pd.DatetimeIndex([d for _, d in Ds]); mm = [m for m, _ in Ds]
    def am(day):
        i = dd.searchsorted(day, side='right') - 1
        return mm[i]
    rows = []
    for j in range(13):
        col = f'{strat}|J{j}|{cell}'
        nav = n[col]
        sig = sorted(pd.to_datetime(p[p.j == j].signal.unique()))
        endj = pd.Timestamp(ends[f'V7_B10|J{j}'])
        pts = list(sig) + ([endj] if endj > sig[-1] else [])
        for a, b in zip(pts[:-1], pts[1:]):
            if a not in nav.index or b not in nav.index: continue
            va, vb = nav.loc[a], nav.loc[b]
            if not (np.isfinite(va) and np.isfinite(vb)): continue
            rs = vb / va - 1
            rb = e.asof(b) / e.asof(a) - 1
            rows.append(dict(j=j, start=a, end=b, month=am(a), rs=rs * 100, rb=rb * 100, ex=(rs - rb) * 100))
    df = pd.DataFrame(rows)
    g = df.groupby('month')
    med = g.ex.median()[g.ex.count() >= 7]
    if return_j: return med, df
    return med
