#!/usr/bin/env python3
"""replicate_o10 r3：統計（自寫）。訊號窗＝訊號日收盤→下一訊號日收盤，最後一窗到 jscan 終點日；洞月份的訊號窗剔除。"""
import json, math, pickle, sys
from pathlib import Path
import numpy as np, pandas as pd

SCR = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/rep_o10')
AG = Path(__file__).resolve().parents[2]
PNL = AG / 'o_v7_listing_exec_engine' / 'data'
ENDS = json.loads((PNL / 'jscan_end_days.json').read_text())
HOLES = {'2022-10'} | {f'2023-{m:02d}' for m in range(1, 10)}
P = pd.read_csv(PNL / 'plans_j7.csv', usecols=['strat', 'j', 'seed', 'signal'])
P = P[(P.strat == 'AB_V7_B10') & (P.seed == 0)]
SIGD = {int(j): pd.DatetimeIndex(g.signal) for j, g in P.groupby('j')}
e50 = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/etf_0050.csv', parse_dates=['date']).set_index('date')['adj'].sort_index()


def tstat(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 3: return float('nan')
    return float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x))))


def nw_t(x, L=None):
    x = np.asarray(x, float); x = x[np.isfinite(x)]; n = len(x)
    if n < 5: return float('nan')
    if L is None: L = int(math.floor(4 * (n / 100) ** (2 / 9)))
    u = x - x.mean(); s = (u @ u) / n
    for l in range(1, L + 1):
        s += 2 * (1 - l / (L + 1)) * (u[l:] @ u[:-l]) / n
    return float(x.mean() / math.sqrt(s / n))


def win(nav, j):
    e = pd.Timestamp(ENDS[f'AB_V7_B10|J{j}'])
    sd = SIGD[j]; sd = sd[sd < e].append(pd.DatetimeIndex([e]))
    E = e50.reindex(e50.index.union(nav.index)).ffill().reindex(nav.index)
    r = nav.reindex(sd).to_numpy(); b = E.reindex(sd).to_numpy()
    df = pd.DataFrame(dict(rs=r[1:] / r[:-1] - 1, rb=b[1:] / b[:-1] - 1, days=(sd[1:] - sd[:-1]).days), index=sd[:-1])
    df['x'] = df.rs - df.rb
    df['hole'] = [str(p) in HOLES for p in df.index.to_period('M')]
    return df


def ann(r, days):
    return float((np.prod(1 + r) ** (365.25 / days.sum()) - 1) * 100) if len(r) else float('nan')


PER = dict(ALL=('2000', '2100'), SEL=('2000', '2019-04-30'), VAL=('2019-05-01', '2023-12-31'), HOLD=('2024-01-01', '2100'),
           OOS=('2019-05-01', '2100'), Y19=('2019-01-01', '2100'), S1=('2000', '2012-12-31'), S2=('2013-01-01', '2018-12-31'),
           S3=('2019-01-01', '2100'))


def stats_j(nav, j, holes=False):
    df = win(nav, j)
    if not holes: df = df[~df.hole]
    o = {}
    for k, (a, b) in PER.items():
        d = df[(df.index >= a) & (df.index <= b)]
        o[f'ann_{k}'] = ann(d.rs, d.days); o[f'b_{k}'] = ann(d.rb, d.days)
        o[f'exc_{k}'] = o[f'ann_{k}'] - o[f'b_{k}']; o[f't_{k}'] = tstat(d.x); o[f'n_{k}'] = len(d)
        o[f'mpp_{k}'] = float(d.x.mean() * 100)
    o['nwt_ALL'] = nw_t(df.x)
    yy = df.groupby(df.index.year).x.sum().sort_values()
    o['t_drop3'] = tstat(df.x[~df.index.year.isin(yy.index[-3:])])
    e = pd.Timestamp(ENDS[f'AB_V7_B10|J{j}']); n = nav.loc[:e]
    E = e50.reindex(e50.index.union(n.index)).ffill().reindex(n.index)
    yrs = (n.index[-1] - n.index[0]).days / 365.25
    o['daily_cagr'] = float(((n.iloc[-1] / n.iloc[0]) ** (1 / yrs) - 1) * 100)
    o['daily_cagr_0050'] = float(((E.iloc[-1] / E.iloc[0]) ** (1 / yrs) - 1) * 100)
    o['maxdd'] = float((n / n.cummax() - 1).min() * 100)
    return o, df


def summ(navd, holes=False):
    per = [stats_j(navd[j][0], j, holes)[0] for j in range(13)]
    out = {}
    for k in per[0]:
        v = np.array([p[k] for p in per], float)
        out[k] = (round(float(np.median(v)), 3), round(float(np.percentile(v, 10)), 3))
    return out


def pair(navA, navB):
    o = {k: [] for k in PER}; t = {k: [] for k in PER}
    for j in range(13):
        a = win(navA[j][0], j); b = win(navB[j][0], j)
        d = (a.rs - b.rs)[~a.hole]
        for k, (s, e) in PER.items():
            x = d[(d.index >= s) & (d.index <= e)]
            o[k].append(x.mean() * 100); t[k].append(tstat(x))
    return {k: (round(float(np.median(o[k])), 3), round(float(np.median(t[k])), 2)) for k in PER}


def load(var, cell):
    return pickle.load(open(SCR / f'nav_{var}_{cell}.pkl', 'rb'))


if __name__ == '__main__':
    var, cell = sys.argv[1], sys.argv[2]
    s = summ(load(var, cell))
    keys = ['ann_ALL', 'b_ALL', 'exc_ALL', 't_ALL', 'nwt_ALL', 't_drop3', 'daily_cagr', 'daily_cagr_0050', 'maxdd', 'n_ALL',
            'exc_SEL', 't_SEL', 'exc_VAL', 't_VAL', 'ann_HOLD', 'b_HOLD', 'exc_HOLD', 't_HOLD', 'exc_OOS', 't_OOS', 'exc_Y19', 't_Y19',
            'exc_S1', 'exc_S2', 'exc_S3', 'mpp_ALL', 'mpp_Y19']
    for k in keys: print(k, s[k])
