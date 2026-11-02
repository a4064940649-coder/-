#!/usr/bin/env python3
"""V7X 統計（口徑以原表 navs_exec7→table 對到 1e−13 確認）：
年化：j 第一個訊號日收盤 → 該 j 的終點日（jscan_end_days.json；官方日＝2026-10-12）；
t：訊號日窗（訊號日收盤→下一訊號日收盤，最後一窗到終點日）超額對 0050 的一般 t；
分段：SEL 年化 [起點, 2018-12-28]、t 窗起點 ≤ 2018-12-31；Y19on 年化 [2019-01-02, 終點]、窗起點 ≥ 2019-01-01；
HOLD [2024-01-02, 終點]；FROM2008 [2008-01-02, 終點]。"""
import json, math
import numpy as np, pandas as pd
import x_data as D

E50 = pd.Series(D.E50, index=D.IDX)
ENDS = json.loads((D.PNL / 'jscan_end_days.json').read_text())
_P = pd.read_csv(D.PNL / 'plans_j7.csv', usecols=['strat', 'j', 'seed', 'signal'])
SIGD = {}
for (st, j), g in _P[_P.seed == 0].groupby(['strat', 'j']):
    SIGD[(st, int(j))] = pd.DatetimeIndex(g.signal)
SIGD[('AB_V7_B10', 'OFF')] = pd.DatetimeIndex(pd.read_csv(D.O7 / 'plan_V7_B10.csv').signal)
SIGD[('AB_V7', 'OFF')] = pd.DatetimeIndex(pd.read_csv(D.O7 / 'plan_V7.csv').signal)


def end_day(strat, j):
    if j == 'OFF': return pd.Timestamp('2026-10-12')
    k = {'AB_V7_B10': 'AB_V7_B10', 'AB_V7': 'AB_V7'}[strat]
    return pd.Timestamp(ENDS[f'{k}|J{j}'])


def tstat(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 3: return float('nan')
    return float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x))))


def cagr(n, a, b):
    yrs = (b - a).days / 365.25
    return float((n[b] / n[a]) ** (1 / yrs) - 1) * 100


def windows(nav, strat, j):
    e = end_day(strat, j)
    sd = SIGD[(strat, j)]
    sd = sd[sd < e].append(pd.DatetimeIndex([e]))
    r = nav.reindex(sd); rb = E50.reindex(sd)
    return ((r.shift(-1) / r - 1) - (rb.shift(-1) / rb - 1)).iloc[:-1]


def stats(nav, strat, j):
    e = end_day(strat, j)
    nav = nav.loc[:e].dropna(); a = nav.index[0]
    x = windows(nav, strat, j)
    yr = lambda s: nav.index[nav.index >= s][0]
    s19, s24, s08 = yr('2019-01-01'), yr('2024-01-01'), yr('2008-01-01')
    sel_end = nav.index[nav.index <= '2018-12-31'][-1]
    out = dict(CAGR=cagr(nav, a, e), E50_CAGR=cagr(E50, a, e), t_0050=tstat(x), mean_pp=float(x.mean() * 100),
               maxDD=float((nav / nav.cummax() - 1).min() * 100),
               CAGR_SEL=cagr(nav, a, sel_end), E50_CAGR_SEL=cagr(E50, a, sel_end), t_SEL=tstat(x[x.index <= '2018-12-31']),
               CAGR_Y19on=cagr(nav, s19, e), E50_CAGR_Y19on=cagr(E50, s19, e), t_Y19on=tstat(x[x.index >= '2019-01-01']),
               CAGR_HOLD=cagr(nav, s24, e), E50_CAGR_HOLD=cagr(E50, s24, e), t_HOLD=tstat(x[x.index >= '2024-01-01']),
               CAGR_FROM2008=cagr(nav, s08, e), t_FROM2008=tstat(x[x.index >= '2008-01-01']),
               CAGR_SELVAL=cagr(nav, a, nav.index[nav.index <= '2019-04-30'][-1]))
    for k in ('', '_SEL', '_Y19on', '_HOLD'):
        out['exc' + k] = out['CAGR' + k] - out['E50_CAGR' + k]
    return out


def jsumm(per_j):
    """per_j：13 個 j 的 stats dict 清單 → 中位數與 p10。"""
    keys = per_j[0].keys(); o = {}
    for k in keys:
        v = np.array([p[k] for p in per_j], float)
        o[k + '_med'] = float(np.median(v)); o[k + '_p10'] = float(np.percentile(v, 10))
    ck = 'CAGR' if 'CAGR' in per_j[0] else 'CAGR_SEL'
    o['CAGR_by_j'] = [p[ck] for p in per_j]
    if 'exc_Y19on' in per_j[0]: o['n_j_exc_Y19on_pos'] = int(sum(p['exc_Y19on'] > 0 for p in per_j))
    return o


def avg_nav(navs):
    R = pd.concat([n.pct_change() for n in navs], axis=1).mean(1).fillna(0)
    return (1 + R).cumprod()
