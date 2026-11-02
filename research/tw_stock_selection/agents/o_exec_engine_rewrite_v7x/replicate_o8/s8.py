#!/usr/bin/env python3
"""統計（自寫）：年化、maxDD、訊號窗 t、分期。"""
import math, json
import numpy as np, pandas as pd
import d8

D = d8.load(); IDX = D['IDX']
E50 = pd.Series(D['E50'], index=IDX)
ENDS = json.loads((d8.O7 / 'data' / 'jscan_end_days.json').read_text())


def cagr(nav, a, b):
    a = pd.Timestamp(a); b = pd.Timestamp(b)
    s = nav[(nav.index >= a) & (nav.index <= b)]
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    return (float(s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1) * 100


def mdd(nav, a, b):
    s = nav[(nav.index >= pd.Timestamp(a)) & (nav.index <= pd.Timestamp(b))]
    return float((s / s.cummax() - 1).min()) * 100


def tstat(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 3: return float('nan')
    return float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x))))


def win_excess(nav, sdays, end):
    D_ = list(pd.DatetimeIndex(sdays)) + [pd.Timestamp(end)]
    D_ = pd.DatetimeIndex(sorted(set(d for d in D_ if d <= pd.Timestamp(end))))
    r = nav.reindex(D_); rb = E50.reindex(D_)
    x = ((r.shift(-1) / r) - (rb.shift(-1) / rb)).iloc[:-1]
    return x


def cell_stats(nav, sdays, end):
    a = pd.Timestamp(sdays[0]); b = pd.Timestamp(end)
    e = E50[(E50.index >= a) & (E50.index <= b)]
    x = win_excess(nav, sdays, end)
    out = dict(CAGR=cagr(nav, a, b), E50=cagr(e, a, b), maxDD=mdd(nav, a, b), t=tstat(x))
    for nm, (p, q) in dict(SEL=(a, '2018-12-28'), Y19=('2019-01-01', b), VAL=('2019-05-01', '2023-12-31'), HOLD=('2024-01-01', b)).items():
        out[f'CAGR_{nm}'] = cagr(nav, p, q); out[f'E50_{nm}'] = cagr(e, p, q)
    out['t_Y19'] = tstat(x[x.index >= pd.Timestamp('2019-01-01')])
    out['t_SEL'] = tstat(x[x.index < pd.Timestamp('2019-01-01')])
    return out


def jmedian(stats_by_j):
    """stats_by_j: list of dict（13 個 j）→ 中位數；超額＝每個 j 的 (年化−同窗 0050) 再取中位數。"""
    S = pd.DataFrame(stats_by_j)
    out = dict(CAGR=S.CAGR.median(), p10=S.CAGR.quantile(0.1), E50=S.E50.median(), t=S.t.median(),
               ex=(S.CAGR - S.E50).median(), ex_SEL=(S.CAGR_SEL - S.E50_SEL).median(), ex_Y19=(S.CAGR_Y19 - S.E50_Y19).median(),
               ex_VAL=(S.CAGR_VAL - S.E50_VAL).median(), HOLD=S.CAGR_HOLD.median(), E50_HOLD=S.E50_HOLD.median(),
               maxDD=S.maxDD.median(), CAGR_SEL=S.CAGR_SEL.median(), t_Y19=S.t_Y19.median())
    return {k: float(v) for k, v in out.items()}
