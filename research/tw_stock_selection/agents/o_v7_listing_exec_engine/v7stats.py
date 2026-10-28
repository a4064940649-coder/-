#!/usr/bin/env python3
"""o_v7 統計工具（由 o_v6 v6stats.py 複製；PER 另加 Y19on＝2019-01 起）（不 import 引擎；只吃淨值與日期）。
窗＝訊號日收盤 → 下一訊號日收盤（與 v4lib.win_ret 相同）；起點歸屬（窗起點落在 [a,b]）＝V4 慣例；
日曆歸屬（o5 教訓 4）：窗終點 ≥ 2024-01-02 的窗全部屬於 HOLD。"""
import math
import numpy as np, pandas as pd

A0, B0 = '2006-04-01', '2026-10-12'
HOLD_CAL_END = pd.Timestamp('2024-01-02')

def tstat(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 3: return float('nan')
    s = x.std(ddof=1)
    return float(x.mean() / s * math.sqrt(len(x))) if s > 0 else float('nan')

def sample(nav, days):
    s = nav.dropna(); s = s[~s.index.duplicated(keep='last')]
    v = s.reindex(s.index.union(days)).ffill().reindex(days)
    v[days < s.index[0]] = np.nan
    return v

def win(nav, sd):
    """sd：訊號日＋評價終點。回傳 DataFrame(index=窗起點, r, end)。"""
    D = pd.DatetimeIndex(sd)
    v = sample(nav, D).to_numpy()
    return pd.DataFrame({'r': v[1:] / v[:-1] - 1, 'end': D[1:]}, index=D[:-1])

def xs(nav, bench, sd, a=A0, b=B0, attr='start'):
    W = win(nav, sd); Wb = win(bench, sd)
    x = (W.r - Wb.r)
    if attr == 'start':
        m = (W.index >= pd.Timestamp(a)) & (W.index <= pd.Timestamp(b))
    else:      # 'end'：窗終點落在 [a, b]
        m = (W.end >= pd.Timestamp(a)) & (W.index <= pd.Timestamp(b))
    return x[m].dropna()

def cagr_mdd(nav, a, b):
    s = nav.dropna(); s = s[(s.index >= pd.Timestamp(a)) & (s.index <= pd.Timestamp(b))]
    r = s.pct_change().dropna()
    if len(r) == 0: return float('nan'), float('nan')
    start = s.index[0]; yrs = (r.index[-1] - start).days / 365.25
    eq = (1 + r).cumprod()
    return float(eq.iloc[-1] ** (1 / yrs) - 1) * 100, float((eq / eq.cummax() - 1).min()) * 100

def cal_t(nav, bench, a=A0, b=B0):
    me = nav.groupby(nav.index.to_period('M')).last(); be = bench.reindex(nav.index).ffill().groupby(nav.index.to_period('M')).last()
    r = (me.pct_change() - be.pct_change()).dropna()
    r = r[(r.index >= pd.Period(a, 'M')) & (r.index <= pd.Period(b, 'M'))]
    return tstat(r.to_numpy())

def endpoint(nav, bench, sd):
    x = xs(nav, bench, sd); o = {'t_all': tstat(x), 'n': int(len(x)), 'mean_pp': float(x.mean() * 100)}
    xsrt = x.sort_values(ascending=False)
    for k in (1, 3, 5, 10): o[f'drop_best{k}'] = tstat(x.drop(xsrt.index[:k]))
    o['drop_last1'] = tstat(x.iloc[:-1])
    o['drop_first_year'] = tstat(x[x.index >= x.index[0] + pd.DateOffset(years=1)])
    o['start_2008'] = tstat(x[x.index >= pd.Timestamp('2008-01-01')])
    o['start_2007'] = tstat(x[x.index >= pd.Timestamp('2007-01-01')])
    yr = x.groupby(x.index.year).sum(); by = int(yr.idxmax())
    o['best_year'] = by; o['drop_best_year'] = tstat(x[x.index.year != by])
    o['best_month'] = str(xsrt.index[0].date()); o['best_month_pp'] = float(xsrt.iloc[0] * 100)
    o['cal_month_t'] = cal_t(nav, bench)
    return o

PER = {'S1': ('2006-04-01', '2013-12-31'), 'S2': ('2014-01-01', '2020-12-31'), 'S3': ('2021-03-01', '2026-10-12'),
       'S4': ('2024-01-01', '2026-10-12'), 'SEL': ('2006-04-01', '2018-12-31'), 'VAL': ('2019-01-01', '2023-12-31'),
       'HOLD': ('2024-01-01', '2026-10-12'), 'ALL': ('2006-04-01', '2026-10-12'), 'POST2019_05': ('2019-05-01', '2026-10-12'),
       'Y19on': ('2019-01-01', '2026-10-12'), 'FROM2008': ('2008-01-01', '2026-10-12')}

def segs(nav, ew, bench, sd):
    out = {}
    for k, (a, b) in PER.items():
        cg, md = cagr_mdd(nav, a, b); x = xs(nav, bench, sd, a, b); xe = xs(nav, ew, sd, a, b)
        out[k] = {'CAGR': cg, 'maxDD': md, 'E50_CAGR': cagr_mdd(bench.reindex(nav.index).ffill(), a, b)[0], 'EW_CAGR': cagr_mdd(ew, a, b)[0],
                  't_0050': tstat(x), 'mean_pp': float(x.mean() * 100), 'n': int(len(x)), 't_EW': tstat(xe), 't_cal': cal_t(nav, bench, a, b)}
    # 日曆歸屬的 HOLD（窗終點 ≥ 2024-01-02）
    x = xs(nav, bench, sd, HOLD_CAL_END, B0, attr='end'); xe = xs(nav, ew, sd, HOLD_CAL_END, B0, attr='end')
    out['HOLD_cal'] = {'t_0050': tstat(x), 'mean_pp': float(x.mean() * 100), 'n': int(len(x)), 't_EW': tstat(xe),
                       'cum_log_vs_0050': float(np.log1p(win(nav, sd).r.reindex(x.index)).sum() - np.log1p(win(bench, sd).r.reindex(x.index)).sum())}
    return out

def yearly(nav, bench, a=A0):
    r = nav.pct_change(); r50 = bench.reindex(nav.index).ffill().pct_change(); m = r.index >= pd.Timestamp(a)
    return {int(y): {'strat': float((1 + g.fillna(0)).prod() - 1) * 100, '0050': float((1 + r50[g.index].fillna(0)).prod() - 1) * 100}
            for y, g in r[m].groupby(r[m].index.year)}

def seed_t(seed_navs, bench, sd):
    ts = np.array([tstat(xs(n, bench, sd)) for n in seed_navs])
    return {'median': float(np.median(ts)), 'min': float(ts.min()), 'max': float(ts.max()), 'n_below_2.0': int((ts < 2.0).sum()),
            'n_below_2.3': int((ts < 2.3).sum()), 'n': int(len(ts))}

def paired(a, b, sd, x0=A0, x1=B0):
    d = (win(a, sd).r - win(b, sd).r)
    d = d[(d.index >= pd.Timestamp(x0)) & (d.index <= pd.Timestamp(x1))].dropna()
    nz = d[d.abs() > 1e-12]
    o = dict(mean_pp=float(d.mean() * 100), t=tstat(d), n=int(len(d)), n_nonzero=int(len(nz)),
             nonzero_months=[str(i.date()) for i in nz.index][:40])
    if len(nz):
        big = nz.abs().idxmax()
        o['largest_month'] = str(big.date()); o['largest_pp'] = float(nz[big] * 100)
        o['t_drop_largest'] = tstat(d.drop(big))
    return o

def mean_nav(rets):
    return (1 + pd.concat(rets, axis=1).mean(axis=1).fillna(0)).cumprod()
