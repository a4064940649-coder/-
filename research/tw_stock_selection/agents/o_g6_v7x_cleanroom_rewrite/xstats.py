# 統計層（clean-room）：13 j 中位數、p10、訊號窗 t、三段、2019 起
import numpy as np, pandas as pd, math
import xdata

def seed_avg(navs):
    """navs: list of arrays（同長度，起點前 NaN）→ 日報酬平均後連乘"""
    M = np.vstack(navs)
    r = M[:, 1:] / M[:, :-1] - 1
    ra = np.nanmean(np.where(np.isfinite(r), r, np.nan), axis=0)
    out = np.full(M.shape[1], np.nan)
    st = int(np.where(np.isfinite(M[0]))[0][0]); out[st] = 1.0
    for t in range(st + 1, M.shape[1]):
        out[t] = out[t - 1] * (1 + (ra[t - 1] if np.isfinite(ra[t - 1]) else 0.0))
    return out

def ann(nav, a, b, dates):
    days = (dates[b] - dates[a]).days
    return (nav[b] / nav[a]) ** (365.25 / days) - 1

def last_le(dates, d):
    return int(np.searchsorted(dates.values, np.datetime64(pd.Timestamp(d)), 'right')) - 1

def tstat(x):
    x = np.asarray(x, float); n = len(x)
    if n < 3: return float('nan')
    return x.mean() / (x.std(ddof=1) / math.sqrt(n))

def metrics(nav, sig_idx, end, E, dates):
    st = sig_idx[0]
    out = {}
    out['ann'] = ann(nav, st, end, dates); out['ann0050'] = ann(E, st, end, dates); out['exc'] = out['ann'] - out['ann0050']
    w = [s for s in sig_idx if s < end] + [end]
    wr = np.array([nav[w[k + 1]] / nav[w[k]] - 1 for k in range(len(w) - 1)])
    we = np.array([E[w[k + 1]] / E[w[k]] - 1 for k in range(len(w) - 1)])
    ws = np.array([dates[w[k]] for k in range(len(w) - 1)])
    ex = wr - we
    out['t'] = tstat(ex); out['n_win'] = len(ex)
    b18 = last_le(dates, '2018-12-31'); b12 = last_le(dates, '2012-12-31')
    out['ann_y19'] = ann(nav, b18, end, dates); out['ann0050_y19'] = ann(E, b18, end, dates); out['exc_y19'] = out['ann_y19'] - out['ann0050_y19']
    out['t_y19'] = tstat(ex[ws >= pd.Timestamp('2019-01-01')])
    out['exc_s1'] = ann(nav, st, b12, dates) - ann(E, st, b12, dates)
    out['exc_s2'] = ann(nav, b12, b18, dates) - ann(E, b12, b18, dates)
    out['exc_s3'] = out['exc_y19']
    seg = nav[st:end + 1]; out['mdd'] = float(np.min(seg / np.maximum.accumulate(seg)) - 1)
    out['win_ex'] = ex.tolist(); out['win_start'] = [str(pd.Timestamp(x).date()) for x in ws]
    return out

def jsummary(per_j):
    """per_j: {j: metrics}；超額＝每個 j 的超額再取中位數"""
    ks = ['ann', 'ann0050', 'exc', 't', 'ann_y19', 'ann0050_y19', 'exc_y19', 't_y19', 'exc_s1', 'exc_s2', 'exc_s3', 'mdd']
    o = {}
    for k in ks:
        v = np.array([per_j[j][k] for j in sorted(per_j)])
        o[k + '_med'] = float(np.median(v))
    o['exc_p10'] = float(np.percentile([per_j[j]['exc'] for j in per_j], 10))
    o['exc_y19_p10'] = float(np.percentile([per_j[j]['exc_y19'] for j in per_j], 10))
    return o
