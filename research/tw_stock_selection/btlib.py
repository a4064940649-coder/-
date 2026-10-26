"""月度選股回測共用函式。

設計重點（避免常見偏誤）：
* 持有期報酬用還原指數（含除權息）；換股日 t 必須有成交才可買；持有期間停止交易或下市者以最後成交價計（不假設 −100%，也不剔除）。
* 年化以實際經過天數計（2024-01 資料缺口使一期跨兩個月）；年度報酬以持有期「結束日」歸年。
* 周轉 = 每期被換掉的檔數比例；成本 = 周轉 × 來回成本（手續費兩邊 + 證交稅 0.3%）。
"""
import math
import numpy as np, pandas as pd

def month_ends(index: pd.DatetimeIndex) -> list:
    s = pd.Series(index, index=index)
    return list(s.groupby([index.year, index.month]).last())

def period_returns(adj: pd.DataFrame, dates: list) -> pd.DataFrame:
    """index=換股日 t（最後一個除外）、columns=股票：t → 下一換股日 的報酬。"""
    A = adj.ffill(limit=30)
    P = A.reindex(dates)
    R = P.shift(-1) / P - 1
    R = R.where(adj.reindex(dates).notna())
    return R.iloc[:-1]

def bench_returns(series: pd.Series, dates: list) -> pd.Series:
    s = series.sort_index().reindex(pd.DatetimeIndex(dates), method='ffill')
    return (s.shift(-1) / s - 1).iloc[:-1]

def _end_index(r: pd.Series, dates: list) -> pd.Series:
    nxt = dict(zip(dates[:-1], dates[1:]))
    return pd.Series(r.values, index=pd.DatetimeIndex([nxt[t] for t in r.index]))

def ann(r: pd.Series, dates: list):
    r = r.dropna()
    if len(r) < 6: return {}
    e = _end_index(r, dates)
    eq = (1 + r).cumprod(); yrs = (e.index[-1] - r.index[0]).days / 365.25
    vol = r.std() * math.sqrt(len(r) / yrs)
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    return {'CAGR': round(float(cagr), 4), 'vol': round(float(vol), 4), 'sharpe0': round(float(cagr / vol), 2) if vol > 0 else None,
            'maxDD': round(float((eq / eq.cummax() - 1).min()), 4), 'periods': int(len(r)), 'years': round(yrs, 2)}

def tstat(x):
    x = pd.Series(x).dropna()
    return round(float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))), 2) if len(x) > 2 and x.std() > 0 else float('nan')

def yearly(r: pd.Series, dates: list):
    e = _end_index(r.dropna(), dates)
    return {int(y): round(float(((1 + g).prod() - 1) * 100), 1) for y, g in e.groupby(e.index.year)}

def run_sort(signal: pd.DataFrame, R: pd.DataFrame, elig: pd.DataFrame, nq=5, top_n=None, rt_cost=0.005, min_names=50):
    """每期依 signal（越大越好）排序：nq 分組等權；若給 top_n，另建前 top_n 檔組合。"""
    groups = {k: {} for k in range(nq)}; top, uni, to, npick = {}, {}, [], []; prev = None
    for t in R.index:
        if t not in signal.index or t not in elig.index: continue
        ok = elig.loc[t].fillna(False).astype(bool) & signal.loc[t].notna() & R.loc[t].notna()
        s = signal.loc[t][ok]
        if s.count() < max(min_names, nq * 5): continue
        r = R.loc[t][ok]; uni[t] = r.mean()
        q = pd.qcut(s.rank(method='first'), nq, labels=False)
        for k in range(nq): groups[k][t] = r[q == k].mean()
        pick = list(s.sort_values(ascending=False).index[:top_n]) if top_n else list(s.index[q == nq - 1])
        top[t] = r[pick].mean(); npick.append(len(pick))
        if prev is not None: to.append(len(set(pick) - prev) / len(pick))
        prev = set(pick)
    res = {f'Q{k+1}': pd.Series(v, dtype=float) for k, v in groups.items()}
    res['universe'] = pd.Series(uni, dtype=float)
    tov = float(np.mean(to)) if to else float('nan')
    res['best_gross'] = pd.Series(top, dtype=float); res['turnover'] = tov; res['n_names'] = float(np.mean(npick)) if npick else 0
    res['best_net'] = res['best_gross'] - tov * rt_cost
    res['best_net_1pct'] = res['best_gross'] - tov * 0.01
    return res

def summarize(res, dates, bench: pd.Series = None, label='', start=None, end=None):
    def cut(x):
        if not isinstance(x, pd.Series): return x
        if start is not None: x = x[x.index >= pd.Timestamp(start)]
        if end is not None: x = x[x.index <= pd.Timestamp(end)]
        return x
    r = {k: cut(v) for k, v in res.items()}
    o = {'label': label, 'turnover': round(res['turnover'], 3), 'n_names': round(res['n_names'], 1)}
    for k in ['best_gross', 'best_net', 'best_net_1pct', 'universe', 'Q1', 'Q5']:
        if k in r: o[k] = ann(r[k], dates)
    sp = (r['Q5'] - r['Q1']).dropna()
    o['Q5_minus_Q1_pct_per_period'] = round(float(sp.mean() * 100), 3); o['t_Q5_Q1'] = tstat(sp)
    ex = (r['best_net'] - r['universe']).dropna()
    o['net_minus_universe_pct_per_period'] = round(float(ex.mean() * 100), 3); o['t_net_vs_universe'] = tstat(ex)
    if bench is not None:
        b = cut(bench).reindex(r['best_net'].index)
        exb = (r['best_net'] - b).dropna()
        o['bench'] = ann(b, dates); o['net_minus_bench_pct_per_period'] = round(float(exb.mean() * 100), 3); o['t_net_vs_bench'] = tstat(exb)
        ys, yb = yearly(r['best_net'], dates), yearly(b, dates)
        o['yearly_net_pct'] = ys; o['yearly_bench_pct'] = yb
        o['years_beat_bench'] = f"{sum(ys[y] > yb.get(y, -1e9) for y in ys)}/{len(ys)}"
    o['yearly_universe_pct'] = yearly(r['universe'], dates)
    return o
