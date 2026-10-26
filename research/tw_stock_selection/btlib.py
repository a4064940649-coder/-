"""月度選股回測共用函式：持有期報酬（含下市股以最後成交價計）、分組／前 N 名組合、周轉、成本、統計。"""
import math
import numpy as np, pandas as pd

def period_returns(adj: pd.DataFrame, dates: list) -> pd.DataFrame:
    """回傳 index=換股日 t、columns=股票 的「t → 下一換股日」報酬；t 之後若停止交易，以最後成交價計（不假設 −100%）。"""
    A = adj.ffill(limit=30)
    P = A.reindex(dates)
    raw_at_t = adj.reindex(dates)
    R = P.shift(-1) / P - 1
    R = R.where(raw_at_t.notna())        # t 當日必須有成交才可買
    return R.iloc[:-1]

def ann(r: pd.Series, ppy=12):
    r = r.dropna()
    if len(r) < 6: return {}
    eq = (1 + r).cumprod(); yrs = len(r) / ppy; vol = r.std() * math.sqrt(ppy)
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    return {'CAGR': round(float(cagr), 4), 'vol': round(float(vol), 4), 'sharpe': round(float(cagr / vol), 2) if vol > 0 else None,
            'maxDD': round(float((eq / eq.cummax() - 1).min()), 4), 'periods': int(len(r))}

def tstat(x):
    x = pd.Series(x).dropna(); return float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))) if len(x) > 2 and x.std() > 0 else float('nan')

def run_sort(signal: pd.DataFrame, R: pd.DataFrame, elig: pd.DataFrame, nq=5, top_n=None, weights=None, rt_cost=0.005, higher_better=True):
    """每個換股日依 signal 排序：nq 分組等權（或 top_n 檔）；回傳各組報酬、周轉、成本後報酬。"""
    out = {k: {} for k in range(nq)}; top = {}; uni = {}; to = []; prev = set(); names = []
    for t in R.index:
        if t not in signal.index: continue
        ok = elig.loc[t] & signal.loc[t].notna() & R.loc[t].notna()
        s = signal.loc[t][ok]
        if not higher_better: s = -s
        if s.count() < max(50, nq * 10): continue
        r = R.loc[t][ok]
        uni[t] = r.mean()
        q = pd.qcut(s.rank(method='first'), nq, labels=False)
        for k in range(nq): out[k][t] = r[q == k].mean()
        if top_n:
            pick = s.sort_values(ascending=False).index[:top_n]
            top[t] = r[pick].mean(); names.append(len(pick))
            to.append(len(set(pick) - prev) / len(pick)); prev = set(pick)
        else:
            pick = set(s.index[q == nq - 1]); to.append(len(pick - prev) / max(len(pick), 1)); prev = pick
    res = {f'Q{k+1}': pd.Series(v) for k, v in out.items()}
    res['universe'] = pd.Series(uni)
    tov = float(np.mean(to)) if to else float('nan')
    best = pd.Series(top) if top_n else res[f'Q{nq}']
    res['best_gross'] = best; res['best_net'] = best - tov * rt_cost; res['turnover'] = tov
    return res

def summarize(res, bench: pd.Series = None, label=''):
    o = {'label': label, 'turnover': round(res['turnover'], 3)}
    for k in ['best_gross', 'best_net', 'universe', 'Q1', 'Q5']:
        if k in res and isinstance(res[k], pd.Series): o[k] = ann(res[k])
    sp = (res.get('Q5', pd.Series(dtype=float)) - res.get('Q1', pd.Series(dtype=float))).dropna()
    if len(sp): o['Q5_Q1_monthly_pct'] = round(float(sp.mean() * 100), 3); o['t_Q5_Q1'] = round(tstat(sp), 2)
    ex = (res['best_net'] - res['universe']).dropna(); o['best_net_minus_universe_pct'] = round(float(ex.mean() * 100), 3); o['t_vs_universe'] = round(tstat(ex), 2)
    if bench is not None:
        b = bench.reindex(res['best_net'].index)
        exb = (res['best_net'] - b).dropna(); o['bench'] = ann(b); o['best_net_minus_bench_pct'] = round(float(exb.mean() * 100), 3); o['t_vs_bench'] = round(tstat(exb), 2)
        yr_s = res['best_net'].groupby(res['best_net'].index.year).apply(lambda g: (1 + g).prod() - 1)
        yr_b = b.groupby(b.index.year).apply(lambda g: (1 + g).prod() - 1)
        o['yearly_net_pct'] = {int(k): round(float(v * 100), 1) for k, v in yr_s.items()}
        o['yearly_bench_pct'] = {int(k): round(float(v * 100), 1) for k, v in yr_b.items()}
        o['years_beat'] = f"{int(sum(yr_s.get(y, 0) > yr_b.get(y, 0) for y in yr_s.index))}/{len(yr_s)}"
    return o
