"""seasonal_statarb 共用：資料載入、可投資條件、鎖漲停排除、月度組合評估。"""
import math, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path('/home/user/-/research/tw_stock_selection')
sys.path.insert(0, str(ROOT))
from btlib import month_ends, period_returns, bench_returns, run_sort, summarize, ann, yearly, tstat  # noqa

P = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel')
OUTDIR = Path(__file__).resolve().parent
E50 = '/home/user/jonatw/finlab-txd/data/raw/etf_0050.csv'
STOCK_LIST = '/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/stock_list.csv'

IS_START, IS_END = '2005-01-01', '2019-04-30'
OOS_START, OOS_END = '2019-05-01', '2026-10-31'


def load(*names):
    return [pd.read_parquet(P / f'{n}.parquet') for n in names]


def e50_series():
    return pd.read_csv(E50, parse_dates=['date']).set_index('date')['adj'].sort_index()


def industries():
    s = pd.read_csv(STOCK_LIST, dtype=str)
    s = s[s.market == 'TWSE']
    return s.set_index('code')['industry']


def locked_limit_up(C, H):
    """收盤鎖漲停（近似）：收盤=最高，且漲幅 ≥ 漲停-0.5%（2015-06-01 前 7%，之後 10%）。"""
    chg = C / C.shift(1) - 1
    thr = pd.Series(np.where(C.index < pd.Timestamp('2015-06-01'), 0.065, 0.095), index=C.index)
    return chg.ge(thr, axis=0) & (C >= H - 1e-9)


def eligibility(me, C, V, lag=0, extra=None):
    """60 日均成交值 ≥ 1,000 萬、股價 ≥ 10；用 lag 日前資訊。回傳 (elig, V60 at me)。"""
    V60 = V.rolling(60, min_periods=50).mean()
    Cs, Vs = (C.shift(lag), V60.shift(lag)) if lag else (C, V60)
    el = (Vs.reindex(me) >= 1e7) & (Cs.reindex(me) >= 10)
    if extra is not None:
        el &= extra
    return el, Vs.reindex(me)


def top_liq(el, v60m, n=300):
    rk = v60m.where(el).rank(axis=1, ascending=False)
    return el & (rk <= n)


def cut(x, a, b):
    return x[(x.index >= pd.Timestamp(a)) & (x.index <= pd.Timestamp(b))]


def eval_monthly(r, me, bench, pool, a, b):
    """r/bench/pool：index=換股日 t 的月報酬 series。回傳年化、MDD、對 0050/同池等權 t 值、逐年。"""
    r, bench, pool = cut(r.dropna(), a, b), cut(bench, a, b), cut(pool, a, b)
    o = {'strat': ann(r, me), 'bench_0050': ann(bench.reindex(r.index), me), 'pool_EW': ann(pool.reindex(r.index), me)}
    eb = (r - bench.reindex(r.index)).dropna(); ep = (r - pool.reindex(r.index)).dropna()
    o['ex_0050_pct_m'] = round(float(eb.mean() * 100), 3); o['t_vs_0050'] = tstat(eb)
    o['ex_EW_pct_m'] = round(float(ep.mean() * 100), 3); o['t_vs_EW'] = tstat(ep)
    ys = yearly(r, me); yb = yearly(bench.reindex(r.index).dropna(), me); yp = yearly(pool.reindex(r.index).dropna(), me)
    o['yearly'] = {y: {'strat': ys[y], '0050': yb.get(y), 'EW': yp.get(y)} for y in ys}
    o['years_beat_0050'] = f"{sum(ys[y] > yb.get(y, 1e9) for y in ys)}/{len(ys)}"
    o['years_beat_EW'] = f"{sum(ys[y] > yp.get(y, 1e9) for y in ys)}/{len(ys)}"
    return o


def fmt(o):
    s, b, p = o['strat'], o['bench_0050'], o['pool_EW']
    return (f"CAGR {s.get('CAGR')} MDD {s.get('maxDD')} | 0050 {b.get('CAGR')} EW {p.get('CAGR')} | "
            f"ex0050 {o['ex_0050_pct_m']}%/m t={o['t_vs_0050']} | exEW {o['ex_EW_pct_m']}%/m t={o['t_vs_EW']} | yrs>0050 {o['years_beat_0050']} >EW {o['years_beat_EW']}")


def pick_portfolio(picks: dict, R: pd.DataFrame, rt_cost=0.005):
    """picks：t → 股票清單（等權）。回傳 gross、net（成本 = 本期新換入比例 × 來回成本）、平均周轉。"""
    g, n1, n2, to, prev = {}, {}, {}, [], None
    for t in R.index:
        p = picks.get(t)
        if not p: continue
        r = R.loc[t, p].dropna()
        if len(r) == 0: continue
        cur = set(r.index); tv = 1.0 if prev is None else len(cur - prev) / len(cur)
        to.append(tv); prev = cur
        g[t] = r.mean(); n1[t] = r.mean() - tv * rt_cost; n2[t] = r.mean() - tv * 0.01
    return {'gross': pd.Series(g, dtype=float), 'net': pd.Series(n1, dtype=float), 'net_1pct': pd.Series(n2, dtype=float),
            'turnover': float(np.mean(to[1:])) if len(to) > 1 else float('nan'), 'n_names': float(np.mean([len(picks[t]) for t in g]))}


def pool_ew(R, el):
    return R.where(el.reindex(R.index).fillna(False).astype(bool)).mean(axis=1)
