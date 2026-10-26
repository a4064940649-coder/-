#!/usr/bin/env python3
"""處置股策略的跨市場樣本外檢驗：同一規則（處置第一天收盤買、最後一天收盤賣）套用在上櫃（TPEx）股。
上櫃價格：tw-stock-screener history.csv.gz（官方日 K，2025-09-24 起），除權息用 exdiv_5y 參考價調整。
上市同期間結果並列對照。異常報酬＝個股 − 同市場等權。輸出 output/bt_disposal_tpex.json
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_twse_panel import compute_returns
X = Path('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data'); OUT = Path(__file__).resolve().parent / 'output'
lst = pd.read_csv(X / 'stock_list.csv', dtype=str)
h = pd.read_csv(X / 'history.csv.gz', dtype={'code': str}, parse_dates=['date']); h = h[(h['volume'] > 0) & (h['close'] > 0)]
ex = pd.read_csv(X / 'extras/exdiv_5y.csv.gz', dtype={'code': str}, parse_dates=['date']); ex = ex[(ex.ref_price > 0) & (ex.prev_close > 0)].copy()
ex['f'] = ex.prev_close / ex.ref_price; ex = ex[(ex.f >= 1) & (ex.f < 3)]; exg = {c: g.sort_values('date') for c, g in ex.groupby('code')}
dp = pd.read_csv(X / 'extras/disposal_5y.csv.gz', dtype={'code': str}, parse_dates=['date', 'start', 'end'])
CAL = pd.DatetimeIndex(sorted(h['date'].unique()))

def build(market):
    codes = set(lst.loc[lst.market == market, 'code'])
    R, V = {}, {}
    for code, g in h[h.code.isin(codes)].groupby('code'):
        g = g.set_index('date').sort_index()
        df = pd.DataFrame({'o': g.open, 'h': g.high, 'c': g.close, 'val': g.volume * g.close, 'x': False, 'chgv': np.nan, 'src': 4.0})
        r, _ = compute_returns(df, [(CAL[0], CAL[-1])], exg.get(code), CAL)
        R[code] = pd.Series(r, index=df.index); V[code] = df['val']
    return pd.DataFrame(R).reindex(CAL), pd.DataFrame(V).reindex(CAL)

def run(market):
    R, V = build(market); adj = (1 + R.fillna(0)).cumprod().where(V.notna())
    liq = V.rolling(20, min_periods=15).mean() >= 1e7
    mkt = R.where(liq).mean(axis=1)
    ev = dp[dp.code.isin(R.columns) & (dp.start >= CAL[20])]
    rows, hold, ent = [], pd.DataFrame(False, index=CAL, columns=R.columns), pd.Series(0.0, index=CAL)
    for _, r in ev.iterrows():
        i0 = CAL.searchsorted(r.start); i1 = CAL.searchsorted(r.end, 'right') - 1
        if i0 >= len(CAL) or i1 <= i0 or not np.isfinite(adj[r.code].iat[i0]) or not np.isfinite(adj[r.code].iat[i1]): continue
        raw = adj[r.code].iat[i1] / adj[r.code].iat[i0] - 1
        m = float(np.prod(1 + mkt.iloc[i0 + 1:i1 + 1].fillna(0)) - 1)
        rows.append({'start': r.start, 'ret': raw, 'abn': raw - m, 'days': i1 - i0})
        hold.iloc[i0 + 1:i1 + 1, hold.columns.get_loc(r.code)] = True; ent.iat[i0] += 1
    d = pd.DataFrame(rows)
    n = hold.sum(axis=1); g = R.where(hold).mean(axis=1)
    cost = (ent.shift(1).fillna(0) / n.replace(0, np.nan)).fillna(0) * 0.005
    p = (g - cost).where(n > 0).fillna(0)
    p = p[p.index >= d.start.min()]
    eq = (1 + p).cumprod(); yrs = (p.index[-1] - p.index[0]).days / 365.25
    byd = d.groupby('start').abn.mean()
    mk = mkt[mkt.index >= d.start.min()].fillna(0); eqm = (1 + mk).cumprod()
    return {'events': int(len(d)), 'mean_pct': round(d.ret.mean() * 100, 2), 'median_pct': round(d.ret.median() * 100, 2), 'hit_rate': round(float((d.ret > 0).mean()), 3),
            'mean_abnormal_pct': round(d.abn.mean() * 100, 2), 't_abnormal_by_date': round(float(byd.mean() / (byd.std() / np.sqrt(len(byd)))), 2),
            'median_days': float(d.days.median()), 'portfolio_return_total': round(float(eq.iloc[-1] - 1), 4), 'portfolio_CAGR': round(float(eq.iloc[-1] ** (1 / yrs) - 1), 4),
            'portfolio_maxDD': round(float((eq / eq.cummax() - 1).min()), 4), 'market_EW_total_same_window': round(float(eqm.iloc[-1] - 1), 4),
            'window': [str(p.index[0].date()), str(p.index[-1].date())],
            'by_quarter_mean_pct': {str(k): round(v * 100, 2) for k, v in d.groupby(d.start.dt.to_period('Q')).ret.mean().items()}}
res = {'TPEX_out_of_sample': run('TPEX'), 'TWSE_same_window': run('TWSE')}
(OUT / 'bt_disposal_tpex.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
for k, v in res.items(): print(k, v)
