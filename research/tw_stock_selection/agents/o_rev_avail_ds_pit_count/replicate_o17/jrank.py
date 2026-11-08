import pandas as pd, numpy as np
n = pd.read_parquet('/home/user/-/research/tw_stock_selection/agents/o_exec_engine_rewrite_v7x/out/navs_oos_R4.parquet')
e = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/etf_0050.csv', parse_dates=['date']).set_index('date')['adj']
def cagr(s):
    s = s.dropna(); y = (s.index[-1] - s.index[0]).days / 365.25
    return (s.iloc[-1] / s.iloc[0]) ** (1 / y) - 1
def cagr_d(s):
    s = s.dropna(); return (s.iloc[-1] / s.iloc[0]) ** (252 / (len(s) - 1)) - 1
for nm, a in (('ALL', None), ('2019_', '2019-01-01')):
    for f in (cagr, cagr_d):
        res = {}
        for j in range(13):
            s = n[f'B10|J{j}|K5T|normal|100萬|GROW']
            if a: s = s[s.index >= a]
            b = e.reindex(s.index).ffill()
            res[j] = (f(s) - f(b)) * 100
        r = pd.Series(res).sort_values(ascending=False)
        rk = list(r.index).index(0) + 1
        x = np.sort(r.values)
        p10 = np.percentile(r.values, 10)
        lo = r.sort_values().index
        print(nm, f.__name__, 'J0 rank', rk, f'{res[0]:+.2f}', 'median J', r.index[6], f'{r.iloc[6]:+.2f}', f'p10={p10:+.2f} between J{lo[1]} J{lo[2]}', 'drop J0 median', f'{np.median(r.drop(0).values):+.2f}')
        print('   ', {f'J{k}': round(v, 2) for k, v in r.items()})
