"""事後診斷（不影響判定）：驗證期的機制還在不在？E 日價格壓力、E+1 回補、相對賣壓（期內三分位）的劑量反應。"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import iflib as L
RES = json.load(open(os.path.join(L.HERE, 'results.json'))); assert 'PREREG' in RES
T = pd.read_csv(os.path.join(L.HERE, 'event_returns_all.csv'), parse_dates=['eff'], dtype={'code': str})
D = L.panel(); A = D['adj'].ffill(limit=30); R = D['ret'].clip(-.5, .5); EW = D['ew']; idx = A.index
def ar(code, E, k):
    j = A.columns.get_loc(code); i = idx.get_loc(E)
    return R.iat[i + k, j] - EW.iat[i + k] if i + k < len(idx) else np.nan
T['AR0'] = [ar(c, e, 0) for c, e in zip(T.code, T.eff)]; T['AR1'] = [ar(c, e, 1) for c, e in zip(T.code, T.eff)]
T['era'] = np.where(T.eff <= '2018-12-31', 'select_2008_2018', 'oos_2019_2026')
out = {}
for era, g in T.groupby('era'):
    g = g.copy(); g['rel_t'] = g.groupby(g.eff.dt.year).pressure.transform(lambda s: pd.qcut(s.rank(method='first'), 3, labels=False) if len(s) >= 3 else 1)
    o = {}
    for c, h in g.groupby('cls'):
        o[c] = {'n': int(len(h)), 'AR0%': round(h.AR0.mean() * 100, 2), 'AR1%': round(h.AR1.mean() * 100, 2), 'vol_ratio_med': round(h.vol_ratio.median(), 1),
                'x5_net': L.stats(h.x_closeE_5, h.m, 0.005)['mean'], 'x20_net': L.stats(h.x_closeE_20, h.m, 0.005)['mean'], 'pressure_med': round(h.pressure.median(), 2)}
    o['rel_tercile_within_year'] = {int(k): {'n': int(len(h)), 'AR0%': round(h.AR0.mean() * 100, 2), 'AR1%': round(h.AR1.mean() * 100, 2),
             'x20_net': L.stats(h.x_closeE_20, h.m, 0.005), 'x5_net': L.stats(h.x_closeE_5, h.m, 0.005)} for k, h in g.groupby('rel_t')}
    o['v20_median_NTD'] = float(g.v20.median())
    out[era] = o
RES['POSTHOC_diagnostics'] = out
json.dump(RES, open(os.path.join(L.HERE, 'results.json'), 'w'), ensure_ascii=False, indent=1, default=str)
for era, o in out.items():
    print('==', era, 'v20 median', round(o['v20_median_NTD'] / 1e6, 1), 'M')
    for c in [k for k in o if k.isupper() or '_DEL' in k]: print(' ', c, o[c])
    for k, v in o['rel_tercile_within_year'].items(): print('  relT', k, v['n'], v['AR0%'], v['AR1%'], 'x5', v['x5_net']['mean'], 'x20', v['x20_net']['mean'], v['x20_net'].get('t_month'))
