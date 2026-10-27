"""刪除承接部位 vs 主策略／同池等權／0050 的同窗比較（月報酬 t）。"""
import os, json, math
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
RES = json.load(open(os.path.join(HERE, 'results.json'))); assert 'PREREG' in RES
S = pd.read_parquet(os.path.join(HERE, 'sleeve_nav.parquet'))
N = pd.read_parquet('/home/user/-/research/tw_stock_selection/agents/main_v2/navs_gapfix.parquet')
E50 = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/etf_0050.csv', parse_dates=['date']).set_index('date').adj.sort_index()
X = pd.concat({'del': S.del_sleeve_normal, 'del_stress': S.del_sleeve_stress, 'main': N['EXFIN|n10|normal'], 'ew': N['EXFIN|EW'], 'e50': E50}, axis=1, sort=True).ffill()
out = {}
for k, (a, b) in {'2010-11_2018': ('2010-11-30', '2018-12-31'), '2019_2026-10-12': ('2018-12-31', '2026-10-12')}.items():
    Y = X.loc[a:b]; M = Y.resample('ME').last().pct_change().dropna()
    yrs = (Y.index[-1] - Y.index[0]).days / 365.25
    o = {c: {'CAGR%': round(float((Y[c].iloc[-1] / Y[c].iloc[0]) ** (1 / yrs) - 1) * 100, 2), 'maxDD_daily%': round(float((Y[c] / Y[c].cummax() - 1).min()) * 100, 1)} for c in Y.columns}
    for ref in ('e50', 'main', 'ew'):
        d = M['del'] - M[ref]; o[f't_del_vs_{ref}'] = round(float(d.mean() / (d.std(ddof=1) / math.sqrt(len(d)))), 2); o[f'mean_excess_vs_{ref}%'] = round(float(d.mean()) * 100, 3)
    out[k] = o
RES['COMPARE'] = out
json.dump(RES, open(os.path.join(HERE, 'results.json'), 'w'), ensure_ascii=False, indent=1, default=str)
print(json.dumps(out, ensure_ascii=False, indent=1))
