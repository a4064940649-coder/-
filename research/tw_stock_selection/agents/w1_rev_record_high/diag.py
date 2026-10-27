#!/usr/bin/env python3
"""補充診斷（看過主結果後加的，不影響判定）：驗證期去掉 2026、逐段月超額、2008-09 影響。讀 nav.parquet 與 _info_*.pkl，輸出 results_diag.json。"""
import json, math
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
nav = pd.read_parquet(HERE / 'nav.parquet')
e50 = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/etf_0050.csv', parse_dates=['date']).set_index('date')['adj'].sort_index().reindex(nav.index, method='ffill')
def t(x):
    x = pd.Series(x).dropna(); return round(float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))), 2)
out = {}
W = {'valid_ex2026_2019-05_2025-12': ('2019-05-01', '2025-12-31'), 'valid_2019-05_2020-12': ('2019-05-01', '2020-12-31'),
     'valid_2021-01_2023-12': ('2021-01-01', '2023-12-31'), 'select_ex2008-09_2010-01_2019-04': ('2010-01-01', '2019-04-30'),
     'y2026': ('2026-01-01', '2026-10-31')}
for name in ('H_all', 'H48', 'H13'):
    info = pd.read_pickle(HERE / f'_info_{name}.pkl')
    D = pd.DatetimeIndex(list(info.d) + [nav.index[-1]])
    o = {}
    for k, (a, b) in W.items():
        def mr(s):
            s = s.dropna(); nd = s.reindex(D); return (nd.shift(-1) / nd - 1).iloc[:-1]
        st = mr(nav[f'{name}_0.001']); ew = mr(nav[f'{name}_EW']); b5 = mr(e50)
        m = (st.index >= pd.Timestamp(a)) & (st.index < pd.Timestamp(b)) & st.notna()
        if name == 'H_all' or True:
            m &= ~((st.index >= pd.Timestamp('2013-02-01')) & (st.index < pd.Timestamp('2014-02-01')))
        r = nav[f'{name}_0.001'].pct_change(); dm = (r.index > pd.Timestamp(a)) & (r.index <= pd.Timestamp(b)) & r.notna()
        yrs = (r.index[dm][-1] - r.index[dm][0]).days / 365.25
        cg = lambda s: round(float((1 + s.pct_change()[dm].fillna(0)).prod() ** (1 / yrs) - 1), 4)
        o[k] = {'n_months': int(m.sum()), 'CAGR': cg(nav[f'{name}_0.001']), '0050_CAGR': cg(e50), 'EW_CAGR': cg(nav[f'{name}_EW']),
                't_vs_EW_m': t((st - ew)[m]), 'excess_vs_EW_pct_m': round(float((st - ew)[m].mean() * 100), 2), 't_vs_0050_m': t((st - b5)[m])}
    out[name] = o
(HERE / 'results_diag.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
for k, v in out.items():
    print(k); [print('  ', kk, vv) for kk, vv in v.items()]
