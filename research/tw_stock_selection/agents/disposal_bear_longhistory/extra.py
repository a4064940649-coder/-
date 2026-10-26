#!/usr/bin/env python3
"""補充：偏誤平移後的 t_month、月平均淨異常、逐年組合 vs 0050。讀 run.py 的結果（重新執行 run.py 的事件部分）。"""
import json, runpy, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
ns = runpy.run_path(str(HERE / 'run.py'), run_name='not_main_but_runs')
tstat = ns['tstat']; G = ns['G']; res = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))
out = {}
for name in ('main_2005-01_2021-09', 'DECISION_bear_union_MA200_or_6periods_2005-2022', '7pct_2005-01_2015-05', '10pct_2015-06_2021-09'):
    df = G[name]; o = {}
    for sh in (0.0, 0.0049, 0.0102):
        net = df.ab - 0.008 + sh
        m = net.groupby(df.date.dt.to_period('M')).mean()
        o[f'shift_{sh*100:.2f}pp'] = {'net_mean_pct': round(net.mean() * 100, 2), 'month_avg_net_pct': round(m.mean() * 100, 2), 't_month': tstat(m), 'median_net_pct': round(net.median() * 100, 2)}
    out[name] = o
nav = pd.read_parquet(HERE / 'nav.parquet')
port = res['portfolio']
yt = {}
for y in range(2005, 2023):
    yt[y] = {'sleeve_cash': port['base_1M_cash_0.8']['yearly'][str(y)]['ret_pct'], 'sleeve_cash_switch': port['switch_MA200_1M_cash_0.8']['yearly'][str(y)]['ret_pct'],
             'sleeve_0050idle': port['base_1M_0050_0.8']['yearly'][str(y)]['ret_pct'], '0050': port['bench']['0050']['yearly'][str(y)]['ret_pct'],
             'TAIEX': port['bench']['TAIEX_price']['yearly'][str(y)]['ret_pct']}
res['extra_bias_shift_t'] = out; res['extra_yearly_table'] = yt
(HERE / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
print(json.dumps(out, indent=0)); print(pd.DataFrame(yt).T)
