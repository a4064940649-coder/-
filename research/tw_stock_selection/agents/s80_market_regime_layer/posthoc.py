#!/usr/bin/env python3
"""S80 事後整理（只讀已存檔的 population.csv、nav_std.csv.gz、results*.json；不重跑引擎、不改判定）。python3 -I -B posthoc.py → results_posthoc.json"""
import json
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
pop = pd.read_csv(HERE / 'population.csv', index_col=0); nav = pd.read_csv(HERE / 'nav_std.csv.gz', index_col=0, parse_dates=True)
REF = [c for c in nav.columns if c not in ('CURRENT', '0050')][0]
cagr = lambda s: float((s.iloc[-1] / s.iloc[0]) ** (365.25 / (s.index[-1] - s.index[0]).days) - 1)
mdd = lambda s: float((s / s.cummax() - 1).min())
out = dict(ref=REF)
# (1) 訓練段拿掉 2008：2008-12-31～2015-12-31（標準路徑）
w = {}
for name, (a, b) in {'2009～2015（拿掉 2008）': ('2008-12-31', '2015-12-31'), '2008 單年': ('2008-01-02', '2008-12-31'), '2010～2015': ('2009-12-31', '2015-12-31')}.items():
    x = nav.loc[a:b]
    w[name] = {c: dict(cagr=cagr(x[c]), maxdd=mdd(x[c])) for c in nav.columns}
    w[name]['rule_minus_current_cagr_pp'] = (w[name][REF]['cagr'] - w[name]['CURRENT']['cagr']) * 100
out['std_path_subwindows'] = w
# (2) 依「空出的錢去哪裡」分組：訓練／驗證的配對差中位數（各格 20 條中位數，再取組內中位數）
o = pop.drop(index='CURRENT').copy()
o['dest'] = [('E50' if '_E50_' in i else 'ALLCASH' if 'ALLCASH' in i else 'CASH') for i in o.index]
cols = ['train_dCAGR', 'train_dMDD', 'valid_dCAGR', 'valid_dMDD', 'valid_dCalmar', 'full_dCAGR', 'full_dMDD']
out['by_dest'] = {k: dict(n=int(len(g)), **{c: float(g[c].median()) for c in cols}, valid_dCAGR_best=float(g['valid_dCAGR'].max()), valid_dCAGR_worst=float(g['valid_dCAGR'].min())) for k, g in o.groupby('dest')}
# (3) 驗證段每少 1 個百分點跌幅要付多少年化（只看跌幅有變淺 ≥ 3 的格）
z = o[o['valid_dMDD'] >= 3]
out['valid_cost_per_pp_dd'] = dict(n=int(len(z)), median=float((-z['valid_dCAGR'] / z['valid_dMDD']).median()), min=float((-z['valid_dCAGR'] / z['valid_dMDD']).min()), max=float((-z['valid_dCAGR'] / z['valid_dMDD']).max()))
# (4) 驗證段 Calmar 差最好的 5 格、年化差最好的 5 格（事後看，不能拿來換格）
out['valid_top5_dCalmar'] = o.sort_values('valid_dCalmar', ascending=False).head(5)[cols].round(3).to_dict('index')
out['valid_top5_dCAGR'] = o.sort_values('valid_dCAGR', ascending=False).head(5)[cols].round(3).to_dict('index')
out['train_top5_RS_cells_in_valid'] = o.sort_values('train_dCalmar', ascending=False).head(5)[cols].round(3).to_dict('index')
out['idle_cash_always'] = o.loc['IDLE_CASH_ALWAYS', cols].round(3).to_dict()
# (5) 現行與凍結格在 2016 起各年誰贏
y = pd.read_csv(HERE / 'yearly.csv')
out['years_rule_beats_current'] = dict(train=int((y[y.year <= 2015].rule_minus_current_pp > 0).sum()), valid=int((y[y.year >= 2016].rule_minus_current_pp > 0).sum()), valid_years=int((y.year >= 2016).sum()))
(HERE / 'results_posthoc.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
print(json.dumps(out, ensure_ascii=False, indent=1))
