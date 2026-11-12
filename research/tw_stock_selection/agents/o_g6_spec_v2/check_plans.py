#!/usr/bin/env python3
"""SPEC v2 小型驗證（只讀，不算報酬）：由既有 orig 名單檔核對 j 日日曆的特例；紙上名單檔的欄位與訊號日數。python3 -I -B check_plans.py"""
import json, gzip, csv, io
from pathlib import Path
import pandas as pd
AG = Path('/home/user/-/research/tw_stock_selection/agents')
d = pd.read_csv(AG / 'o_v7_listing_exec_engine/data/plans_j7.csv', usecols=['strat', 'j', 'seed', 'signal', 'exec'])
out = {'strats_in_file': sorted(d.strat.unique().tolist())}
d = d[(d.strat == 'AB_V7_B10') & (d.seed == 0)]
per = {}
for j, g in d.groupby('j'):
    s = pd.to_datetime(g.signal)
    f = lambda ym: [x for x in g.signal if x.startswith(ym)]
    per[int(j)] = dict(n=len(g), first=str(s.min().date()), last=str(s.max().date()), sig_2026_09=f('2026-09'), sig_2024_01=f('2024-01'), sig_2019_02=f('2019-02'),
                       sig_2023_04=f('2023-04'), sig_2020_02=f('2020-02'), sig_2018_01=f('2018-01'),
                       months_missing=sorted(set(str(p) for p in pd.period_range('2006-04', '2026-09', freq='M')) - set(str(p) for p in s.dt.to_period('M'))))
out['AB_V7_B10_seed0_by_j'] = per
L = list(csv.DictReader(io.TextIOWrapper(gzip.open(AG / 'o_g6_allpit_list_trace_stage2/out/k_paper_lists.csv.gz'), encoding='utf-8')))
c0 = [r['signal'] for r in L if r['mask'] == 'M0' and r['variant'] == 'C0']
out['paper_C0'] = dict(n=len(c0), first=min(c0), last=max(c0), n_not_day11=len([x for x in c0 if x[-2:] != '11']), columns=list(L[0].keys()))
Path(__file__).with_name('check_plans.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
print({k: (v['n'], v['first'], v['last'], v['sig_2026_09']) for k, v in per.items()}, out['paper_C0'])
