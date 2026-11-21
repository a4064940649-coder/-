#!/usr/bin/env python3
"""S86：把 results.json 整理成報告用表格（印到螢幕，並存 summary_tables.json）。"""
import json, numpy as np, pandas as pd
from pathlib import Path
H = Path(__file__).resolve().parent
R = json.load(open(H / 'results.json')); out = {}
pc = lambda x: round(x * 100, 2)
print('== 各年股利占淨值 %')
Y = pd.DataFrame(R['dividends']['yearly']); cols = [c for c in Y.columns if c not in ('year',)]
Z = Y.copy()
for c in cols:
    if not c.startswith('ntd'): Z[c] = (Z[c] * 100).round(2)
    else: Z[c] = Z[c].round(0)
print(Z.to_string(index=False)); out['yearly'] = Z.to_dict('records')
print('2019on avg', {k: pc(v) for k, v in R['dividends']['avg_2019on'].items()})
print('2016on avg', {k: pc(v) for k, v in R['dividends']['avg_2016on'].items()})
print('q7', R['q7']['event_counts'], round(R['q7']['capture_ratio_value_weighted'], 3))
print('== NHI static')
for k, v in R['nhi']['static'].items():
    print(k, {n: (round(x['prem_pct_per_year'] * 100, 4), round(x['prem_ntd_per_year']), round(x['n_hit_per_year'], 1), round(x['n_pay_per_year'], 1), round(x['div_ntd_per_year']), round(x['max_single_ntd'])) for n, x in v.items()})
for k, v in R['nhi']['path'].items():
    print(k, 'strat', pc(v['strat_nhi']['c19'] - v['strat_base']['c19']), pc(v['strat_nhi']['full'] - v['strat_base']['full']), 'e50', pc(v['e50_nhi']['c19'] - v['e50_base']['c19']), pc(v['e50_nhi']['full'] - v['e50_base']['full']), round(v['strat_nhi_ntd']), round(v['e50_nhi_ntd']))
print('== fee path (2015-07 起, 正式零股制度)')
for k, v in R['fee']['path'].items(): print(k, pc(v['c19']), pc(v['full']), round(v['fee_ntd']))
print('== fee first-year mean')
fy = R['fee']['first_year']; t = {}
for cap in ('30萬', '50萬', '100萬'):
    b = fy[f'{cap}|0.28|低消20']['mean_ret']
    t[cap] = {k.split('|', 1)[1]: round((v['mean_ret'] - b) * 100, 2) for k, v in fy.items() if k.startswith(cap + '|')}
    t[cap]['基準第一年平均%'] = pc(b); print(cap, t[cap])
out['fee_first_year_vs_28_min20'] = t
print('== oddlot (4 位移平均, 零股全期可買)')
o = {}
for k, v in R['oddlot'].items():
    o[k] = dict(c19=pc(np.mean([x['c19'] for x in v.values()])), full=pc(np.mean([x['full'] for x in v.values()])))
for cap in ('30萬', '50萬', '100萬'):
    b = o[f'{cap}|基準']; print(cap, {k.split('|')[1]: (v['c19'], round(v['c19'] - b['c19'], 2), round(v['full'] - b['full'], 2)) for k, v in o.items() if k.startswith(cap)})
out['oddlot'] = o
print('== combined')
C = {}
for nm, v in R['combined'].items():
    mean = lambda d, f: float(np.mean([x[f] for x in d.values()]))
    r = dict(std=dict(strat_before=(pc(v['before']['0']['c19']), pc(v['before']['0']['full'])), strat_after=(pc(v['after']['0']['c19']), pc(v['after']['0']['full']))),
             e50_before=(pc(v['e50_before']['c19']), pc(v['e50_before']['full'])), e50_after=(pc(v['e50_after']['c19']), pc(v['e50_after']['full'])),
             mean4=dict(strat_before=(pc(mean(v['before'], 'c19')), pc(mean(v['before'], 'full'))), strat_after=(pc(mean(v['after'], 'c19')), pc(mean(v['after'], 'full'))),
                        strat_after_upper_mixed=(pc(mean(v['after_upper_mixed'], 'c19')), pc(mean(v['after_upper_mixed'], 'full'))),
                        strat_after_plus_oddlot=(pc(mean(v['after_plus_oddlot'], 'c19')), pc(mean(v['after_plus_oddlot'], 'full')))),
             maxdd_after_std=pc(v['after']['0']['maxdd']), maxdd_before_std=pc(v['before']['0']['maxdd']), e50_maxdd_after=pc(v['e50_after']['maxdd']),
             steps_std={k: (pc(x['c19'] - v['before']['0']['c19']), pc(x['full'] - v['before']['0']['full'])) for k, x in v['steps'].items()},
             e50_steps={k: (pc(x['c19'] - v['e50_before']['c19']), pc(x['full'] - v['e50_before']['full'])) for k, x in v['e50_steps'].items()},
             ntd=v['after_ntd'], e50_ntd=v['e50_after_ntd'], per_shift_after_c19=[pc(x['c19']) for x in v['after'].values()], per_shift_before_c19=[pc(x['c19']) for x in v['before'].values()])
    m4 = r['mean4']
    r['lead_mean4'] = dict(before=(round(m4['strat_before'][0] - r['e50_before'][0], 2), round(m4['strat_before'][1] - r['e50_before'][1], 2)),
                           after=(round(m4['strat_after'][0] - r['e50_after'][0], 2), round(m4['strat_after'][1] - r['e50_after'][1], 2)))
    r['lead_std'] = dict(before=(round(r['std']['strat_before'][0] - r['e50_before'][0], 2), round(r['std']['strat_before'][1] - r['e50_before'][1], 2)),
                         after=(round(r['std']['strat_after'][0] - r['e50_after'][0], 2), round(r['std']['strat_after'][1] - r['e50_after'][1], 2)))
    C[nm] = r; print(nm); [print('  ', k, x) for k, x in r.items()]
out['combined'] = C
json.dump(out, open(H / 'summary_tables.json', 'w'), ensure_ascii=False, indent=1, default=str)
