#!/usr/bin/env python3
"""挑選期＋驗證期（長期營收）：全部規則、選 K、寫判定（RUN）。保留期在 holdout.py。"""
import sys, json, hashlib, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import bd_core as K
import bd_exp as E

HERE = K.HERE; T0 = time.time()
R = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))
s = json.dumps(R['PREREG'], ensure_ascii=False, sort_keys=True)
assert hashlib.sha256(s.encode()).hexdigest() == R['PREREG_sha256'], 'PREREG 被改過'
PER = {'SEL': ('2006-04-01', '2013-12-31'), 'VAL': ('2014-01-01', '2020-12-31'), 'SELVAL': ('2006-04-01', '2020-12-31')}
navs = {}
YL = K.yoy_from_long()
res, plans = E.evaluate(YL, '2005-01-01', '2020-12-31', PER, navs, 'long')
E.add_matched(res, PER)
print('evaluated', f'{time.time()-T0:.0f}s', flush=True)

def verdict(ru, rule, slip=0.001, alt='cash'):
    o = ru[f'{rule}|{alt}|{slip}']; b = ru[f'BASE|cash|{slip}']
    sel = (o['SEL']['Calmar'] > b['SEL']['Calmar']) and (o['SEL']['t_vs_base'] > -1)
    val = (o['VAL']['Calmar'] > b['VAL']['Calmar']) and (o['VAL']['CAGR'] >= b['VAL']['CAGR'] - 0.03)
    return sel, val

RUN = {}
for u, ru in res.items():
    cal = {k: ru[f'D1_K{k}|cash|0.001']['SEL']['Calmar'] for k in E.KS}
    kbest = max(cal, key=lambda k: cal[k])
    sel_rules = {'D1': f'D1_K{kbest}', 'D2': 'D2', 'D3': 'D3'}
    vv = {}
    for fam, rule in sel_rules.items():
        sel, val = verdict(ru, rule)
        sel_s, val_s = verdict(ru, rule, slip='stress')
        if sel and val: v = '待觀察' if not val_s else '可用'
        elif sel or val: v = '待觀察'
        else: v = '不可用'
        sel5, val5 = verdict(ru, rule, alt='0050')
        vv[fam] = {'rule': rule, 'SEL_ok': sel, 'VAL_ok': val, 'stress_SEL_ok': sel_s, 'stress_VAL_ok': val_s, 'verdict': v,
                   '0050版_SEL_ok': sel5, '0050版_VAL_ok': val5}
    # 其他 K 只報告
    other = {f'D1_K{k}': dict(zip(('SEL_ok', 'VAL_ok'), verdict(ru, f'D1_K{k}'))) for k in E.KS}
    RUN[u] = {'SEL_Calmar_by_K': cal, 'K_selected': kbest, 'verdicts': vv, 'all_K_report_only': other}
    print(u, RUN[u], flush=True)

R['RUN'] = RUN; R['RESULTS_SELVAL'] = res; R['EXPOSURE_PLANS_long'] = plans
R['B_series'] = {str(d.date()): [int(b), int(p)] for d, (b, p) in E.BSER.iterrows()}
R.pop('HOLDOUT', None)
(HERE / 'results.json').write_text(json.dumps(R, ensure_ascii=False, indent=1, default=float), encoding='utf-8')
pd.DataFrame(navs).to_parquet(HERE / 'navs_selval.parquet')
print('done', f'{time.time()-T0:.0f}s')
