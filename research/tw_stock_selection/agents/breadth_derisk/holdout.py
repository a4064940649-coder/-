#!/usr/bin/env python3
"""保留期 2021-03～2026-10（rev_5y）：判定寫完後才算，只報告。"""
import sys, json, hashlib, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd
import bd_core as K
import bd_exp as E
HERE = K.HERE; T0 = time.time()
R = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))
s = json.dumps(R['PREREG'], ensure_ascii=False, sort_keys=True)
assert hashlib.sha256(s.encode()).hexdigest() == R['PREREG_sha256']
assert 'RUN' in R and all('verdicts' in v for v in R['RUN'].values()), '判定尚未寫入，不能算保留期'
PER = {'HOLD': ('2021-03-01', '2026-10-31'), 'HOLD_2024_2026': ('2024-01-01', '2026-10-31')}
navs = {}
Y5 = K.yoy_from_rev5y()
res, plans = E.evaluate(Y5, '2021-01-01', None, PER, navs, 'r5y')
E.add_matched(res, PER)
R['HOLDOUT'] = res; R['EXPOSURE_PLANS_r5y'] = plans
(HERE / 'results.json').write_text(json.dumps(R, ensure_ascii=False, indent=1, default=float), encoding='utf-8')
pd.DataFrame(navs).to_parquet(HERE / 'navs_hold.parquet')
print('done', f'{time.time()-T0:.0f}s')
