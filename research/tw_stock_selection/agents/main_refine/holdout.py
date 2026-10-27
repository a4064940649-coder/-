#!/usr/bin/env python3
"""main_refine 第 2 階段（判定寫入後才執行）：保留期 2021-03～2026-10 統計、逐年、最終建議版本。另附 2006～2020 合併段（補充，不參與判定）。"""
import json, sys
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.dont_write_bytecode = True
import rlib as R

res = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))
assert 'RUN' in res and all('verdict' in v for k, v in res['RUN']['variants'].items() if k != 'BASE'), '判定尚未寫入'
ND = pd.read_parquet(HERE / 'navs.parquet')
HS = {'HOLD': ('2021-03-01', '2026-10-31'), 'SUPP_2006_2020': ('2006-04-01', '2020-12-31'), 'SUPP_2024_2026': ('2024-01-01', '2026-10-31')}
out = {'variants': {}}
names = list(res['RUN']['variants'])
for name in names:
    fe = pd.Timestamp(res['RUN']['info'][name]['first_exec']); fb = pd.Timestamp(res['RUN']['info']['BASE']['first_exec'])
    v = {'segments': {}, 'paired_vs_BASE': {}, 'paired_vs_BASE_stress': {}}
    for sk, (a, b) in HS.items():
        a2 = max(pd.Timestamp(a), fe + pd.offsets.MonthBegin(1)) if fe > fb else pd.Timestamp(a)
        for ck in ('slip0.001', 'stress'):
            v['segments'][f'{sk}|{ck}'] = R.seg_stats(ND[f'{name}|{ck}'], a2, b)
        if name != 'BASE':
            v['paired_vs_BASE'][sk] = R.paired(ND[f'{name}|slip0.001'], ND['BASE|slip0.001'], a2, b)
            v['paired_vs_BASE_stress'][sk] = R.paired(ND[f'{name}|stress'], ND['BASE|stress'], a2, b)
    v['yearly_pct_2021plus'] = {y: x for y, x in R.yearly(ND[f'{name}|slip0.001']).items() if y >= 2021}
    out['variants'][name] = v
    print(name, {k: (x['CAGR'], x['maxDD'], x['Calmar'], x['t_vs_0050_m']) for k, x in v['segments'].items() if x and 'slip' in k},
          {k: (x['diff_pct_m'], x['t_paired_m']) for k, x in v['paired_vs_BASE'].items()}, flush=True)
e = R.E50.reindex(ND.index)
out['0050'] = {sk: R.seg_stats(e, a, b) for sk, (a, b) in HS.items()}
out['0050_yearly_pct_2021plus'] = {y: x for y, x in R.yearly(e.loc[ND.index[0]:]).items() if y >= 2021}
# 最終建議（事前規則）
ok = [k for k, v in res['RUN']['variants'].items() if v.get('verdict') == '可用' and out['variants'][k]['paired_vs_BASE']['HOLD']['diff_pct_m'] >= 0]
out['usable_and_hold_nonneg'] = ok
out['FINAL'] = ok[0] if len(ok) == 1 else ('需合併檢定：' + ','.join(ok) if ok else 'BASE（原版 n10，無改良通過）')
res['HOLDOUT'] = out
(HERE / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
print('FINAL', out['FINAL'])
