#!/usr/bin/env python3
"""補充統計（PREREG 之後，用已存的淨值）：挑選期／驗證期／保留期切分、月均超額、修補版對現行版的配對 t（均值與波動拆解）。寫進 results.json 的 EXTRAS。"""
import sys, json, hashlib
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import osurv_core as O
V, K = O.V, O.K
RJ = HERE / 'results.json'
RES = json.loads(RJ.read_text(encoding='utf-8'))
assert RES.get('PREREG') and 'RESULTS' in RES
SPL = {'SEL': ('2006-04-01', '2019-04-30', '2006-04-01'), 'VAL': ('2019-05-01', '2023-12-31', '2019-05-01'), 'HOLD': ('2024-01-01', '2026-10-12', '2024-01-01'),
       'ALL': ('2006-04-01', '2026-10-12', '2006-04-01')}
VERS = ['ORIG', 'R_DEL', 'B05_DEL', 'B10_DEL', 'R_ALL', 'PX_DEL']
NV = {v: pd.read_parquet(HERE / f'navs_{v}.parquet') for v in VERS}
S = V.V2('gapfix', *O.load_yl('ORIG'), exfin=True); _, _, SD = S.bounds()
out = {}
for v in VERS:
    for cf in ('EXFIN|n10', 'ALL|n10', 'EXFIN|n30'):
        u = cf.split('|')[0]
        nav, ew = NV[v][f'{cf}|normal'].dropna(), NV[v][f'{u}|EW'].dropna()
        for k, (a, b, ad) in SPL.items():
            st = V.seg_stats(nav, ew, a, b, SD, ad=ad)
            o = K.stats(nav, a, b, SD, K.E50, ew)
            out[f'{v}|{cf}|{k}'] = {'CAGR': round(st['CAGR'], 2), 'maxDD': round(st['maxDD'], 2), 't_0050': round(st['t_0050'], 3), 't_EW': round(st['t_EW'], 3),
                                    't_0050_cal': round(st['t_0050_cal'], 3), 'ex_0050_month_pp': round(o['ex_0050_m'] * 100, 3), 'n_m': st['n_m']}
# 配對：修補版 − 現行版（訊號窗月報酬）
def wr(nav):
    D = pd.DatetimeIndex([d for d in SD if d in nav.index]); nd = nav.reindex(D)
    return (nd.shift(-1) / nd - 1).iloc[:-1]
PAIR = {}
E50w = wr(K.E50)
for v in VERS[1:]:
    for cf in ('EXFIN|n10', 'ALL|n10', 'EXFIN|n30'):
        a, b = wr(NV[v][f'{cf}|normal'].dropna()), wr(NV['ORIG'][f'{cf}|normal'].dropna())
        x = (a - b).dropna()
        xa, xb = (a - E50w).dropna(), (b - E50w).dropna()
        for k, (lo, hi, _) in SPL.items():
            m = (x.index >= pd.Timestamp(lo)) & (x.index <= pd.Timestamp(hi))
            ma = (xa.index >= pd.Timestamp(lo)) & (xa.index <= pd.Timestamp(hi)); mb = (xb.index >= pd.Timestamp(lo)) & (xb.index <= pd.Timestamp(hi))
            PAIR[f'{v}|{cf}|{k}'] = {'mean_diff_pp_month': round(float(x[m].mean()) * 100, 3), 't_paired': round(K.tstat(x[m]), 3), 'n_months_differ': int((x[m].abs() > 1e-12).sum()),
                                     'ex0050_mean_pp': [round(float(xa[ma].mean()) * 100, 3), round(float(xb[mb].mean()) * 100, 3)],
                                     'ex0050_sd_pp': [round(float(xa[ma].std()) * 100, 3), round(float(xb[mb].std()) * 100, 3)]}
RES['EXTRAS'] = {'splits': out, 'paired_vs_ORIG': PAIR}
RJ.write_text(json.dumps(RES, ensure_ascii=False, indent=1, default=lambda x: x.item() if hasattr(x, 'item') else str(x)), encoding='utf-8')
for v in VERS:
    print(v, {k: (out[f'{v}|EXFIN|n10|{k}']['CAGR'], out[f'{v}|EXFIN|n10|{k}']['t_0050'], out[f'{v}|EXFIN|n10|{k}']['ex_0050_month_pp'], out[f'{v}|EXFIN|n10|{k}']['maxDD']) for k in SPL})
for k, v in PAIR.items():
    if 'EXFIN|n10' in k or k.endswith('|ALL'): print(k, v)
