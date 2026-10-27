#!/usr/bin/env python3
"""事後描述（不參與判定）：KY_ALL 與 MAIN_R 名單不同的月份，新進股（KY 與非 KY）對被擠掉股的持有期報酬差。寫進 results.json 的 POSTHOC。"""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'o_rev_survivorship'))
import osurv_core as OS
sys.path.pop(0)
import numpy as np, pandas as pd
V, K = OS.V, OS.K
RJ = HERE / 'results.json'; R = json.loads(RJ.read_text(encoding='utf-8')); assert 'RESULTS' in R
YD, Y5 = OS.load_yl('R_DEL'); YA, _ = OS.load_yl('R_ALL')
SR = OS.V2B('gapfix', YD, Y5, exfin=True); SA = OS.V2B('gapfix', YA, Y5, exfin=True)
pr, pa = SR.plan(10), SA.plan(10)
tend = K.IDX.get_loc(SR.end_day)
_, per_r = V.stock_contrib(pr, tend); _, per_a = V.stock_contrib(pa, tend)
rows = []
for (d, pk_r, r_r), (d2, pk_a, r_a) in zip(per_r, per_a):
    assert d == d2
    mr = dict(zip(pk_r.tolist(), r_r)); ma = dict(zip(pk_a.tolist(), r_a))
    ins = [j for j in ma if j not in mr]; outs = [j for j in mr if j not in ma]
    if not ins: continue
    rows.append({'sig': str(d.date()), 'in': [K.COLS[j] for j in ins], 'in_ky': [K.COLS[j] in OS.FOREIGN for j in ins],
                 'in_ret': float(np.mean([ma[j] for j in ins])), 'out': [K.COLS[j] for j in outs], 'out_ret': float(np.mean([mr[j] for j in outs])),
                 'ky_in_ret': float(np.mean([ma[j] for j in ins if K.COLS[j] in OS.FOREIGN])) if any(K.COLS[j] in OS.FOREIGN for j in ins) else np.nan})
df = pd.DataFrame(rows); df['diff'] = df['in_ret'] - df['out_ret']
ky_ev = [(r['sig'], c, float(ma_ret)) for r in rows for c, k, ma_ret in zip(r['in'], r['in_ky'], [np.nan] * len(r['in'])) if k]
out = {'n_months_differ': int(len(df)), 'mean_in_minus_out_pp': round(float(df['diff'].mean()) * 100, 3), 't_by_month': round(K.tstat(df['diff']), 3),
       'share_negative': round(float((df['diff'] < 0).mean()), 3), 'mean_in_ret_pp': round(float(df['in_ret'].mean()) * 100, 3), 'mean_out_ret_pp': round(float(df['out_ret'].mean()) * 100, 3),
       'mean_ky_in_ret_pp': round(float(df['ky_in_ret'].mean()) * 100, 3),
       'worst5': df.nsmallest(5, 'diff')[['sig', 'in', 'out', 'diff']].assign(diff=lambda x: (x['diff'] * 100).round(2)).to_dict('records'),
       'best5': df.nlargest(5, 'diff')[['sig', 'in', 'out', 'diff']].assign(diff=lambda x: (x['diff'] * 100).round(2)).to_dict('records'),
       'note': '事後描述，PREREG 之後才寫，不參與判定；新進股包含因 KY 加入而改變排名的非 KY 股（百分位重排）。'}
R['POSTHOC_KY'] = out
RJ.write_text(json.dumps(R, ensure_ascii=False, indent=1, default=lambda x: x.item() if hasattr(x, 'item') else str(x)), encoding='utf-8')
print(json.dumps(out, ensure_ascii=False, default=str, indent=0))
