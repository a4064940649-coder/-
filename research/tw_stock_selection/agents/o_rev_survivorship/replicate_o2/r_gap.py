#!/usr/bin/env python3
"""複製員：缺口統計。每個訊號日，營收以外條件都合格（前 300、s52、非處置、隔日有開盤）但 3 月年增率缺值的股票。"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from r_lib import *

YO, YD, YA = load_yoy('ORIG'), load_yoy('R_DEL'), load_yoy('R_ALL')
_, Y5, _ = V.load_revenue()
S = V2B('gapfix', YO, Y5)
END_T = IDX.get_loc(IDX[IDX >= pd.Timestamp('2026-10-01')][0])
rows = []
for d in S.sig_days:
    if d < pd.Timestamp('2012-10-01'): continue
    t, e, ok = S.base_parts(d)
    m = pd.Period(d, 'M')
    Y = YO if d < V.SWITCH else Y5
    y0 = K.yoy3_for_month(Y, m)
    yd = K.yoy3_for_month(YD if d < V.SWITCH else Y5, m)
    ya = K.yoy3_for_month(YA if d < V.SWITCH else Y5, m)
    miss = ok & ~np.isfinite(y0)
    dl = LASTT < END_T
    for j in np.where(miss)[0]:
        rows.append((str(d.date()), COLS[j], bool(FORMASK[j]), bool(dl[j]), bool(V.FINMASK[j]), bool(np.isfinite(yd[j])), bool(np.isfinite(ya[j]))))
G = pd.DataFrame(rows, columns=['d', 'code', 'foreign', 'delisted', 'fin', 'fill_del', 'fill_all'])
G.to_csv(H / 'my_gap.csv', index=False)
G['d'] = pd.to_datetime(G['d'])
out = {}
w = (G.d >= '2013-04-01') & (G.d <= '2018-12-31')
days = [d for d in S.sig_days if pd.Timestamp('2013-04-01') <= d <= pd.Timestamp('2018-12-31')]
for nm, m in (('dom_delisted', ~G.foreign & G.delisted), ('ky_delisted', G.foreign & G.delisted), ('dom_any', ~G.foreign), ('ky_any', G.foreign)):
    c = G[w & m].groupby('d').size().reindex(pd.DatetimeIndex(days), fill_value=0)
    out[nm + '_2013-04_2018'] = {'mean': float(c.mean()), 'min': int(c.min()), 'max': int(c.max())}
w2 = G.d >= '2013-01-01'
for nm, m, f in (('dom_delisted', ~G.foreign & G.delisted, 'fill_del'), ('ky_delisted', G.foreign & G.delisted, 'fill_all'),
                 ('dom_any', ~G.foreign, 'fill_del'), ('ky_any', G.foreign, 'fill_all')):
    x = G[w2 & m]
    out[nm + '_coverage_2013_2026'] = {'n': int(len(x)), 'filled': int(x[f].sum()), 'rate': float(x[f].mean())}
    for a, b in (('2013-01-01', '2015-03-31'), ('2015-04-01', '2019-05-31'), ('2021-01-01', '2026-12-31')):
        xx = x[(x.d >= a) & (x.d <= b)]
        out[nm + f'_cov_{a[:7]}_{b[:7]}'] = {'n': int(len(xx)), 'rate': float(xx[f].mean()) if len(xx) else None}
print(json.dumps(out, indent=1))
(H / 'my_gap.json').write_text(json.dumps(out, indent=1))
