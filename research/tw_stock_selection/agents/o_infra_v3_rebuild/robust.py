#!/usr/bin/env python3
"""穩健性與診斷（PREREG 之後；只讀 run_P*.json／navs）：python3 -I -B robust.py → results_robust.json
(1) V3（排除金融 n10）分產業貢獻（毛報酬近似）；(2) 逐年 0050；(3) 營收面板「下市前截斷」：終止上市 4 碼股票的最後營收月與最後交易日的落差；
(4) V3 對 V2 的配對月差 t（ALL、P1、S3）。"""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
RES = json.loads((HERE / 'results.json').read_text(encoding='utf-8')); assert RES.get('PREREG')
import o3_core as O
V, K = O.V, O.K
O.apply_patch()
R1 = json.loads((HERE / 'run_P1.json').read_text(encoding='utf-8'))['runs']
R0 = json.loads((HERE / 'run_P0.json').read_text(encoding='utf-8'))['runs']
DW = RES['RESULTS']['D_worse']
k3 = f'P1|R1|X1|D{DW}|EXFIN|n10|normal'; k2 = 'P0|R0|X0|Dv2|EXFIN|n10|normal'
out = {}
# (1) 分產業
picks = R1[k3]['picks']
sd = sorted(picks)
YL, Y5, _ = V.load_revenue()
ind = V.SL['industry']
con = {}; cnt = {}
ex = [K.IDX[K.IDX > pd.Timestamp(d)][0] for d in sd]
for i, d in enumerate(sd[:-1]):
    t, t2 = K.IDX.get_loc(ex[i]), K.IDX.get_loc(ex[i + 1])
    pk = [K.CI[c] for c in picks[d]]
    for j in pk:
        g = K.Cn[t, j] / K.On[t, j] * np.prod(K.GROW[t + 1:t2 + 1, j])
        if K.TRADED[t2, j] and np.isfinite(K.On[t2, j]): g *= K.On[t2, j] / K.Cn[t2, j]
        it = ind.get(K.COLS[j], '?') if isinstance(ind.get(K.COLS[j], '?'), str) else '?'
        con[it] = con.get(it, 0) + (g - 1) / len(pk); cnt[it] = cnt.get(it, 0) + 1
tot = sum(con.values())
out['industry'] = sorted([{'industry': k, 'contrib_pp': v * 100, 'share': v / tot, 'slots': cnt[k]} for k, v in con.items()], key=lambda x: -x['contrib_pp'])[:12]
# (2) 0050 逐年
E50 = K.E50
y = E50.groupby(E50.index.year).last(); y0 = E50.groupby(E50.index.year).first()
nav = pd.read_parquet(HERE / 'navs_P1.parquet')[k3].dropna()
e = E50.reindex(nav.index)
out['yearly_0050'] = {int(a): round(float(b) * 100, 1) for a, b in (e.groupby(e.index.year).last() / e.groupby(e.index.year).first().shift(0) - 1).items()}
ee = e.copy(); yy = {}
for yr, g in ee.groupby(ee.index.year):
    prev = ee[ee.index < g.index[0]]
    base = prev.iloc[-1] if len(prev) else g.iloc[0]
    yy[int(yr)] = round(float(g.iloc[-1] / base - 1) * 100, 1)
out['yearly_0050'] = yy
# (3) 營收截斷
d = pd.read_csv('/home/user/niklavsd/bottleneck-signals/data/processed/tw_revenue.csv', dtype={'code': str})
lastrev = d.groupby('code')['month'].max()
dl = pd.read_csv(HERE / 'delist_official.csv', dtype=str)
dl = dl[(dl['four'] == 'True') & (dl['in_panel'] == 'True')]
dl['date'] = pd.to_datetime(dl['date'])
YR = O.load_rdel()
rows = []
lt = O.last_trade_idx()
for c, dd in zip(dl['code'], dl['date']):
    if dd < pd.Timestamp('2006-01-01'): continue
    j = K.CI[c]; ltd = K.IDX[lt[j]] if K.IDX[lt[j]] <= dd else K.IDX[np.where(K.TRADED[:, j] & (K.IDX <= dd))[0][-1]]
    lr = pd.Timestamp(lastrev[c]) if c in lastrev.index else None
    lrr = YR[c].dropna().index.max().to_timestamp() if c in YR.columns and YR[c].notna().any() else None
    rows.append({'code': c, 'delist': str(dd.date()), 'last_trade': str(ltd.date()), 'last_rev_orig': str(lr.date()) if lr is not None else None,
                 'gap_months_orig': (ltd.year - lr.year) * 12 + ltd.month - lr.month if lr is not None else None,
                 'gap_months_rdel': (ltd.year - lrr.year) * 12 + ltd.month - lrr.month if lrr is not None else None})
T = pd.DataFrame(rows)
out['rev_truncation'] = {'n': int(len(T)), 'gap_ge3_orig': int((T['gap_months_orig'].fillna(999) >= 3).sum()), 'gap_ge3_rdel': int((T['gap_months_rdel'].fillna(999) >= 3).sum()),
                         'median_gap_orig': float(T['gap_months_orig'].median()), 'by_period_ge3_orig': T.assign(y=T['delist'].str[:4].astype(int)).groupby(pd.cut(T['delist'].str[:4].astype(int), [2005, 2012, 2018, 2026]))['gap_months_orig'].apply(lambda s: int((s.fillna(999) >= 3).sum())).astype(int).to_dict(),
                         'examples': T.sort_values('delist').head(0).to_dict(orient='records')}
out['rev_truncation']['by_period_ge3_orig'] = {str(k): v for k, v in out['rev_truncation']['by_period_ge3_orig'].items()}
# (4) 配對月差
n3 = pd.read_parquet(HERE / 'navs_P1.parquet')[k3].dropna(); n2 = pd.read_parquet(HERE / 'navs_P0.parquet')[k2].dropna()
S = V.V2('gapfix', YL, Y5, exfin=True); sdays = S.sig_days + [S.end_day]
D = pd.DatetimeIndex([x for x in sdays if x in n3.index])
r3 = (n3.reindex(D).shift(-1) / n3.reindex(D) - 1).iloc[:-1]; r2 = (n2.reindex(D).shift(-1) / n2.reindex(D) - 1).iloc[:-1]
dd_ = (r3 - r2).dropna()
def sl(a, b): x = dd_[(dd_.index >= a) & (dd_.index <= b)]; return {'mean_pp': float(x.mean() * 100), 't': K.tstat(x), 'n': int(len(x)), 'nonzero': int((x.abs() > 1e-12).sum())}
out['paired_V3_minus_V2'] = {'ALL': sl('2006-04-01', '2026-10-12'), 'P1': sl('2006-04-01', '2020-12-31'), 'S1': sl('2006-04-01', '2013-12-31'), 'S2': sl('2014-01-01', '2020-12-31'), 'S3': sl('2021-03-01', '2026-10-12')}
(HERE / 'results_robust.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding='utf-8')
print(json.dumps({k: v for k, v in out.items()}, ensure_ascii=False, default=float)[:3000])
