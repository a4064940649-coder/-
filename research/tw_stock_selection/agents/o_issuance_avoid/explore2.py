# -*- coding: utf-8 -*-
"""Exploration part 2 (selection period only): 52w-ratio-matched CARs (same pool, same s52 quintile),
and main-strategy picks that face an upcoming SEO ex-rights / CB listing inside the next holding period."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import oi_lib as L
K, V, IDX, COLS = L.K, L.V, L.IDX, L.COLS
X = pd.read_csv(L.HERE / 'explore_events_sel.csv', dtype={'code': str})
END_T = L.tpos(L.SEL_END) - 1
spec = [('SEO', 'recv', (0, 2)), ('SEO', 'recv', (1, 60)), ('SEO', 'recv', (1, 120)), ('SEO', 'exr', (0, 20)), ('SEO', 'exr', (-10, -1)),
        ('CB', 'recv', (1, 60)), ('CB', 'list', (-20, -6)), ('CB', 'list', (-5, 20)), ('CB', 'list', (1, 60)),
        ('SDIV', 'recv', (0, 2)), ('SDIV', 'recv', (1, 60)), ('SDIV', 'exr', (0, 20)), ('ECB', 'recv', (1, 60)), ('GDR', 'recv', (1, 60))]
out = {'note': 'CAR_m = sum(ret) - mean sum(ret) of same-pool peers in the same s52 quintile (day before anchor)'}
cols = {}
for et, an, (a, b) in spec:
    sub = X[(X['etype'] == et) & X[f't_{an}'].notna()] if f't_{an}' in X else X.iloc[0:0]
    vals = []; keys = []; s52 = []
    for r in sub.itertuples():
        t0 = L.tpos(getattr(r, f't_{an}'))
        if t0 + b > END_T: continue
        v = L.car_matched(L.CI[r.code], t0, a, b)
        vals.append(v); keys.append(r.ym); s52.append(r.s52)
    k = f'{et}|{an}|[{a},{b}]'
    out[k] = L.cluster_t(vals, keys)
    v = np.array(vals, float); s = np.array(s52, float)
    out[k + '|s52>=0.9'] = L.cluster_t(v[s >= 0.9], np.array(keys)[s >= 0.9])
    out[k + '|s52<0.9'] = L.cluster_t(v[s < 0.9], np.array(keys)[s < 0.9])
    print(k, {kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in out[k].items()}, flush=True)

# upcoming-event exposure of main-strategy picks (selection period)
E = L.load_events()
seo = E[(E['etype'] == 'SEO') & E['eff'].notna()].copy()
seo['exr_t'] = [L.exrights_date(j, e) for j, e in zip(seo['j'], seo['eff'])]
cb = E[(E['etype'] == 'CB') & E['list_date'].notna()].copy()
res = {}
for n in (10, 30):
    P = pd.read_csv(L.HERE / f'explore_picks_n{n}.csv', dtype={'code': str}, parse_dates=['d'])
    flags = []
    for r in P.itertuples():
        t = L.tpos(r.d)
        s = seo[(seo['code'] == r.code) & (seo['eff'] <= r.d)]
        up_exr = any((x is not None) and (x > t) and (x <= t + 25) for x in s['exr_t'])
        c = cb[(cb['code'] == r.code)]
        known = c['eff'].fillna(c['list_date'] - pd.Timedelta(days=28))   # TPEx-only: listing - median lag (not causal; flagged)
        up_list = bool(((c['list_date'] > r.d) & (c['list_date'] <= r.d + pd.Timedelta(days=35)) & c['eff'].notna() & (c['eff'] <= r.d)).any())
        rec_list = bool(((c['list_date'] <= r.d) & (c['list_date'] >= r.d - pd.Timedelta(days=30))).any())
        flags.append((up_exr, up_list, rec_list))
    P[['up_exr', 'up_list', 'rec_list']] = pd.DataFrame(flags, index=P.index)
    d = {}
    for f in ('up_exr', 'up_list', 'rec_list'):
        g = P[P[f]].groupby(P['d'].dt.to_period('M'))['rel'].mean()
        d[f] = dict(n=int(P[f].sum()), months=int(len(g)), mean_rel=float(g.mean()) if len(g) else None, t=L.tstat(g.to_numpy()))
    res[f'n{n}'] = d
    print(n, d)
out['picks_upcoming'] = res
L.dump(out, 'explore2.json')
