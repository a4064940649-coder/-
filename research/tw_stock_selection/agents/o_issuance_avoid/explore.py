# -*- coding: utf-8 -*-
"""Exploration on the SELECTION period only (events whose CAR window ends <= 2019-04-30; main-strategy signal days
<= 2019-03 so that the holding period ends inside the selection period). Output explore.json, explore_events_sel.csv."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import oi_lib as L
K, V, IDX, COLS = L.K, L.V, L.IDX, L.COLS

E = L.load_events()
END_T = L.tpos(L.SEL_END) - 1          # last trading index inside selection period
WIN = {'SEO': {'recv': [(0, 2), (-20, -1), (-60, -1), (3, 60), (1, 60), (1, 120)], 'eff': [(0, 2), (1, 60)], 'exr': [(-10, -1), (0, 20), (0, 60)]},
       'CB': {'recv': [(0, 2), (-60, -1), (3, 60), (1, 60)], 'eff': [(0, 2), (1, 60)], 'list': [(-20, -6), (-5, 20), (-5, -1), (0, 20), (1, 60)]},
       'ECB': {'recv': [(0, 2), (-60, -1), (3, 60), (1, 60)]},
       'GDR': {'recv': [(0, 2), (-60, -1), (3, 60), (1, 60)]},
       'SDIV': {'recv': [(0, 2), (-60, -1), (3, 60), (1, 60)], 'exr': [(0, 20), (0, 60)]}}

rows = []
issu = E[E['etype'].isin(['SEO', 'CB', 'ECB', 'GDR'])]
for r in E.itertuples():
    if r.anchor < L.EV_START or r.anchor > L.SEL_END: continue
    j = r.j
    anchors = {}
    if pd.notna(r.recv): anchors['recv'] = L.tpos(r.recv)
    if pd.notna(r.eff): anchors['eff'] = L.tpos(r.eff)
    if pd.notna(r.list_date): anchors['list'] = L.tpos(r.list_date)
    if r.etype in ('SEO', 'SDIV') and pd.notna(r.eff):
        x = L.exrights_date(j, r.eff)
        if x is not None: anchors['exr'] = x
    a0 = anchors.get('recv', anchors.get('list'))
    if a0 is None or a0 < 130: continue
    rec = dict(event_id=r.event_id, code=r.code, etype=r.etype, anchor=r.anchor, ym=str(pd.Timestamp(r.anchor).to_period('M')),
               proceeds=r.proceeds, src=r.src)
    rec.update(L.chars(j, a0))
    rec['dose_adv'] = r.proceeds / rec['adv60'] if (pd.notna(r.proceeds) and rec['adv60'] and np.isfinite(rec['adv60'])) else np.nan
    prev = issu[(issu['code'] == r.code) & (issu['anchor'] < r.anchor) & (issu['anchor'] >= r.anchor - pd.Timedelta(days=3 * 365))]
    rec['repeat'] = bool(len(prev) > 0)
    rec['fin'] = r.code in V.FIN
    for an, wins in WIN.get(r.etype, {}).items():
        t0 = anchors.get(an)
        if t0 is None: continue
        rec[f't_{an}'] = str(IDX[t0].date())
        for a, b in wins:
            if t0 + b > END_T: continue          # window must end inside the selection period
            rec[f'car_{an}_{a}_{b}'] = L.car(j, t0, a, b)
            if (a, b) in ((1, 60), (0, 60), (1, 120)):
                rec[f'bhar_{an}_{a}_{b}'] = L.bhar(j, t0, a, b)
    rows.append(rec)
X = pd.DataFrame(rows)
X.to_csv(L.HERE / 'explore_events_sel.csv', index=False)

out = {'n_events_sel': X['etype'].value_counts().to_dict(), 'note': 'CAR = sum(ret - same-pool EW), in fraction; t_cl = t of anchor-month means'}
# (a) CAR table by type x anchor x window
tab = {}
for et, d in WIN.items():
    sub = X[X['etype'] == et]
    for an, wins in d.items():
        for a, b in wins:
            c = f'car_{an}_{a}_{b}'
            if c not in sub: continue
            tab[f'{et}|{an}|[{a},{b}]'] = L.cluster_t(sub[c], sub['ym'])
            if f'bhar_{an}_{a}_{b}' in sub:
                tab[f'{et}|{an}|BHAR[{a},{b}]'] = L.cluster_t(sub[f'bhar_{an}_{a}_{b}'], sub['ym'])
out['car'] = tab

# subgroups (P2 dose, P3)
def grp(sub, col, key, q=3):
    s = sub.dropna(subset=[col, key])
    if len(s) < 3 * q: return {}
    lab = pd.qcut(s[col].rank(method='first'), q, labels=[f'q{i + 1}' for i in range(q)])
    o = {}
    for g, x in s.groupby(lab, observed=True):
        o[str(g)] = dict(L.cluster_t(x[key], x['ym']), cut_lo=float(x[col].min()), cut_hi=float(x[col].max()))
    return o

sg = {}
for et, key in [('SEO', 'car_recv_1_60'), ('SEO', 'car_recv_0_2'), ('CB', 'car_list_-5_20'), ('CB', 'car_recv_1_60'), ('CB', 'car_list_1_60'), ('SDIV', 'car_recv_1_60')]:
    sub = X[X['etype'] == et]
    if key not in sub: continue
    d = {}
    for col in ('dose_adv', 'pre6m', 's52', 'adv60'):
        d[col] = grp(sub, col, key)
    s = sub.dropna(subset=[key])
    d['s52>=0.9'] = L.cluster_t(s[s['s52'] >= 0.9][key], s[s['s52'] >= 0.9]['ym'])
    d['s52<0.9'] = L.cluster_t(s[s['s52'] < 0.9][key], s[s['s52'] < 0.9]['ym'])
    d['in_pool'] = L.cluster_t(s[s['in_pool']][key], s[s['in_pool']]['ym'])
    d['not_pool'] = L.cluster_t(s[~s['in_pool']][key], s[~s['in_pool']]['ym'])
    d['repeat'] = L.cluster_t(s[s['repeat']][key], s[s['repeat']]['ym'])
    d['first'] = L.cluster_t(s[~s['repeat']][key], s[~s['repeat']]['ym'])
    d['nonfin'] = L.cluster_t(s[~s['fin']][key], s[~s['fin']]['ym'])
    hi = s[(s['s52'] >= 0.9) & (s['pre6m'] >= s['pre6m'].median())]
    d['P3_overlap(s52>=0.9 & pre6m>=median)'] = L.cluster_t(hi[key], hi['ym'])
    sg[f'{et}|{key}'] = d
out['subgroups'] = sg

# CAR path (cumulative mean AR by day) for SEO recv, CB list, SDIV recv
def path(et, an, lo=-60, hi=120):
    sub = X[(X['etype'] == et) & X.get(f't_{an}', pd.Series(dtype=str)).notna()]
    acc = []
    for r in sub.itertuples():
        t0 = L.tpos(getattr(r, f't_{an}')); j = L.CI[r.code]
        if t0 + hi > END_T or t0 + lo < 1: continue
        ar = L.R[t0 + lo:t0 + hi + 1, j] - np.nan_to_num(L.PEW[t0 + lo:t0 + hi + 1])
        acc.append(np.nan_to_num(ar))
    if not acc: return {}
    m = np.cumsum(np.mean(acc, axis=0))
    return {'n': len(acc), 'days': list(range(lo, hi + 1, 10)), 'cum_mean_ar': [round(float(m[k - lo]), 4) for k in range(lo, hi + 1, 10)]}
out['paths'] = {'SEO|recv': path('SEO', 'recv'), 'CB|list': path('CB', 'list'), 'CB|recv': path('CB', 'recv'), 'SDIV|recv': path('SDIV', 'recv'), 'ECB|recv': path('ECB', 'recv')}

# ---------------- (b)(c) main strategy picks in the selection period ----------------
ms = {}
issue_types = ['SEO', 'CB', 'ECB', 'GDR']
Ei = E[E['etype'].isin(issue_types)].copy()
Ei['known'] = Ei['recv'].fillna(Ei['list_date'])     # first date we can be sure the event is public
by_code = {c: g for c, g in Ei.groupby('code')}
for exfin in (True,):
    S = L.strategy(exfin=exfin)
    t0, t1, sd = S.bounds()
    for n in (10, 30):
        pl, _ = L.plan_avoid(S, n)
        con, per = V.stock_contrib(pl, t1)
        recs = []
        for (d, pk, r) in per:
            if d > pd.Timestamp('2019-03-31'): continue
            pool_ret = np.mean(r)
            for j, x in zip(pk, r):
                c = COLS[j]; g = by_code.get(c)
                past90 = fut90 = False; past_type = None
                if g is not None:
                    pk_ = g[(g['known'] <= d) & (g['known'] >= d - pd.Timedelta(days=90))]
                    past90 = len(pk_) > 0
                    past_type = ','.join(sorted(set(pk_['etype']))) if past90 else None
                    fu = g[(g['known'] > d) & (g['known'] <= d + pd.Timedelta(days=90))]
                    fut90 = len(fu) > 0
                recs.append(dict(d=d, code=c, r=x, rel=x - pool_ret, past90=past90, fut90=fut90, past_type=past_type))
        P = pd.DataFrame(recs)
        P['ym'] = P['d'].dt.to_period('M').astype(str)
        thr = P['r'].quantile(0.9)
        d_ = {}
        d_['n_pick_months'] = int(len(P))
        d_['share_past90'] = float(P['past90'].mean()); d_['share_fut90'] = float(P['fut90'].mean())
        d_['share_any_pm90'] = float((P['past90'] | P['fut90']).mean())
        d_['n_past90'] = int(P['past90'].sum())
        d_['months_with_past90'] = int(P[P['past90']]['ym'].nunique())
        # P5: next-period return of event picks vs other picks of the same month (paired by month)
        pm = P[P['past90']].groupby('ym')['rel'].mean()
        d_['P5_rel_past90'] = dict(mean=float(pm.mean()) if len(pm) else np.nan, t=L.tstat(pm.to_numpy()), n_months=int(len(pm)))
        fm = P[P['fut90']].groupby('ym')['rel'].mean()
        d_['rel_fut90'] = dict(mean=float(fm.mean()) if len(fm) else np.nan, t=L.tstat(fm.to_numpy()), n_months=int(len(fm)))
        for et in issue_types:
            q = P[P['past_type'].fillna('').str.contains(et)].groupby('ym')['rel'].mean()
            d_[f'P5_rel_past90_{et}'] = dict(mean=float(q.mean()) if len(q) else np.nan, t=L.tstat(q.to_numpy()), n_months=int(len(q)))
        # right tail
        d_['top10pct_cut'] = float(thr)
        d_['share_past90_in_top10pct'] = float(P[P['r'] >= thr]['past90'].mean())
        d_['share_past90_in_rest'] = float(P[P['r'] < thr]['past90'].mean())
        d_['share_fut90_in_top10pct'] = float(P[P['r'] >= thr]['fut90'].mean())
        d_['share_fut90_in_rest'] = float(P[P['r'] < thr]['fut90'].mean())
        d_['past90_examples'] = P[P['past90']].sort_values('r')[['d', 'code', 'r', 'past_type']].astype(str).values.tolist()[:5] + \
            P[P['past90']].sort_values('r')[['d', 'code', 'r', 'past_type']].astype(str).values.tolist()[-5:]
        ms[f'exfin_n{n}'] = d_
        P.to_csv(L.HERE / f'explore_picks_n{n}.csv', index=False)
out['main_strategy_sel'] = ms
L.dump(out, 'explore.json')
import json
print(json.dumps({k: out[k] for k in ['n_events_sel', 'car']}, ensure_ascii=False, indent=0, default=str)[:6000])
print(json.dumps(out['main_strategy_sel'], ensure_ascii=False, default=str)[:4000])
