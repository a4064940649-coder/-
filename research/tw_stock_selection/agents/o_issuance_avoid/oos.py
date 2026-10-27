# -*- coding: utf-8 -*-
"""Out-of-sample run of the frozen spec (V1_both_90d). Refuses to run unless PREREG exists in results.json.
Runs validation (2019-05 ~ 2026-10-12, split 2019-05~2023-12 / 2024-01~2026-10-12) once; no spec change."""
import sys, json, hashlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
HERE = Path(__file__).resolve().parent
res = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))
assert 'PREREG' in res and res.get('PREREG_sha256'), 'PREREG missing: run prereg.py first'
blob = json.dumps(res['PREREG'], ensure_ascii=False, sort_keys=True).encode('utf-8')
assert hashlib.sha256(blob).hexdigest() == res['PREREG_sha256'], 'PREREG was modified'
import numpy as np, pandas as pd
import oi_lib as L
import iterate as IT
K, V, IDX, COLS, N = L.K, L.V, L.IDX, L.COLS, L.N

CFG = IT.VERSIONS['V1_both_90d']
SEGS = {'SEL': ('2006-04-01', '2019-04-30'), 'VAL': ('2019-05-01', '2026-10-12'), 'VAL_a_19_23': ('2019-05-01', '2023-12-31'),
        'VAL_b_24_26': ('2024-01-01', '2026-10-12'), 'ALL': ('2006-04-01', '2026-10-12')}
SIGWIN = {'SEL': ('2006-04-01', '2019-03-31'), 'VAL': ('2019-05-01', '2026-10-12'), 'VAL_a_19_23': ('2019-05-01', '2023-12-31'),
          'VAL_b_24_26': ('2024-01-01', '2026-10-12'), 'ALL': ('2006-04-01', '2026-10-12')}
S = L.strategy(exfin=True)
t0, t1, sd = S.bounds()
out = {'PREREG_sha256': res['PREREG_sha256'], 'strategy': {}, 'robust': {}}

def paired(nav, base, a, b):
    mr = L.monthly_rets(nav, sd); mb = L.monthly_rets(base, sd); m50 = L.bench_monthly(sd).reindex(mr.index)
    sel = (mr.index >= pd.Timestamp(a)) & (mr.index <= pd.Timestamp(b))
    d = (mr - mb)[sel].dropna(); xv = (mr - m50)[sel].dropna(); xb = (mb - m50)[sel].dropna()
    yrs = d.groupby(d.index.year).sum()
    best = int(yrs.idxmax()) if len(yrs) else None
    dex = d[d.index.year != best]
    return dict(paired_mean_pp=d.mean() * 100, paired_sd_pp=d.std() * 100, paired_t=L.tstat(d.to_numpy()), n_m=len(d), n_nonzero=int((d.abs() > 1e-12).sum()),
                best_year=best, paired_ex_best_year_mean_pp=dex.mean() * 100, paired_ex_best_year_t=L.tstat(dex.to_numpy()),
                yearly_sum_pp={int(k): round(v * 100, 3) for k, v in yrs.items()},
                ver_vs0050=dict(mean_pp=xv.mean() * 100, sd_pp=xv.std() * 100, t=L.tstat(xv.to_numpy())),
                base_vs0050=dict(mean_pp=xb.mean() * 100, sd_pp=xb.std() * 100, t=L.tstat(xb.to_numpy())))

avoid = IT.make_avoid(CFG)
navs = {}
for n in (10, 30):
    bpl, _ = L.plan_avoid(S, n)
    vpl, _ = L.plan_avoid(S, n, avoid, sell_held=False)
    ew = L.ew_nav(S, bpl)
    o = {}
    for cost in ('normal', 'stress'):
        base, _ = L.run_nav(S, bpl, cost); nav, lg = L.run_nav(S, vpl, cost)
        navs[f'n{n}|{cost}|base'] = base; navs[f'n{n}|{cost}|V1'] = nav
        for sname, (a, b) in SEGS.items():
            sb = L.seg(base, ew, a, b, sd); sv = L.seg(nav, ew, a, b, sd)
            pw = paired(nav, base, *SIGWIN[sname])
            o[f'{cost}|{sname}'] = dict(base=sb, V1=sv, dCAGR_pp=sv['CAGR'] - sb['CAGR'], dMDD_pp=sv['maxDD'] - sb['maxDD'], paired=pw)
    navs[f'n{n}|EW'] = ew
    # affected pick-months and their contributions
    con_b, per_b = V.stock_contrib(bpl, t1); con_v, per_v = V.stock_contrib(vpl, t1)
    rb = {d: dict(zip(pk, r)) for d, pk, r in per_b}; rv = {d: dict(zip(pk, r)) for d, pk, r in per_v}
    aff = []
    for d in rb:
        rem = set(rb[d]) - set(rv.get(d, {})); add = set(rv.get(d, {})) - set(rb[d])
        if not rem: continue
        k = len(rb[d])
        c = (np.mean([rv[d][j] for j in add]) if add else 0) - np.mean([rb[d][j] for j in rem])
        for j in rem:
            aff.append(dict(d=d, code=COLS[j], name=V.SL['name'].get(COLS[j], ''), ind=V.SL['industry'].get(COLS[j], ''), r_removed=rb[d][j],
                            r_added=np.mean([rv[d][x] for x in add]) if add else np.nan, contrib_pp=c * len(rem) / k * 100 / len(rem)))
    A = pd.DataFrame(aff)
    A.to_csv(HERE / f'oos_affected_n{n}.csv', index=False)
    for sname, (a, b) in SIGWIN.items():
        s = A[(A['d'] >= pd.Timestamp(a)) & (A['d'] <= pd.Timestamp(b))] if len(A) else A
        o[f'affected|{sname}'] = dict(n_pick_months=int(len(s)), n_months=int(s['d'].nunique()) if len(s) else 0,
                                      mean_removed_ret=float(s['r_removed'].mean()) if len(s) else None, mean_added_ret=float(s['r_added'].mean()) if len(s) else None,
                                      sum_contrib_pp=float(s['contrib_pp'].sum()) if len(s) else 0.0,
                                      sum_contrib_ex_top_pp=float(s['contrib_pp'].sum() - s['contrib_pp'].max()) if len(s) else 0.0,
                                      top=s.sort_values('contrib_pp', ascending=False)[['d', 'code', 'name', 'r_removed', 'r_added', 'contrib_pp']].astype(str).head(3).values.tolist() if len(s) else [],
                                      bottom=s.sort_values('contrib_pp')[['d', 'code', 'name', 'r_removed', 'r_added', 'contrib_pp']].astype(str).head(3).values.tolist() if len(s) else [],
                                      by_industry=s.groupby('ind')['contrib_pp'].agg(['size', 'sum']).round(3).to_dict(orient='index') if len(s) else {})
    out['strategy'][f'n{n}'] = o
    print(n, {k: (round(v['dCAGR_pp'], 2), round(v['paired']['paired_mean_pp'], 3), round(v['paired']['paired_t'], 2), round(v['V1']['CAGR'], 1), round(v['base']['CAGR'], 1), round(v['V1']['maxDD'], 1), round(v['base']['maxDD'], 1))
              for k, v in o.items() if '|' in k and not k.startswith('affected')}, flush=True)
    print(n, {k: (v['n_pick_months'], round(v['sum_contrib_pp'], 2)) for k, v in o.items() if k.startswith('affected')}, flush=True)
pd.DataFrame(navs).to_parquet(HERE / 'oos_navs.parquet')

# ---------------- gates ----------------
g = {}
n10 = out['strategy']['n10']
g['sel_paired_t>=1.5'] = n10['normal|SEL']['paired']['paired_t'] >= 1.5
g['val_paired>0_and_t>=1.0'] = n10['normal|VAL']['paired']['paired_mean_pp'] > 0 and n10['normal|VAL']['paired']['paired_t'] >= 1.0
g['ALL_CAGR>=base+0.3pp'] = n10['normal|ALL']['dCAGR_pp'] >= 0.3
g['ALL_MDD_not_worse_than_2pp'] = n10['normal|ALL']['dMDD_pp'] >= -2.0
g['ALL_ex_best_year>0'] = n10['normal|ALL']['paired']['paired_ex_best_year_mean_pp'] > 0
g['affected_pick_months_ALL>=30'] = n10['affected|ALL']['n_pick_months'] >= 30
out['gates_n10'] = g
ex = json.loads((HERE / 'explore.json').read_text(encoding='utf-8'))
out['gate_event_level_selection'] = dict(
    P1_SEO_recv_1_60=ex['car']['SEO|recv|[1,60]'], P2_CB_list_m5_20=ex['car']['CB|list|[-5,20]'], P4_SDIV_recv_1_60=ex['car']['SDIV|recv|[1,60]'],
    passed=bool((ex['car']['SEO|recv|[1,60]']['mean'] <= -0.015 and ex['car']['SEO|recv|[1,60]']['t_cl'] <= -2) or
                (ex['car']['CB|list|[-5,20]']['mean'] <= -0.015 and ex['car']['CB|list|[-5,20]']['t_cl'] <= -2)))

# ---------------- event-level out of sample ----------------
E = L.load_events()
END = L.tpos(pd.Timestamp('2026-10-22')) - 1
evs = {}
spec = [('SEO', 'recv', (0, 2)), ('SEO', 'recv', (1, 60)), ('CB', 'list', (-20, -6)), ('CB', 'list', (-5, 20)), ('CB', 'list', (1, 60)), ('CB', 'recv', (1, 60)), ('SDIV', 'recv', (1, 60))]
for et, an, (a, b) in spec:
    for pname, (pa, pb) in {'VAL_a_19_23': ('2019-05-01', '2023-12-31'), 'VAL_b_24_26': ('2024-01-01', '2026-10-22'), 'VAL': ('2019-05-01', '2026-10-22')}.items():
        sub = E[(E['etype'] == et)]
        col = {'recv': 'recv', 'list': 'list_date'}[an]
        sub = sub[sub[col].notna() & (sub[col] >= pa) & (sub[col] <= pb)]
        v1 = []; v2 = []; ks = []
        for r in sub.itertuples():
            t = L.tpos(getattr(r, col))
            if t is None or t + b > END: continue
            v1.append(L.car(r.j, t, a, b)); v2.append(L.car_matched(r.j, t, a, b)); ks.append(str(pd.Timestamp(getattr(r, col)).to_period('M')))
        evs[f'{et}|{an}|[{a},{b}]|{pname}'] = dict(poolEW=L.cluster_t(v1, ks), s52matched=L.cluster_t(v2, ks))
out['event_oos'] = evs
for k, v in evs.items():
    print(k, 'pool', round(v['poolEW']['mean'] * 100, 2) if v['poolEW']['n'] else None, round(v['poolEW']['t_cl'], 2) if v['poolEW']['n'] else None, v['poolEW']['n'],
          '| matched', round(v['s52matched']['mean'] * 100, 2) if v['s52matched']['n'] else None, round(v['s52matched']['t_cl'], 2) if v['s52matched']['n'] else None)
res['OOS'] = out
L.dump(res, 'results.json')
print(json.dumps(out['gates_n10']), out['gate_event_level_selection']['passed'])
