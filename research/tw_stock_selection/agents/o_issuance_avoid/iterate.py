# -*- coding: utf-8 -*-
"""Model iterations on the SELECTION period only (main strategy, ex-financials, gapfix, official tie-break).
All versions and the selection rule are declared here BEFORE running; every result is written to iterations.json.

Event 'known' date (causal):
  FSC events: 收文日 recv (the board resolution was announced on MOPS earlier, so recv <= signal day => public).
  TPEx-only CBs (FSC mirror gaps): listing date.
Selection rule (fixed in advance): among versions whose n10 avoid list touches >= 30 pick-months in the selection
period, choose the highest paired-difference t (version minus original, signal-window monthly net returns,
2006-04 ~ 2019-03 signal days). If no version reaches t >= 1.5 the best one is still frozen for the record, and
the conclusion is decided by the pre-registered gates.
"""
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import oi_lib as L
K, V, IDX, COLS, N = L.K, L.V, L.IDX, L.COLS, L.N

VERSIONS = {
    'V1_both_90d':        dict(types=('SEO', 'CB'), mode='window', days=90),
    'V2_SEO_90d':         dict(types=('SEO',), mode='window', days=90),
    'V3_CB_90d':          dict(types=('CB',), mode='window', days=90),
    'V4_both_30d':        dict(types=('SEO', 'CB'), mode='window', days=30),
    'V5_both_60d':        dict(types=('SEO', 'CB'), mode='window', days=60),
    'V6_pending_nextper': dict(types=('SEO', 'CB'), mode='pending'),
    'V7_both_90d_dose':   dict(types=('SEO', 'CB'), mode='window', days=90, dose_min=20.0),
    'V8_both_90d_sell':   dict(types=('SEO', 'CB'), mode='window', days=90, sell_held=True),
}
SEL_A, SEL_B = '2006-04-01', '2019-03-31'

E = L.load_events()
E = E[E['etype'].isin(['SEO', 'CB'])].copy()
E['known'] = E['recv'].fillna(E['list_date'])
E['dose'] = np.nan
for i, r in E.iterrows():
    t = L.tpos(r['known'])
    if t is None or t < 2 or not np.isfinite(r['proceeds']): continue
    v = L.V60[t - 1, r['j']]
    if np.isfinite(v) and v > 0: E.at[i, 'dose'] = r['proceeds'] / v
E['exr_t'] = [L.exrights_date(j, e) if (et == 'SEO' and pd.notna(e)) else None for j, e, et in zip(E['j'], E['eff'], E['etype'])]

def make_avoid(cfg):
    sub = E[E['etype'].isin(cfg['types'])]
    if cfg.get('dose_min'):
        sub = sub[sub['dose'] >= cfg['dose_min']]
    cache = {}
    def f(d):
        if d in cache: return cache[d]
        m = np.zeros(N, bool)
        if cfg['mode'] == 'window':
            x = sub[(sub['known'] <= d) & (sub['known'] >= d - pd.Timedelta(days=cfg['days']))]
            m[x['j'].to_numpy(int)] = True
        else:
            t = IDX.get_loc(d)
            s = sub[(sub['etype'] == 'SEO') & sub['eff'].notna() & (sub['eff'] <= d) & (sub['eff'] >= d - pd.Timedelta(days=60))]
            for r in s.itertuples():
                if r.exr_t is None or r.exr_t > t: m[r.j] = True          # ex-rights not yet happened by d
            c = sub[sub['etype'] == 'CB']
            c1 = c[c['eff'].notna() & (c['eff'] <= d) & (c['eff'] >= d - pd.Timedelta(days=90)) & ((c['list_date'].isna()) | (c['list_date'] > d - pd.Timedelta(days=30)))]
            c2 = c[c['eff'].isna() & (c['list_date'] <= d) & (c['list_date'] >= d - pd.Timedelta(days=30))]
            m[c1['j'].to_numpy(int)] = True; m[c2['j'].to_numpy(int)] = True
        cache[d] = m
        return m
    return f

def window_stats(nav, base, sd, a, b):
    mr = L.monthly_rets(nav, sd); mb = L.monthly_rets(base, sd); m50 = L.bench_monthly(sd).reindex(mr.index)
    sel = (mr.index >= pd.Timestamp(a)) & (mr.index <= pd.Timestamp(b))
    d = (mr - mb)[sel].dropna()
    x = (mr - m50)[sel].dropna()
    yrs = d.groupby(d.index.year).sum()
    best = yrs.idxmax() if len(yrs) else None
    d_ex = d[d.index.year != best] if best is not None else d
    return dict(paired_mean_pp=float(d.mean() * 100), paired_sd_pp=float(d.std() * 100), paired_t=L.tstat(d.to_numpy()), n_m=int(len(d)),
                n_nonzero=int((d.abs() > 1e-12).sum()), paired_ex_best_year_mean_pp=float(d_ex.mean() * 100), best_year=int(best) if best else None,
                t_0050=L.tstat(x.to_numpy()), ex_0050_mean_pp=float(x.mean() * 100), ex_0050_sd_pp=float(x.std() * 100))

if __name__ == '__main__':
    T0 = time.time()
    S = L.strategy(exfin=True)
    t0, t1, sd = S.bounds()
    out = {'declared': {k: {kk: (list(vv) if isinstance(vv, tuple) else vv) for kk, vv in v.items()} for k, v in VERSIONS.items()},
           'selection_rule': __doc__, 'period': [SEL_A, SEL_B], 'results': {}}
    for n in (10, 30):
        bpl, _ = L.plan_avoid(S, n)
        base, _ = L.run_nav(S, bpl)
        ew = L.ew_nav(S, bpl)
        bs = L.seg(base, ew, '2006-04-01', '2019-04-30', sd)
        out['results'][f'BASE_n{n}'] = dict(seg=bs)
        print(n, 'BASE', round(bs['CAGR'], 2), round(bs['maxDD'], 1), round(bs['t_0050'], 2), f'{time.time() - T0:.0f}s', flush=True)
        for name, cfg in VERSIONS.items():
            pl, skipped = L.plan_avoid(S, n, make_avoid(cfg), sell_held=cfg.get('sell_held', False))
            nav, lg = L.run_nav(S, pl)
            sg = L.seg(nav, ew, '2006-04-01', '2019-04-30', sd)
            ws = window_stats(nav, base, sd, SEL_A, SEL_B)
            # affected pick-months: names in the ORIGINAL picks that the version removed
            bp = {p[3]: set(p[1]) for p in bpl}; vp = {p[3]: set(p[1]) for p in pl}
            aff = [(d, j) for d in bp if pd.Timestamp(SEL_A) <= d <= pd.Timestamp(SEL_B) for j in bp[d] - vp.get(d, set())]
            out['results'][f'{name}_n{n}'] = dict(seg=sg, paired=ws, affected_pick_months=len(aff), affected_months=len({d for d, _ in aff}),
                                                  dCAGR_pp=sg['CAGR'] - bs['CAGR'], dMDD_pp=sg['maxDD'] - bs['maxDD'], turnover=lg['turnover'])
            print(n, name, 'CAGR', round(sg['CAGR'], 2), 'd', round(sg['CAGR'] - bs['CAGR'], 2), 'MDD', round(sg['maxDD'], 1), 'pair', round(ws['paired_mean_pp'], 3),
                  't', round(ws['paired_t'], 2), 'aff', len(aff), f'{time.time() - T0:.0f}s', flush=True)
    cand = {k[:-4]: v for k, v in out['results'].items() if k.endswith('_n10') and not k.startswith('BASE') and v['affected_pick_months'] >= 30}
    best = max(cand, key=lambda k: cand[k]['paired']['paired_t']) if cand else None
    out['chosen'] = best
    out['chosen_reason'] = 'highest selection-period paired t (n10) among versions with >= 30 affected pick-months'
    out['n_versions'] = len(VERSIONS)
    L.dump(out, 'iterations.json')
    print('chosen', best)
