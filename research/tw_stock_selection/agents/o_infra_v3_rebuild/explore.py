#!/usr/bin/env python3
"""探索（描述統計，不算任何策略淨值）：python3 -I -B explore.py → explore.json
E1 價格修補的 32 檔：每個換股日有幾檔進得了前 300 池（價格、流動性、隔日有成交）、缺不缺營收、若營收百分位設 0.5／1.0 會排第幾；下市前 6 個月報酬。
E2 「持有中永久停止交易」事件（R_DEL＋修補＋界限 pfix 1.0 的排除金融 n10，最大曝險情境）：持有時點、最後收盤、對價、出清差異。
E3 近似處置名單：2006-04～2021-09 每個換股日有幾檔合格池股票會被剔除；其中幾檔原本會進 n10／n30（用 V2 名單判斷）。
E4 機制描述（挑選期 2006-04～2019-04）：(a) 併購標的最後一個換股日的 52 週比；(b) 近似處置股在換股日的 52 週比百分位與下月報酬減池內等權（描述、不分群 t）。"""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import o3_core as O
V, K = O.V, O.K
O.apply_patch()
E = O.load_events()
YR = O.load_rdel(); YL, Y5, _ = V.load_revenue()
GC = O.delist_codes()
FOREIGN = set(json.loads((O.ORS / 'data_freeze.json').read_text(encoding='utf-8'))['foreign_codes'])
DM = pd.read_csv(HERE / 'delist_coverage.csv', dtype={'code': str}).set_index('code')
lt = O.last_trade_idx()
out = {}

# ---------- E1 ----------
PC = list(DM.index)
O.set_disposal('v2')
Sb = {pf: O.V3('gapfix', YR, Y5, exfin=True, gapcodes={c: GC[c] for c in PC}, pfix=pf, foreign=FOREIGN) for pf in (0.5, 1.0)}
rows = []
for d in Sb[1.0].sig_days:
    if not (pd.Timestamp('2019-05-17') <= d <= pd.Timestamp('2023-12-31')): continue
    for pf, S in Sb.items():
        cs = S.cross_section(d)
        od = S.order(cs, 'official')[:int(cs['el'].sum())]
        rk = {K.COLS[j]: i + 1 for i, j in enumerate(od)}
        for c in PC:
            j = K.CI[c]
            if cs['top'][j]:
                rows.append({'sig': str(d.date()), 'code': c, 'pfix': pf, 'in_top300': True, 'gap_eligible': bool(cs['gm'][j]), 'has_rev': bool(np.isfinite(cs['y3'][j])) if pf < 1 else None,
                             'rank': rk.get(c), 'pool': int(cs['el'].sum()), 's52': float(cs['s52'][j]) if np.isfinite(cs['s52'][j]) else None})
e1 = pd.DataFrame(rows)
pre6 = {}
for c in PC:
    j = K.CI[c]; L = lt[j]
    if L < 0: continue
    a0 = K.MYADJ.iloc[max(L - 126, 0):L + 1, j].dropna()
    pre6[c] = {'name': DM.loc[c, 'name'], 'type': E.loc[c, 'type'] if c in E.index else None, 'last_trade': str(K.IDX[L].date()),
               'ret_6m_before_last_trade': float(a0.iloc[-1] / a0.iloc[0] - 1) if len(a0) > 20 else None}
out['E1'] = {'n_rows': int(len(e1)),
             'stock_signal_in_top300_pfix1': int((e1['pfix'] == 1.0).sum()) if len(e1) else 0,
             'by_code': {c: {'n_sig_in_top300': int(((e1.code == c) & (e1.pfix == 1.0)).sum()),
                             'n_gap_eligible': int(((e1.code == c) & (e1.pfix == 1.0) & e1.gap_eligible).sum()),
                             'best_rank_pfix1': (int(e1[(e1.code == c) & (e1.pfix == 1.0)]['rank'].min()) if e1[(e1.code == c) & (e1.pfix == 1.0)]['rank'].notna().any() else None),
                             'n_top10_pfix1': int(((e1.code == c) & (e1.pfix == 1.0) & (e1['rank'] <= 10)).sum()),
                             'n_top30_pfix1': int(((e1.code == c) & (e1.pfix == 1.0) & (e1['rank'] <= 30)).sum()),
                             'n_top10_pfix05': int(((e1.code == c) & (e1.pfix == 0.5) & (e1['rank'] <= 10)).sum()),
                             'n_top30_pfix05': int(((e1.code == c) & (e1.pfix == 0.5) & (e1['rank'] <= 30)).sum())}
                         for c in sorted(set(e1.code))} if len(e1) else {},
             'pre_delist_6m': pre6,
             'per_signal_in_top300_mean': float(e1[e1.pfix == 1.0].groupby('sig').size().reindex([str(d.date()) for d in Sb[1.0].sig_days if pd.Timestamp('2019-05-17') <= d <= pd.Timestamp('2023-12-31')]).fillna(0).mean()) if len(e1) else 0}
print('E1', {k: v for k, v in out['E1'].items() if k in ('n_rows', 'stock_signal_in_top300_pfix1', 'per_signal_in_top300_mean')}, flush=True)
for c, v in out['E1']['by_code'].items(): print('   ', c, DM.loc[c, 'name'], v, pre6.get(c, {}).get('ret_6m_before_last_trade'))

# ---------- E2 ----------
S = O.V3('gapfix', YR, Y5, exfin=True, gapcodes=GC, pfix=1.0, foreign=FOREIGN)
t0, t1, sd = S.bounds(); pl = S.plan(10, 'official')
_, lg = O.simulate([(p[0], p[1]) for p in pl], 0, 0, t0, t1)
xr = O.exit_rules(E, 'main')
_, lgx = O.simulate([(p[0], p[1]) for p in pl], 0, 0, t0, t1, xrules=xr)
dead = {}
for d, codes, sh in lg['frozen']:
    t = K.IDX.get_loc(pd.Timestamp(d))
    for c in codes:
        j = K.CI[c]
        if lt[j] < t:
            x = dead.setdefault(c, {'first_frozen_rebal': d, 'n_frozen_rebal': 0, 'max_share': 0.0}); x['n_frozen_rebal'] += 1; x['max_share'] = max(x['max_share'], sh)
e2 = []
for c, x in dead.items():
    e = E.loc[c] if c in E.index else None
    held = [str(p[3].date()) for p in pl if K.CI[c] in set(p[1])]
    e2.append({'code': c, 'name': e['name'] if e is not None else '', 'type': e['type'] if e is not None else 'missing', 'last_trade': str(K.IDX[lt[K.CI[c]]].date()),
               'last_close': float(K.Cn[lt[K.CI[c]], K.CI[c]]), 'cash': (float(e['cash']) if e is not None and pd.notna(e['cash']) else None),
               'exit_factor_main': xr[K.CI[c]][0] if K.CI[c] in xr else None, 'held_signals': held[-3:], **x})
out['E2'] = {'n_frozen_rebal_days_noX': len(lg['frozen']), 'n_frozen_rebal_days_X': len(lgx['frozen']), 'n_dead_codes': len(dead),
             'max_frozen_share_noX': max([f[2] for f in lg['frozen']], default=0), 'exits': lgx['exits'], 'events': e2}
print('E2 frozen rebal days', len(lg['frozen']), '→ with X', len(lgx['frozen']), 'dead codes', len(dead))
for r in e2: print('   ', r)

# ---------- E3 ----------
H = O.load_disposal_hist()
O.set_disposal('none')
S0 = V.V2('gapfix', YL, Y5, exfin=True)
pl10 = {p[3]: set(p[1]) for p in S0.plan(10, 'official')}
pl30 = {p[3]: set(p[1]) for p in S0.plan(30, 'official')}
e3 = []
for d in S0.sig_days:
    if d >= O.D5_START: break
    cs = S0.cross_section(d)
    m = O._mask_from(H, d) & cs['el']
    e3.append({'sig': str(d.date()), 'n_pool_flagged': int(m.sum()), 'in_n10': int(len(set(np.where(m)[0]) & pl10.get(d, set()))),
               'in_n30': int(len(set(np.where(m)[0]) & pl30.get(d, set())))})
e3 = pd.DataFrame(e3); e3['year'] = e3['sig'].str[:4]
out['E3'] = {'per_year': e3.groupby('year')[['n_pool_flagged', 'in_n10', 'in_n30']].sum().to_dict(orient='index'),
             'mean_pool_flagged_per_signal': float(e3['n_pool_flagged'].mean()), 'total_in_n10': int(e3['in_n10'].sum()), 'total_in_n30': int(e3['in_n30'].sum()),
             'signals': int(len(e3))}
print('E3', {k: v for k, v in out['E3'].items() if k != 'per_year'})
print('   ', out['E3']['per_year'])

# ---------- E4（挑選期 2006-04～2019-04） ----------
# (a) 併購標的在最後一個換股日的 52 週比
O.set_disposal('v2')
S1 = V.V2('gapfix', YL, Y5, exfin=False)
a = {}
for c in E.index:
    if E.loc[c, 'type'] != 'cash' or c not in K.CI: continue
    j = K.CI[c]; L = lt[j]
    if L < 0 or K.IDX[L] > pd.Timestamp('2019-04-30'): continue
    prev = [d for d in S1.sig_days if K.IDX.get_loc(d) < L]
    if not prev: continue
    t = K.IDX.get_loc(prev[-1]); a[c] = {'sig': str(prev[-1].date()), 's52': float(S1.S52[t][j]) if np.isfinite(S1.S52[t][j]) else None,
                                          'cash_over_last_close': float(E.loc[c, 'cash_over_last_close']) if pd.notna(E.loc[c, 'cash_over_last_close']) else None}
# (b) 近似處置股：換股日池內 52 週比百分位、下一個換股日之前的報酬減池內等權
b = []
sdays = [d for d in S0.sig_days if d <= pd.Timestamp('2019-04-30')]
for i, d in enumerate(sdays[:-1]):
    cs = S0.cross_section(d)
    m = O._mask_from(H, d) & cs['el']
    if not m.any(): continue
    t, t2 = cs['e'], S0.cross_section(sdays[i + 1])['e']
    g = np.prod(K.GROW[t + 1:t2 + 1], axis=0) * K.Cn[t] / K.On[t]
    el = cs['el']
    pr52 = pd.Series(np.where(el, cs['s52'], np.nan)).rank(pct=True).to_numpy()
    ew = np.nanmean(np.where(el, g, np.nan))
    for j in np.where(m)[0]: b.append({'sig': str(d.date()), 'code': K.COLS[j], 'pct_s52': float(pr52[j]), 'ex_ret': float(g[j] - ew), 'score': float(cs['sc'][j])})
b = pd.DataFrame(b)
out['E4'] = {'merger_targets_s52_last_signal': a,
             'disposal_proxy_in_pool_2006_2019': {'n': int(len(b)), 'mean_pct_s52': float(b['pct_s52'].mean()) if len(b) else None,
                                                  'mean_next_ret_minus_poolEW': float(b['ex_ret'].mean()) if len(b) else None,
                                                  'median_next_ret_minus_poolEW': float(b['ex_ret'].median()) if len(b) else None,
                                                  'mean_score': float(b['score'].mean()) if len(b) else None,
                                                  'by_limit_regime': {k: {'n': int(len(g_)), 'mean_ex': float(g_['ex_ret'].mean())} for k, g_ in b.groupby(b['sig'] >= '2015-06-01')} if len(b) else {}}}
print('E4a', a); print('E4b', out['E4']['disposal_proxy_in_pool_2006_2019'])
(HERE / 'explore.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=lambda x: x.item() if hasattr(x, 'item') else str(x)), encoding='utf-8')
