#!/usr/bin/env python3
"""複製員：探索期（訊号日 2007-01～2019-03）年輕股殘差、P4 入選股。v2 正式版股池（排除金融、gapfix）。"""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / 'main_v2')); import v2_core as V; K = V.K; sys.path.pop(0)
import numpy as np, pandas as pd
L = pd.read_csv(HERE / 'my_listings.csv', dtype={'code': str}, parse_dates=['first']).set_index('code')
FIRST = np.array([L.loc[c, 'first'] for c in K.COLS], dtype='datetime64[ns]')
KIND = np.array([L.loc[c, 'kind'] for c in K.COLS])
YL, Y5, _ = V.load_revenue()
S = V.V2('gapfix', YL, Y5, exfin=True)
sd = S.sig_days
def per_ret(t, t2, pk):
    r = K.Cn[t, pk] / K.On[t, pk]
    if t2 > t: r = r * np.prod(K.GROW[t + 1:t2 + 1, pk], axis=0)
    tr = K.TRADED[t2, pk] & np.isfinite(K.On[t2, pk])
    r = np.where(tr, r * K.On[t2, pk] / np.where(tr, K.Cn[t2, pk], 1.0), r)
    return r - 1
rows = []
for i, d in enumerate(sd[:-1]):
    if not (pd.Timestamp('2006-04-01') <= d <= pd.Timestamp('2019-03-31')): continue
    cs = S.cross_section(d); el = np.where(cs['el'])[0]
    t = cs["e"]; t2 = K.IDX.get_loc(sd[i + 1]) + 1
    r = per_ret(t, t2, el)
    pr52 = pd.Series(cs['s52'][el]).rank(pct=True).to_numpy(); prr = pd.Series(cs['y3'][el]).rank(pct=True).to_numpy()
    age = (np.datetime64(d) - FIRST[el]).astype('timedelta64[D]').astype(float) / 365.25
    for j, a, b, c, rr, ag in zip(el, pr52, prr, cs['sc'][el], r, age):
        rows.append((d, K.COLS[j], KIND[j], ag, a, b, c, rr))
X = pd.DataFrame(rows, columns=['d', 'code', 'kind', 'age', 'pr52', 'prr', 'sc', 'r'])
X['new'] = X.kind.isin(['ipo', 'transfer', 'reorg', 'dr'])
X['young'] = X.kind.isin(['ipo', 'transfer']) & (X.age >= 1) & (X.age < 3)
X = X[np.isfinite(X.r)]
E = X[(X.d >= '2007-01-01')]
out = {'n_months': int(E.d.nunique()), 'pool_mean': round(E.groupby('d').size().mean(), 1),
       'young_mean': round(E.groupby('d').young.sum().mean(), 2),
       'young_ipo_mean': round(E[E.young & (E.kind == 'ipo')].groupby('d').size().reindex(E.d.unique(), fill_value=0).mean(), 2),
       'young_tr_mean': round(E[E.young & (E.kind == 'transfer')].groupby('d').size().reindex(E.d.unique(), fill_value=0).mean(), 2)}
def resid(g):
    A = np.column_stack([np.ones(len(g)), g.pr52, g.prr]); b, *_ = np.linalg.lstsq(A, g.r.to_numpy(), rcond=None)
    return g.r - A @ b
E = E.assign(res=E.groupby('d', group_keys=False).apply(resid))
def gap(mA, mB):
    s = E[mA].groupby('d').res.mean() - E[mB].groupby('d').res.mean()
    s = s.dropna(); return {'pp': round(s.mean() * 100, 2), 't': round(K.tstat(s), 2), 'n_m': len(s)}
old3 = ~E.new | (E.age >= 3)
out['young_vs_3plus'] = gap(E.young, old3)
out['young_ipo_vs_3plus'] = gap(E.young & (E.kind == 'ipo'), old3)
out['young_tr_vs_3plus'] = gap(E.young & (E.kind == 'transfer'), old3)
m35 = E.new & (E.age >= 3) & (E.age < 5); o5 = ~E.new | (E.age >= 5)
out['3to5_vs_5plus'] = gap(m35, o5)
out['ipo_3to4_vs_5plus'] = gap(E.new & (E.kind == 'ipo') & (E.age >= 3) & (E.age < 4), o5)
out['ipo_2to4_vs_4plus'] = gap(E.new & (E.kind == 'ipo') & (E.age >= 2) & (E.age < 4), ~E.new | (E.age >= 4))
# Fama-MacBeth
co = []
for d, g in E.groupby('d'):
    A = np.column_stack([np.ones(len(g)), g.pr52, g.prr, g.young.astype(float)]); b, *_ = np.linalg.lstsq(A, g.r.to_numpy(), rcond=None); co.append(b[3])
out['FM_young'] = {'pp': round(np.mean(co) * 100, 2), 't': round(K.tstat(np.array(co)), 2)}
# P4：原版 n10 入選股（挑選期 2006-04～2019-03）
pl = S.plan(10, 'official')
pk = []
for p in pl:
    if pd.Timestamp('2006-04-01') <= p[3] <= pd.Timestamp('2019-03-31'):
        for j in p[1]: pk.append((p[3], K.COLS[j]))
P = pd.DataFrame(pk, columns=['d', 'code']).merge(X[['d', 'code', 'young', 'r', 'kind', 'age']], on=['d', 'code'], how='left')
out['P4_young_by_kind'] = P[P.young].kind.value_counts().to_dict(); out['P4_new_1to3_by_kind'] = P[(P.age >= 1) & (P.age < 3) & (P.kind != 'old')].kind.value_counts().to_dict()
out['P4_n_slots'] = len(P); out['P4_young'] = int(P.young.sum()); out['P4_share'] = round(P.young.mean() * 100, 1)
g = P.groupby('d').apply(lambda x: x[x.young].r.mean() - x[~x.young].r.mean()).dropna()
out['P4_young_minus_other'] = {'pp': round(g.mean() * 100, 2), 't': round(K.tstat(g), 2), 'n_m': len(g)}
print(json.dumps(out, ensure_ascii=False, indent=0))
(HERE / 'r_explore.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
