# -*- coding: utf-8 -*-
"""Adversarial statistical review (o2) of o_issuance_avoid. Read-only: reads researcher outputs, writes nothing.
Run: python3 -I -B review_stat_o2.py"""
import json, math
from pathlib import Path
import numpy as np, pandas as pd

H = Path(__file__).resolve().parent
ex = json.load(open(H / 'explore.json'))
res = json.load(open(H / 'results.json'))
X = pd.read_csv(H / 'explore_events_sel.csv', dtype={'code': str}, parse_dates=['anchor', 't_recv', 't_exr', 't_list', 't_eff'])

def norm_p(t):
    return math.erfc(abs(t) / math.sqrt(2))

def tstat(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return x.mean() / (x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 2 else np.nan

def cl_t(df, col):
    s = df[[col, 'ym']].dropna()
    g = s.groupby('ym')[col].mean()
    return len(s), s[col].mean() * 100, tstat(g.to_numpy())

print('=== A. multiple testing over event-window table (selection, cluster t) ===')
rows = [(k, v['t_cl']) for k, v in ex['car'].items() if 'BHAR' not in k and v.get('t_cl') is not None]
m = len(rows)
ps = sorted([(norm_p(t), k, t) for k, t in rows])
print('windows:', m, ' Bonferroni 5% |t| cut ~', round(3.19 if m == 36 else 3.2, 2))
for i, (p, k, t) in enumerate(ps[:10]):
    holm = p * (m - i)
    print(f'{k:28s} t={t:6.2f} p={p:.4f} holm_p={min(1, holm):.3f}', 'PASS' if holm < 0.05 else 'fail')
nsub = sum(len(v) for v in ex['subgroups'].values())
print('subgroup cells looked at (top level keys):', nsub)

print('\n=== B. SEO recv [0,+2] by year / leave-one-year-out ===')
S = X[(X.etype == 'SEO') & X['car_recv_0_2'].notna()].copy()
S['y'] = S['anchor'].dt.year
by = S.groupby('y')['car_recv_0_2'].agg(['count', 'mean'])
by['sum_pp'] = by['count'] * by['mean'] * 100
print(by.round(4).to_string())
print('all:', [round(v, 3) for v in cl_t(S, 'car_recv_0_2')])
for y in sorted(S.y.unique()):
    n, mu, t = cl_t(S[S.y != y], 'car_recv_0_2')
    print(f' drop {y}: n={n} mean={mu:.2f}% t={t:.2f}')
top2 = by['sum_pp'].nsmallest(2).index.tolist()
n, mu, t = cl_t(S[~S.y.isin(top2)], 'car_recv_0_2')
print(f' drop 2 most negative years {top2}: n={n} mean={mu:.2f}% t={t:.2f}')
# winsorize
w = S['car_recv_0_2'].clip(S['car_recv_0_2'].quantile(.01), S['car_recv_0_2'].quantile(.99))
print(' median %:', round(S['car_recv_0_2'].median() * 100, 2), ' share<0:', round((S['car_recv_0_2'] < 0).mean(), 3))
# SEO is anchored on FSC recv date; check pre-window
for c in ['car_recv_-20_-1', 'car_recv_-60_-1']:
    print(' pre-window', c, [round(v, 3) for v in cl_t(S.dropna(subset=[c]), c)])

print('\n=== C. "ex-rights" anchor: is it the SEO ex-rights date or a cash-dividend ex-date? ===')
E = X[(X.etype == 'SEO') & X['t_exr'].notna()].copy()
lag = (E['t_exr'] - E['t_eff']).dt.days
print(' eff->exr calendar days p10/50/90:', np.nanpercentile(lag, [10, 50, 90]).round(1))
mo = E['t_exr'].dt.month.value_counts().sort_index()
print(' exr month distribution:', mo.to_dict())
summer = E['t_exr'].dt.month.isin([6, 7, 8, 9])
print(' share exr in Jun-Sep (cash-dividend season):', round(summer.mean(), 3))
for nm, sub in [('Jun-Sep', E[summer]), ('other months', E[~summer])]:
    for c in ['car_exr_-10_-1', 'car_exr_0_20']:
        print(f'  {nm:12s} {c}:', [round(v, 3) for v in cl_t(sub.dropna(subset=[c]), c)])

print('\n=== D. CB pre-listing run-up: conditioning on completed listing ===')
C = X[(X.etype == 'CB')].copy()
print(' CB rows with recv:', C['t_recv'].notna().sum(), ' with listing:', C['t_list'].notna().sum())
lagcl = (C['t_list'] - C['t_recv']).dt.days.dropna()
print(' recv->listing days p10/50/90:', np.nanpercentile(lagcl, [10, 50, 90]).round(1))
f = pd.read_csv(H / 'fsc_cases_all.csv', dtype={'code': str})
f = f[f.ctype == '上市']
it = f['item'].astype(str)
for nm, mask in [('CB', it.str.contains('轉換公司債')), ('SEO', it.str.startswith('現金'))]:
    g = f[mask]
    bad = ~g['result'].isin(['生效', '核減'])
    print(f' {nm}: filings={len(g)}  not effective (withdrawn/revoked/rejected)={bad.sum()}  share={bad.mean():.3f}')

print('\n=== E. strategy paired diffs: year concentration ===')
st = res['OOS']['strategy']
for n in ('n10', 'n30'):
    for per in ('normal|SEL', 'normal|VAL', 'normal|ALL'):
        d = st.get(n, {}).get(per)
        if not d: continue
        p = d['paired']; ys = p['yearly_sum_pp']
        tot = sum(ys.values())
        print(f' {n} {per}: mean={p["paired_mean_pp"]:.3f}pp t={p["paired_t"]:.2f} n_m={p["n_m"]} nonzero={p.get("n_nonzero")} sum={tot:.1f}pp')
        print('   yearly:', ys)
        ymax = max(ys, key=lambda k: abs(ys[k]))
        print(f'   largest |year| {ymax}: {ys[ymax]:.1f}pp = {abs(ys[ymax]) / max(1e-9, sum(abs(v) for v in ys.values())):.2f} of gross')
print(' keys available:', {n: list(st[n].keys()) for n in st})

print('\n=== F. n10 validation affected list: concentration ===')
A = pd.read_csv(H / 'oos_affected_n10.csv', dtype={'code': str}, parse_dates=['d'])
V = A[A.d >= '2019-05-01']
print(' val affected rows:', len(V), ' sum contrib pp:', round(V.contrib_pp.sum(), 2))
g = V.groupby('code')['contrib_pp'].sum().sort_values()
print(' by code (worst 5 / best 5):', g.head(5).round(2).to_dict(), g.tail(5).round(2).to_dict())
print(' ex worst code:', round(g.iloc[1:].sum(), 2), ' ex worst+best:', round(g.iloc[1:-1].sum(), 2))
print(' share of rows with removed name beating added name:', round((V.r_removed > V.r_added).mean(), 3))
