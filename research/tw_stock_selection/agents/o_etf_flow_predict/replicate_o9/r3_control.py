import sys, json
sys.path.insert(0, '/home/user/-/research/tw_stock_selection/agents/o_etf_flow_predict/replicate_o9')
import pandas as pd, numpy as np
import rlib as R

OUT = R.AG + '/o_etf_flow_predict/replicate_o9/'
td = R.tdays()
adj = R.panel('adj').loc[td]; cr = R.panel('close_raw').loc[td]; op = R.panel('open').loc[td]
val = R.panel('value').loc[td]; ret = R.panel('ret').loc[td]
hi52 = adj.rolling(250, min_periods=200).max()
ratio52 = adj / hi52
r20 = adj / adj.shift(20) - 1
sig60 = ret.rolling(60, min_periods=40).std()
v60 = val.rolling(60, min_periods=40).mean()
Q = 1e6
FEE = 0.001425 * 0.28


def tick(p):
    return 0.01 if p < 10 else 0.05 if p < 50 else 0.1 if p < 100 else 0.5 if p < 500 else 1.0 if p < 1000 else 5.0


def side_cost(code, C, entry_open):
    s = sig60.at[C, code]; v = v60.at[C, code]
    imp = 0.001 + 0.5 * s * np.sqrt(Q / (0.05 * v))
    return max(imp, tick(entry_open) / entry_open) + FEE


def win_ret(code, d0, d1):
    """open of d0 -> close of d1, dividend-adjusted"""
    o = op.at[d0, code]; c0 = cr.at[d0, code]; a0 = adj.at[d0, code]; a1 = adj.at[d1, code]
    if any(pd.isna(x) for x in (o, c0, a0, a1)) or o <= 0:
        return np.nan
    return a1 / (o * a0 / c0) - 1


def next_td(d):
    return td[td.searchsorted(d, side='right')]


def prev_td(d):  # last td strictly before d
    return td[td.searchsorted(d, side='left') - 1]


ev = R.events(); ev = ev[ev.etf == 'TW50']
pred = pd.read_csv(OUT + 'r1_tw50_v4_preds.csv', dtype={'code': str})
rows = []
for rv, tf, C in R.tw50_reviews():
    e = ev[ev.review == rv]
    if len(e):
        ann = pd.Timestamp(e.ann.iloc[0]); eff = pd.Timestamp(e.eff.iloc[0])
    else:
        eff = tf; ann = tf - pd.Timedelta(days=14)
    ann = R.last_td_le(ann); eff = R.last_td_le(eff)
    d0 = next_td(C)
    ends = {'W1': prev_td(ann), 'W2': ann, 'W3': prev_td(eff)}
    p = pred[pred.review == rv]
    if not len(p):
        continue
    mem = R.tw50_members_at(C)
    evset = set(e.code) | set(p.code)
    liq = (v60.loc[C] >= 1e7) & (cr.loc[C] >= 10)
    cand = [x for x in liq[liq].index if x not in mem and x not in evset and len(x) == 4 and not x.startswith('0') and not x.startswith('91')]
    F = pd.DataFrame({'a': ratio52.loc[C, cand], 'b': r20.loc[C, cand]}).dropna()
    Z = (F - F.mean()) / F.std()
    for _, r in p.iterrows():
        x = r.code
        if x not in adj.columns or pd.isna(ratio52.at[C, x]) or pd.isna(op.at[d0, x]):
            rows.append(dict(review=rv, code=x, kind=r.kind, ok=False)); continue
        zx = (np.array([ratio52.at[C, x], r20.at[C, x]]) - F.mean().values) / F.std().values
        dist = np.sqrt(((Z.values - zx) ** 2).sum(1))
        ctrl = list(Z.index[np.argsort(dist)[:3]])
        rec = dict(review=rv, code=x, kind=r.kind, hit=r.hit, rank=r['rank'], C=str(C.date()), ann=str(ann.date()), ok=True, ctrl=','.join(ctrl))
        o = op.at[d0, x]
        cs = side_cost(x, C, o)
        rt = 2 * cs + 0.003
        for w, d1 in ends.items():
            rs = win_ret(x, d0, d1)
            rc = np.nanmean([win_ret(c, d0, d1) for c in ctrl])
            rec[w + '_gross'] = rs - rc
            rec[w + '_net'] = rs - rc - rt
            rec[w + '_stress'] = rs - rc - rt - 0.008
        rec['rt_cost'] = rt
        rec['D_dummy'] = None
        rows.append(rec)
df = pd.DataFrame(rows)
df.to_csv(OUT + 'r3_control_events.csv', index=False)


def summ(d, col):
    d = d.dropna(subset=[col])
    g = d.groupby('review')[col].mean()
    return dict(n=len(d), n_rev=int(d.review.nunique()), mean=d[col].mean(), median=d[col].median(),
                t_cluster=R.tstat(g.values), t_event=R.tstat(d[col].values), mean_of_rev=g.mean())


res = {}
for kind in ['add', 'del']:
    d = df[(df.kind == kind) & df.ok]
    print(kind, 'preds', int((df.kind == kind).sum()), 'with returns', len(d), 'reviews', d.review.nunique())
    for col in ['W1_gross', 'W1_net', 'W2_gross', 'W2_net', 'W2_stress', 'W3_gross', 'W3_net']:
        s = summ(d, col); res[f'{kind}_{col}'] = s
        print(' ', col, {k: round(v * (100 if k in ('mean', 'median', 'mean_of_rev') else 1), 2) for k, v in s.items()})
d = df[(df.kind == 'add') & df.ok]
for lab, m in [('sel<=2019-04', d.C <= '2019-04-30'), ('2019+', d.C > '2019-04-30')]:
    s = summ(d[m], 'W2_net'); res['add_W2_net_' + lab] = s
    print(lab, {k: round(v * (100 if k in ('mean', 'median', 'mean_of_rev') else 1), 2) for k, v in s.items()})
# leave-one-review-out
g = d.groupby('review')['W2_net']
ts = []
for rv in d.review.unique():
    dd = d[d.review != rv]; ts.append(R.tstat(dd.groupby('review')['W2_net'].mean().values))
ts_ev = [R.tstat(d[d.review != rv]['W2_net'].values) for rv in d.review.unique()]
yr = d.C.str[:4]
worst = d.groupby(yr)['W2_net'].mean().idxmin()
res['loo_t_cluster'] = (min(ts), max(ts)); res['loo_t_event'] = (min(ts_ev), max(ts_ev))
res['drop_worst_year'] = dict(year=worst, t_cluster=R.tstat(d[yr != worst].groupby('review')['W2_net'].mean().values), t_event=R.tstat(d[yr != worst]['W2_net'].values))
print('LOO t cluster', res['loo_t_cluster'], 'event', res['loo_t_event'], 'drop worst year', res['drop_worst_year'])
print('mean rt cost add', d.rt_cost.mean())
json.dump(res, open(OUT + 'r3_control.json', 'w'), indent=1, default=float)
