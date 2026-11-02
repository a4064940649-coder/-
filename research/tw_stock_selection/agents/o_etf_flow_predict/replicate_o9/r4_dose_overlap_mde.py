import sys, json
sys.path.insert(0, '/home/user/-/research/tw_stock_selection/agents/o_etf_flow_predict/replicate_o9')
import pandas as pd, numpy as np
from statistics import NormalDist
import rlib as R

OUT = R.AG + '/o_etf_flow_predict/replicate_o9/'
td = R.tdays()
val = R.panel('value').loc[td]
res = {}
# ---- 0056 dose: AUM points (億) taken as the only available inputs (no monthly AUM data exists)
pts = {'2017-12': 106.5, '2021-10': 1000.0, '2022-10': 1591.0, '2026-03': 5210.0, '2026-07': 7184.0}
ps = pd.Series(pts); ps.index = pd.to_datetime(ps.index)  # first of month


def aum_at(d, mode='lin'):
    x = (ps.index - pd.Timestamp('2000-01-01')).days.values.astype(float)
    y = ps.values if mode == 'lin' else np.log(ps.values)
    v = np.interp((pd.Timestamp(d) - pd.Timestamp('2000-01-01')).days, x, y)
    return (v if mode == 'lin' else np.exp(v)) * 1e8


ev = R.events()
e56 = ev[(ev.etf == '0056') & (ev.action == 'add')]
for mode in ['lin', 'log']:
    rows = []
    for _, r in e56.iterrows():
        C = R.last_td_le(pd.Timestamp(r.cutoff))
        adv20 = val.loc[:C, r.code].tail(20).mean() if r.code in val.columns else np.nan
        rows.append(dict(review=r.review, code=r.code, D=aum_at(C, mode) / 50 / adv20))
    d = pd.DataFrame(rows)
    res['D0056_' + mode] = dict(n=int(d.D.notna().sum()), median=d.D.median(), p25=d.D.quantile(.25), p75=d.D.quantile(.75), max=d.D.max(),
                                by_review=d.groupby('review').D.median().round(2).to_dict())
    print('0056 D', mode, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in res['D0056_' + mode].items()})
# ---- TW50 predicted adds: weight and D per 1000億
pred = pd.read_csv(OUT + 'r1_tw50_v4_preds.csv', dtype={'code': str})
rows = []
for rv, tf, C in R.tw50_reviews():
    p = pred[pred.review == rv]
    if not len(p[p.kind == 'add']):
        continue
    mc, rk = R.universe_mcap(C, 0, True)
    mem = R.tw50_members_at(C)
    adds = list(p[p.kind == 'add'].code); dels = set(p[p.kind == 'del'].code)
    new = [m for m in mem if m not in dels] + adds
    tot = mc.reindex(new).sum()
    for x in adds:
        w = mc[x] / tot
        adv20 = val.loc[:C, x].tail(20).mean()
        rows.append(dict(review=rv, code=x, w=w, D1000=w * 1000e8 / adv20))
d = pd.DataFrame(rows)
res['tw50_dose'] = dict(n=len(d), w_median=d.w.median(), D1000_median=d.D1000.median())
print('TW50', res['tw50_dose'])
# ---- overlap with V7 B10 seed0 holdings on announcement date
plan = pd.read_csv(R.AG + '/o_v7_listing_exec_engine/plan_AB_V7_B10_s0.csv', parse_dates=['signal', 'exec'])


def held(d, strict=False):
    pl = plan[plan['exec'] < d] if strict else plan[plan['exec'] <= d]
    return set(pl.iloc[-1].picks.split(',')) if len(pl) else set()


ov = {}
for strict in [False, True]:
    tag = 'exec<ann' if strict else 'exec<=ann'
    o = {}
    for kind in ['add', 'del']:
        h = n = 0
        for _, r in pred[pred.kind == kind].iterrows():
            e = ev[(ev.etf == 'TW50') & (ev.review == r.review)]
            tf = [t for rv, t, c in R.tw50_reviews() if rv == r.review][0]
            ann = R.last_td_le(pd.Timestamp(e.ann.iloc[0])) if len(e) else R.last_td_le(tf - pd.Timedelta(days=14))
            n += 1; h += r.code in held(ann, strict)
        o['tw50_pred_' + kind] = (h, n)
    for kind in ['add', 'del']:
        x = ev[(ev.etf == '0056') & (ev.action == kind)]
        h = sum(r.code in held(R.last_td_le(pd.Timestamp(r.ann)), strict) for _, r in x.iterrows())
        o['0056_' + kind] = (int(h), len(x))
    ov[tag] = o
    print('overlap', tag, o)
res['overlap'] = ov
# ---- P4 MDE: sigma of monthly excess of random 30 of top-150 vs EW top-150
adj = R.panel('adj').loc[td]
me = pd.Series(td, index=td).groupby(td.to_period('M')).last()
me = me[(me >= pd.Timestamp('2013-01-31'))]
sig = {}
for start in ['2013-01', '2019-05']:
    for seed in range(5):
        rng = np.random.default_rng(seed); ex = []
        for i in range(len(me) - 1):
            d0, d1 = me.iloc[i], me.iloc[i + 1]
            if d0 < pd.Timestamp(start):
                continue
            mc, rk = R.universe_mcap(d0, 0, True)
            U = rk[rk <= 150].index
            r = (adj.loc[d1, U] / adj.loc[d0, U] - 1).dropna()
            pick = rng.choice(r.index, 30, replace=False)
            ex.append(r[pick].mean() - r.mean())
        sig.setdefault(start, []).append(np.std(ex, ddof=1))
k = NormalDist().inv_cdf(0.975) + NormalDist().inv_cdf(0.8)
res['P4'] = {s: dict(sigma=np.mean(v) * 100, MDE={T: k * np.mean(v) * 100 / np.sqrt(T) for T in (36, 48, 90)}) for s, v in sig.items()}
res['P4_given_sigma_2.081'] = {T: k * 2.081 / np.sqrt(T) for T in (36, 48, 90)}
print('P4', res['P4'], res['P4_given_sigma_2.081'])
json.dump(res, open(OUT + 'r4_dose_overlap_mde.json', 'w'), indent=1, default=float)
