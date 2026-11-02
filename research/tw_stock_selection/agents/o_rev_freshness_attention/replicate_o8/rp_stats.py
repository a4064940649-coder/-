#!/usr/bin/env python3
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
D = pd.concat([pd.read_parquet(HERE / f'rp_alpha_s{s}.parquet').assign(seed=s) for s in range(5)])
cols = ['a_gross', 'a_net', 'a_stress']
A = D.groupby(['leg', 'grp', 'm', 'j'])[cols + ['n', 'N']].mean().reset_index()
for c in cols: A[c] *= 100
A['mp'] = pd.PeriodIndex(A['m'], freq='M')
# 最後一個有效錨定月沒有「下一個有效錨定月」→ 無 α（研究員口徑：2026-10 不是官方錨定月）
A = A[A.mp < pd.Period('2026-09', 'M')].copy() if '--keeplast' not in sys.argv else A
def tst(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return x.mean() / (x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 2 else np.nan
def slopes(leg, g, col='a_net'):
    x = A[(A.leg == leg) & (A.grp == g)]
    out = {}
    for m, y in x.groupby('mp'):
        if len(y) < 10: continue
        jj = y.j.to_numpy(float); a = y[col].to_numpy()
        out[m] = np.polyfit(jj, a, 1)[0]
    return pd.Series(out).sort_index()
P = lambda s: pd.Period(s, 'M')
PER = {'post': (P('2019-05'), P('2026-09')), 'pre': (P('2006-04'), P('2018-12')), 'VAL': (P('2019-05'), P('2023-12')), 'HOLD': (P('2024-01'), P('2026-09'))}
def sel(s, k): a, b = PER[k]; return s[(s.index >= a) & (s.index <= b)]
def summ(s): return dict(mean=round(float(s.mean()), 4), t=round(float(tst(s)), 2), n=len(s))
def welch(a, b):
    return (a.mean() - b.mean()) / np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
R = {}
def block(leg, hi, lo, col='a_net', keys=('post', 'pre', 'VAL', 'HOLD')):
    bh, bl = slopes(leg, hi, col), slopes(leg, lo, col)
    I = (bh - bl).dropna()
    o = {}
    for k in keys:
        o[k] = dict(b_hi=summ(sel(bh, k)), b_lo=summ(sel(bl, k)), I=summ(sel(I, k)))
    if 'pre' in keys and len(sel(I, 'pre')) > 2:
        o['DDD'] = dict(mean=round(float(sel(I, 'post').mean() - sel(I, 'pre').mean()), 4), t=round(float(welch(sel(I, 'post'), sel(I, 'pre'))), 2))
    return o, bh, bl, I
o, bh, bl, I = block('main', 'A_hi', 'A_lo')
R['A_main_net'] = o
bm = slopes('main', 'A_mid')
R['A_mid'] = {k: summ(sel(bm, k)) for k in ('post', 'pre')}
x = sel(bl, 'post'); se = x.std(ddof=1) / np.sqrt(len(x))
R['b_lo_post_CI'] = [round(x.mean() - 1.96 * se, 3), round(x.mean() + 1.96 * se, 3)]
R['ratio_lo_hi'] = round(abs(sel(bl, 'post').mean()) / abs(sel(bh, 'post').mean()), 2)
R['A_main_gross'] = block('main', 'A_hi', 'A_lo', 'a_gross', ('post',))[0]
R['A_main_stress'] = block('main', 'A_hi', 'A_lo', 'a_stress', ('post',))[0]
# j α levels
lv = {}
for k in ('pre', 'post'):
    a, b = PER[k]
    x = A[(A.leg == 'main') & (A.mp >= a) & (A.mp <= b)]
    lv[k] = {g: round(float(x[x.grp == g].a_net.mean()), 3) for g in ('A_lo', 'A_mid', 'A_hi')}
    lv[k + '_byj'] = {g: [round(float(v), 2) for v in x[x.grp == g].groupby('j').a_net.mean()] for g in ('A_lo', 'A_hi')}
R['levels'] = lv
# 副指標
R['M50'] = block('main', 'M50_in', 'M50_out')[0]
R['FO'] = block('main', 'FO_hi', 'FO_lo', keys=('post',))[0]
Ifo = block('main', 'FO_hi', 'FO_lo', keys=('post',))[3]
Ia_fo = I[I.index >= P('2021-10')]
R['FO_window'] = dict(FO_I=summ(Ifo[(Ifo.index >= P('2021-10')) & (Ifo.index <= P('2026-09'))]), A_I=summ(Ia_fo[Ia_fo.index <= P('2026-09')]))
R['sub_b_post'] = {g: summ(sel(slopes('main', g), 'post')) for g in ('M50_in', 'FO_hi', 'COMBO')}
R['legs'] = {lg: block(lg, 'A_hi', 'A_lo', keys=('post', 'pre'))[0] for lg in ('rev', 'px')}
# 穩健性：逐年剔除、年份占比
Ip = sel(I, 'post')
yrs = sorted(set(Ip.index.year))
R['drop_year'] = {int(y): round(float(Ip[Ip.index.year != y].mean()), 4) for y in yrs}
R['year_share'] = {int(y): round(float(Ip[Ip.index.year == y].sum() / Ip.sum()), 3) for y in yrs}
# 母體
x = A[(A.leg == 'main') & (A.grp == 'A_lo') & (A.mp >= PER['post'][0]) & (A.mp <= PER['post'][1])]
R['pop'] = dict(N_lo_mean=round(float(x.N.mean()), 1), n_lo_mean=round(float(x.n.mean()), 2), n_months_post=int(len(Ip)), n_months_pre=int(len(sel(I, 'pre'))))
# 逐檔貢獻（A_hi − A_lo；slope 權重；seed 平均）
C = pd.concat([pd.read_parquet(HERE / f'rp_contrib_s{s}.parquet') for s in range(5)])
C['c'] = C['c'] * 100 / 5
C['mp'] = pd.PeriodIndex(C['m'], freq='M')
C = C[(C.mp >= PER['post'][0]) & (C.mp <= PER['post'][1]) & C.mp.isin(Ip.index)]
# 該月 j 權重
jw = {}
for m, y in A[(A.leg == 'main') & (A.grp == 'A_hi') & A.mp.isin(Ip.index)].groupby('mp'):
    jj = y.j.to_numpy(float); w = (jj - jj.mean()) / ((jj - jj.mean()) ** 2).sum(); jw.update({(m, int(a)): b for a, b in zip(jj, w)})
C['w'] = [jw.get((m, j), 0.0) for m, j in zip(C.mp, C.j)]
C['sg'] = np.where(C.grp == 'A_hi', 1.0, -1.0)
C['x'] = C.c * C.w * C.sg / len(Ip)
cs = C.groupby('code').x.sum()
resid = Ip.mean() - cs.sum()
top = cs.abs().sort_values(ascending=False)
R['stock'] = dict(I=round(float(Ip.mean()), 4), sum_contrib=round(float(cs.sum()), 4), resid_cost=round(float(resid), 4),
                  top5=[(c, round(float(cs[c]), 4), round(float(cs[c] / Ip.mean()), 3)) for c in top.index[:5]],
                  I_drop_top=round(float(Ip.mean() - cs[top.index[0]]), 4))
rng = np.random.default_rng(0); v = cs.to_numpy(); bs = []
for _ in range(5000):
    k = rng.integers(0, len(v), len(v)); bs.append(v[k].sum() + resid)
R['stock']['boot_CI'] = [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]
print(json.dumps(R, ensure_ascii=False, indent=1, default=str))
json.dump(R, open(HERE / 'rp_results.json', 'w'), ensure_ascii=False, indent=1, default=str)
