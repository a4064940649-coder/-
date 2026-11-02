#!/usr/bin/env python3
"""複製員 o7：零衝擊正式欄統計（自寫統計）。AB／B10 淨值＝軌 B 自寫引擎 40 seeds 平均；滑價格點用軌 A。"""
import sys, json, math
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import rb_core as B
OUT = HERE / 'out'; AG = B.AG
NA = pd.read_parquet(OUT / 'navs_a.parquet'); NB = pd.read_parquet(OUT / 'navs_b.parquet')
SD = B.official_days() + [B.END]
def r(x, k=4): return None if x is None or not np.isfinite(x) else round(float(x), k)
nav = {}
for k in ('V7', 'V7_B10'):
    for c in ('normal', 'stress', 'EW'):
        nav[f'AB_{k}|{c}'] = B.avg_nav([NB[f'AB_{k}|{c}|s{s}'].dropna() for s in range(40)])
        d = float((nav[f'AB_{k}|{c}'] - NA[f'AB_{k}|{c}'].reindex(nav[f'AB_{k}|{c}'].index)).abs().max())
        print('A vs B', k, c, d)
for k in ('V7', 'V7_B10'):
    for c in ('normal', 'stress'): nav[f'{k}|{c}'] = NA[f'{k}|{c}'].dropna()
res = {}
A0, A1 = '2006-04-01', '2026-10-12'
E50 = B.E50
for k in ('AB_V7', 'AB_V7_B10', 'V7', 'V7_B10'):
    for c in ('normal', 'stress'):
        s = B.seg(nav[f'{k}|{c}'], SD, A0, A1); res[f'{k}|{c}|ALL'] = {kk: r(v) for kk, v in s.items()}
res['0050_CAGR'] = r(B.cagr_mdd(E50.reindex(nav['V7|normal'].index), A0, A1)[0])
res['EW_CAGR(AB_V7)'] = r(B.cagr_mdd(nav['AB_V7|EW'], A0, A1)[0])
PER = {'SEL': ('2006-04-01', '2018-12-31'), 'VAL': ('2019-01-01', '2023-12-31'), 'POST2019': ('2019-01-01', A1), 'POST2019-05': ('2019-05-01', A1),
       'FROM2008': ('2008-01-01', A1), 'HOLD': ('2024-01-01', A1)}
for k in ('AB_V7', 'AB_V7_B10'):
    for p, (a, b) in PER.items():
        s = B.seg(nav[f'{k}|normal'], SD, a, b); s['E50_CAGR'] = B.cagr_mdd(E50.reindex(nav[f'{k}|normal'].index), a, b)[0]
        res[f'{k}|{p}'] = {kk: r(v) for kk, v in s.items()}
# 端點（AB）
x = B.xs(nav['AB_V7|normal'], SD, A0, A1)
ep = {}
for m in (1, 3, 5, 10): ep[f'drop_best{m}'] = r(B.tstat(x.drop(x.nlargest(m).index)))
ep['drop_first_year'] = r(B.tstat(x[x.index >= x.index[0] + pd.DateOffset(years=1)]))
ep['drop_first12'] = r(B.tstat(x.iloc[12:]))
ep['drop_last1'] = r(B.tstat(x.iloc[:-1]))
n = nav['AB_V7|normal']; e = E50.reindex(n.index)
me = n.groupby(n.index.to_period('M')).last(); be = e.groupby(e.index.to_period('M')).last()
cm = (me.pct_change() - be.pct_change()).dropna(); ep['calendar_month_t'] = r(B.tstat(cm)); ep['calendar_n'] = len(cm)
res['endpoints_AB'] = ep
# 成本兩平：t 對單邊滑價線性內插
SL = sorted({float(c.split('slip')[1]) for c in NA.columns if '|slip' in c})
def be(k, tgt):
    ts = [B.seg(NA[f'{k}|slip{s}'].dropna(), SD, A0, A1)['t'] for s in SL]
    for i in range(len(SL) - 1):
        if (ts[i] - tgt) * (ts[i + 1] - tgt) <= 0:
            return r(100 * (SL[i] + (tgt - ts[i]) * (SL[i + 1] - SL[i]) / (ts[i + 1] - ts[i])), 4)
    return None
res['breakeven_slip_pct'] = {f'{k}|t{t}': be(k, t) for k in ('V7', 'V7_B10', 'AB_V7', 'AB_V7_B10') for t in (2.3, 2.0)}
# V7−V4、V7−V6（AB，一般成本；V4／V6 淨值取各自研究的凍結輸出）
V4N = pd.read_parquet(AG / 'o_v4_revenue_patch_official' / 'navs_v4.parquet'); V6N = pd.read_parquet(AG / 'o_v6_price_panel_official' / 'navs_v6.parquet')
SD4 = list(pd.to_datetime(pd.read_csv(AG / 'o_v4_revenue_patch_official' / 'plan_V4.csv')['signal'])) + [B.END]
SD6 = list(pd.to_datetime(pd.read_csv(AG / 'o_v6_price_panel_official' / 'plan_V6.csv')['signal'])) + [B.END]
print('grids equal V4/V6/V7', SD4 == SD, SD6 == SD)
def paired(na, nb, sd):
    D = pd.DatetimeIndex([d for d in sd if d in na.index])
    ra = (na.reindex(D).shift(-1) / na.reindex(D) - 1).iloc[:-1]; rb = (nb.reindex(D).shift(-1) / nb.reindex(D) - 1).iloc[:-1]
    d = (ra - rb); d = d[(d.index >= A0)]
    nz = d[d.abs() > 1e-12]
    o = dict(ann_pp=r(B.cagr_mdd(na, A0, A1)[0] - B.cagr_mdd(nb, A0, A1)[0]), mean_pp=r(d.mean() * 100, 5), t=r(B.tstat(d)), n_nonzero=len(nz),
             first=str(nz.index.min().date()) if len(nz) else None, last=str(nz.index.max().date()) if len(nz) else None,
             max_month=str(d.idxmax().date()), t_drop_max=r(B.tstat(d.drop(d.idxmax()))))
    return o
for k in ('AB_V7', 'AB_V7_B10', 'V7'):
    k4 = k.replace('V7', 'V4'); k6 = k.replace('V7', 'V6')
    res[f'{k}-V4'] = paired(nav[f'{k}|normal'], V4N[f'{k4}|normal'].dropna(), SD)
    res[f'{k}-V6'] = paired(nav[f'{k}|normal'], V6N[f'{k6}|normal'].dropna(), SD)
# 2006～2023 對 V6 日淨值
for k in ('V7', 'AB_V7', 'AB_V7_B10'):
    a = nav[f'{k}|normal']; b = V6N[k.replace('V7', 'V6') + '|normal'].reindex(a.index)
    m = a.index < pd.Timestamp('2024-01-01'); res[f'pre2024_maxabs_vs_V6|{k}'] = float((a[m] - b[m]).abs().max())
# 保留期：日曆歸屬（窗終點 ≥ 2024-01-02）、平穩 bootstrap（自寫；種子 777＋L）
def wins(n, sd):
    D = pd.DatetimeIndex([d for d in sd if d in n.index]); v = n.reindex(D).to_numpy(); b = E50.reindex(D).to_numpy()
    w = pd.DataFrame({'start': D[:-1], 'end': D[1:], 'r': v[1:] / v[:-1] - 1, 'b': b[1:] / b[:-1] - 1})
    w['x'] = w.r - w.b; w['lx'] = np.log1p(w.r) - np.log1p(w.b); return w
def sboot(c, n, L, Bn, seed):
    rng = np.random.default_rng(seed); m = len(c); tot = np.zeros(Bn); pos = rng.integers(0, m, Bn)
    for t in range(n):
        if t > 0:
            jump = rng.random(Bn) < 1.0 / L
            pos = np.where(jump, rng.integers(0, m, Bn), (pos + 1) % m)
        tot += c[pos]
    return tot
CUT = pd.Timestamp('2024-01-02')
for k in ('AB_V7', 'AB_V7_B10', 'AB_V7|stress'):
    nn = nav[k if '|' in k else f'{k}|normal']; w = wins(nn, SD)
    h = w[w.end >= CUT]; c = w[(w.start >= A0) & (w.end < CUT)]
    act = h.lx.sum(); o = dict(n_hold=len(h), n_ctrl=len(c), mean_x_pp=r(h.x.mean() * 100), t=r(B.tstat(h.x)), cum_log=r(act))
    for L in (3, 6, 12):
        for lab, sh in (('raw', 0.0), ('shrink50', 0.5), ('zero', 1.0)):
            cc = c.lx.to_numpy() - sh * c.lx.mean()
            o[f'p_lower_L{L}_{lab}'] = r((sboot(cc, len(h), L, 10000, 777 + L) <= act).mean())
    res[f'HOLD_cal|{k}'] = o
# 周轉
ab = json.load(open(OUT / 'ab_b.json'))
res['nblk_AB_V7_seeds'] = [ab['nblk'][f'V7|s{s}'] for s in range(40)][:5]
json.dump(res, open(OUT / 'stats.json', 'w'), ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False, indent=1))
