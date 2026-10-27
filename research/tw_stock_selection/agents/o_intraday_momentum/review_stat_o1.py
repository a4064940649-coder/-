#!/usr/bin/env python3
"""反方審查（資料與統計）o_intraday_momentum：P4b（買進改隔日收盤）的統計脆弱性。只讀，不寫研究員的檔案，只印出結果。"""
import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import olib as L
K, V = L.K, L.V
IDX, COLS, N = L.IDX, L.COLS, L.N
YL, Y5, _ = V.load_revenue()
S = V.V2('gapfix', YL, Y5, exfin=True)
pl = S.plan(10, 'official')
t0, t1, sd = S.bounds()
S52, V60 = V.rolling_signals('gapfix')
pp = [(p[0], p[1]) for p in pl]
nons3 = {p[0] for p in pl if not (L.SRC3[0] <= IDX[p[0]] <= L.SRC3[1])}

def tt(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return round(L.tstat(x), 2), round(L.nw_t(x), 2), len(x), round(float(x.mean()) * 100, 3)

# ---------- 1. 每筆：新進股 O->C 相對同日池平均 O->C（扣掉當日市場盤中漂移）----------
rows = []; prev = None
for (e, pk, pool, d) in pl:
    cur = set(pk.tolist())
    if prev is not None:
        el, rk = L.base_pool(IDX.get_loc(d), S52, V60)
        oc_all = K.Cn[e] / K.On[e] - 1
        okp = el & np.isfinite(oc_all) & K.TRADED[e]
        pool_oc = np.nanmean(oc_all[okp])
        # 前 300 池在「非換股日」的平均盤中（前 5 個交易日）作對照
        nb = []
        for k in range(1, 6):
            x = K.Cn[e - k] / K.On[e - k] - 1; nb.append(np.nanmean(x[okp & np.isfinite(x)]))
        hold_oc = np.nanmean(oc_all[list(cur & prev)]) if len(cur & prev) else np.nan
        s3 = L.SRC3[0] <= IDX[e] <= L.SRC3[1]
        for j in cur - prev:
            rows.append(dict(d=d, e=IDX[e], code=COLS[j], imp=K.On[e, j] / K.Cn[e, j] - 1, oc=oc_all[j], pool_oc=pool_oc,
                             pre5=np.nanmean(nb), hold_oc=hold_oc, src3=s3))
    prev = cur
R = pd.DataFrame(rows)
R['per'] = np.where(R.d <= L.SEL_LAST_SIG, 'sel', 'val')
R = R[(R.d <= V.END1012) & (R.d >= pd.Timestamp('2006-04-01'))]
for per in ('sel', 'val'):
    g = R[(R.per == per) & ~R.src3]
    m = g.groupby('d').agg(oc=('oc', 'mean'), pool=('pool_oc', 'first'), pre5=('pre5', 'first'), hold=('hold_oc', 'first'), imp=('imp', 'mean'))
    print(f'[{per}] months', len(m), 'trades', len(g))
    print('  new O->C            t,nw,n,mean%', tt(m.oc))
    print('  pool O->C same day  ', tt(m.pool))
    print('  pool O->C prev5 days', tt(m.pre5))
    print('  held-over O->C      ', tt(m.hold))
    print('  new minus pool      ', tt(m.oc - m.pool))
    print('  imp (O/C-1)         ', tt(m.imp))
    if per == 'val':
        y = m.groupby(m.index.year)['imp'].mean() * 100
        print('  imp by year', y.round(3).to_dict())
        for yy in (2021, 2022):
            mm = m[m.index.year != yy]; print(f'  imp ex {yy}', tt(mm.imp))
        srt = m.imp.sort_values(ascending=False)
        for k in (1, 3, 5):
            print(f'  imp ex top{k} months', tt(m.imp.drop(srt.index[:k])), 'dropped', [str(x.date()) for x in srt.index[:k]])
        # 月份 bootstrap
        rng = np.random.default_rng(0); x = m.imp.to_numpy()
        bs = np.array([rng.choice(x, len(x)).mean() for _ in range(20000)])
        print('  bootstrap P(mean<=0.20%)', float((bs <= 0.002).mean()), 'P(<=0)', float((bs <= 0).mean()))
        post = g[g.e >= pd.Timestamp('2025-09-24')]
        print('  post-src3 (src4) trades', len(post), 'imp mean%', round(post.imp.mean() * 100, 3) if len(post) else None)

# ---------- 2. 淨值配對差：零填補、逐年、剔除 2021 ----------
bb, ss = K.rates(0.001)
base = L.simulate_exec(pp, bb, ss, t0, t1)
nav = L.simulate_exec(pp, bb, ss, t0, t1, 'close', 'open', apply_days=nons3)
nav_all = L.simulate_exec(pp, bb, ss, t0, t1, 'close', 'open')
for nm, nv in (('ex_src3', nav), ('all_applied', nav_all)):
    pv, d = L.paired(nv, base, sd, L.VAL_FIRST, V.END1012)
    print(f'\n[VAL paired {nm}]', {k: (round(v, 3) if isinstance(v, float) else v) for k, v in pv.items()})
    if nm == 'ex_src3':
        nz = d[d.abs() > 1e-9]
        print('  zero-ish months', int((d.abs() <= 1e-9).sum()), '| non-zero only', tt(nz), '| |d|<1e-4 months', int((d.abs() < 1e-4).sum()))
        print('  by year mean%', (d.groupby(d.index.year).mean() * 100).round(3).to_dict())
        print('  by year sum%', (d.groupby(d.index.year).sum() * 100).round(2).to_dict())
        for yy in (2021,):
            print(f'  ex {yy}', tt(d[d.index.year != yy]))
        srt = d.sort_values(ascending=False)
        for k in (1, 3, 5):
            print(f'  ex top{k}', tt(d.drop(srt.index[:k])), [str(x.date()) for x in srt.index[:k]])
        print('  2019-05..2023-12 only', tt(d[d.index <= pd.Timestamp('2023-12-31')]))
        print('  2022-01..2026-10 only', tt(d[d.index >= pd.Timestamp('2022-01-01')]))
# 全期
pv, d = L.paired(nav, base, sd, pd.Timestamp('2006-04-01'), V.END1012)
print('\n[ALL paired ex_src3]', tt(d))
print('  ex 2021', tt(d[d.index.year != 2021]))
print('  2014-2020', tt(d[(d.index >= pd.Timestamp('2014-01-01')) & (d.index <= pd.Timestamp('2020-12-31'))]))
print('  2012-2020', tt(d[(d.index >= pd.Timestamp('2012-01-01')) & (d.index <= pd.Timestamp('2020-12-31'))]))
roll = d.rolling(36).mean() * 100
print('  rolling-36m mean% at year ends', roll.groupby(roll.index.year).last().round(3).to_dict())
# CAGR 貢獻
def cagr(nv, a, b):
    x = nv[(nv.index >= pd.Timestamp(a)) & (nv.index <= pd.Timestamp(b))]
    yrs = (x.index[-1] - x.index[0]).days / 365.25
    return (x.iloc[-1] / x.iloc[0]) ** (1 / yrs) - 1
print('  CAGR open/buyclose ALL', round(cagr(base, '2006-04-01', '2026-10-12') * 100, 2), round(cagr(nav, '2006-04-01', '2026-10-12') * 100, 2))
# 把 2021 年的配對差歸零後的 CAGR（以月配對差重組）
ra = L.sig_window_returns(base, sd); rb = L.sig_window_returns(nav, sd).reindex(ra.index)
dd = (rb - ra); dd0 = dd.copy(); dd0[dd0.index.year == 2021] = 0
alt = (1 + ra + dd0).dropna(); bas = (1 + ra).dropna(); full = (1 + rb).dropna()
yrs = (sd[-1] - sd[0]).days / 365.25
print('  monthly-chain CAGR base/buyclose/buyclose-with-2021-zeroed', round((bas.prod() ** (1 / yrs) - 1) * 100, 2), round((full.prod() ** (1 / yrs) - 1) * 100, 2), round((alt.prod() ** (1 / yrs) - 1) * 100, 2))
