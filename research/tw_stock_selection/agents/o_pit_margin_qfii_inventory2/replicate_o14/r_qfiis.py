import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json
import numpy as np, pandas as pd
from r_lib import *
import r_src as S

D, first, last, calall = denominators()
idx, cols = D.index, D.columns
Dv = D.values
delisted = (last.reindex(cols) < pd.Timestamp('2026-10-01')).values
W = value_panel(calall)
pool = pool300(D, W).values
q = S.snaps('qfii-twse')
v = S.vf()
out = {}
Q1 = S.wide(q[q.foreign_pct.notna()], 'foreign_pct').reindex(index=idx, columns=cols)
Q2 = S.wide(v[v.foreign_ratio.notna()], 'foreign_ratio').reindex(index=idx, columns=cols)
have = Q1.notna().values | Q2.notna().values
daily = {}
for s in SEG:
    sm = seg_mask(idx, s)[:, None]
    daily[s] = {nm: round(100 * cov(have, den & sm)[0], 2) for nm, den in
                [('all', Dv), ('pool', pool), ('delisted', Dv & delisted[None, :])]}
out['daily'] = daily
print('daily', daily)

# monthly formation-day variant: 11th or first trading day after
months = pd.period_range('2006-01', '2026-10', freq='M')
rows = []
for m in months:
    d0 = pd.Timestamp(m.year, m.month, 11)
    i = idx.searchsorted(d0)
    if i >= len(idx):
        continue
    fd = idx[i]
    den = Dv[i]
    h = have[i]
    rows.append(dict(month=str(m), fd=fd, n=int(den.sum()), scov=int((h & den).sum()),
                     pool_n=int(pool[i].sum()), pool_cov=int((h & pool[i]).sum()),
                     del_n=int((den & delisted).sum()), del_cov=int((h & den & delisted).sum())))
mf = pd.DataFrame(rows)
mf['ok'] = mf.scov / mf.n
snapdates = set(q.date.unique())
mf['has_snap'] = mf.fd.isin(snapdates)
mon = {}
for s in SEG:
    a, b = SEG[s]
    x = mf[(mf.month >= a[:7]) & (mf.month <= b[:7])]
    mon[s] = dict(months=len(x), months_cov=int((x.scov > 0).sum()), months_ge95=int((x.ok >= .95).sum()),
                  stock_month_cov=round(100 * x.scov.sum() / x.n.sum(), 2),
                  pool=round(100 * x.pool_cov.sum() / max(x.pool_n.sum(), 1), 2),
                  delisted=round(100 * x.del_cov.sum() / max(x.del_n.sum(), 1), 2))
x = mf[mf.month >= '2014-01']
mon['2014-01~2026-10'] = dict(months=len(x), all=round(100 * x.scov.sum() / x.n.sum(), 3),
                             pool=round(100 * x.pool_cov.sum() / x.pool_n.sum(), 3),
                             delisted=round(100 * x.del_cov.sum() / x.del_n.sum(), 3),
                             fd_eq_snapdate=int(x.has_snap.sum()))
out['monthly'] = mon
print('monthly', mon)

# voidful vs snapshot overlap
m = q.merge(v, on=['date', 'code'])
out['vf_vs_snap'] = dict(n=len(m), ratio=round(100 * (m.foreign_pct == m.foreign_ratio).mean(), 3),
                         issued=round(100 * (m.issued == m.total_shares).mean(), 3),
                         held=round(100 * (m.foreign_held == m.foreign_shares).mean(), 3))
mc = m[m.code.str.match(COMMON)]
out['vf_vs_snap_common'] = dict(n=len(mc), ratio=round(100 * (mc.foreign_pct == mc.foreign_ratio).mean(), 3))
# range checks (snapshot + voidful)
rr = {}
for nm, d, r, h_, i_ in [('snap', q, 'foreign_pct', 'foreign_held', 'issued'), ('vf', v, 'foreign_ratio', 'foreign_shares', 'total_shares')]:
    rr[nm] = dict(n=len(d), ratio_in_range=round(100 * d[r].between(0, 100).mean(), 4),
                  held_le_issued=round(100 * (d[h_] <= d[i_]).mean(), 4))
out['range'] = rr
print(json.dumps({k: out[k] for k in ('vf_vs_snap', 'vf_vs_snap_common', 'range')}, ensure_ascii=False))

# shares vs TEJ monthly anchor (from 2014)
tm = pd.read_csv(TEJM, dtype={'code': str})
tm['me'] = pd.to_datetime(tm.month_end)
tm = tm[tm.code.str.match(COMMON)]
qs = q[q.code.str.match(COMMON)].copy()
res_sh = {}
for mode in ('prev_month_end', 'same_month_end'):
    qq = qs.copy()
    if mode == 'prev_month_end':
        qq['me'] = (qq.date.dt.to_period('M') - 1).dt.to_timestamp(how='end').dt.normalize()
    else:
        qq['me'] = qq.date.dt.to_period('M').dt.to_timestamp(how='end').dt.normalize()
    tm2 = tm.copy(); tm2['me'] = tm2.me.dt.normalize()
    mm = qq.merge(tm2[['code', 'me', 'shares_k', 'market']], on=['code', 'me'])
    mm = mm[(mm.issued > 0) & (mm.shares_k > 0)]
    ld = (np.log(mm.issued) - np.log(mm.shares_k * 1000)).abs()
    res_sh[mode] = dict(n=len(mm), median_pct=round(100 * ld.median(), 5), p90_pct=round(100 * ld.quantile(.9), 4),
                        within_0p1=round(100 * (ld <= 0.001).mean(), 2))
    if mode == 'prev_month_end':
        anchored = set(mm.code)
# delisted (in 2014+ denominator) with no TEJ anchor at all
d14 = D.loc['2014-01-02':]
codes14 = d14.columns[d14.any()]
dl14 = [c for c in codes14 if last[c] < pd.Timestamp('2026-10-01')]
tejcodes = set(tm.code)
res_sh['delisted_2014_n'] = len(dl14)
res_sh['delisted_no_tej_any'] = sum(c not in tejcodes for c in dl14)
res_sh['delisted_no_pair'] = sum(c not in anchored for c in dl14)
dl_all = [c for c in cols if last[c] < pd.Timestamp('2026-10-01')]
res_sh['delisted_all_no_tej_any'] = sum(c not in tejcodes for c in dl_all)
out['shares'] = res_sh
print('shares', res_sh)
json.dump(out, open(f'{HERE}/my_qfiis.json', 'w'), ensure_ascii=False, indent=1, default=str)
