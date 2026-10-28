import sys; sys.path.insert(0, "/home/user/-/research/tw_stock_selection/agents/o_rev_event_trigger/replicate_o5")
import rcore as C
import pandas as pd, numpy as np
nw = pd.read_parquet(C.AG/'o_v4_revenue_patch_official'/'raw_news_mentions.parquet')
nw['dt'] = pd.to_datetime(nw['date'])
nw['today'] = nw['snip'].str.extract(r'今\s*[\(（]\s*(\d{1,2})\s*[\)）]\s*日')[0].astype(float)
s = nw[nw['today'].notna() & (nw['today'] == nw['dt'].dt.day)].copy()
s['m'] = [pd.Period(y, 'M') + 1 for y in s['ym']]
s['D'] = [C.deadline(m) for m in s['m']]
s = s.drop_duplicates(['code', 'ym'])
s = s[s['dt'] <= s['D']]
print('n strict (<=D)', len(s), 'share at D', (s['dt'] == s['D']).mean())
s['k'] = [C.TX.get_loc(D) - C.TX.get_loc(C.TX[C.TX.searchsorted(dt)]) for D, dt in zip(s['D'], s['dt'])]
# y3 pool percentile at official signal day
rows = []
for m, g in s.groupby('m'):
    d = [x for x in C.OFF_DAYS if pd.Period(x, 'M') == m]
    if not d: continue
    cs = C.OFF.cross_section(d[0]); pool = cs['top'] & ~C.L.FIN2MASK & ~C.L.FORMASK & np.isfinite(cs['y3'])
    pr = pd.Series(np.where(pool, cs['y3'], np.nan), index=C.COLS).rank(pct=True)
    for c, k in zip(g['code'], g['k']):
        if c in pr.index and np.isfinite(pr[c]): rows.append((str(m), c, k, pr[c]))
X = pd.DataFrame(rows, columns=['m', 'code', 'k', 'pr'])
print('early>=5', X[X.k >= 5].pr.mean(), 'at D', X[X.k == 0].pr.mean(), len(X))
g = X.assign(e=X.k >= 5, z=X.k == 0).groupby('m').apply(lambda q: q[q.e].pr.mean() - q[q.z].pr.mean()).dropna()
print('monthly diff', g.mean(), C.tstat(g.values), len(g))
