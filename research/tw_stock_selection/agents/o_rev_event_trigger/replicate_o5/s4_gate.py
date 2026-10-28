import sys; sys.path.insert(0, "/home/user/-/research/tw_stock_selection/agents/o_rev_event_trigger/replicate_o5")
import re, json
import rcore as C
import pandas as pd, numpy as np
nw = pd.read_parquet(C.AG/'o_v4_revenue_patch_official'/'raw_news_mentions.parquet')
nw['dt'] = pd.to_datetime(nw['date'])
nw['today'] = nw['snip'].str.extract(r'今\s*[\(（]\s*(\d{1,2})\s*[\)）]\s*日')[0].astype(float)
rows = []
for d in C.OFF_DAYS:
    m = pd.Period(d, 'M')
    if not (2011 <= m.year <= 2018): continue
    cs = C.OFF.cross_section(d)
    pool = cs['top'] & ~C.L.FIN2MASK & ~C.L.FORMASK
    for j in np.where(pool)[0]: rows.append((str(m), C.COLS[j]))
P = pd.DataFrame(rows, columns=['m', 'code'])
print('pool stock-months', len(P))
P['ym'] = [str(pd.Period(x, 'M') - 1) for x in P.m]
D = {m: C.deadline(pd.Period(m, 'M')) for m in P.m.unique()}
D2 = {m: C.TX[C.TX.get_loc(D[m]) + 2] for m in D}
g = nw.groupby(['code', 'ym'])
exact = loose = 0; strict_ts = []
idx = {k: v for k, v in g}
for m, c, ym in zip(P.m, P.code, P.ym):
    x = idx.get((c, ym))
    if x is None: continue
    ex = x[x['today'].notna() & (x['today'] == x['dt'].dt.day)]
    if len(ex): exact += 1
    if (x['dt'] <= D2[m]).any(): loose += 1
print('exact', exact, exact / len(P), 'loose', loose, loose / len(P))
# 來源缺口上界：該月份（錨定月 m）有任何新聞（營收月 m−1 的提及）
ymhas = set(nw['ym'])
ub = P['ym'].isin(ymhas).mean(); print('upper bound (ym has any mention)', ub)
mdate = set(nw['dt'].dt.to_period('M').astype(str))
print('upper bound (month m has any news date)', P['m'].isin(mdate).mean())
e2 = 0; e3 = 0
for m, c, ym in zip(P.m, P.code, P.ym):
    x = idx.get((c, ym))
    if x is None: continue
    if x['today'].notna().any(): e2 += 1
    if x['snip'].str.contains('今').any(): e3 += 1
print('exact no daymatch', e2 / len(P), 'any 今', e3 / len(P))
# upper bound: months where loose coverage > 0
P['lo'] = [ (idx.get((c, ym)) is not None) for c, ym in zip(P.code, P.ym)]
mm = P.groupby('m')['lo'].any(); print('months with any mention among pool', P['m'].map(mm).mean())
