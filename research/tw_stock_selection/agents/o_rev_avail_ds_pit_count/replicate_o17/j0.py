import sys
sys.path.insert(0, '/home/user/-/research/tw_stock_selection/agents/main_v2')
import pandas as pd, numpy as np
import v2_core as V
sys.path.insert(0, '/home/user/-/research/tw_stock_selection/agents/o_rev_avail_ds_pit_count/replicate_o17')
import r1
P = pd.read_csv('/home/user/-/research/tw_stock_selection/agents/o_v7_listing_exec_engine/data/plans_j7.csv', dtype=str)
P = P[(P.strat == 'AB_V7') & (P.seed == '0')].copy()
P['n'] = P.picks.fillna('').map(lambda s: len([x for x in s.split(',') if x]))
P['j'] = P.j.astype(int); P['anchor'] = P.signal.str[:7]
# J0 = 期限日？
j0 = P[P.j == 0]
dl = [V.TCAL[V.TCAL >= pd.Timestamp(int(s[:4]), int(s[5:7]), 10)][0].date().isoformat() for s in j0.signal]
print('J0 months', len(j0), 'J0==first trading day>=10th', np.mean(np.array(dl) == j0.signal.values), [ (a,b) for a,b in zip(j0.signal,dl) if a!=b][:5])
for nm, m in (('ALL', P.anchor >= '0'), ('2019_', P.anchor >= '2019'), ('2006_2013', P.anchor < '2014'), ('2019_2024-05', (P.anchor >= '2019') & (P.anchor < '2024-06')), ('2024-06_', P.anchor >= '2024-06')):
    q = P[m]; a = q[q.j == 0].n.sum(); b = q.n.sum()
    print(nm, a, b, a / b, q[q.j == 0].anchor.nunique())
q = j0[j0.signal >= '2019'].copy()
rows = [(s, c) for s, p in zip(q.signal, q.picks.fillna('')) for c in p.split(',') if c]
st = [r1.classify(c, str(pd.Period(s, 'M') - 1), s)[0] for s, c in rows]
vc = pd.Series(st).value_counts(); print('J0 2019+ slots', len(rows), vc.to_dict())
allj = P[P.anchor >= '2019'].n.sum()
print('upper share of all j', (len(rows) - vc.get('AVAIL', 0)) / allj, 'lower', vc.get('UNAVAIL', 0) / allj)
R = pd.DataFrame(rows, columns=['sig', 'code']); R['st'] = st
R['src'] = [';'.join(r1.classify(c, str(pd.Period(s, 'M') - 1), s)[1]) for s, c in rows]
print(R[R.st == 'UNDET'].sig.value_counts().to_dict())
print(pd.crosstab(R.sig.str[:4], R.st))
print(R[R.sig.isin(['2026-10-12', '2024-06-11'])].to_string())
