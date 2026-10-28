import sys, pickle; sys.path.insert(0, '/home/user/-/research/tw_stock_selection/agents/o_v4_revenue_patch_official/replicate_o4')
import rcore as K, pandas as pd, numpy as np
SP = '/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/'
R = pickle.load(open(SP + 'rep_o4_batch.pkl', 'rb'))
pl = R['B10'][0][0]
prev = set()
for p in pl:
    e = p['i_e']
    for j in prev:
        tr = np.isfinite(K.On[e, j]) and np.isfinite(K.Cn[e, j])
        if not tr:
            print(K.IDX[e].date(), K.COLS[j], 'last', K.IDX[K.LASTT[j]].date(), 'act', K.x_action(j, e), 'gapend', K.GAP.get(K.COLS[j]))
    prev = set(p['picks'])
from collections import Counter
c = Counter(K.COLS[j] for p in pl for j in p['picks'] if j in p['gap'])
print(sum(c.values()), c.most_common(8))
c = Counter(K.COLS[j] for p in R['BQ'][0][0] for j in p['picks'] if K.COLS[j] in K.GAP and not np.isfinite(K.y3(K.YF if K.IDX[p['i_d']]<K.SWITCH else K.Y5, pd.Period(K.IDX[p['i_d']],'M'))[j]))
print('BQ', sum(c.values()), c)
c = Counter(K.COLS[j] for p in R['B05'][0][0] for j in p['picks'] if j in p['gap']); print('B05', sum(c.values()))
