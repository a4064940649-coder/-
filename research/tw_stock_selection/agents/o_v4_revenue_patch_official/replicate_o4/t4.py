import sys, pickle; sys.path.insert(0, '/home/user/-/research/tw_stock_selection/agents/o_v4_revenue_patch_official/replicate_o4')
import rcore as K, pandas as pd, numpy as np
D = pickle.load(open('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/rep_o4_main.pkl','rb'))
o = K.summary(D['nav'], D['pl'], D['ew'], full=True)
for k, v in o.items(): print(k, v)
best = o['best'][0]; p=[p for p in D['pl'] if K.IDX[p['i_d']]==best][0]; print([K.COLS[j] for j in p['picks']])
