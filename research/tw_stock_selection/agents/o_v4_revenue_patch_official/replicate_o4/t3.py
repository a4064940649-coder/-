import sys, pickle; sys.path.insert(0, '/home/user/-/research/tw_stock_selection/agents/o_v4_revenue_patch_official/replicate_o4')
import rcore as K, pandas as pd, numpy as np
D = pickle.load(open('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/rep_o4_main.pkl','rb'))
e = pd.Series(K.E50, index=K.IDX)
a = pd.Timestamp('2006-04-11')
for name, s in [('v4', D['nav']), ('ew', D['ew']), ('0050', e)]:
    R = s[K.END]/s[a]; y=(K.END-a).days/365.25
    print(name, R**(1/y)-1, R**(1/((K.END-a).days/365))-1)
