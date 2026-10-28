import sys, pickle; sys.path.insert(0, '/home/user/-/research/tw_stock_selection/agents/o_v4_revenue_patch_official/replicate_o4')
import rcore as K, pandas as pd, numpy as np
D = pickle.load(open('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/rep_o4_main.pkl','rb'))
e = pd.Series(K.E50, index=K.IDX)
for name, s in [('v4', D['nav']), ('ew', D['ew']), ('0050', e)]:
    x = s[(s.index >= K.A0) & (s.index <= K.END)]
    i0 = s.index.get_loc(x.index[0]); base = s.iloc[i0-1]; R = x.iloc[-1]/base
    nd = len(x); cd = (x.index[-1]-s.index[i0-1]).days; cd2=(x.index[-1]-x.index[0]).days
    print(name, R, [R**(1/y)-1 for y in (cd/365.25, cd/365, nd/252, cd2/365.25, (nd-1)/252)])
