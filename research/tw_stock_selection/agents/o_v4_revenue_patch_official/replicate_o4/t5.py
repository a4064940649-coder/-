import sys, pickle; sys.path.insert(0, '/home/user/-/research/tw_stock_selection/agents/o_v4_revenue_patch_official/replicate_o4')
import rcore as K, pandas as pd, numpy as np
D = pickle.load(open('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/rep_o4_main.pkl','rb'))
for k,(a,b) in K.SEGS.items():
    for nm, s in [('v4', D['nav']), ('50', K.E50S)]:
        x = s[(s.index>=a)&(s.index<=b)]; p = s[s.index<a].index[-1]
        r1 = (x.iloc[-1]/s[p])**(365.25/(x.index[-1]-p).days)-1
        r2 = (x.iloc[-1]/x.iloc[0])**(365.25/(x.index[-1]-x.index[0]).days)-1
        print(k, nm, round(r1*100,2), round(r2*100,2), end=' | ')
    print()
