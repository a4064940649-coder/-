import sys, pickle; sys.path.insert(0, '.')
import pandas as pd
import rcore5 as R
S = pickle.load(open('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/rep_o5/s1.pkl','rb'))
nav = S['NAV']['AB_V5']
for a in R.IDX[(R.IDX>='2021-02-20')&(R.IDX<='2021-03-20')]:
    o=[]
    for s in [nav, R.E50]:
        x = s[(s.index>=a)&(s.index<=R.END)]
        o.append((x.iloc[-1]/x.iloc[0])**(365.25/(x.index[-1]-x.index[0]).days)-1)
    print(a.date(), '%.4f %.4f'%tuple(o))
