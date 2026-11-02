import sys; sys.path.insert(0,'/home/user/-/research/tw_stock_selection/agents/o_disposal_official_fullhist/replicate_o9')
import ev
from ev import *
d=load_events()
base=run(d)
d2=d.copy(); d2['adopt']=True
alt=run(d2)
k=lambda df:set(zip(df.code,df.s))
o=lambda df:df[(df.s>='2006-04-01')&(df.s<='2021-09-30')]
extra=o(alt)[[x not in k(base) for x in zip(o(alt).code,o(alt).s)]]
for r in extra.itertuples():
    g=d[d.code==r.code].sort_values('start')
    g=g[(g.start>=r.start-pd.Timedelta(days=40))&(g.start<=r.start+pd.Timedelta(days=5))]
    print(r.code,r.s.date(),round(r.net*100,2)); print(g[['announce','start','end','si','ei','valid','adopt','kind','nth','second']].to_string())
