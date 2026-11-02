import sys; sys.path.insert(0,'/home/user/-/research/tw_stock_selection/agents/o_disposal_official_fullhist/replicate_o9')
import ev
from ev import *
d=load_events()
base=run(d); base=base[(base.s>='2006-04-01')&(base.s<='2021-09-30')]
# variant: no bear filter -> list events near bear boundary
nb=run(d,use_bear=False); nb=nb[(nb.s>='2006-04-01')&(nb.s<='2021-09-30')]
print('nobear',len(nb))
# variant bear using panel previous day mapped to index
def bear2(t):
    p=IDX[t-1]; q=tw.index[tw.index<=p]; return bool(bear.loc[q[-1]])
ev.bear_before=bear2
v=run(d); v=v[(v.s>='2006-04-01')&(v.s<='2021-09-30')]; print('bear2',len(v),summ(v))
# bear with today's close (lookahead check)
def bear3(t):
    q=tw.index[tw.index<=IDX[t]]; return bool(bear.loc[q[-1]])
ev.bear_before=bear3
v=run(d); v=v[(v.s>='2006-04-01')&(v.s<='2021-09-30')]; print('bear3',len(v),summ(v))
# chain on all records not adopt
d2=d.copy(); d2['adopt']=True
ev.bear_before=bear2
v=run(d2); v=v[(v.s>='2006-04-01')&(v.s<='2021-09-30')]; print('nochain',len(v),summ(v))
print(tw.loc['2006-03-25':'2006-04-05'])
