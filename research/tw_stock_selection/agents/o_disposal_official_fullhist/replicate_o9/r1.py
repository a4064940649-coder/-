import sys; sys.path.insert(0,'/home/user/-/research/tw_stock_selection/agents/o_disposal_official_fullhist/replicate_o9')
from ev import *
for sm,ci in [('cal',True),('cal',False)]:
    d=load_events(sm,ci); tr=run(d)
    o=tr[(tr.s>='2006-04-01')&(tr.s<='2021-09-30')]
    print(sm,ci,len(d),d.valid.sum(),d.adopt.sum(),summ(o),summ(o,'gross'),summ(o,'n50'))
