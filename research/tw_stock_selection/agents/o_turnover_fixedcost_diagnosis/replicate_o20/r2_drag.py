import sys; sys.path.insert(0,"/home/user/-/research/tw_stock_selection/agents/o_turnover_fixedcost_diagnosis/replicate_o20")
from lib import *
cols=['cost','tax','fee','fee_min_ex','tick','imp','odd','cost_h05','cost_h10']
for tier in ['100n','30n']:
    t=load_tl(tier)
    for per in ['Y19on','POSTODD','FULL','S0618']:
        d=drag(t,per,cols); print(tier,per,d[cols].median().round(3).to_dict())
