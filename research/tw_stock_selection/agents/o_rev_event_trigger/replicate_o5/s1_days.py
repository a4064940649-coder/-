import sys; sys.path.insert(0, "/home/user/-/research/tw_stock_selection/agents/o_rev_event_trigger/replicate_o5")
import time; T=time.time()
import rcore as C
import pandas as pd, numpy as np
print('load', time.time()-T, len(C.OFF_DAYS), C.OFF.end_day)
# official vs J0/J+1 mapping
cnt={'J+1':0,'J0':0,'other':[]}
for d in C.OFF_DAYS+[C.OFF.end_day]:
    m=pd.Period(d,'M')
    if d==C.j_day(m,1): cnt['J+1']+=1
    elif d==C.j_day(m,0): cnt['J0']+=1
    else: cnt['other'].append((str(m),str(d.date()),str(C.deadline(m).date())))
print(cnt)
for j in (0,1):
    sd,end,am,sk=C.build_days(j); print(j,len(sd),end,sk)
# union calendar vs taiex
