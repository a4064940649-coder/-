import sys; sys.path.insert(0, '/home/user/-/research/tw_stock_selection/agents/o_largecap_leg/replicate_o3')
import json, pickle, time
import pandas as pd, numpy as np
from rlib import *
t0=time.time()
D = Data(); print('load', time.time()-t0)
pickle.dump(D, open('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/rep_o3_data.pkl','wb'))
b = bench0050(D.nidx)
for s in SEG: print(s, round(seg_nav_cagr(b, D.sigs, s)*100,2), len(seg_mask(mret(b,D.sigs),s)))
