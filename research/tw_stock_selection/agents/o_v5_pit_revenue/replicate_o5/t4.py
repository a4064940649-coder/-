import sys; sys.path.insert(0, '.')
import numpy as np
import rcore5 as R
sys.path.insert(0, '/home/user/-/research/tw_stock_selection/agents/main_v2'); import v2_core as V2; sys.path.pop(0)
d = np.where(V2.PCOL != R.PCOL)[0]
print(len(d), [(R.COLS[j], V2.PCOL[j], R.PCOL[j], V2.IND[j], R._IND[j]) for j in d[:10]])
print(len(V2.FIN), len(R.FIN), list(V2.PDL['p_by_industry'].items())[:5])
