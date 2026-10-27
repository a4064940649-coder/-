import sys; sys.path.insert(0, "/home/user/-/research/tw_stock_selection/agents/o_disposal_rejudge/replicate_o3")
import numpy as np, pandas as pd
import r_core as R
L = R.prep(pd.read_csv(R.HERE / 'my_list_AP.csv', dtype={'code': str}, parse_dates=['ann', 'start', 'end']))
nav = {v: R.simulate(R.plan(v, 10, L=L)) for v in ('E0', 'E1', 'E5')}
for v in ('E0', 'E5'): print('AP', v, R.paired(nav[v], nav['E1'], 'sel'))
print('AP E1 all', R.cagr(nav['E1'], 'all'))
