# K（保管員）：只列密封目標檔的結構（欄名、列數），不輸出數值
import pandas as pd, sys
p = '/home/user/-/research/tw_stock_selection/agents/o_exec_engine_rewrite_v7x/out/navs_oos_R4.parquet'
with open(p, 'rb') as f: x = pd.read_parquet(f)
print(x.shape, list(x.columns[:12]), '...', len(x.columns), type(x.index), str(x.index[0])[:10], str(x.index[-1])[:10])
