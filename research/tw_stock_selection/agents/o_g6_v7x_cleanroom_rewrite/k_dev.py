# K：開發切片比對，只回傳每個 j 的年化差（pp），不回傳 t 或水準
import pandas as pd, numpy as np, pickle, sys
ver = sys.argv[1]
SCR = '/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/g6A/'
a = pickle.load(open(SCR + f'dev_{ver}.pkl', 'rb'))
with open('/home/user/-/research/tw_stock_selection/agents/o_exec_engine_rewrite_v7x/out/navs_oos_R4.parquet', 'rb') as f: K = pd.read_parquet(f)
dates = a['dates']; e = a['end']
out = {}
for j, nav in a['navs'].items():
    st = int(np.where(np.isfinite(nav))[0][0])
    days = (dates[e] - dates[st]).days
    mine = (nav[e] / nav[st]) ** (365.25 / days) - 1
    k = K[f'B10|J{j}|K5T|normal|100萬|GROW'].reindex(dates)
    theirs = (k.iloc[e] / k.iloc[st]) ** (365.25 / days) - 1
    out[f'J{j}'] = round(100 * (mine - theirs), 3)
print('dev', ver, 'ann diff pp (mine - orig):', out)
open('/home/user/-/research/tw_stock_selection/agents/o_g6_v7x_cleanroom_rewrite/logs/k_dev_' + ver + '.txt', 'w').write(str(out))
