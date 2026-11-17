#!/usr/bin/env python3
"""S55 事後（不在 PREREG 判定內）：7% 漲跌停下「開盤即漲停」的 5 筆買進明細與損益。python3 -I -B posthoc.py → results_posthoc.json"""
import json
import pandas as pd, numpy as np
A='/home/user/-/research/tw_stock_selection/agents/s55_pre2015_oos/'
t=pd.read_csv(A+'oos_trades.csv.gz',dtype={'code':str})
z=np.load('/tmp/claude-0/-home-user/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s55/cache_pre/panel_cs.npz'); idx=pd.DatetimeIndex(z['IDX']); cols=list(z['COLS'])
nav=pd.read_csv(A+'oos_nav.csv.gz',index_col=0,parse_dates=True)
t0=list(idx).index(nav.index[0])
O=z['O'];C=z['C'];CHG=z['CHG']
def tick(p): return np.select([p<10,p<50,p<100,p<500,p<1000],[0.01,0.05,0.1,0.5,1.0],5.0)
rows=[]
for r in t.itertuples():
    tt=int(r.ent_t); j=cols.index(r.code)
    ref=C[tt-1,j] if not np.isfinite(CHG[tt,j]) else C[tt,j]-CHG[tt,j]
    u=np.floor(ref*1.07/tick(ref*1.07)+1e-9)*tick(ref*1.07)
    if np.isfinite(O[tt,j]) and O[tt,j]>=u-1e-9: rows.append((r.code,str(idx[tt].date()),O[tt,j],ref,r.pnl,r.proceeds/r.cost-1))
print(pd.DataFrame(rows,columns=['code','date','open','ref','pnl','ret']).to_string()); print('sum pnl',sum(x[4] for x in rows))
json.dump(dict(note='事後；這 5 筆實際上多半買不到（7% 漲停）；剔除它們不在引擎內重跑，只列損益', trades=[dict(code=a,date=b,open=float(c),ref=float(d),pnl=float(e),ret=float(f)) for a,b,c,d,e,f in rows], sum_pnl=float(sum(x[4] for x in rows))), open(A+'results_posthoc.json','w'), ensure_ascii=False, indent=1)
