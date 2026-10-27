import sys; sys.path.insert(0,'/home/user/-/research/tw_stock_selection/agents/w1_granville_pullback_ma60')
import run as R, numpy as np, pandas as pd
m=(R.IDX>='2005-01-01')&(R.IDX<='2019-04-30')
yrs=14.3
print('EW oo CAGR', np.exp(np.log1p(R.ew_oo[m]).sum()/yrs)-1, 'EW cc', np.exp(np.log1p(R.ew_cc[m]).sum()/yrs)-1)
df=pd.read_csv(R.L.P.parent/'w1_granville_primary_events.csv.gz',parse_dates=['sig','entry'],dtype={'code':str})
row=df.sample(1,random_state=3).iloc[0]; print(row.to_dict())
j=R.cols.index(row['code']); t=R.IDX.get_loc(row['sig'])
MA=R.ADJ[row['code']].rolling(60).mean()
a=R.ADJ[row['code']]; v=R.VOL[row['code']]
print('MA_t',MA.iloc[t],'MA_t-20',MA.iloc[t-20],'max20/MA',a.iloc[t-19:t+1].max()/MA.iloc[t],'dist',a.iloc[t]/MA.iloc[t]-1,
 'min5', (a.iloc[t-4:t+1]/MA.iloc[t-4:t+1]).min(),'vol ratio', v.iloc[t]/v.iloc[t-19:t+1].mean(),'up',a.iloc[t]>a.iloc[t-1],'V60',R.V60[row['code']].iloc[t],'close',R.CR[row['code']].iloc[t])
e=t+1; print('entry open', R.AOn[e,j], 'exit', R.PX[e+20,j], R.IDX[e+20], 'raw', R.PX[e+20,j]/R.AOn[e,j]-1)
print(df['raw'].describe()); print(df.nlargest(3,'raw')[['code','sig','raw']])
# gross by hold path: how much is from entry overnight vs rest
# 診斷：t 收盤進場（不可成交）vs t+1 開盤
import math
out=[]
for _,r in df.iterrows():
    j=R.cols.index(r['code']); t=R.IDX.get_loc(r['sig']); e=t+1; x=e+r['hold']
    gap=R.AOn[e,j]/R.ACn[t,j]-1
    out.append((r['sig'], gap, R.ew_oo[t-0] if False else 0))
g=pd.DataFrame(out,columns=['sig','gap','z'])
for a,b in [('2005-01-01','2019-04-30'),('2019-05-01','2026-10-31')]:
    s=g[(g.sig>=a)&(g.sig<=b)]; print(a, 'mean overnight gap to entry %.3f%%'%(s.gap.mean()*100))
