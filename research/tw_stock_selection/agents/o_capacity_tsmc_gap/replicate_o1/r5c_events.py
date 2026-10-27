import sys, json, pickle
import numpy as np, pandas as pd
sys.path.insert(0, '/home/user/-/research/tw_stock_selection/agents/o_capacity_tsmc_gap/replicate_o1')
import rcore as R
from r2_tsmc import crosses
from r5_attr import stock_win, summarize_month, Sl, PK, buy, sell, ewnav, ew_win, period_mask
for n in (10,30):
  picks=PK[(n,0.0)]
  rows=[]
  for k,d in enumerate(Sl):
    cr=crosses[d]; pk=picks[d]; prev=set(picks[Sl[k-1]]) if k>0 else set(); nx=set(picks[Sl[k+1]]) if k+1<len(Sl) else set()
    W={m:stock_win(d,m) for m in ('oo','cc','oc')}
    rk=cr.vrank
    for j in pk:
        b = buy if j not in prev else 0; s = sell if j not in nx else 0
        net=(1+W['oo'][j])*(1-s)/(1+b)-1
        g = 'A' if rk[j]>100 else 'B'
        g4 = '1-50' if rk[j]<=50 else ('51-100' if rk[j]<=100 else ('101-200' if rk[j]<=200 else '201-300'))
        lo,hi={'1-50':(1,50),'51-100':(51,100),'101-200':(101,200),'201-300':(201,300)}[g4]
        gm = cr.el&(rk>=lo)&(rk<=hi)
        gm2 = cr.el&((rk>100) if g=='A' else (rk<=100))
        row=dict(d=d,j=j,g=g,g4=g4,net=net,gross=W['oo'][j])
        for m in ('oo','cc','oc'):
            row['pool_'+m]=np.nanmean(W[m][cr.el]); row['g4_'+m]=np.nanmean(W[m][gm]); row['g2_'+m]=np.nanmean(W[m][gm2])
        row['pool_nav']=ew_win(ewnav,d)
        rows.append(row)
  E=pd.DataFrame(rows); E['d']=pd.to_datetime(E['d'])
  E.to_pickle(R.HERE/f'events_n{n}.pkl')
  for m in ('oo','cc','oc'):
    E['xg']=E['net']-E['g2_'+m]
    M=E.groupby(['d','g'])['xg'].mean().unstack()
    x=(M['A']-M['B']).dropna()
    print(n, 'within-group EW', m, {p:(round(x[period_mask(x.index,p)].mean()*100,3), round(R.tstat(x[period_mask(x.index,p)]),3)) for p in ('SEL','VAL','ALL')})
  V=E[period_mask(E['d'],'VAL')&(E['g4']=='1-50')]
  for m in ('oo','cc','oc'):
    print(n,'VAL 1-50 event-weighted: picks-pool', m, round((V['net']-V['pool_'+m]).mean()*100,3), 'gross', round((V['gross']-V['pool_'+m]).mean()*100,3), 'grpEW-pool', round((V['g4_'+m]-V['pool_'+m]).mean()*100,3))
