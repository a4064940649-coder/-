import numpy as np, pandas as pd, json
exec(open('r1_nav.py').read().split("for v in")[0])
v='REAL_opt'
for mode in ['lastpre','first19','first19_open']:
  rows=[]
  for j in range(13):
    s=nav[f'{v}|{j}'].dropna(); end=pd.Timestamp(jend[f'V7_B10|J{j}']); s=s[s.index<=end]
    if mode=='lastpre': pre=s[s.index<'2019-01-01'].index[-1]
    else: pre=s[s.index>='2019-01-01'].index[0]
    s19=s[s.index>=pre]
    for ym in ['days','tdays252']:
      yrs=(s19.index[-1]-s19.index[0]).days/365.25 if ym=='days' else (len(s19)-1)/252
      c=(s19.iloc[-1]/s19.iloc[0])**(1/yrs)-1; e=e50.reindex(s19.index); ce=(e.iloc[-1]/e.iloc[0])**(1/yrs)-1
      rows.append((ym,j,c*100,ce*100,(c-ce)*100))
  d=pd.DataFrame(rows,columns=['ym','j','c','e','x'])
  print(mode, d.groupby('ym')[['c','e','x']].median().round(3).to_dict('index'))
# 0050 from 2019-01-01 to 2026-10-22 days
e=e50.dropna(); 
for a in ['2018-12-28','2019-01-02']:
  for b in ['2026-10-12','2026-10-22']:
    x=e[(e.index>=a)&(e.index<=b)]; yrs=(x.index[-1]-x.index[0]).days/365.25; print(a,b,((x.iloc[-1]/x.iloc[0])**(1/yrs)-1)*100)
# monthly t variants: excess of daily returns, grouped by month
for v in ['REAL_opt','REAL_p90','REAL_carry']:
  ts=[]
  for j in range(13):
    s=nav[f'{v}|{j}'].dropna(); end=pd.Timestamp(jend[f'V7_B10|J{j}']); s=s[s.index<=end]
    r=s.pct_change(); re=e50.reindex(s.index).pct_change(); ex=(r-re).dropna()
    m=ex.groupby(ex.index.to_period('M')).sum(); ts.append(m.mean()/m.std()*np.sqrt(len(m)))
    lm=(np.log1p(r)-np.log1p(re)).dropna().groupby(ex.index.to_period('M')).sum()
  print(v,'t daily-sum monthly median',np.median(ts))
