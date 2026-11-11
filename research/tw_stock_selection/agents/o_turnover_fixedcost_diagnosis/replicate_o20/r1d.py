import numpy as np, pandas as pd, json
exec(open('r1_nav.py').read().split("for v in")[0])
def tt(x): x=x.dropna(); return x.mean()/x.std(ddof=1)*np.sqrt(len(x))
for v in ['REAL_opt','REAL_p90','REAL_carry']:
  A=[];B=[];L=[]
  for j in range(13):
    s=nav[f'{v}|{j}'].dropna(); end=pd.Timestamp(jend[f'V7_B10|J{j}']); s=s[s.index<=end]; e=e50.reindex(s.index)
    m=s.resample('ME').last(); me=e.resample('ME').last()
    A.append((m.pct_change()-me.pct_change()).rename(j))
    r=s.pct_change(); re=e.pct_change(); L.append((r-re).rename(j))
  A=pd.concat(A,axis=1); L=pd.concat(L,axis=1)
  # t via 21-day blocks? daily t with month cluster
  d=L.mean(axis=1).dropna()
  print(v,'meanj monthly',round(tt(A.mean(axis=1)),3),'medj',round(tt(A.median(axis=1)),3),'daily t',round(tt(d),3),
   'daily per j median',round(np.median([tt(L[j]) for j in range(13)]),3))
  # avg nav
  navs=pd.concat([nav[f'{v}|{j}'] for j in range(13)],axis=1)
