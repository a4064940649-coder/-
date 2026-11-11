import numpy as np, pandas as pd, json
exec(open('r1_nav.py').read().split("for v in")[0])
def tt(x): x=x.dropna(); return x.mean()/x.std(ddof=1)*np.sqrt(len(x))
out={}
for v in ['REAL_opt','REAL_p90','REAL_carry']:
  res={k:[] for k in ['A','B','C','D','E']}
  for j in range(13):
    s=nav[f'{v}|{j}'].dropna(); end=pd.Timestamp(jend[f'V7_B10|J{j}']); s=s[s.index<=end]
    e=e50.reindex(s.index)
    m=s.resample('ME').last(); me=e.resample('ME').last()
    m0=pd.concat([s.iloc[:1],m]); me0=pd.concat([e.iloc[:1],me])
    res['A'].append(tt(m.pct_change()-me.pct_change()))
    res['B'].append(tt(m0.pct_change()-me0.pct_change()))
    res['C'].append(tt(np.log(m0).diff()-np.log(me0).diff()))
    res['D'].append(tt(np.log(m).diff()-np.log(me).diff()))
    # daily compounding within month
    r=s.pct_change(); re=e.pct_change()
    res['E'].append(tt((r-re).groupby(r.index.to_period('M')).apply(lambda x:(1+x).prod()-1)))
  out[v]={k:round(np.median(x),3) for k,x in res.items()}
print(out)
