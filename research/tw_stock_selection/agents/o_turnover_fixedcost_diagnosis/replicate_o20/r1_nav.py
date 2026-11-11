import numpy as np, pandas as pd, json
B='/home/user/-/research/tw_stock_selection/agents/'
z=np.load(B+'o_exec_engine_rewrite_v7x/data/x_cache.npz',allow_pickle=True)
idx=pd.DatetimeIndex(z['IDX']); e50=pd.Series(z['E50'],idx)
nav=pd.read_parquet(B+'o_turnover_fixedcost_diagnosis/data/navs_100n_avg20.parquet')
nav.index=pd.to_datetime(nav.index)
jend=json.load(open(B+'o_v7_listing_exec_engine/data/jscan_end_days.json'))
def cagr(s,a,b):
    s=s.dropna(); s=s[(s.index>=a)&(s.index<=b)]
    return (s.iloc[-1]/s.iloc[0])**(365.25/(s.index[-1]-s.index[0]).days)-1, s.index[0], s.index[-1]
for v in ['REAL_opt','REAL_p90','REAL_carry']:
  rows=[]
  for j in range(13):
    s=nav[f'{v}|{j}'].dropna(); end=pd.Timestamp(jend[f'V7_B10|J{j}'])
    s=s[s.index<=end]
    # y19 base: last date before 2019
    for mode in ['lastpre','first19']:
      pass
    pre=s[s.index<'2019-01-01'].index[-1]
    s19=s[s.index>=pre]
    yrs=(s19.index[-1]-s19.index[0]).days/365.25
    c=(s19.iloc[-1]/s19.iloc[0])**(1/yrs)-1
    e=e50.reindex(s19.index); ce=(e.iloc[-1]/e.iloc[0])**(1/yrs)-1
    yrsF=(s.index[-1]-s.index[0]).days/365.25
    cf=(s.iloc[-1]/s.iloc[0])**(1/yrsF)-1
    ef=e50.reindex(s.index); cef=(ef.iloc[-1]/ef.iloc[0])**(1/yrsF)-1
    mdd=(s/s.cummax()-1).min()
    # monthly excess t
    m=s.resample('ME').last(); me=ef.resample('ME').last()
    ex=(m.pct_change()-me.pct_change()).dropna()
    t=ex.mean()/ex.std()*np.sqrt(len(ex))
    rows.append(dict(j=j,c19=c*100,e19=ce*100,x19=(c-ce)*100,cf=cf*100,ef=cef*100,xf=(cf-cef)*100,mdd=mdd,t=t,start=s.index[0].date()))
  d=pd.DataFrame(rows); print(v); print(d.round(3).to_string()); print('median',d.median(numeric_only=True).round(3).to_dict()); print('p10 xf',np.percentile(d.xf,10), 'p10 x19',np.percentile(d.x19,10))
