# 用研究員的平均淨值（cap_avg_navs）獨立重算年化、MDD、對 0050 t（統計層複製，非模擬器）
import pandas as pd, numpy as np, json
B='/home/user/-/research/tw_stock_selection/agents/o_open_impact_calib/'
OUT=B+'replicate_o6/'
nav=pd.read_parquet(B+'data/cap_avg_navs.parquet')
e=pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/etf_0050.csv',parse_dates=['date']).set_index('date')['adj']
pl=pd.read_csv(B+'data/plans_ab.csv',parse_dates=['signal','exec'])
print('exec dates vary by seed?', pl.groupby('signal').exec.nunique().max(), pl.groupby('seed').size().unique()[:3])
sig=sorted(pl.signal.unique()); 
t0,t1=pd.Timestamp('2006-04-11'),pd.Timestamp('2026-10-12')
nav=nav.loc[t0:t1]; e=e.reindex(nav.index).ffill()
def ann(x): return (x.iloc[-1]/x.iloc[0])**(365.25/(x.index[-1]-x.index[0]).days)-1
def mdd(x): return (x/x.cummax()-1).min()
def tstat(v): v=np.asarray(v); return v.mean()/v.std(ddof=1)*np.sqrt(len(v))
bounds=[d for d in sig if t0<=d<=t1]+[t1]
bounds=sorted(set(bounds))
def wins(x):
    xs=x.reindex(bounds,method='ffill'); return xs.pct_change().dropna()
ew=wins(e)
rows={}
for c in nav.columns:
    s=nav[c]; ex=wins(s)-ew
    rows[c]=dict(ann=ann(s)*100,mdd=mdd(s)*100,t=tstat(ex),n=len(ex))
R=pd.DataFrame(rows).T
for k in ['AB|M3V_point','AB|M3V_p10','AB|M3V_p90','B10|M3V_point','B10|M3V_p90','B10|M3V_p10','AB|STRESS','B10|STRESS','AB|O5']:
    sub=R[R.index.str.startswith(k+'|')]
    print(k); print(sub.round(3).to_string())
R.to_csv(OUT+'r6_stats.csv')
