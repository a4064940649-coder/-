# 逐 seed t<2 個數、分段 t、拿掉最好 10 窗（以研究員淨值；統計層）
import pandas as pd, numpy as np
B='/home/user/-/research/tw_stock_selection/agents/o_open_impact_calib/'
n=pd.read_parquet(B+'data/cap_navs_main.parquet'); a=pd.read_parquet(B+'data/cap_avg_navs.parquet')
e=pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/etf_0050.csv',parse_dates=['date']).set_index('date')['adj']
pl=pd.read_csv(B+'data/plans_ab.csv',parse_dates=['signal'])
t0,t1=pd.Timestamp('2006-04-11'),pd.Timestamp('2026-10-12')
bounds=sorted(set([d for d in pl.signal.unique() if t0<=d<=t1]+[t1]))
def wins(x): x=x.dropna(); return x.reindex(bounds,method='ffill').pct_change().dropna()
ew=wins(e)
def ts(v): v=np.asarray(v); return v.mean()/v.std(ddof=1)*np.sqrt(len(v))
for tn in ['30萬','100萬','300萬','1000萬']:
  for ver in ['GROW','FIX']:
    cs=[c for c in n.columns if c.startswith(f'AB|M3V_point|{tn}|{ver}|')]
    tt=[ts(wins(n[c].loc[t0:t1])-ew) for c in cs]
    ex=wins(a[f'AB|M3V_point|{tn}|{ver}'].loc[t0:t1])-ew
    seg=lambda lo,hi: ts(ex[(ex.index>lo)&(ex.index<=hi)])
    drop10=ts(ex.sort_values().iloc[:-10])
    print(tn,ver,'seeds',len(cs),'t<2:',sum(np.array(tt)<2),'| drop10',round(drop10,2),'FROM2008',round(seg(pd.Timestamp('2008-01-01'),t1),2),
          'Y19_26',round(seg(pd.Timestamp('2019-01-01'),t1),2),'HOLD24',round(seg(pd.Timestamp('2024-01-01'),t1),2))
