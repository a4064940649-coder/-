#!/usr/bin/env python3
"""敏感度（固定區塊 6／12）的 12 個月分位精確值：固定區塊 12 的 12 個月＝129 個循環窗之一（離散）、固定區塊 6＝兩塊組合，可窮舉。用來判斷原研究與複製在這些格的差異是抽樣誤差。需先跑 rep_engine.py（讀暫存區 navs_daily.npz）。在暫存區目錄執行。"""
import numpy as np, pandas as pd
z=np.load('navs_daily.npz'); IDX=pd.DatetimeIndex(z['IDX'])
MN=pd.DataFrame(z['NAV'],index=IDX).groupby(IDX.to_period('M')).last(); MB=pd.Series(z['E50'],index=IDX).groupby(IDX.to_period('M')).last()
m=[x for x in MN.index if x>=pd.Period('2016-01','M')]
s=MN.loc[m].to_numpy(); b=MB.loc[m].to_numpy(); rs=s[1:]/s[:-1]-1; rb=b[1:]/b[:-1]-1; T=len(rb)
# exact b12 12m 0050
I=(np.arange(T)[:,None]+np.arange(12)[None,:])%T
cb=np.prod(1+rb[I],axis=1)-1
print('b12 exact 0050 12m p75/p90', np.percentile(cb,[75,90]))
cs=np.prod(1+rs[I,:],axis=1)-1   # (T,20)
print('b12 exact strat 12m p75/p90/p95', np.percentile(cs.ravel(),[75,90,95]))
# b6 exact: 2 blocks of 6 -> T^2 combos x 20 paths
I6=(np.arange(T)[:,None]+np.arange(6)[None,:])%T
g=np.prod(1+rs[I6,:],axis=1)  # (T,20)
x=(g[:,None,:]*g[None,:,:]).ravel()-1
print('b6 exact strat 12m p90/p95', np.percentile(x,[90,95]))
