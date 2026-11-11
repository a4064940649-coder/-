# -*- coding: utf-8 -*-
"""B 第 2 步：第一個分歧日的持股與資料。python3 -I -B b2_firstdiv.py"""
import json, numpy as np, pandas as pd
AG='/home/user/-/research/tw_stock_selection/agents'
SCR='/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad'
z0=np.load(SCR+'/rep_o19/rdata_strict.npz',mmap_mode='r')
print(z0.files)
dates=pd.DatetimeIndex(z0['dates'].astype('datetime64[D]')); codes=list(z0['codes'])
R4=pd.read_parquet(AG+'/o_exec_engine_rewrite_v7x/out/navs_oos_R4.parquet')
p=pd.read_csv(AG+'/o_v7_listing_exec_engine/data/plans_j7.csv',dtype={'picks':str}); p=p[p.strat=='AB_V7_B10']
print(p.columns.tolist(), p.shape)
for cell,(c,t) in {'N100':('normal','100萬'),'N1000':('normal','1000萬')}.items():
    z=np.load(f'{SCR}/rep_o19/navs_{cell}_strict.npz')
    for j in [6,8,0]:
        eqs=z[f'eq_j{j}']; s0=int(z[f's0_j{j}'])
        b=R4[f'B10|J{j}|K5T|{c}|{t}|GROW']
        rb=b.pct_change()
        rs=pd.DataFrame(eqs.T,index=dates).pct_change()
        rm=rs.mean(1)
        d=(rm-rb.reindex(dates)).dropna(); fd=d[d.abs()>1e-9].index[:3]
        for dd in fd[:2]:
            i=dates.get_loc(dd)
            print(cell,j,dd.date(),'avg diff',round(d[dd]*1e4,4),'bp; rep seeds ret dispersion',np.round(rs.loc[dd].values*100,3)[:20])
