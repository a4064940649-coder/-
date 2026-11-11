# -*- coding: utf-8 -*-
"""B 第 4 步：中位 j（N1000 j8、S100 j6）淨值比的最大跳動日。python3 -I -B b4_jumps.py"""
import json, numpy as np, pandas as pd
AG='/home/user/-/research/tw_stock_selection/agents'
SCR='/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad'
R4=pd.read_parquet(AG+'/o_exec_engine_rewrite_v7x/out/navs_oos_R4.parquet')
z0=np.load(SCR+'/rep_o19/rdata_strict.npz',mmap_mode='r')
dates=pd.DatetimeIndex(z0['dates'].astype('datetime64[D]'))
b3=json.load(open(AG+'/o_g6_allpit_list_trace_stage2/out/b3_perj.json'))
out={}
for cell,c,t,j in [('N1000','normal','1000萬',8),('S100','stress','100萬',6),('N100','normal','100萬',6),('N1000','normal','1000萬',3)]:
    print(cell,'perj exc diff',b3[cell]['exc']['perj_diff'])
    z=np.load(f'{SCR}/rep_o19/navs_{cell}_strict.npz'); eqs=z[f'eq_j{j}']; s0=int(z[f's0_j{j}'])
    rs=pd.DataFrame(eqs.T,index=dates).pct_change().iloc[s0+1:]
    rm=rs.mean(axis=1); rb=R4[f'B10|J{j}|K5T|{c}|{t}|GROW'].pct_change().reindex(rm.index)
    dl=np.log1p(rm)-np.log1p(rb)
    top=dl.abs().sort_values(ascending=False).head(8)
    cum=dl.cumsum()
    print(cell,j,'cum log diff end',round(cum.iloc[-1]*100,3),'%; top days:',[(str(d.date()),round(dl[d]*1e4,2)) for d in top.index])
    # 各 seed 在該日的報酬離散（看是否單一 seed）
    for d in top.index[:3]:
        v=rs.loc[d].values*100
        print('   ',d.date(),'rep seeds ret % min/med/max',np.round([v.min(),np.median(v),v.max()],3))
    out[f'{cell}|J{j}']=dict(cum_log_diff_end=float(cum.iloc[-1]),top=[(str(d.date()),float(dl[d])) for d in top.index],
                             by_year={int(y):float(v) for y,v in dl.groupby(dl.index.year).sum().items()})
    print('   by year bp', {y:round(v*1e4,1) for y,v in out[f'{cell}|J{j}']['by_year'].items() if abs(v)>3e-4})
json.dump(out,open(AG+'/o_g6_allpit_list_trace_stage2/out/b4_jumps.json','w'),indent=1)
