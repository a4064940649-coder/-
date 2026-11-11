# -*- coding: utf-8 -*-
"""B 第 1 步：複製員 rengine（rep_o19 cache，orig 名單）對原引擎 navs_oos_R4（orig×orig）逐 j 淨值對齊。
不 import 任何研究員或複製員 .py；只讀 npz／parquet／json 資料檔。python3 -I -B b1_navdiff.py"""
import json, numpy as np, pandas as pd
AG='/home/user/-/research/tw_stock_selection/agents'
SCR='/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad'
R4=pd.read_parquet(AG+'/o_exec_engine_rewrite_v7x/out/navs_oos_R4.parquet')
ENDS=json.load(open(AG+'/o_v7_listing_exec_engine/data/jscan_end_days.json'))
z0=np.load(SCR+'/rep_o19/rdata_strict.npz',mmap_mode='r')
dates=z0['dates'].astype('datetime64[D]'); E=z0['E'].astype(float)
CELLS={'N100':('normal','100萬'),'N300':('normal','300萬'),'N1000':('normal','1000萬'),'S100':('stress','100萬')}
out={}
for cell,(c,t) in CELLS.items():
    z=np.load(f'{SCR}/rep_o19/navs_{cell}_strict.npz')
    res={}
    for j in range(13):
        eqs=z[f'eq_j{j}']; s0=int(z[f's0_j{j}'])
        r=eqs[:,s0+1:]/eqs[:,s0:-1]-1; rm=r.mean(0)
        nav=np.full(len(dates),np.nan); nav[s0]=1; nav[s0+1:]=np.cumprod(1+rm)
        a=pd.Series(nav,index=pd.DatetimeIndex(dates))
        b=R4[f'B10|J{j}|K5T|{c}|{t}|GROW']
        idx=a.index.intersection(b.dropna().index)
        aa=a.reindex(idx); bb=b.reindex(idx)
        end=pd.Timestamp(ENDS[f'AB_V7_B10|J{j}'])
        rel=(aa/bb-1)
        relx=rel[rel.index<=end]
        bad=relx[relx.abs()>1e-9]
        first=str(bad.index[0].date()) if len(bad) else None
        # 日報酬差第一個 > 1e-9 的日子
        ra=aa.pct_change(); rb=bb.pct_change(); dr=(ra-rb)[ra.index<=end].dropna()
        fd=dr[dr.abs()>1e-9]
        res[j]=dict(s0=str(a.index[s0].date()), r4_first=str(b.dropna().index[0].date()), end=str(end.date()),
                    first_nav_div=first, first_ret_div=str(fd.index[0].date()) if len(fd) else None,
                    n_ret_div_days=int(len(fd)), rel_end=float(relx.iloc[-1]), max_abs_rel=float(relx.abs().max()))
    out[cell]=res
    print(cell, [(j,res[j]['first_ret_div'],res[j]['n_ret_div_days'],round(res[j]['rel_end']*100,3)) for j in range(13)])
json.dump(out,open(AG+'/o_g6_allpit_list_trace_stage2/out/b1_navdiff.json','w'),indent=1,ensure_ascii=False)
