# -*- coding: utf-8 -*-
"""C：成交值名次（vrank）兩引擎差異，以規格文字「V60 ≥ 1e7 且收盤 ≥ 10，依 V60 排名次」從共用面板重算抽樣裁定。python3 -I -B c_vrank.py"""
import numpy as np, pandas as pd
AG='/home/user/-/research/tw_stock_selection/agents'; PAN='/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel/'
X=np.load(AG+'/o_exec_engine_rewrite_v7x/data/x_cache.npz'); R=np.load('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/rep_o19/rdata_strict.npz')
xi=pd.DatetimeIndex(X['IDX']); cols=list(X['COLS'])
VRo=X['VRp'][1:]; VRr=R['VR'][:-1]; V60=X['V60p'][1:]; C=X['C'][:-1]  # 皆為 t−1 的值
mis=np.argwhere(np.isfinite(VRo)&np.isfinite(VRr)&(np.abs(VRo-VRr)>0.5))
rng=np.random.default_rng(0); smp=mis[rng.choice(len(mis),200,replace=False)]
cl=pd.read_parquet(PAN+'close_raw.parquet'); val=pd.read_parquet(PAN+'value.parquet')
good=cl.notna().sum(axis=1)>=300; G=good[good].index; v60=val.loc[G].rolling(60,min_periods=60).mean()
win={'orig':0,'rep':0,'neither':0}; ex=[]
for t,n in smp:
    d=xi[t]   # 這是 t−1 日（x 的 t+1 取用）
    if d not in v60.index: win['neither']+=1; continue
    row=v60.loc[d]; c=cl.loc[d]; el=row[(row>=1e7)&(c>=10)]
    code=cols[n]
    if code not in el.index: win['neither']+=1; ex.append((str(d.date()),code,'not eligible',VRo[t,n],VRr[t,n])); continue
    rk=float((el>el[code]).sum()+1)
    if abs(rk-VRo[t,n])<0.5: win['orig']+=1
    elif abs(rk-VRr[t,n])<0.5: win['rep']+=1
    else: win['neither']+=1; ex.append((str(d.date()),code,rk,VRo[t,n],VRr[t,n]))
print('cells finite both',int((np.isfinite(VRo)&np.isfinite(VRr)).sum()),'mismatch',len(mis),'sample 200 →',win); print(ex[:8])
