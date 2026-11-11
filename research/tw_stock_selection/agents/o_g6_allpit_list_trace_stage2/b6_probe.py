# -*- coding: utf-8 -*-
"""B：兩引擎輸入資料（原引擎 x_cache.npz、複製員 rdata_strict.npz）在指定日×股的逐欄比對。只讀資料檔。
python3 -I -B b6_probe.py DATE CODE[,CODE...] [ndays]"""
import sys, numpy as np, pandas as pd
AG='/home/user/-/research/tw_stock_selection/agents'
SCR='/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad'
X=np.load(AG+'/o_exec_engine_rewrite_v7x/data/x_cache.npz',mmap_mode='r')
Rz=np.load(SCR+'/rep_o19/rdata_strict.npz',mmap_mode='r')
xi=pd.DatetimeIndex(X['IDX']); xc=list(X['COLS']); ri=pd.DatetimeIndex(Rz['dates'].astype('datetime64[D]')); rc=list(Rz['codes'])
d=pd.Timestamp(sys.argv[1]); codes=sys.argv[2].split(','); nd=int(sys.argv[3]) if len(sys.argv)>3 else 1
pairs=[('C','C'),('O','O'),('PC','PC'),('CF','CF'),('ADJ','ADJ'),('G','G'),('V','V'),('V60','V60'),('V60p','V60'),('VRp','VR'),('SIGp','SIG'),('F','FL')]
for c in codes:
    for k in range(-nd+1,1) if nd>0 else [0]:
        a=xi.get_loc(d)+k; b=ri.get_loc(xi[a])
        row=[]
        for kx,kr in pairs:
            vx=X[kx][a,xc.index(c)] if c in xc else np.nan
            vr=Rz[kr][b,rc.index(c)] if c in rc else np.nan
            # 複製員 V60/VR/SIG 在 t−1 取用；原引擎 V60p/VRp/SIGp 是否已位移，另列 t−1
            row.append(f'{kx}:{vx:.6g}/{kr}:{vr:.6g}' + ('' if (np.isnan(vx) and np.isnan(vr)) or (np.isfinite(vx) and np.isfinite(vr) and abs(vx-vr)<=1e-9*max(1,abs(vx))) else ' **'))
        vrm1=Rz['VR'][b-1,rc.index(c)]; v60m1=Rz['V60'][b-1,rc.index(c)]
        print(xi[a].date(),c,' | '.join(row),f'| rep VR[t-1]={vrm1:.6g} V60[t-1]={v60m1:.6g}')
