# -*- coding: utf-8 -*-
"""S17 獨立複製：凍結後追加（不在 my_numbers.json 內）。NOMU − STD 在 2019-01～2025-12-30（去掉 2026 未滿一年）的 20 條路徑逐條配對。
用法：python3 -I -B rep_s17_extra.py <scratch_dir>（需先建面板）。結果：W10_70 16／20、+1.84；WN_70 18／20、+3.44 個百分點。"""
import sys
sys.path.insert(0, "/home/user/-/research/tw_stock_selection/agents/s17_surge_definition_family/replicate")
import numpy as np, rep_s17_lib as K
S,Q,R,L=K.S,K.Q,K.R,K.L
sc=sys.argv[1]; P=R.build(sc); K.attach_e50_fixed(P,sc); excl=S.innov_mask_fn(P,sc)
Rv,mi0=L.load_rev(P); F,aux=K.family(Rv,mi0)
i19=L.didx(P,"2018-12-28"); i25=L.didx(P,"2025-12-31")-1; print(P.dstr[i25])
pl={(d,sh):Q.cell_plans(Q.make_lists(P,F[d],mi0,shift=sh,excl=excl),"C10",x=0.10) for d in ("STD","NOMU") for sh in range(4)}
for cell,sz in (("W10_70","W10"),("WN_70","WN")):
    dd=[]
    for sh in range(4):
        for cp in (6e5,8e5,1e6,1.5e6,2e6):
            a=K.simulate(P,*pl[("NOMU",sh)],sat_w=0.7,sizing=sz,capital=cp,**K.ST)["nav"]; b=K.simulate(P,*pl[("STD",sh)],sat_w=0.7,sizing=sz,capital=cp,**K.ST)["nav"]
            dd.append(L.ann(P,a,i19,i25)-L.ann(P,b,i19,i25))
    dd=np.array(dd); print(cell,"2019-2025 NOMU-STD n_pos",(dd>0).sum(),"median",round(np.median(dd)*100,2))
