import sys, math
import numpy as np, pandas as pd
sys.path.insert(0, '/home/user/-/research/tw_stock_selection/agents/o_capacity_tsmc_gap/replicate_o1')
import rcore as R
from r5_attr import stock_win, summarize_month, Sl
crA={d:R.Cross(d,exfin=False) for d in Sl}
groups=[(1,50),(51,100),(101,200),(201,300)]
def ordered(cr,g):
    pos={j:k for k,j in enumerate(cr.order)}; return np.array(sorted(g,key=lambda j:pos[j]))
sels={'floor':lambda cr,g: ordered(cr,g)[:len(g)//5],'ceil':lambda cr,g: ordered(cr,g)[:math.ceil(len(g)/5)],'pct>0.8':lambda cr,g: g[pd.Series(cr.score[g]).rank(pct=True).to_numpy()>0.8],'pct>=0.8':lambda cr,g: g[pd.Series(cr.score[g]).rank(pct=True).to_numpy()>=0.8]}
for wn in ('oo','cc','oc'):
    W={d:stock_win(d,wn) for d in Sl}
    for sn,sf in sels.items():
        res={}
        for lo,hi in groups:
            ser={}
            for d in Sl:
                cr=crA[d]; r=W[d]; g=np.where(cr.el&(cr.vrank>=lo)&(cr.vrank<=hi))[0]
                if len(g)<5: continue
                ser[d]=np.nanmean(r[sf(cr,g)])-np.nanmean(r[g])
            s=summarize_month(ser); res[f'{lo}-{hi}']=(round(s['SEL']['pp'],2),round(s['SEL']['t'],2),round(s['VAL']['pp'],2),round(s['VAL']['t'],2))
        print(wn,sn,res)
