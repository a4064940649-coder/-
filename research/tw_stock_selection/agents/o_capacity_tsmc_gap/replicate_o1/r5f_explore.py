import sys, math
import numpy as np, pandas as pd
sys.path.insert(0, '/home/user/-/research/tw_stock_selection/agents/o_capacity_tsmc_gap/replicate_o1')
import rcore as R
from r2_tsmc import crosses
from r5_attr import stock_win, summarize_month, Sl
groups=[(1,50),(51,100),(101,200),(201,300)]
def h21(d,h=21,start='open'):
    i=R.POS[d]; a=R.AO[i+1] if start=='open' else R.AC[i]; e=min(i+1+h, len(R.IDX)-1) if start=='open' else min(i+h,len(R.IDX)-1)
    return R.AC[e]/a-1
def ordered(cr,g):
    pos={j:k for k,j in enumerate(cr.order)}; return np.array(sorted(g,key=lambda j:pos[j]))
sels={'floor':lambda cr,g: ordered(cr,g)[:len(g)//5],'ceil':lambda cr,g: ordered(cr,g)[:math.ceil(len(g)/5)],'pct>0.8':lambda cr,g: g[pd.Series(cr.score[g]).rank(pct=True).to_numpy()>0.8],'pct>=0.8':lambda cr,g: g[pd.Series(cr.score[g]).rank(pct=True).to_numpy()>=0.8]}
wins={'cc':lambda d: stock_win(d,'cc'),'oc':lambda d: stock_win(d,'oc'),'o21':lambda d:h21(d,21),'o20':lambda d:h21(d,20),'c21':lambda d:h21(d,21,'close'),'o21c':lambda d: h21(d,20)}
for wn,wf in wins.items():
    W={d:wf(d) for d in Sl}
    for sn,sf in sels.items():
        res={}
        for lo,hi in groups:
            ser={}
            for d in Sl:
                cr=crosses[d]; r=W[d]; g=np.where(cr.el&(cr.vrank>=lo)&(cr.vrank<=hi))[0]
                if len(g)<5: continue
                ser[d]=np.nanmean(r[sf(cr,g)])-np.nanmean(r[g])
            s=summarize_month(ser); res[f'{lo}-{hi}']=(round(s['SEL']['pp'],2),round(s['SEL']['t'],2),round(s['VAL']['pp'],2),round(s['VAL']['t'],2))
        print(wn,sn,res)
