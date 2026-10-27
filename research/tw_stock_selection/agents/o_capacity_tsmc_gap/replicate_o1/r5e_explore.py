import sys, math
import numpy as np, pandas as pd
sys.path.insert(0, '/home/user/-/research/tw_stock_selection/agents/o_capacity_tsmc_gap/replicate_o1')
import rcore as R
from r2_tsmc import crosses
from r5_attr import stock_win, summarize_month, Sl, buy, sell
groups=[(1,50),(51,100),(101,200),(201,300)]
W={d:stock_win(d,'oo') for d in Sl}
RT=(1-sell)/(1+buy)-1
def run(sel, net=False):
    res={}
    for lo,hi in groups:
        ser={}
        for d in Sl:
            cr=crosses[d]; r=W[d]
            g=np.where(cr.el&(cr.vrank>=lo)&(cr.vrank<=hi))[0]
            if len(g)<5: continue
            top=sel(cr,g)
            rt=r[top]
            if net: rt=(1+rt)*(1+RT)-1
            ser[d]=np.nanmean(rt)-np.nanmean(r[g])
        s=summarize_month(ser); res[f'{lo}-{hi}']=(s['SEL']['pp'],s['SEL']['t'],s['VAL']['pp'],s['VAL']['t'])
    return res
def ordered(cr,g):
    pos={j:k for k,j in enumerate(cr.order)}; return np.array(sorted(g,key=lambda j:pos[j]))
V={
 'floor_order': lambda cr,g: ordered(cr,g)[:len(g)//5],
 'ceil_order': lambda cr,g: ordered(cr,g)[:math.ceil(len(g)/5)],
 'round_order': lambda cr,g: ordered(cr,g)[:int(round(len(g)/5))],
 'pct>=0.8': lambda cr,g: g[pd.Series(cr.score[g]).rank(pct=True).to_numpy()>=0.8],
 'pct>0.8': lambda cr,g: g[pd.Series(cr.score[g]).rank(pct=True).to_numpy()>0.8],
 'qcut': lambda cr,g: g[pd.qcut(pd.Series(cr.score[g]).rank(method='first'),5,labels=False).to_numpy()==4],
 'quantile0.8': lambda cr,g: g[cr.score[g]>=np.quantile(cr.score[g],0.8)],
}
for k,f in V.items():
    print(k, run(f))
print('floor net', run(V['floor_order'], True))
