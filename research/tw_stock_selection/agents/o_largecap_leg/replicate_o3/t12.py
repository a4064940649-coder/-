import sys; sys.path.insert(0, '/home/user/-/research/tw_stock_selection/agents/o_largecap_leg/replicate_o3')
import pickle, json, math; from rlib import *
SC='/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/'
D=pickle.load(open(SC+'rep_o3_data.pkl','rb')); N=pd.read_parquet('navs_rep2.parquet'); sigs=D.sigs
B=bench0050(D.nidx); rb=mret(B,sigs)
def pd_(a,b,segs=('SEL','VAL','VAL_A','HOLD','ALL')):
    d=mret(N[a],sigs)-mret(N[b],sigs); return {s:(round(seg_mask(d,s).mean()*100,3),round(tstat(seg_mask(d,s)),2)) for s in segs}
print('Lnorm-POOLzero',pd_('L|normal','POOL|zero'))
print('Lstress-POOLzero',pd_('L|stress3','POOL|zero'))
print('LXnorm-POOLXzero',pd_('LX|normal','POOLX|zero'))
e=mret(N['POOL|zero'],sigs)-rb; print('POOLzero-0050', round(seg_mask(e,'SEL').mean()*100,3), round(tstat(seg_mask(e,'SEL')),2))
# 0050 mix variants: rebalance at exec day close, and with cost on 0050 trades
M=N['MAIN|normal']
def mix_at(points, cost=0.0):
    out=pd.Series(index=M.index,dtype=float); base=1.0; pts=points+[END]
    out[M.index<pts[0]]=(0.7*M/M[M.index[0]]+0.3*B/B[B.index[0]])[M.index<pts[0]]
    base=out[M.index<pts[0]].iloc[-1]
    for i in range(len(pts)-1):
        s,e=pts[i],pts[i+1]; m=(M.index>s)&(M.index<=e)
        out[m]=base*(0.7*M[m]/M[s]+0.3*B[m]/B[s]); base=out[e]
    return out
ex=[D.next_day(t) for t in sigs if D.next_day(t)<=END]
for nm,v in [('exec_close',mix_at(ex))]:
    r=mret(v,sigs)-rb
    print(nm,{s:(round(seg_nav_cagr(v,None,s)*100,2),round(tstat(seg_mask(r,s)),3),round(seg_mask(r,s).std()*math.sqrt(12)*100,2)) for s in ['SEL','VAL','HOLD','ALL']})
