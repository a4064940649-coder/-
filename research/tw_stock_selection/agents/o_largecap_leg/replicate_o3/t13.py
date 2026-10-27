import sys; sys.path.insert(0, '/home/user/-/research/tw_stock_selection/agents/o_largecap_leg/replicate_o3')
import pickle, json, math; from rlib import *
SC='/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/'
D=pickle.load(open(SC+'rep_o3_data.pkl','rb')); N=pd.read_parquet('navs_rep2.parquet'); sigs=D.sigs
PL=pickle.load(open(SC+'rep_o3_plans.pkl','rb'))
B=bench0050(D.nidx); rb=mret(B,sigs)
# 池等權不擋漲停、零成本
tg={t:{c:1/len(PL['pool'][t]) for c in PL['pool'][t]} for t in PL['sigs']}
nv,_=D.simulate(tg,'zero')
tgx={t:{c:1/len(PL['poolx'][t]) for c in PL['poolx'][t]} for t in PL['sigs']}
nvx,_=D.simulate(tgx,'zero')
for nm,a,b in [('L-poolnoLU',N['L|normal'],nv),('Ls-poolnoLU',N['L|stress3'],nv),('LX-poolxnoLU',N['LX|normal'],nvx)]:
    d=mret(a,sigs)-mret(b,sigs); print(nm,{s:(round(seg_mask(d,s).mean()*100,3),round(tstat(seg_mask(d,s)),2)) for s in ['SEL','VAL','VAL_A','HOLD','ALL']})
d=mret(nv,sigs)-rb; print('pool-0050', round(seg_mask(d,'SEL').mean()*100,3), round(tstat(seg_mask(d,'SEL')),2))
# 0050 混合含再平衡成本
M=N['MAIN|normal']; pts=[x for x in sigs if x<=END]
for cm in [(CB+CS)/2, CB+CS]:
    out=pd.Series(index=M.index,dtype=float); base=1.0
    for i,s in enumerate(pts):
        e=pts[i+1] if i+1<len(pts) else END
        m=(M.index>=s)&(M.index<=e)
        if i>0:
            va=out[s]*0  # placeholder
        v=base*(0.7*M[m]/M[s]+0.3*B[m]/B[s]); out[m]=v
        # 期末漂移：0050 袖子偏離 30% 的金額
        if e in v.index and i+1<len(pts):
            wb=0.3*B[e]/B[s]*base/v[e]; amt=abs(wb-0.3)
            base=v[e]*(1-amt*cm)
        else: base=v.iloc[-1]
    r=mret(out,sigs)-rb
    print(cm,{s:(round(seg_nav_cagr(out,None,s)*100,2),round(tstat(seg_mask(r,s)),3),round(seg_mask(r,s).std()*math.sqrt(12)*100,2)) for s in ['SEL','VAL','HOLD','ALL']})
