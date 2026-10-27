import sys; sys.path.insert(0, '/home/user/-/research/tw_stock_selection/agents/o_largecap_leg/replicate_o3')
import pickle, json, math; from rlib import *
SC='/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/'
D=pickle.load(open(SC+'rep_o3_data.pkl','rb')); sigs=D.sigs; N=pd.read_parquet('navs_rep2.parquet')
PL=pickle.load(open(SC+'rep_o3_plans.pkl','rb')); ss=PL['sigs']
names=dict(pd.read_csv(EX+'stock_list.csv',dtype=str)[['code','name']].values)
# 每窗：隔日開盤 → 下一窗隔日開盤（最後一窗到 END 收盤）的個股報酬
ex=[D.next_day(t) for t in ss]
def wret(c,i):
    a=D.adjopen.at[ex[i],c]
    if i+1<len(ss): b=D.adjopen.at[ex[i+1],c]
    else: b=D.adjf.at[END,c]
    if not (b>0): b=D.adjf[c].loc[:ex[i+1] if i+1<len(ss) else END].dropna().iloc[-1]
    return b/a-1
val=[i for i,t in enumerate(ss) if pd.Timestamp('2019-01-01')<=t<=pd.Timestamp('2026-09-30')]
contrib={}; Lg=[]; Pg=[]; rows=[]
for i in val:
    t=ss[i]; pr={c:wret(c,i) for c in PL['pool'][t]}; pe=np.nanmean(list(pr.values()))
    lr={c:pr[c] if c in pr else wret(c,i) for c in PL['L'][t]}
    rows.append((t,lr,pe))
    for c,v in lr.items(): contrib[c]=contrib.get(c,0)+v/5*100
top=sorted(contrib.items(),key=lambda x:-x[1])[:5]
print([(c,names.get(c),round(v,1)) for c,v in top])
g=np.mean([np.mean(list(lr.values()))-pe for t,lr,pe in rows])*100
top5={c for c,_ in top}
g2=np.mean([np.mean([pe if c in top5 else v for c,v in lr.items()])-pe for t,lr,pe in rows])*100
# 成本拖累：L 零成本 − L 一般 的 VAL 月均
dr=(seg_mask(mret(N['L|zero'],sigs),'VAL')-seg_mask(mret(N['L|normal'],sigs),'VAL')).mean()*100
print('gross ex',round(g,3),'drop5',round(g2,3),'drag',round(dr,3),'net',round(g2-dr,3))
for nm,f in [('ex_pool',lambda v,pe: (v-pe)/5*100),('log',lambda v,pe: np.log1p(v)/5*100),('logex',lambda v,pe:(np.log1p(v)-np.log1p(pe))/5*100)]:
    cc={}
    for t,lr,pe in rows:
        for c,v in lr.items(): cc[c]=cc.get(c,0)+f(v,pe)
    tp=sorted(cc.items(),key=lambda x:-x[1])[:5]; t5={c for c,_ in tp}
    g2=np.mean([np.mean([pe if c in t5 else v for c,v in lr.items()])-pe for t,lr,pe in rows])*100
    print(nm,[(c,names.get(c),round(v,1)) for c,v in tp], round(g2,3), round(g2-dr,3))
lg=np.mean([np.mean(list(lr.values())) for t,lr,pe in rows])*100
ln=seg_mask(mret(N['L|normal'],sigs),'VAL').mean()*100; lz=seg_mask(mret(N['L|zero'],sigs),'VAL').mean()*100
print('Lgross window',round(lg,3),'Lnet',round(ln,3),'Lzero',round(lz,3),'drag gross-net',round(lg-ln,3))
