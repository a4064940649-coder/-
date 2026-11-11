# -*- coding: utf-8 -*-
"""B：對齊實驗 V0～V3（複製員引擎修改副本）逐 j 對原引擎 R4 的差（研究員自寫統計，日曆年化、per-j 終點）。
python3 -I -B b9_align_stats.py"""
import json, os, numpy as np, pandas as pd
AG='/home/user/-/research/tw_stock_selection/agents'; SCR='/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad'
R4=pd.read_parquet(AG+'/o_exec_engine_rewrite_v7x/out/navs_oos_R4.parquet')
ENDS=json.load(open(AG+'/o_v7_listing_exec_engine/data/jscan_end_days.json'))
z0=np.load(SCR+'/rep_o19/rdata_strict.npz',mmap_mode='r'); dates=pd.DatetimeIndex(z0['dates'].astype('datetime64[D]')); E=pd.Series(z0['E'].astype(float),index=dates)
p=pd.read_csv(AG+'/o_v7_listing_exec_engine/data/plans_j7.csv',dtype={'picks':str}); p=p[(p.strat=='AB_V7_B10')&(p.seed==0)]
PL={j:sorted(pd.to_datetime(g.signal).tolist()) for j,g in p.groupby('j')}
def ann(nav,a,b): return (nav[b]/nav[a])**(365.25/(b-a).days)-1
def stats(nav,j):
    nav=nav.dropna(); s0=nav.index[0]; end=pd.Timestamp(ENDS[f'AB_V7_B10|J{j}'])
    sig=PL[j]; wins=[(a,b) for a,b in zip(sig,sig[1:]+[end]) if a<end]
    ex=np.array([(nav[b]/nav[a]-1)-(E[b]/E[a]-1) for a,b in wins]); y0=dates[dates>=pd.Timestamp('2019-01-01')][0]
    return dict(exc=100*(ann(nav,s0,end)-ann(E,s0,end)), t=ex.mean()/(ex.std(ddof=1)/np.sqrt(len(ex))), exc_y19=100*(ann(nav,y0,end)-ann(E,y0,end)))
CN={'N100':('normal','100萬'),'N300':('normal','300萬'),'N1000':('normal','1000萬'),'S100':('stress','100萬'),'O30':('normal','30萬')}
out={}
for cell in ['N1000','S100','N100','O30']:
    c,t=CN[cell]; B=[]
    for j in range(13):
        b=R4[f'B10|J{j}|K5T|{c}|{t}|GROW']; B.append(b)
    for tag in ['V0','V1','V2','V3']:
        f=f'{SCR}/o_g6s2/navs_{cell}_{tag}.npz'
        if not os.path.exists(f): continue
        z=np.load(f); A=[];O=[]
        for j in range(13):
            eqs=z[f'eq_j{j}']; s0=int(z[f's0_j{j}']); r=eqs[:,s0+1:]/eqs[:,s0:-1]-1; rm=r.mean(0)
            nav=np.full(len(dates),np.nan); nav[s0]=1; nav[s0+1:]=np.cumprod(1+rm); a=pd.Series(nav,index=dates)
            b=B[j].reindex(dates); b=b/b[a.index[s0]]; b[b.index<a.index[s0]]=np.nan
            A.append(stats(a,j)); O.append(stats(b,j))
        d={}
        for k in ['exc','t','exc_y19']:
            va=np.array([x[k] for x in A]); vb=np.array([x[k] for x in O]); dj=va-vb
            d[k]=dict(rep_med=float(np.median(va)),orig_med=float(np.median(vb)),diff_of_med=float(np.median(va)-np.median(vb)),
                      perj_med=float(np.median(dj)),perj_maxabs=float(np.abs(dj).max()),rep_p10=float(np.percentile(va,10)),orig_p10=float(np.percentile(vb,10)),
                      med_j=(int(np.argsort(va)[6]),int(np.argsort(vb)[6])),perj=[round(float(x),4) for x in dj])
        out[f'{cell}|{tag}']=d
        print(cell,tag,{k:(round(d[k]['rep_med'],3),round(d[k]['orig_med'],3),round(d[k]['diff_of_med'],3),round(d[k]['perj_med'],3),round(d[k]['perj_maxabs'],3)) for k in d})
json.dump(out,open(AG+'/o_g6_allpit_list_trace_stage2/out/b9_align.json','w'),indent=1)
