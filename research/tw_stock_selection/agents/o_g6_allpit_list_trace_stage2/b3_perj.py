# -*- coding: utf-8 -*-
"""B 第 3 步：同一套統計程式（研究員自寫，日曆年化）分別算複製員與原引擎的逐 j 數字，報逐 j 差分布與中位 j 換位。
python3 -I -B b3_perj.py"""
import json, numpy as np, pandas as pd
AG='/home/user/-/research/tw_stock_selection/agents'
SCR='/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad'
OUT=AG+'/o_g6_allpit_list_trace_stage2/out'
R4=pd.read_parquet(AG+'/o_exec_engine_rewrite_v7x/out/navs_oos_R4.parquet')
ENDS=json.load(open(AG+'/o_v7_listing_exec_engine/data/jscan_end_days.json'))
z0=np.load(SCR+'/rep_o19/rdata_strict.npz',mmap_mode='r')
dates=pd.DatetimeIndex(z0['dates'].astype('datetime64[D]')); E=pd.Series(z0['E'].astype(float),index=dates)
p=pd.read_csv(AG+'/o_v7_listing_exec_engine/data/plans_j7.csv',dtype={'picks':str}); p=p[(p.strat=='AB_V7_B10')&(p.seed==0)]
PL={j:sorted(pd.to_datetime(g.signal).tolist()) for j,g in p.groupby('j')}
def ann(nav,a,b): return (nav[b]/nav[a])**(365.25/(b-a).days)-1
def stats(nav,j):
    nav=nav.dropna(); s0=nav.index[0]; end=pd.Timestamp(ENDS[f'AB_V7_B10|J{j}'])
    sig=[d for d in PL[j]]; wins=[(a,b) for a,b in zip(sig,sig[1:]+[end]) if a<end]
    ex=np.array([(nav[b]/nav[a]-1)-(E[b]/E[a]-1) for a,b in wins])
    y0=dates[dates>=pd.Timestamp('2019-01-01')][0]
    yr={}
    for y in range(2006,2027):
        a=nav.index[nav.index<pd.Timestamp(f'{y}-01-01')]; a=a[-1] if len(a) else s0
        b=min(pd.Timestamp(f'{y}-12-31'),end); b=nav.index[nav.index<=b][-1]
        if b<=a: continue
        yr[y]=(nav[b]/nav[a]-1)-(E[b]/E[a]-1)
    return dict(exc=ann(nav,s0,end)-ann(E,s0,end), t=ex.mean()/(ex.std(ddof=1)/np.sqrt(len(ex))),
                exc_y19=ann(nav,y0,end)-ann(E,y0,end), yr=yr)
CELLS={'N100':('normal','100萬'),'N300':('normal','300萬'),'N1000':('normal','1000萬'),'S100':('stress','100萬'),'O30':('normal','30萬')}
res={}
for cell,(c,t) in CELLS.items():
    z=np.load(f'{SCR}/rep_o19/navs_{cell}_strict.npz')
    A=[];B=[]
    for j in range(13):
        eqs=z[f'eq_j{j}']; s0=int(z[f's0_j{j}'])
        r=eqs[:,s0+1:]/eqs[:,s0:-1]-1; rm=r.mean(0)
        nav=np.full(len(dates),np.nan); nav[s0]=1; nav[s0+1:]=np.cumprod(1+rm)
        a=pd.Series(nav,index=dates)
        b=R4[f'B10|J{j}|K5T|{c}|{t}|GROW'].reindex(dates)
        b=b/b[a.index[s0]]; b[b.index<a.index[s0]]=np.nan
        A.append(stats(a,j)); B.append(stats(b,j))
    d={}
    for k in ['exc','t','exc_y19']:
        va=np.array([x[k] for x in A]); vb=np.array([x[k] for x in B]); sc=1 if k=='t' else 100
        dj=(va-vb)*sc
        d[k]=dict(rep_med=float(np.median(va)*sc), orig_med=float(np.median(vb)*sc), diff_of_med=float((np.median(va)-np.median(vb))*sc),
                  perj_diff_median=float(np.median(dj)), perj_diff_maxabs=float(np.abs(dj).max()), perj_diff=[round(float(x),4) for x in dj],
                  rep_med_j=int(np.argsort(va)[6]), orig_med_j=int(np.argsort(vb)[6]),
                  rep_p10=float(np.percentile(va,10)*sc), orig_p10=float(np.percentile(vb,10)*sc))
    # 逐年中位差（逐 j 年度超額差的 13 j 中位數）
    yd={}
    for y in range(2006,2027):
        dj=[(A[j]['yr'].get(y,np.nan)-B[j]['yr'].get(y,np.nan))*100 for j in range(13)]
        yd[y]=float(np.nanmedian(dj))
    d['yearly_median_diff_pp']=yd
    res[cell]=d
    print(cell, {k:(round(d[k]['rep_med'],3),round(d[k]['orig_med'],3),round(d[k]['diff_of_med'],3),round(d[k]['perj_diff_median'],3),round(d[k]['perj_diff_maxabs'],3),d[k]['rep_med_j'],d[k]['orig_med_j']) for k in ['exc','t','exc_y19']})
    print('  yearly', {y:round(v,3) for y,v in yd.items() if abs(v)>0.05})
json.dump(res,open(OUT+'/b3_perj.json','w'),indent=1)
