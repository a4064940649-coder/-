# -*- coding: utf-8 -*-
"""研究員比對：紙上三遮罩 M0／M1／M2 × opt／p90／carry；ia 與日曆年化並列；三段、2019 起、拿掉最好 3 年 t、成本兩平點；M1→M2 名單差。
python3 -I -B p2_paper_compare.py"""
import json, numpy as np, pandas as pd
AG='/home/user/-/research/tw_stock_selection/agents'; O=AG+'/o_g6_allpit_list_trace_stage2/out'
Z=np.load(O+'/k_paper_windows.npz'); L=pd.read_csv(O+'/k_paper_lists.csv.gz',dtype=str)
d0,d1=pd.Timestamp('2006-04-11'),pd.Timestamp('2026-10-12'); yrs=(d1-d0).days/365.25
def tstat(x): x=np.asarray(x); return x.mean()/(x.std(ddof=1)/np.sqrt(len(x)))
res={}
for m in ['M0','M1','M2']:
    for v in ['ALLPIT_REAL_opt','ALLPIT_REAL_p90','ALLPIT_REAL_carry']:
        for c in ['normal','stress']:
            k=f'{m}|{v}|{c}'; r=Z[k+'|r']; r50=Z[k+'|r50']; per=pd.PeriodIndex(Z[k+'|per'],freq='M')
            g=np.prod(1+r); g5=np.prod(1+r50); n=len(r)
            ex=r-r50; yr=per.year
            ann_ia=(g**(12/n)-1)*100; ann_cal=(g**(1/yrs)-1)*100; a50_ia=(g5**(12/n)-1)*100; a50_cal=(g5**(1/yrs)-1)*100
            def seg(a,b):
                mk=(per>=pd.Period(a,'M'))&(per<=pd.Period(b,'M')); nn=mk.sum()
                return (np.prod(1+r[mk])**(12/nn)-np.prod(1+r50[mk])**(12/nn))*100
            ye=pd.Series(ex).groupby(yr).sum(); best3=ye.sort_values().index[-3:]
            t_x3=tstat(ex[~np.isin(yr,best3)])
            # 成本兩平點：每窗單邊周轉 τ，額外單邊成本 x → r−2xτ；求 ia 年化超額＝0
            Lv=L[(L['mask']==m)&(L.variant==v)].sort_values('signal'); P=[set(s.split(',')) for s in Lv.picks]
            tau=np.array([1.0]+[len(P[i]-P[i-1])/10 for i in range(1,len(P))])[:n]
            def exc_at(x): rr=r-2*x*tau; return (np.prod(1+rr)**(12/n)-g5**(12/n))
            lo,hi=0.0,0.05
            for _ in range(60):
                mid=(lo+hi)/2
                if exc_at(mid)>0: lo=mid
                else: hi=mid
            res[k]=dict(ann_ia=ann_ia,ann_cal=ann_cal,exc_ia=ann_ia-a50_ia,exc_cal=ann_cal-a50_cal,t=tstat(ex),
                        seg1=seg('2006-01','2012-12'),seg2=seg('2013-01','2018-12'),seg3=seg('2019-01','2026-12'),t_x_best3=t_x3,
                        turnover_oneway_month=float(tau[1:].mean()),breakeven_extra_oneway_pct=lo*100)
for k,v in res.items():
    if k.endswith('normal'): print(k,{a:round(b,3) for a,b in v.items()})
# 名單差
def pk(m,v): x=L[(L['mask']==m)&(L.variant==v)]; return dict(zip(x.signal,[set(s.split(',')) for s in x.picks]))
diffs={}
for v in ['ALLPIT_REAL_opt','ALLPIT_REAL_p90','ALLPIT_REAL_carry']:
    for a,b in [('M0','M1'),('M1','M2'),('M0','M2')]:
        A,B=pk(a,v),pk(b,v); ds=[(d,sorted(A[d]-B[d]),sorted(B[d]-A[d])) for d in A if A[d]!=B.get(d)]
        diffs[f'{v}|{a}->{b}']=dict(n_days=len(ds),n_slots=sum(len(x[1]) for x in ds),pre_2021_09_24=sum(len(x[1]) for x in ds if x[0]<='2021-09-23'),days=ds)
        print(v,a,'->',b,'days',len(ds),'slots',diffs[f'{v}|{a}->{b}']['n_slots'],'pre-2021-09-24 slots',diffs[f'{v}|{a}->{b}']['pre_2021_09_24'])
json.dump(dict(stats=res,list_diffs=diffs),open(O+'/p2_paper_compare.json','w'),ensure_ascii=False,indent=1,default=float)
