import pandas as pd, numpy as np, pickle, sys, json
OUT='/home/user/-/research/tw_stock_selection/agents/o_lc_sector_rotation_pit/replicate_o12/'
tag=sys.argv[1]
W=pickle.load(open(OUT+f'win_{tag}.pkl','rb'))
def nw_t(x,L=6):
    x=np.asarray(x)-np.mean(x); n=len(x); g0=(x*x).sum()/n; s=g0
    for l in range(1,L+1): s+=2*(1-l/(L+1))*(x[l:]*x[:-l]).sum()/n
    return np.mean(np.asarray(x)+0)/1  # placeholder
def stats(e):
    e=np.asarray(e); n=len(e); m=e.mean(); sd=e.std(ddof=1)
    return dict(n=n,mean_pp=m*100,t=m/(sd/np.sqrt(n)),IR=m/sd*np.sqrt(12),TE=sd*np.sqrt(12)*100)
def nwt(e,L=6):
    e=np.asarray(e); n=len(e); m=e.mean(); x=e-m; v=(x*x).sum()/n
    for l in range(1,L+1): v+=2*(1-l/(L+1))*(x[l:]*x[:-l]).sum()/n
    return m/np.sqrt(v/n)
def ann(r): r=np.asarray(r); return (np.prod(1+r)**(12/len(r))-1)*100
PER={'VAL':('2017-01','2026-10'),'FORMAL':('2014-01','2026-10'),'SEL':('2014-01','2016-12'),'S1418':('2014-01','2018-12'),'S19':('2019-01','2026-10')}
res={}
for pn,(a,b) in PER.items():
    rows=[]
    for j,df in W.items():
        x=df[(df.am>=a)&(df.am<=b)]
        if len(x)<10: continue
        r=dict(j=j,n=len(x),ann=ann(x.normal),ann0050=ann(x.e0050),annB=ann(x.B_normal))
        for nm,(s,bm) in {'v0050':('normal','e0050'),'vB':('normal','B_normal'),'zvB':('zero','B_zero'),'sv0050':('stress','e0050'),'svB':('stress','B_stress'),'Bv0050':('B_normal','e0050'),'BzvB0050':('B_zero','e0050')}.items():
            st=stats(x[s]-x[bm]);
            for k_,v in st.items(): r[f'{nm}_{k_}']=v
            r[f'{nm}_nwt']=nwt(x[s]-x[bm])
        r['turn']=x.turn.mean()*100; r['Bturn']=x.B_turn.mean()*100
        r['cost_pp']=(x.zero-x.normal).mean()*100
        rows.append(r)
    R=pd.DataFrame(rows)
    med=R.median(); p10=R.quantile(0.1)
    res[pn]=dict(med=med.round(3).to_dict(),p10={k:round(p10[k],3) for k in ['v0050_mean_pp','vB_mean_pp']},nj=len(R))
# drop best 3 years vs 0050 (per j) 
for pn,(a,b) in [('VAL',PER['VAL']),('FORMAL',PER['FORMAL'])]:
    ts=[];yrs=[]
    for j,df in W.items():
        x=df[(df.am>=a)&(df.am<=b)].copy(); x['y']=x.am.str[:4]; x['ex']=x.normal-x.e0050
        yb=x.groupby('y').ex.sum().sort_values(ascending=False).index[:3]
        z=x[~x.y.isin(yb)].ex; ts.append(z.mean()/(z.std(ddof=1)/np.sqrt(len(z)))); yrs.append(tuple(yb))
    res[pn]['drop3_t_med']=float(np.median(ts)); res[pn]['drop3_years']={','.join(k):v for k,v in pd.Series(yrs).value_counts().head(3).items()}
# yearly B vs 0050 2024
yy=[]
for j,df in W.items():
    x=df[df.am.str[:4]=='2024']; yy.append(((1+x.B_normal).prod()-(1+x.e0050).prod())*100)
res['B_v0050_2024_pp_med']=float(np.median(yy))
print(json.dumps({k:v for k,v in res.items()},ensure_ascii=False,indent=0,default=str))
json.dump(res,open(OUT+f'stats_{tag}.json','w'),ensure_ascii=False,indent=1,default=str)
