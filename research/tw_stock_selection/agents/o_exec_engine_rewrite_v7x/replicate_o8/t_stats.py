import sys; sys.path.insert(0,'.')
import json, numpy as np, pandas as pd
import s8, d8
O7=d8.O7
N=pd.read_parquet(O7/'navs_exec7.parquet'); T=json.load(open(O7/'results.json'))['RESULTS_EXEC']['table']
P=pd.read_csv(O7/'data'/'plans_j7.csv',usecols=['strat','j','seed','signal'])
P=P[P.seed==0]; SD={(s,j):list(pd.to_datetime(g.signal)) for (s,j),g in P.groupby(['strat','j'])}
PA=pd.read_csv(O7/'data'/'plans_ab7.csv',usecols=['strat','seed','signal']); PA=PA[PA.seed==0]
for s,g in PA.groupby('strat'): SD[(s,-1)]=list(pd.to_datetime(g.signal))
mx={}; 
for c in N.columns:
    k,jj=c.split('|')[:2]; st='AB_V7' if k=='AB' else 'AB_V7_B10'
    if jj=='OFF': sd=SD[(st,-1)]; end='2026-10-12'
    else: j=int(jj[1:]); sd=SD[(st,j)]; end=s8.ENDS[f'{st}|J{j}']
    nav=N[c].dropna(); o=s8.cell_stats(nav,sd,end); r=T[c]
    for a,b in [('CAGR','CAGR'),('maxDD','maxDD'),('t','t_0050'),('CAGR_SEL','CAGR_SEL'),('CAGR_Y19','CAGR_Y19on'),('t_Y19','t_Y19on'),('CAGR_HOLD','CAGR_HOLD'),('E50','E50_CAGR'),('t_SEL','t_SEL')]:
        if b in r: mx[a]=max(mx.get(a,0),abs(o[a]-r[b]))
print(len(N.columns), mx)
import itertools
c='B10|J3|K5T|normal|300萬|GROW'; nav=N[c].dropna(); r=T[c]; sd=SD[('AB_V7_B10',3)]; end=s8.ENDS['AB_V7_B10|J3']
print(r['CAGR_Y19on'], r['CAGR_HOLD'], r.get('E50_CAGR_Y19on'), r.get('E50_CAGR_HOLD'))
cands=[d for d in sd if pd.Timestamp('2018-12-01')<=d<=pd.Timestamp('2019-02-01')]+[pd.Timestamp('2018-12-28'),pd.Timestamp('2019-01-02')]
for d in cands: print('Y19',d.date(), s8.cagr(nav,d,end), s8.cagr(s8.E50,d,end))
cands=[d for d in sd if pd.Timestamp('2023-12-01')<=d<=pd.Timestamp('2024-02-01')]+[pd.Timestamp('2023-12-29'),pd.Timestamp('2024-01-02')]
for d in cands: print('HOLD',d.date(), s8.cagr(nav,d,end), s8.cagr(s8.E50,d,end))
