# -*- coding: utf-8 -*-
"""B 第 5 步：逐換股日比對持股／現金（原引擎 K 輸出 vs 複製員引擎輸出；兩者都只讀 json）。
python3 -I -B b5_trace_cmp.py J SEED COST TIER CELL [maxshow]"""
import sys, json, gzip, numpy as np, pandas as pd
O='/home/user/-/research/tw_stock_selection/agents/o_g6_allpit_list_trace_stage2/out'
j,seed,cost,tier,cell=int(sys.argv[1]),int(sys.argv[2]),sys.argv[3],int(float(sys.argv[4])),sys.argv[5]
mx=int(sys.argv[6]) if len(sys.argv)>6 else 5
K=json.load(gzip.open(f'{O}/k_orig_J{j}_s{seed}_{cost}_{tier}.json.gz','rt'))
R=json.load(gzip.open(f'{O}/rep_J{j}_s{seed}_{cell}.json.gz','rt'))
kn=pd.Series(K['nav']); kn.index=pd.to_datetime(kn.index)
rq=pd.Series(R['eq']); rq.index=pd.to_datetime(rq.index); rn=rq/rq.iloc[0]
idx=kn.index.intersection(rn.index)
dr=(rn.reindex(idx).pct_change()-kn.reindex(idx).pct_change()).dropna()
print('final nav rep/orig', round(rn.reindex(idx).iloc[-1],6), round(kn.reindex(idx).iloc[-1],6), 'first ret diff>1e-9:', dr[dr.abs()>1e-9].index[:3].date.tolist(), 'n', int((dr.abs()>1e-9).sum()))
kr={r['date']:r for r in K['rec'] if r['kind']=='reb'}
rr={r['date']:r for r in R['rec']}
n=0; res=[]
for d in sorted(set(kr)&set(rr)):
    a=kr[d]['S']; b=rr[d]['S']
    ks=set(a)|set(b)
    diff={c:(round(a.get(c,0),3),round(b.get(c,0),3)) for c in ks if abs(a.get(c,0)-b.get(c,0))>=0.5}
    eqk=kn.get(pd.Timestamp(d)); 
    cd=kr[d]['cash']-rr[d]['cash']*1.0
    if diff:
        res.append(dict(date=d,diff=diff,cash_orig=kr[d]['cash'],cash_rep=rr[d]['cash'],pend_orig=kr[d]['pend']))
        if n<mx: print(d,'shares orig vs rep',diff,'cash',round(kr[d]['cash'],1),round(rr[d]['cash'],1),'pend_orig',kr[d]['pend'])
        n+=1
print('rebalance dates with share diffs:',n,'of',len(set(kr)&set(rr)))
json.dump(res,open(f'{O}/b5_cmp_J{j}_s{seed}_{cell}.json','w'),indent=1)
