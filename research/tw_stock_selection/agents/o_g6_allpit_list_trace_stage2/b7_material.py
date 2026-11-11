# -*- coding: utf-8 -*-
"""B：逐換股日「重大分歧」（持股價值差合計 > 門檻×淨值）清單與事件分類。python3 -I -B b7_material.py J SEED COST TIER CELL THR"""
import sys, json, gzip, numpy as np, pandas as pd
AG='/home/user/-/research/tw_stock_selection/agents'; O=AG+'/o_g6_allpit_list_trace_stage2/out'
j,seed,cost,tier,cell,thr=int(sys.argv[1]),int(sys.argv[2]),sys.argv[3],int(float(sys.argv[4])),sys.argv[5],float(sys.argv[6])
X=np.load(AG+'/o_exec_engine_rewrite_v7x/data/x_cache.npz',mmap_mode='r'); xi=pd.DatetimeIndex(X['IDX']); xc={c:i for i,c in enumerate(X['COLS'])}
CF=X['CF']; TR=np.isfinite(X['C'])
K=json.load(gzip.open(f'{O}/k_orig_J{j}_s{seed}_{cost}_{tier}.json.gz','rt')); R=json.load(gzip.open(f'{O}/rep_J{j}_s{seed}_{cell}.json.gz','rt'))
kr={r['date']:r for r in K['rec'] if r['kind']=='reb'}; rr={r['date']:r for r in R['rec']}
rows=[]
for d in sorted(set(kr)&set(rr)):
    t=xi.get_loc(pd.Timestamp(d)); a=kr[d]['S']; b=rr[d]['S']
    val=lambda S: sum(v*CF[t,xc[c]] for c,v in S.items())
    nav=val(a)+kr[d]['cash']
    diffs={c:(a.get(c,0)-b.get(c,0))*CF[t,xc[c]] for c in set(a)|set(b)}
    tot=sum(abs(v) for v in diffs.values())/nav
    if tot>thr:
        big=sorted(diffs.items(),key=lambda x:-abs(x[1]))[:4]
        tags=[]
        for c,v in big:
            ti=xc[c]; tags.append(f"{c}:{v/nav*100:+.2f}%{'(停牌)' if not TR[t,ti] else ''}")
        rows.append(dict(date=d,tot=tot,big=tags,cash_o=kr[d]['cash'],cash_r=rr[d]['cash'],pend_o=kr[d]['pend']))
print(f'J{j} s{seed} {cell}: material dates (>{thr*100}% NAV):',len(rows))
for r in rows[:12]: print(' ',r['date'],round(r['tot']*100,2),'%',r['big'],'pend_o',dict(list(r['pend_o'].items())[:6]))
json.dump(rows,open(f'{O}/b7_material_J{j}_s{seed}_{cell}.json','w'),indent=1)
