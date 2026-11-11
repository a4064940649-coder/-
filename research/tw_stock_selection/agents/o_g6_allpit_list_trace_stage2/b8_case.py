# -*- coding: utf-8 -*-
"""B：單一分歧個案：兩引擎在區間內的持股軌跡＋名單＋可交易性。python3 -I -B b8_case.py J SEED COST TIER CELL D0 D1 CODES"""
import sys, json, gzip, numpy as np, pandas as pd
AG='/home/user/-/research/tw_stock_selection/agents'; O=AG+'/o_g6_allpit_list_trace_stage2/out'
j,seed,cost,tier,cell,d0,d1,codes=int(sys.argv[1]),int(sys.argv[2]),sys.argv[3],int(float(sys.argv[4])),sys.argv[5],sys.argv[6],sys.argv[7],sys.argv[8].split(',')
K=json.load(gzip.open(f'{O}/k_orig_J{j}_s{seed}_{cost}_{tier}.json.gz','rt')); R=json.load(gzip.open(f'{O}/rep_J{j}_s{seed}_{cell}.json.gz','rt'))
X=np.load(AG+'/o_exec_engine_rewrite_v7x/data/x_cache.npz',mmap_mode='r'); xi=pd.DatetimeIndex(X['IDX']); xc=list(X['COLS'])
p=pd.read_csv(AG+'/o_v7_listing_exec_engine/data/plans_j7.csv',dtype={'picks':str}); g=p[(p.strat=='AB_V7_B10')&(p.j==j)&(p.seed==seed)]
print(g[(g['exec']>=d0)&(g['exec']<=d1)][['signal','exec','picks']].to_string(index=False))
for r in K['rec']:
    if d0<=r['date']<=d1 and (r['kind']=='reb' or any(c in r['pend'] for c in codes)):
        print('K',r['date'],r['kind'],{c:(r.get('pre',{}).get(c),r['S'].get(c),r['pend'].get(c)) for c in codes},'cash',round(r['cash']))
for r in R['rec']:
    if d0<=r['date']<=d1: print('R',r['date'],{c:r['S'].get(c) for c in codes},'cash',round(r['cash']))
m=(xi>=d0)&(xi<=d1)
for c in codes:
    i=xc.index(c); print(c,'C',[(d.strftime('%m-%d'),None if np.isnan(v) else v) for d,v in zip(xi[m],X['C'][m,i])][:25])
    print(c,'V60p',[None if np.isnan(v) else round(v/1e6,1) for v in X['V60p'][m,i]][:25])
