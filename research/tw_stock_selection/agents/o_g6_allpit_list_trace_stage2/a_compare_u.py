# -*- coding: utf-8 -*-
"""揭盲比對（研究員）：A 自建 U_opt 放回槽 vs ER2 官方 U_opt_slot_list；相異槽分類。python3 -I -B a_compare_u.py"""
import json, os, collections
import numpy as np, pandas as pd
AG='/home/user/-/research/tw_stock_selection/agents'; HERE=AG+'/o_g6_allpit_list_trace_stage2'
E=json.load(open(AG+'/o_rev_trunc_2009_erratum/ERRATUM2.json'))
A=json.load(open(HERE+'/out/a_U_2006_2013.json'))
off=set(map(tuple,E['U_opt_slot_list'])); mine=set(map(tuple,A['opt']['slots']))
ins={(r['sig'],r['code']):r for r in E['U_opt_inserted']}
only_mine=sorted(mine-off); only_off=sorted(off-mine)
print('official',len(off),'mine',len(mine),'common',len(off&mine),'only_mine',len(only_mine),'only_off',len(only_off),'jaccard',round(len(off&mine)/len(off|mine),4))
c=collections.Counter(x for _,x in only_mine); print('only_mine by code',c.most_common(12))
c2=collections.Counter(x for _,x in only_off); print('only_off by code',c2.most_common(12))
print('only_off causes',collections.Counter(ins[k]['cause'] for k in only_off if k in ins))
json.dump(dict(only_mine=only_mine,only_off=only_off,official_n=len(off),mine_n=len(mine)),open(HERE+'/out/a_compare_u.json','w'),indent=1)
