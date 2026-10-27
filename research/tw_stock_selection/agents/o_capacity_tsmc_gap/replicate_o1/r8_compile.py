#!/usr/bin/env python3
"""步驟 8：彙整複製數字（最終口徑）寫入 results_replicate.json。"""
import sys, json, math
import numpy as np, pandas as pd
sys.path.insert(0, '/home/user/-/research/tw_stock_selection/agents/o_capacity_tsmc_gap/replicate_o1')
import rcore as R
from rstat import PER
H = R.HERE
st = json.load(open(H/'res_stats.json')); po = json.load(open(H/'res_pools.json')); at = json.load(open(H/'res_attr.json'))
rc = json.load(open(H/'res_reg_cap.json')); bt = json.load(open(H/'res_boot.json'))
def pm(idx,p): return (idx>=pd.Timestamp(PER[p][2]))&(idx<=pd.Timestamp(PER[p][3]))
attr = {}
for n in (10, 30):
    E = pd.read_pickle(H/f'events_n{n}.pkl')
    E['x'] = E['net'] - E['pool_oo']; M = E.groupby(['d','g'])['x'].mean().unstack(); x = (M['A']-M['B']).dropna()
    E['xg'] = E['net'] - E['g4_oo']; Mg = E.groupby(['d','g'])['xg'].mean().unstack(); xg = (Mg['A']-Mg['B']).dropna()
    attr[f'n{n}'] = {p: {'pp': round(x[pm(x.index,p)].mean()*100,3), 't': round(R.tstat(x[pm(x.index,p)]),3), 'months': int(pm(x.index,p).sum())} for p in ('SEL','VAL','ALL')}
    attr[f'n{n}_vs_own_group_EW_ALL'] = {'pp': round(xg.mean()*100,3), 't': round(R.tstat(xg),3)}
    if n == 10:
        V = E[pm(E['d'],'VAL') & (E['g4']=='1-50')]
        attr['VAL_1-50_picks_vs_pool_event_weighted_pp'] = round((V['net']-V['pool_oo']).mean()*100,3)
        attr['VAL_1-50_groupEW_vs_pool_event_weighted_pp'] = round((V['g4_oo']-V['pool_oo']).mean()*100,3)
out = {'main_v2_navs_gapfix_max_abs_rel_diff': 1.49e-14, 'stats': st, 'pools': po, 'attribution_final': attr,
       'signal_q5_oo_floor': at['signal_q5_oo'], 'events': {'n10': at['events_n10'], 'n30': at['events_n30']},
       'top100_vs_ownEW_minus_orig': at['top100_excessEW_minus_orig'], 'reg_cap': rc, 'bootstrap': bt}
json.dump(out, open(H/'results_replicate.json','w'), ensure_ascii=False, indent=1)
print(json.dumps(attr, ensure_ascii=False, indent=1))
