#!/usr/bin/env python3
"""探索敏感度：報酬定義（成交日開盤→下一成交日開盤 vs 訊號日收盤→下一訊號日收盤）、年齡分組限續抱股。"""
import sys, json, pickle
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import r4_core as C
K = C.K
P = pickle.load(open(HERE / 'plans.pkl', 'rb'))
rows = P['PL']['BASE']
G, Cn, On, TR = K.GROW, K.Cn, K.On, K.TRADED
SA, SB = pd.Timestamp('2006-04-01'), pd.Timestamp('2018-12-31')
def fwd_oo(i, js):
    t = rows[i]['e']; t2 = rows[i + 1]['e']; js = np.asarray(js, int)
    r = Cn[t, js] / On[t, js] * np.prod(G[t + 1:t2 + 1, js], axis=0)
    trd = TR[t2, js] & np.isfinite(On[t2, js])
    return np.where(trd, r * On[t2, js] / np.where(trd, Cn[t2, js], 1), r) - 1
def fwd_cc(i, js):
    t = rows[i]['e'] - 1; t2 = rows[i + 1]['e'] - 1; js = np.asarray(js, int)
    return np.prod(G[t + 1:t2 + 1, js], axis=0) - 1
res = {}
for lab, fwd in (('open_open', fwd_oo), ('close_close', fwd_cc)):
    age = {}; recs = []
    for i, p in enumerate(rows):
        newage = {int(j): age.get(int(j), 0) + 1 for j in p['picks']}
        if SA <= p['d'] <= SB and i > 0 and i + 1 < len(rows):
            hp = set(p['held_prev'])
            nj = [int(j) for j in p['picks'] if int(j) not in hp]
            for j, r in zip(nj, fwd(i, nj)): recs.append(dict(d=p['d'], kind='new', rank=p['rank'].get(j, np.nan), s52=p['s52'][j], age=0, r=r))
            oj = [int(j) for j in p['held_prev'] if TR[p['e'], j] and np.isfinite(On[p['e'], j])]
            for j, r in zip(oj, fwd(i, oj)): recs.append(dict(d=p['d'], kind='old', rank=p['rank'].get(j, np.nan), s52=p['s52'][j], age=age.get(j, 0), r=r))
        age = newage
    R = pd.DataFrame(recs); old = R[R.kind == 'old']; new = R[R.kind == 'new']
    nm = new.groupby('d')['r'].mean()
    o = {'new_pooled': new['r'].mean() * 100, 'new_monthly': nm.mean() * 100}
    def g(m, name):
        gg = old[m & old.r.notna()]; gm = gg.groupby('d')['r'].mean(); dd = (gm - nm.reindex(gm.index)).dropna()
        o[name] = (round(dd.mean() * 100, 2), round(C.tstat(dd), 2), len(dd), len(gg))
    rk, s5, ag = old['rank'], old['s52'], old['age']
    g(rk <= 10, 'top10'); g((rk > 10) & (rk <= 30) & (s5 >= .9), 'P2a'); g(s5 < 0.85, 's<0.85')
    g((s5 < 0.85) & (rk > 10), 's<0.85&rk>10'); g((s5 < 0.85) & ((rk > 10) | rk.isna()), 's<0.85&(rk>10|inel)')
    for a in (1, 2, 3): g((rk <= 10) & (ag == a), f'top10 age=={a}')
    g((rk <= 10) & (ag >= 3), 'top10 age>=3'); g((rk <= 10) & (ag >= 4), 'top10 age>=4')
    for a in (1, 2): g(ag == a, f'all age=={a}')
    g(ag >= 3, 'all age>=3')
    res[lab] = o
    print(lab, o, flush=True)
