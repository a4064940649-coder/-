#!/usr/bin/env python3
"""策略執行與彙總（紙上引擎 vrun；可執行版另見 lc_exec.py）。"""
import numpy as np, pandas as pd
import lc_lib as X

JS = list(range(13))
GRID = [dict(kind=K, n=n, wt=wt, cap=cap, carrier=car)
        for K in ('IND', 'CTRL') for n in (1, 2, 3) for wt in ('cap', 'eq') for cap in (0.15, 0.25) for car in ('PURE', 'SAT')]
REFS = [dict(kind='B', carrier='PURE'), dict(kind='IND_ONLY', carrier='PURE'), dict(kind='IND_ONLY', carrier='SAT')]
DEFAULT = dict(kind='IND', n=2, wt='cap', cap=0.15, carrier='SAT')

def name(sp):
    if sp['kind'] == 'B': return 'B'
    if sp['kind'] == 'IND_ONLY': return f"IND_ONLY_{sp['carrier']}"
    s = f"{sp['kind']}_n{sp['n']}_{sp['wt']}_c{int(round(sp['cap'] * 100))}_{sp['carrier']}"
    if sp.get('buf', 1) != 1: s += f"_b{sp['buf']}"
    if sp.get('overflow', 'prop') != 'prop': s += f"_{sp['overflow']}"
    return s

def make(sp, **kw):
    a = dict(sp); a.update(kw)
    return X.Strat(a.get('kind'), n=a.get('n', 2), wt=a.get('wt', 'cap'), cap=a.get('cap', 0.15), carrier=a.get('carrier', 'SAT'),
                   k=a.get('k'), buf=a.get('buf', 1), sector_rank_fn=a.get('srf'), score_fn=a.get('sf'), overflow=a.get('overflow', 'prop'))

def plans(sp, days, **kw):
    s = make(sp, **kw); P = []; I = []
    for d in days[:-1]:
        cs = X.cs_cached(d); idx, w, info = s.step(cs); P.append((idx, w)); I.append(info)
    return P, I

_RC = {}
def run(sp, j, per, cost='normal'):
    key = (X.MODE['m'], name(sp), j, per, cost)
    if key in _RC: return _RC[key]
    days = X.days_for(j, per)
    pl, info = plans(sp, days)
    R, TO = X.vrun(days, pl, cost)
    b = X.bench_win(days)
    out = dict(days=days, r=R[0], to=TO[0], b=b, x=R[0] - b, plans=pl, info=info)
    _RC[key] = out
    return out

def summ_j(xs):
    """xs：{j: 超額序列} → j 中位數與 p10。"""
    st = {j: X.stats(x) for j, x in xs.items()}
    def med(k): return float(np.median([st[j][k] for j in st]))
    def p10(k): return float(np.percentile([st[j][k] for j in st], 10))
    return dict(mean_pp_jmed=med('mean_pp'), mean_pp_p10=p10('mean_pp'), t_jmed=med('t'), t_p10=p10('t'), IR_jmed=med('IR'), TE_jmed=med('TE'),
                n=int(np.median([st[j]['n'] for j in st])), per_j={str(j): {k: round(v, 4) for k, v in st[j].items()} for j in st})
