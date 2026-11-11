# -*- coding: utf-8 -*-
"""A：ER2 文字定義的 U_opt／U_p90／U_carry（2006～2013）放回名單（自建 C0 v1 凍結後一次跑）。
python3 -I -B a_build_u.py"""
import sys, json, os, hashlib
sys.argv = [sys.argv[0], 'v1']
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import a_build as B   # 本研究員自寫
HERE = B.HERE
IT = json.load(open(os.path.join(HERE, 'a_iterations.json')))
assert any(x['version'] == 'v1' and x['jaccard_mean'] >= 0.99 for x in IT), 'C0 閘門未過'
cl = B.cl
last_trade = {c: cl[c].last_valid_index() for c in B.cols}
rvm = {c: (s.first_valid_index(), s.last_valid_index()) for c, s in B.RV.items()}
PRI = ['TRUNC', 'IFRS13', 'PRE12', 'HOLE', 'NEWLIST', 'BASE_LE0']
GOV = {'TRUNC', 'IFRS13', 'PRE12', 'HOLE', 'NEVER_DOM'}
def cause(c, m):
    if c not in B.RV.columns or rvm[c][0] is None: return 'NEVER_DOM'
    f, l = rvm[c]; lt = last_trade[c].to_period('M') if last_trade[c] is not None else None
    cs = set()
    for k in (1, 2, 3, 13, 14, 15):
        x = m - k
        lev = B.RV[c].get(x, np.nan)
        if np.isfinite(lev):
            if k <= 3:
                b = B.RV[c].get(x - 12, np.nan)
                if np.isfinite(b) and b <= 0: cs.add('BASE_LE0')
            continue
        if x > l:
            if lt is not None and pd.Period('2009-12', 'M') <= lt <= pd.Period('2012-12', 'M'): cs.add('TRUNC')
            elif x >= pd.Period('2013-01', 'M') and l >= pd.Period('2012-12', 'M'): cs.add('IFRS13')
            else: cs.add('PRE12')
        elif x < f: cs.add('NEWLIST')
        else: cs.add('HOLE')
    for p in PRI:
        if p in cs: return p
    return None
y3cache = {}
def y3_at(m):
    if m not in y3cache:
        y3cache[m] = B.YOY.loc[[m - 1, m - 2, m - 3]].mean(skipna=False) if (m - 3) in B.YOY.index else pd.Series(dtype=float)
    return y3cache[m]
def fill_gov(x, d):
    return 1.0 if cause(x, d.to_period('M')) in GOV else None
def fill_carry(x, d):
    if cause(x, d.to_period('M')) not in GOV: return None
    m = d.to_period('M')
    for k in range(1, 25):
        s = y3_at(m - k)
        v = s.get(x, np.nan)
        if np.isfinite(v): return float(v)
    return None
OUT = {}
for kind, fn in [('opt', fill_gov), ('p90', fill_gov), ('carry', fill_carry)]:
    L, A = B.build(kind, fn)
    slots = [(d, x) for d in L for x in L[d] if x in A[d]]
    n0612 = sum(1 for d, x in slots if d <= '2012-12-31')
    OUT[kind] = dict(lists=L, slots=slots, n_2006_2012=n0612, n_2006_2013=len(slots))
    print(kind, 'U slots 2006-2012', n0612, '2006-2013', len(slots))
C0 = json.load(open(os.path.join(HERE, 'out/a_C0_v1.json')))
for k in OUT: OUT[k]['changed'] = int(sum(len(set(C0[d]) - set(OUT[k]['lists'][d])) for d in C0))
rows = []
for k in OUT:
    for d, L in OUT[k]['lists'].items():
        rows.append(dict(variant=f'U_{k}', signal=d, picks=','.join(L), inserted=','.join(x for x in L if x in dict.fromkeys(y for dd, y in OUT[k]['slots'] if dd == d))))
for d, L in C0.items(): rows.append(dict(variant='C0', signal=d, picks=','.join(L), inserted=''))
pd.DataFrame(rows).to_csv(os.path.join(HERE, 'out/a_lists_2006_2013.csv.gz'), index=False)
json.dump({k: dict(n_2006_2012=v['n_2006_2012'], n_2006_2013=v['n_2006_2013'], changed=v['changed'], slots=v['slots']) for k, v in OUT.items()},
          open(os.path.join(HERE, 'out/a_U_2006_2013.json'), 'w'), ensure_ascii=False, indent=1)
print('changed', {k: OUT[k]['changed'] for k in OUT})
