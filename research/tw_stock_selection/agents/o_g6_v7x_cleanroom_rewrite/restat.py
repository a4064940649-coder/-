# 凍結後 F5（統計層）只重算統計：以正式執行存下的 seed 平均淨值（formal_navs.pkl）＋ F5 後的 xstats 重算 Track E／L 各格
# 用法：python3 -I -B audit_run.py J restat.py <navs pkl> <tag> [my_plans pkl]
import sys, os, json, pickle, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import xdata, xstats, xlists
NP, TAG = sys.argv[1], sys.argv[2]
MPP = sys.argv[3] if len(sys.argv) > 3 else xdata.SCR + 'my_plans_AB_V7_B10.pkl'
navs = pickle.load(open(NP, 'rb'))
X = xdata.load(); dates = X['dates']; E = X['E']; T = len(dates)
KD = pickle.load(open(xdata.SCR.replace('g6A/', 'g6K_lists/') + 'k_all_lists.pkl', 'rb'))
MP = pickle.load(open(MPP, 'rb'))
L = xlists.Lists()
def end_for_j(j):
    L.jdays(0); tv = L.tx.values
    k = int(np.searchsorted(tv, np.datetime64(pd.Timestamp('2026-10-12')), 'left')) + j
    if k >= len(tv): return T - 1
    return min(int(np.searchsorted(dates.values, tv[k], 'left')), T - 1)
ENDS = {j: end_for_j(j) for j in range(13)}
idx = lambda d: int(np.searchsorted(dates.values, np.datetime64(pd.Timestamp(d)), 'left'))
SIG = {}
for v, plans in KD['variants'].items():
    for j in range(13): SIG[(v, j)] = [idx(s) for s, e, pk in plans[(j, 0)]]
for j in range(13): SIG[('mine', j)] = [idx(s) for s, e, pk in MP[(j, 0)]]
OUT = {}
for c in sorted(set(k[0] for k in navs)):
    if not (c.startswith('E|') or c.startswith('L|')): continue
    var = c.split('|')[1] if c.startswith('E|') else 'mine'
    per_j = {j: xstats.metrics(navs[(c, j)], SIG[(var, j)], ENDS[j], E, dates) for j in range(13) if (c, j) in navs}
    s = xstats.jsummary(per_j)
    s['exc_y19_frozenbase_med'] = float(np.median([per_j[j]['exc_y19_frozenbase'] for j in per_j]))
    OUT[c] = {'sum': s, 'per_j_exc': {j: per_j[j]['exc'] for j in per_j}, 'per_j_y19': {j: per_j[j]['exc_y19'] for j in per_j}}
R = json.load(open(os.path.join(HERE, 'results.json')))
R[f'RESTAT_{TAG}'] = {'source_navs': NP, 'cells': OUT}
json.dump(R, open(os.path.join(HERE, 'results.json'), 'w'), ensure_ascii=False, indent=1, default=float)
for c in OUT:
    if c.startswith('E|REAL') and c.endswith('N100'):
        s = OUT[c]['sum']; print(c, 'exc', round(100 * s['exc_med'], 3), 'y19', round(100 * s['exc_y19_med'], 3), 'ann_y19', round(100 * s['ann_y19_med'], 3), '0050', round(100 * s['ann0050_y19_med'], 3))
