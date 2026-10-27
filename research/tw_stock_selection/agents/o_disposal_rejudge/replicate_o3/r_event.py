import sys; sys.path.insert(0, "/home/user/-/research/tw_stock_selection/agents/o_disposal_rejudge/replicate_o3")
import pickle, math
import numpy as np, pandas as pd
import r_core as R
D = pickle.load(open(R.HERE / 'my_runs.pkl', 'rb')); PL, NAV = D['PL'], D['NAV']
IDX, Cn, On, G, TR = R.IDX, R.Cn, R.On, R.GROW, R.TRADED
def seg(j, ta, tb, from_open=True, to_open=False):
    """ta 日（開盤或收盤）到 tb 日（收盤或開盤）的報酬"""
    r = (Cn[ta, j] / On[ta, j]) if from_open else 1.0
    if tb > ta: r = r * np.prod(G[ta + 1:tb + 1, j], axis=0)
    if to_open: r = r * On[tb, j] / Cn[tb, j]
    return r - 1
pl = PL[('E0', 10)]
rows = []
for i, p in enumerate(pl):
    if p['d'] > pd.Timestamp('2021-08-31'): break
    e = p['e']; e2 = pl[i + 1]['e']
    aff = [j for j in p['pick'] if j in p['st']]
    if not aff: continue
    oth = [j for j in p['pick'] if j not in p['st']]
    for j in aff:
        nth, rem, L, a, b = p['st'][j]
        tend = R.IDX.searchsorted(b, side='right') - 1; tend = min(tend, e2 - 1)
        ok = np.isfinite(On[e2, oth]) & np.isfinite(On[e2, j])
        hold = seg(j, e, e2, True, True) - np.nanmean(seg(np.array(oth), e, e2, True, True))
        ins = seg(j, e, tend, True, False) - np.nanmean(seg(np.array(oth), e, tend, True, False))
        aft = ((1 + seg(j, e, e2, True, True)) / (1 + seg(j, e, tend, True, False)) - 1) - np.nanmean((1 + seg(np.array(oth), e, e2, True, True)) / (1 + seg(np.array(oth), e, tend, True, False)) - 1)
        thr = math.ceil(0.6 * L)
        if rem == 0:
            ins = np.nan
            aft = seg(j, tend, e2, False, True) - np.nanmean(seg(np.array(oth), tend, e2, False, True))
        rows.append(dict(d=p['d'], code=R.COLS[j], nth=nth, rem=rem, L=L, grp=('R0' if rem == 0 else ('R1' if rem < thr else 'R2')), hold=hold, ins=ins, aft=aft))
ev = pd.DataFrame(rows); print(ev.round(4).to_string())
def clus(df, col):
    m = df.groupby('d')[col].mean().dropna(); return round(m.mean() * 100, 2), round(R.tstat(m), 2), len(m)
for col in ('hold', 'ins', 'aft'):
    print(col, 'all', clus(ev, col), 'R1', clus(ev[ev.grp == 'R1'], col), 'R2', clus(ev[ev.grp == 'R2'], col))
# 剔除股減替補股、累積超額路徑
P1 = {p['d']: p for p in PL[('E1', 10)]}
rep = []; paths = []
for i, p in enumerate(pl):
    if p['d'] > pd.Timestamp('2021-08-31'): break
    aff = [j for j in p['pick'] if j in p['st']]
    if not aff: continue
    e = p['e']; e2 = pl[i + 1]['e']
    q = P1[p['d']]; e2b = [x for x in PL[('E1', 10)] if x['d'] > p['d']][0]['e']
    repl = [j for j in q['pick'] if j not in set(p['pick'])]
    ra = np.mean(seg(np.array(aff), e, e2, True, True)); rr = np.nanmean(seg(np.array(repl), e, e2b, True, True))
    rep.append((p['d'], ra - rr, len(aff), len(repl)))
    oth = np.array([j for j in p['pick'] if j not in p['st']])
    for j in aff:
        cj = [seg(j, e, e + k, True, False) for k in range(0, 25)]
        co = [np.nanmean(seg(oth, e, e + k, True, False)) for k in range(0, 25)]
        paths.append(np.array(cj) - np.array(co))
r = pd.DataFrame(rep, columns=['d', 'x', 'na', 'nr']); print(r)
print('drop minus replacement', round(r.x.mean() * 100, 2), round(R.tstat(r.x), 2))
pm = np.mean(paths, axis=0) * 100
print('path (day k = through close of e+k-1):', {k + 1: round(v, 2) for k, v in enumerate(pm)})
