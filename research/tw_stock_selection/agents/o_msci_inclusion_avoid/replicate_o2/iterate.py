"""鄰近版本 A1、A2、B1、C1（C2 需要 K_class 壓力指標，規格未給定義，不重做）。"""
import sys, os, json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import mainlib as M
V, K = M.V, M.K
PL = M.plans(); t0, t1, sd = PL['t0'], PL['t1'], PL['sd']; ew = PL['ew']
ev = pd.read_csv(M.HERE / 'events_rep.csv', dtype={'code': str}, parse_dates=['E'])
ev = ev[(ev.family == 'MSCI') & ev.code.isin(M.COLS)]
CI = {c: j for j, c in enumerate(M.COLS)}
def sellmap(classes):
    d = {}
    for _, r in ev[ev.cls.isin(classes)].iterrows(): d.setdefault(int(r.Ei), set()).add(CI[r.code])
    return d
SELL_STD = sellmap(['STD']); EXCL = sellmap(['STD', 'SC'])
def plan_A(n, classes):
    x = ev[ev.cls.isin(classes)]
    out = []; held = set(); nchg = 0
    for (e, pk, pool, d) in PL[n]:
        order, gu = PL['cand'][e]
        bad = {CI[r.code] for _, r in x.iterrows() if int(r.Ai) <= e <= int(r.Ei)}
        pick = []
        for j in order:
            if j in bad: continue
            if gu[j] and j not in held: continue
            pick.append(j)
            if len(pick) == n: break
        if set(pick) != set(pk): nchg += 1
        out.append((e, np.array(pick, int), pool, d)); held = set(pick)
    return out, nchg
SEG = {'select': ('2010-11-01', '2018-12-31'), 'judge': ('2019-01-01', '2026-10-12'), 'cross': ('2006-04-01', '2026-10-12')}
res = {}
for n in (10, 30):
    navO, _ = M.simulate(PL[n], 'normal', t0, t1, mode='none')
    base = {s: V.seg_stats(navO, ew, a, b, sd)['CAGR'] for s, (a, b) in SEG.items()}
    pA1, c1 = plan_A(n, ['STD']); pA2, c2 = plan_A(n, ['STD', 'SC'])
    vers = {'A1': (pA1, None, 'none'), 'A2': (pA2, None, 'none'), 'B1': (PL[n], SELL_STD, 'repl'), 'B2': (PL[n], SELL_STD, '0050'), 'C1': (pA1, SELL_STD, 'repl')}
    for k, (pl, sells, mode) in vers.items():
        nav, log = M.simulate(pl, 'normal', t0, t1, sells=sells, mode=mode, cand=PL['cand'], excl=EXCL)
        r = {s: round(V.seg_stats(nav, ew, a, b, sd)['CAGR'] - base[s], 2) for s, (a, b) in SEG.items()}
        r['n_sell'] = sum(1 for x in log if x['kind'] == 'sell'); r['n_plan_changed'] = {'A1': c1, 'A2': c2, 'C1': c1}.get(k, 0)
        res[f'n{n}|{k}'] = r
        print(n, k, r)
json.dump(res, open(M.HERE / 'iterate_rep.json', 'w'), indent=1)
