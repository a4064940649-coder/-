#!/usr/bin/env python3
"""複製員：迭代版本中規格可由文字推得者（A1 推測＝跳過上市 [1,3) 年的初次上市＋轉上市股；A5 凍結規格）。挑選期 n10、v2、一般成本。"""
import sys, json, runpy
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.argv = ['r_oos.py', 'v2']
sys.path.insert(0, str(HERE.parent.parent / 'main_v2')); import v2_core as V; K = V.K; sys.path.pop(0)
import numpy as np, pandas as pd
L = pd.read_csv(HERE / 'my_listings.csv', dtype={'code': str}, parse_dates=['first']).set_index('code')
FIRST = np.array([L.loc[c, 'first'] for c in K.COLS], dtype='datetime64[ns]'); KIND = np.array([L.loc[c, 'kind'] for c in K.COLS])
YL, Y5, _ = V.load_revenue(); S = V.V2('gapfix', YL, Y5, exfin=True); t0, t1, sd = S.bounds()
def plan(skipf):
    out = []; held = set(); nsk = 0
    for d in S.sig_days:
        cs = S.cross_section(d); el = cs['el']
        if el.sum() < 50: continue
        gu = K.gap_up_open(cs['e']); sk = skipf(d) if skipf else np.zeros(K.N, bool)
        pick = []
        for j in S.order(cs, 'official')[:int(el.sum())]:
            if gu[j] and j not in held: continue
            if sk[j] and j not in held:
                if d < pd.Timestamp('2019-04-01'): nsk += 1
                continue
            pick.append(j)
            if len(pick) == 10: break
        out.append((cs['e'], np.array(pick, int), np.where(el)[0], d)); held = set(pick)
    return out, nsk
def sk(kinds, lo, hi):
    def f(d):
        a = (np.datetime64(d) - FIRST).astype('timedelta64[D]').astype(float) / 365.25
        return np.isin(KIND, kinds) & (a >= lo) & (a < hi)
    return f
b, s = K.rates(0.001)
pl0, _ = plan(None)
ew, _ = K.simulate([(p[0], p[2]) for p in pl0], 0, 0, t0, t1)
nav0 = K.simulate([(p[0], p[1]) for p in pl0], b, s, t0, t1)[0]
def wr(nav):
    D = pd.DatetimeIndex([d for d in sd if d in nav.index]); nd = nav.reindex(D); return (nd.shift(-1) / nd - 1).iloc[:-1]
res = {'orig_SEL_CAGR': round(V.seg_stats(nav0, ew, '2006-04-01', '2019-04-30', sd)['CAGR'], 2)}
for nm, f in {'A1_guess_ipo+tr_1to3': sk(['ipo', 'transfer'], 1, 3), 'A5': sk(['ipo'], 2, 4)}.items():
    pl, nsk = plan(f)
    nav = K.simulate([(p[0], p[1]) for p in pl], b, s, t0, t1)[0]
    x = (wr(nav) - wr(nav0)); x = x[(x.index >= '2006-04-01') & (x.index <= '2019-03-31')]
    res[nm] = {'SEL_CAGR': round(V.seg_stats(nav, ew, '2006-04-01', '2019-04-30', sd)['CAGR'], 2), 'pair_pp': round(x.mean() * 100, 3), 't': round(K.tstat(x), 2), 'skips_SEL': nsk}
print(json.dumps(res, ensure_ascii=False))
(HERE / 'r_iter.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
