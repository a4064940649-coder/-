#!/usr/bin/env python3
"""S25 事後描述（未登記，判定之後才跑，不影響判定、不得拿來換規則）：其餘 3 條候選在驗證窗（重新起跑）20 條路徑的表現。
輸出 results_posthoc.json。results.json 必須已存在。python3 -I -B posthoc.py"""
import sys, json, time, resource, gc
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s25lib
from s25lib import s17lib, s10lib, s4lib, stlib, COSTS

H = Path(__file__).resolve().parent
assert (H / 'results.json').exists(), '先跑 validate.py'
t_all = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
P = s25lib.load()
INNOV, _ = s4lib.innov_panel(P)
U = stlib.universe(P, 'twse') & ~INNOV
del INNOV
RT = s25lib.surge(P)
s17lib.e50_fix_suspension(P)
FD = s25lib.load_factors()
lastb = lambda d: P.IDX[P.IDX < pd.Timestamp(d)][-1]
TB, DE = lastb('2022-01-01'), P.IDX[-1]
CAPS = (6e5, 8e5, 1e6, 1.5e6, 2e6)
ratio = json.loads((H / 'results.json').read_text())['meta']['fresh_capital_scale']


def sim(S1, cap):
    S1 = s25lib.fresh_from(S1, s25lib.VALID_FIRST_MONTH)
    S2, ekw = s10lib.variant_S(S1, 'E0')
    return s10lib.simulate_s10(P, S2, U, cost=COSTS['strict'], capital=cap / ratio, sat_w=0.7, sizing='W10', lock=False, **ekw)[0]


CFG = ['CUR'] + [r['id'] for r in s25lib.RULES]
RD = {r['id']: r for r in s25lib.RULES}
M = {c: {} for c in CFG}
for s in range(4):
    S = s10lib.make_signal_s10(P, U, RT, shift=s)
    F = s25lib.trend_features(S, FD)
    for cap in CAPS:
        for c in CFG:
            S1 = S if c == 'CUR' else s25lib.apply_regime(S, F, s25lib.triggered(F, RD[c]), RD[c]['action'])[0]
            nav = sim(S1, cap)
            sg = s17lib.seg_dd(nav)
            M[c][f'{int(cap/1e4)}萬|+{s}'] = dict(valid=stlib.cagr(nav, TB, DE), dd=stlib.maxdd(nav, TB, DE), seg2022=sg['2022'], seg2024=sg['2024-07~2025-04'])
    gc.collect()
keys = list(M['CUR'])
out = {}
for c in CFG[1:]:
    o = {}
    for f in ('valid', 'dd', 'seg2022', 'seg2024'):
        d = np.array([M[c][k][f] - M['CUR'][k][f] for k in keys]) * 100
        o[f] = dict(median_pp=float(np.median(d)), npos=int((d > 0).sum()))
    out[c] = o
res = dict(note='未登記事後描述；不影響判定', valid_fresh_diff_vs_CUR=out, meta=dict(runtime_s=round(time.time() - t_all, 1), maxrss_mb=rss()))
(H / 'results_posthoc.json').write_text(json.dumps(res, ensure_ascii=False, indent=1))
print(json.dumps(res, ensure_ascii=False))
