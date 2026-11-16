#!/usr/bin/env python3
"""S32 事後描述（驗證期判定之後；未登記，不改判定）：4 條候選＋對照的全期連續主跑（位移 0、100 萬）與 20 條路徑中位數：
2019 起／全期年化、最大回撤、Calmar、最賺 5 筆占比、CAPM β、單檔最大權重、有效持股數。輸出 results_posthoc.json。python3 -I -B posthoc.py"""
import sys, json, time, resource, gc
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s32lib
from s32lib import s17lib, s10lib, s4lib, stlib, COSTS

H = Path(__file__).resolve().parent
assert (H / 'results.json').exists(), '驗證期還沒跑：事後描述必須在判定之後'
t_all = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
P = s32lib.load()
INNOV, _ = s4lib.innov_panel(P)
U = stlib.universe(P, 'twse') & ~INNOV
del INNOV
RT = s32lib.surge(P)
s17lib.e50_fix_suspension(P)
bench = pd.Series(P.E50[P.t0:], index=P.IDX[P.t0:])
lastb = lambda d: P.IDX[P.IDX < pd.Timestamp(d)][-1]
D0, DE, D19 = P.IDX[P.t0], P.IDX[-1], lastb('2019-01-01')
CELL = dict(sat_w=0.7, sizing='W10', lock=False)
CFG = {'CUR': None, **{r['id']: r for r in s32lib.RULES}, s32lib.CONTROL['id']: s32lib.CONTROL}
out = {c: {} for c in CFG}
for s in range(4):
    S = s10lib.make_signal_s10(P, U, RT, shift=s)
    for c, rule in CFG.items():
        S1, _ = s32lib.build_wmul(P, S, rule)
        S2, ekw = s10lib.variant_S(S1, 'E0')
        for cap in (6e5, 8e5, 1e6, 1.5e6, 2e6):
            nav, lg, st = s32lib.simulate_w(P, S2, U, cost=COSTS['strict'], capital=cap, **CELL, **ekw)
            npd = st['nposd']
            out[c][f'{int(cap/1e4)}萬|+{s}'] = dict(since2019=stlib.cagr(nav, D19, DE), full=stlib.cagr(nav, D0, DE), maxdd=stlib.maxdd(nav, D0, DE),
                                                    calmar=s32lib.calmar(nav, D0, DE), top5=s17lib.top5_share(lg, nav, cap), beta=s32lib.capm_beta(nav, bench, D0, DE)['beta'],
                                                    maxw_max=float(st['maxw'].max()), maxw_mean=float(st['maxw'][npd > 0].mean()), effn=float((1 / st['hhi'][npd > 0]).mean()),
                                                    seg=s17lib.seg_dd(nav))
            del nav, lg, st
    gc.collect()
keys = list(out['CUR'])
res = {'main_100萬_+0': {c: out[c]['100萬|+0'] for c in CFG}, 'paths_median': {}, 'paired_vs_CUR': {}}
for c in CFG:
    res['paths_median'][c] = {f: float(np.median([out[c][k][f] for k in keys])) for f in out[c][keys[0]] if f != 'seg'}
    if c != 'CUR':
        o = {}
        for f in ('since2019', 'full', 'maxdd', 'calmar', 'top5', 'beta', 'maxw_max', 'effn'):
            d = np.array([out[c][k][f] - out['CUR'][k][f] for k in keys])
            o[f] = dict(median=float(np.median(d)), npos=int((d > 0).sum()))
        for sg in s17lib.SEGMENTS:
            d = np.array([out[c][k]['seg'][sg] - out['CUR'][k]['seg'][sg] for k in keys])
            o['seg_' + sg] = dict(median=float(np.median(d)), npos=int((d > 0).sum()))
        res['paired_vs_CUR'][c] = o
res['meta'] = dict(runtime_s=round(time.time() - t_all, 1), maxrss_mb=rss(), note='事後描述，不參與選擇與判定；全期連續（訓練＋驗證混在一起）')
(H / 'results_posthoc.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
for c in CFG:
    print(c, {k: round(v, 4) for k, v in res['paths_median'][c].items()})
for c, o in res['paired_vs_CUR'].items():
    print(c, {k: (round(v['median'], 4), v['npos']) for k, v in o.items()})
print(res['meta'])
