#!/usr/bin/env python3
"""同分排序的影響：分數＝0.5×百分位＋0.5×百分位，常常剛好同分落在第 10／30 名邊界。比較不同同分規則與 30 組隨機。"""
import sys, json; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd, ia_core as K
res = {}
SETS = {'P2_rev5y': (K.yoy_from_rev5y(), '2021-01-01', '2021-03-01', '2026-10-31'),
        'P1_long': (K.yoy_from_long(), '2005-01-01', '2006-04-01', '2020-12-31')}
b, s = K.rates(0.001)
for tag, (Y, st, a, e) in SETS.items():
    res[tag] = {}
    # 邊界同分頻率
    S = K.Strategy(Y, st)
    t0 = K.IDX.get_loc(S.sig_days[0]); t1 = K.IDX.get_loc(S.end_day); sd = S.sig_days + [S.end_day]
    tie = {10: 0, 30: 0}; tot = 0
    for d in S.sig_days:
        if d > pd.Timestamp(e): break
        cs = S.cross_section(d); sc = np.sort(np.round(cs['sc'][cs['el']], 10))[::-1]; tot += 1
        for n in (10, 30):
            if len(sc) > n and sc[n - 1] == sc[n]: tie[n] += 1
    res[tag]['boundary_tie_share'] = {n: round(tie[n] / tot, 3) for n in tie}
    print(tag, 'boundary ties', res[tag]['boundary_tie_share'], flush=True)
    for n in (30, 10):
        out = {}
        ew = None
        for tb in ['stable', 'reverse', 's52', 'yoy'] + [f'rand:{i}' for i in range(30)]:
            S = K.Strategy(Y, st, tiebreak=tb)
            pl = S.plan(n)
            if ew is None: ew, _ = K.simulate([(p[0], p[2]) for p in pl], 0, 0, t0, t1)
            nav, _ = K.simulate([(p[0], p[1]) for p in pl], b, s, t0, t1)
            o = K.stats(nav, a, e, sd, K.E50, ew)
            out[tb] = (round(o['CAGR'] * 100, 2), round(o['t_0050'], 2), round(o['t_EW'], 2))
        rc = np.array([v[0] for k, v in out.items() if k.startswith('rand')]); rt = np.array([v[1] for k, v in out.items() if k.startswith('rand')])
        res[tag][f'n{n}'] = {'rules': {k: v for k, v in out.items() if not k.startswith('rand')},
                             'random30': {'CAGR_min': rc.min(), 'CAGR_p50': float(np.median(rc)), 'CAGR_max': rc.max(), 'CAGR_sd': round(float(rc.std()), 2),
                                          't0050_min': rt.min(), 't0050_max': rt.max()}}
        print(tag, n, res[tag][f'n{n}'], flush=True)
(K.HERE / 'results_tiebreak.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding='utf-8')
