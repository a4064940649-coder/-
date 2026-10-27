#!/usr/bin/env python3
"""診斷：用修正前的 value（value_orig_dupbug.parquet）重跑，確認 exec_rebuild 47.3%/57.5% 與本稽核差距的來源。"""
import sys, json; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd, ia_core as K
res = {}
b, s = K.rates(0.001)
for tag, vfile in (('value_fixed', 'value.parquet'), ('value_dupbug', 'value_orig_dupbug.parquet')):
    K.V = pd.read_parquet(K.P / vfile); K._RS.clear()
    for ytag, Y, st, a, e in (('P2_rev5y', K.yoy_from_rev5y(), '2021-01-01', '2021-03-01', '2026-10-31'), ('P2_long', K.yoy_from_long(), '2021-01-01', '2021-03-01', '2026-10-31')):
        for tb in ('stable', 'reverse'):
            S = K.Strategy(Y, st, tiebreak=tb)
            t0 = K.IDX.get_loc(S.sig_days[0]); t1 = K.IDX.get_loc(S.end_day); sd = S.sig_days + [S.end_day]
            for n in (30, 10):
                pl = S.plan(n)
                ew, _ = K.simulate([(p[0], p[2]) for p in pl], 0, 0, t0, t1)
                nav, _ = K.simulate([(p[0], p[1]) for p in pl], b, s, t0, t1)
                o = K.stats(nav, a, e, sd, K.E50, ew)
                res[f'{tag}|{ytag}|{tb}|n{n}'] = K.short(o)
                print(tag, ytag, tb, n, K.short(o), flush=True)
(K.HERE / 'results_oldvalue.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding='utf-8')
