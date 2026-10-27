#!/usr/bin/env python3
"""獨立重建主策略：三組期間 × n30/n10 × 引擎選項。輸出 results_main.json、navs.parquet、picks.json。"""
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import ia_core as K

OUT = K.HERE
T0 = time.time()
res = {}; navs = {}; picks = {}

def run(tag, Y, start, a, b, sigmode='strict', freeze=True, delist_zero=False, ns=(30, 10), slips=(0.001,), use_disposal=True, block=True, w52=0.5):
    S = K.Strategy(Y, start, sigmode=sigmode, use_disposal=use_disposal, w52=w52)
    t0 = K.IDX.get_loc(S.sig_days[0]); t1 = K.IDX.get_loc(S.end_day)
    sd = S.sig_days + [S.end_day]
    out = {}
    for n in ns:
        pl = S.plan(n, block=block)
        ew, _ = K.simulate([(p[0], p[2]) for p in pl], 0.0, 0.0, t0, t1, freeze=freeze)
        navs[f'{tag}|EW'] = ew
        picks[f'{tag}|n{n}'] = {str(K.IDX[p[0]].date()): [K.COLS[j] for j in p[1]] for p in pl}
        for slip in slips:
            bb, ss = K.rates(slip)
            nav, lg = K.simulate([(p[0], p[1]) for p in pl], bb, ss, t0, t1, freeze=freeze, delist_zero=delist_zero)
            key = f'{tag}|n{n}|{slip}'; navs[key] = nav
            o = K.stats(nav, a, b, sd, K.E50, ew)
            o['turnover'] = round(lg['turnover'], 3); o['blocked'] = S.info[f'blocked_n{n}']; o['frozen_events'] = lg['frozen'][:50]
            o['n_frozen_events'] = len(lg['frozen'])
            out[f'n{n}|{slip}'] = o
            print(key, K.short(o), f'{time.time()-T0:.0f}s', flush=True)
    res[tag] = out
    return S

if __name__ == '__main__':
    YL = K.yoy_from_long(); Y5 = K.yoy_from_rev5y()
    run('P2_rev5y', Y5, '2021-01-01', '2021-03-01', '2026-10-31', slips=(0.001, 0.003, 'stress'))
    run('P2_rev5y_nofreeze', Y5, '2021-01-01', '2021-03-01', '2026-10-31', freeze=False)
    run('P2_rev5y_lenient', Y5, '2021-01-01', '2021-03-01', '2026-10-31', sigmode='lenient')
    run('P2_long', YL, '2021-01-01', '2021-03-01', '2026-10-31')
    run('P1_long', YL, '2005-01-01', '2006-04-01', '2020-12-31', slips=(0.001, 0.003, 'stress'))
    run('P1_long_nofreeze', YL, '2005-01-01', '2006-04-01', '2020-12-31', freeze=False)
    run('P1_long_delist0', YL, '2005-01-01', '2006-04-01', '2020-12-31', delist_zero=True)
    run('P1_long_lenient', YL, '2005-01-01', '2006-04-01', '2020-12-31', sigmode='lenient')
    (OUT / 'results_main.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    pd.DataFrame(navs).to_parquet(OUT / 'navs.parquet')
    (OUT / 'picks.json').write_text(json.dumps(picks, ensure_ascii=False), encoding='utf-8')
    print('done', time.time() - T0)
