#!/usr/bin/env python3
"""跑一個設定（run）：B10 j 0～12 × seeds 0～19 × 32 格；存 j 平均淨值與統計。"""
import sys, json, time, pickle
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import d8, e8, s8, g8_zero as Z
from multiprocessing import Pool
CELLS = [(m, c, ti, a) for m in ('K5T', 'TICK') for c in ('normal', 'stress') for ti in ('30萬', '100萬', '300萬', '1000萬') for a in ('FIX', 'GROW')]
RUNS = {
 'R0': {},
 'R1': dict(F1=True),
 'R2': dict(F1=True, F2=True),
 'R3': dict(F1=True, F2=True, F3=0.5),
 'R4': dict(F1=True, F2=True, F3=0.5, F4=True),
 'R4c': dict(F1=True, F2=True, F3=0.5, F4=True, f2fix=True),
 'R0_nan': dict(zcap_nan='none'),
 'R4_z025': dict(F1=True, F2=True, F3=0.25, F4=True),
 'R4_z100': dict(F1=True, F2=True, F3=1.0, F4=True),
}
PJ = None
def job(a):
    key, run, cells = a; st, j, sd = key
    plan = PJ[key]; t1 = e8.TPOS[pd.Timestamp(s8.ENDS[f'{st}|J{j}'])]
    out = {}; acc = {}
    for cell in cells:
        m, c, ti, ac = cell
        cfg = dict(model=m, cost=c, tier=ti, acct=ac, **RUNS[run]); cfg['track_qoa'] = (ti == '1000萬' and ac == 'GROW' and m == 'K5T' and c == 'normal')
        S = e8.Sim(plan, t1, cfg); r = S.run(); out[cell] = r.to_numpy()
        acc[cell] = dict(n_disp=S.acc['n_disp'], n_mdelay=S.acc['n_mdelay'], n_cap=S.acc['n_cap'],
                         qoa_med=float(np.median(S.acc['qoa'])) if S.acc['qoa'] else None, qoa_p90=float(np.quantile(S.acc['qoa'], 0.9)) if S.acc['qoa'] else None)
    return key, out, acc, (plan[0][0] - 1, t1)
if __name__ == '__main__':
    run = sys.argv[1]; cells = CELLS if len(sys.argv) < 3 else [c for c in CELLS if sys.argv[2] in '|'.join(c)]
    js = range(13); seeds = range(20)
    if len(sys.argv) >= 4: js = [int(x) for x in sys.argv[3].split(',')]
    if len(sys.argv) >= 5: seeds = range(int(sys.argv[4]))
    PJ = {k: v for k, v in Z.load_plans('j').items() if k[0] == 'AB_V7_B10' and k[1] in js and k[2] in seeds}
    tt = time.time()
    with Pool(4) as P: R = P.map(job, [(k, run, cells) for k in PJ], chunksize=1)
    navs = {}; stats = {}; accs = {}
    for j in js:
        rr = [r for r in R if r[0][1] == j]; t0, t1 = rr[0][3]
        sd = [p[1] for p in PJ[rr[0][0]]]
        for cell in cells:
            m = np.mean([r[1][cell] for r in rr], axis=0); m[0] = 0.0
            nav = pd.Series(np.cumprod(1 + m), index=e8.IDX[t0:t1 + 1])
            navs[(j,) + cell] = nav
            stats['|'.join((f'J{j}',) + cell)] = s8.cell_stats(nav, sd, s8.ENDS[f'AB_V7_B10|J{j}'])
            a = [r[2][cell] for r in rr]
            accs['|'.join((f'J{j}',) + cell)] = dict(n_disp=float(np.mean([x['n_disp'] for x in a])), n_mdelay=float(np.mean([x['n_mdelay'] for x in a])),
                 n_cap=float(np.mean([x['n_cap'] for x in a])), qoa_med=(float(np.median([x['qoa_med'] for x in a])) if a[0]['qoa_med'] is not None else None),
                 qoa_p90=(float(np.median([x['qoa_p90'] for x in a])) if a[0]['qoa_p90'] is not None else None))
    tag = run if len(sys.argv) < 3 else run + '_' + '_'.join(sys.argv[2:])
    pickle.dump(dict(navs=navs, stats=stats, acc=accs), open(HERE / 'out' / f'run_{tag}.pkl', 'wb'))
    print(run, 'done', time.time() - tt, flush=True)
