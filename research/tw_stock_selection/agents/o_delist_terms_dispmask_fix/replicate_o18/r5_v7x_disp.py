#!/usr/bin/env python3
"""o18 題 2 獨立複製（步驟 5）：V7X 名單 orig／union／replace（r4 重建）× 引擎 orig／v3（r2 規則）× 4 格。逐 j 統計全部存檔。"""
import sys, json, time
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r2_v7x as R2
sys.path.pop(0)
D, Z, S = R2.D, R2.Z, R2.S
WK = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/o18_delistterms/replicate_o18')

def load_plans(kind):
    d = pd.read_csv(WK / f'plans_{kind}.csv', dtype={'picks': str})
    out = {}
    for (j, s), g in d.groupby(['j', 'seed'], sort=False):
        out[(int(j), int(s))] = sorted([(int(e), np.array([D.CI[c] for c in pk.split(',')], int), D.TPOS[pd.Timestamp(sg)]) for sg, e, pk in zip(g.signal, g.exec_idx, g.picks)], key=lambda x: x[0])
    return out

ENG = {}; PL = {}
def _job(a):
    kind, key, eng, cell = a
    m, co, tier, tl = cell
    cfg = dict(R2.BASE, tier=tier, model=m, cost=co); st = ENG[eng]
    s = Z.CSim(PL[kind][key], cfg, settle=st, cmode='orig' if st is None else 'consideration')
    return kind, key, eng, tl + '|' + co, s.run()

if __name__ == '__main__':
    t0 = time.time()
    ENG['orig'] = None
    for b in ('main', 'lo', 'hi'): ENG['v3_' + b] = R2.v3_settle(b)
    for k in ('orig', 'union', 'replace'): PL[k] = load_plans(k)
    PLAN_ENG = {'orig': ['orig', 'v3_main'], 'union': ['orig', 'v3_main', 'v3_lo', 'v3_hi'], 'replace': ['orig', 'v3_main']}
    jobs = [(k, key, e, c) for k, es in PLAN_ENG.items() for e in es for c in R2.CELLS for key in PL[k]]
    navs = {}
    with Pool(4) as pool:
        for k, key, e, cl, nav in pool.imap_unordered(_job, jobs, chunksize=4):
            navs.setdefault((k, e, cl, key[0]), []).append(nav)
    print('runs', round(time.time() - t0), flush=True)
    res = {}
    for (k, e, cl, j) in sorted({x[:3] + (0,) for x in navs}):
        per = [S.stats(S.avg_nav(navs[(k, e, cl, jj)]), 'AB_V7_B10', jj) for jj in range(13)]
        sm = S.jsumm(per)
        r = {f: v for f, v in sm.items() if f.endswith('_med') or f.endswith('_p10')}
        r['by_j'] = {f: [p[f] for p in per] for f in ('exc', 't_0050', 'exc_Y19on', 't_Y19on', 'exc_SEL')}
        res[f'{k}|{e}|{cl}'] = r
    # Δ union／replace 對 orig（同引擎）
    dl = {}
    for k in ('union', 'replace'):
        for e in PLAN_ENG[k]:
            for c in R2.CELLS:
                cl = c[3] + '|' + c[1]
                if f'orig|{e}|{cl}' not in res: continue
                a, b = res[f'{k}|{e}|{cl}'], res[f'orig|{e}|{cl}']
                dd = {}
                for f in ('exc', 'exc_Y19on', 't_0050'):
                    x = np.array(a['by_j'][f]) - np.array(b['by_j'][f])
                    dd[f + '_med_diff'] = a[f + '_med'] - b[f + '_med']; dd[f + '_diff_med'] = float(np.median(x)); dd[f + '_diff_mean'] = float(np.mean(x))
                dl[f'{k}-orig|{e}|{cl}'] = dd
    res['delta'] = dl
    (HERE / 'r5_v7x_disp.json').write_text(json.dumps(res, ensure_ascii=False, indent=1))
    for k, v in res.items():
        if 'exc_med' in v: print(k, *(f'{x}={v[x]:.3f}' for x in ('exc_med', 't_0050_med', 'exc_Y19on_med')))
    for k, v in dl.items(): print(k, {a: round(b, 3) for a, b in v.items()})
    print('time', round(time.time() - t0))
