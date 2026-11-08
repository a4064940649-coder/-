#!/usr/bin/env python3
"""o17 獨立複製 r5：(a) c50 另一解讀（入帳後到再投入前仍以 0050 計值，A16 式）；(b) 2432 改判 unknown 的 lo。只跑 100萬 一般。"""
import sys, json
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import r3_v7x as R
S = R.S
class SimA16(R.SimC):
    pass
def job(args):
    key, pl = args
    cfg = dict(R.BASE, tier=1e6, cost='normal')
    out = {}
    R.UNKNOWN_EXTRA.clear(); R.UNKNOWN_EXTRA.add('2432')
    out['lo_2432unk'] = R.SimC(pl, dict(cfg, variant='lo')).run()
    out['hi_2432unk'] = R.SimC(pl, dict(cfg, variant='hi')).run()
    R.UNKNOWN_EXTRA.clear()
    return key, out
if __name__ == '__main__':
    P = R.D.plans('AB_V7_B10', js=list(range(13)), seeds=list(range(20)))
    navs = {}
    with Pool(4) as pool:
        for key, out in pool.imap_unordered(job, sorted(P.items()), chunksize=2):
            for k, n in out.items(): navs.setdefault((key[0], k), []).append(n)
    base = pd.read_parquet(HERE / 'r3_navs_v7x.parquet')
    res = {}
    for e in ('lo_2432unk', 'hi_2432unk', 'orig', 'lo'):
        per = []
        for j in range(13):
            nav = S.avg_nav(navs[(j, e)]) if (j, e) in navs else base[f'B10|J{j}|K5T|normal|100萬|GROW|{e}']
            per.append(S.stats(nav, 'AB_V7_B10', j))
        sm = S.jsumm(per); res[e] = {k: sm[k] for k in ('exc_med', 't_0050_med', 'exc_Y19on_med')}
    for e in ('lo_2432unk', 'hi_2432unk', 'lo'):
        res[e + '_delta'] = {k: res[e][k] - res['orig'][k] for k in res['orig']}
    print(json.dumps(res, ensure_ascii=False, indent=1))
    (HERE / 'r5_v7x_sens.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
