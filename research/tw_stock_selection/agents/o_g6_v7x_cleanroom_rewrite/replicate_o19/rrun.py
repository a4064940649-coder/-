# -*- coding: utf-8 -*-
"""批次：orig 名單（plans_j7 AB_V7_B10，j0～12、seeds 0～19）× 本引擎。
python3 -I -B rrun.py CELL [datamode] [jlist] [nseeds]
CELL: N100 N300 N1000 S100 W30 O30
"""
import sys, os, time, json
import numpy as np, pandas as pd
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rengine as RE

SCR = '/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/rep_o19'
CELLS = {'N100': (100e4, False, False), 'N300': (300e4, False, False), 'N1000': (1000e4, False, False),
         'S100': (100e4, True, False), 'W30': (30e4, False, True), 'O30': (30e4, False, False)}
_D = None


def init(path):
    global _D
    _D = RE.Data(path)


def load_plans(D, j, seed):
    p = pd.read_csv(RE.AG + '/o_v7_listing_exec_engine/data/plans_j7.csv', dtype={'picks': str})
    return p


def job(args):
    cell, j, seed, plan = args
    tier, stress, w30 = CELLS[cell]
    cfg = RE.Cfg(tier, stress, w30, os.environ.get('RVAR', ''))
    pl = []
    for sig, exe, picks in plan:
        si = _D.dpos[np.datetime64(sig)]; ei = _D.dpos[np.datetime64(exe)]
        pl.append((si, ei, [_D.col[c] for c in picks.split(',')]))
    s0, eq = RE.run_sim(_D, pl, cfg)
    return cell, j, seed, s0, eq


if __name__ == '__main__':
    cell = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else 'strict'
    js = [int(x) for x in sys.argv[3].split(',')] if len(sys.argv) > 3 and sys.argv[3] != 'all' else list(range(13))
    ns = int(sys.argv[4]) if len(sys.argv) > 4 else 20
    path = f'{SCR}/rdata_{mode}.npz'
    p = pd.read_csv(RE.AG + '/o_v7_listing_exec_engine/data/plans_j7.csv', dtype={'picks': str})
    p = p[p.strat == 'AB_V7_B10']
    tasks = []
    for j in js:
        for s in range(ns):
            g = p[(p.j == j) & (p.seed == s)].sort_values('signal')
            tasks.append((cell, j, s, list(zip(g.signal, g['exec'], g.picks))))
    t0 = time.time()
    res = {}
    with Pool(4, initializer=init, initargs=(path,)) as pool:
        for cell_, j, s, s0, eq in pool.imap_unordered(job, tasks):
            res[(j, s)] = (s0, eq)
    out = {}
    for j in js:
        eqs = np.array([res[(j, s)][1] for s in range(ns)])
        s0 = res[(j, 0)][0]
        r = eqs[:, 1:] / eqs[:, :-1] - 1
        out[f'eq_j{j}'] = eqs
        out[f's0_j{j}'] = np.array(s0)
    np.savez(f'{SCR}/navs_{cell}_{mode}{os.environ.get("RTAG", "")}.npz', **out)
    print('done', cell, mode, len(tasks), 'sims', round(time.time() - t0, 1), 's')
