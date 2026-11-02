#!/usr/bin/env python3
"""replicate_o10 r2：用 V7X x_exec.py（只 import）跑 13 個 j。
用法：r2_run.py VAR[,VAR] CELL[,CELL] [TAG]；CELL＝100G／100F／300G／300F／1000G／1000F（＋S＝壓力成本，例如 100GS）。
環境變數：OTPX_DELIST_F（下櫃出清比例）、OTPX_OA_MULT（上櫃開盤量占比倍數）。輸出 scratchpad/rep_o10/nav_<VAR>_<CELL><TAG>.pkl"""
import sys, os, pickle, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import multiprocessing as mp
import x_data as D
sys.path.append(str(HERE.parents[1] / 'o_exec_engine_rewrite_v7x'))
import x_exec as X

OAM = float(os.environ.get('OTPX_OA_MULT', '1.0'))
if OAM != 1.0:
    X.OASH = np.where(D.TPX, X.OASH * OAM, X.OASH)


def cfg_of(cell):
    st = cell.endswith('S'); c = cell[:-1] if st else cell
    tier = {'100': 1e6, '300': 3e6, '1000': 1e7, '30': 3e5, '50': 5e5}[c[:-1]]
    return dict(tier=tier, mode={'G': 'GROW', 'F': 'FIX'}[c[-1]], model='K5T', cost='stress' if st else 'normal',
                f1=True, f2=True, f3=0.5, f4=True)


_JOBS = {}


def _one(key):
    var, cell, j = key
    pl = _JOBS[var][j]
    nav, acc = X.run_one(pl, cfg_of(cell))
    return key, nav.astype('float64'), acc


if __name__ == '__main__':
    vs = sys.argv[1].split(','); cells = sys.argv[2].split(','); tag = sys.argv[3] if len(sys.argv) > 3 else ''
    for v in vs:
        _JOBS[v] = pickle.load(open(D.SCR / f'plans_{v}.pkl', 'rb'))
    keys = [(v, c, j) for v in vs for c in cells for j in range(13)]
    t0 = time.time(); res = {}
    with mp.get_context('fork').Pool(4) as pool:
        for key, nav, acc in pool.imap_unordered(_one, keys):
            res[key] = (nav, acc)
    for v in vs:
        for c in cells:
            pickle.dump({j: res[(v, c, j)] for j in range(13)}, open(D.SCR / f'nav_{v}_{c}{tag}.pkl', 'wb'))
    print('done', len(keys), round(time.time() - t0), flush=True)
