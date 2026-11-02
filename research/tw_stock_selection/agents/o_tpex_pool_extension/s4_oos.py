#!/usr/bin/env python3
"""第 4 步：樣本外（全期模擬一次，不再改規格）。先斷言 PREREG 存在、frozen_spec.txt 雜湊相符。
用法：python3 -I -B s4_oos.py main         → out/navs_oos.parquet、out/acc_oos.json
      OTPX_DELIST_F=0.5 python3 -I -B s4_oos.py sens_d05   → out/navs_sens_d05.parquet（M、BASE 的 100 萬 GROW 一般）
      OTPX_DELIST_F=0.0 python3 -I -B s4_oos.py sens_d00
      python3 -I -B s4_oos.py sens_oa05     → 上櫃開盤量占比 ×0.5"""
import sys, json, hashlib, time
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
RES = json.loads((HERE / 'results.json').read_text())
assert 'PREREG' in RES, 'PREREG 不存在'
assert hashlib.sha256((HERE / 'frozen_spec.txt').read_bytes()).hexdigest() == RES['PREREG']['spec_sha256'], 'frozen_spec 雜湊不符'
import otlib as L
PR = RES['PREREG']; IT = json.loads((HERE / 'iterations.json').read_text())
MODE = sys.argv[1]
if MODE == 'sens_oa05':
    L.X.OASH[L.D.MK] *= 0.5
FULL = [(t, m, c) for t in ('100萬', '300萬', '1000萬') for m in ('FIX', 'GROW') for c in ('normal', 'stress')]
SAT = [('30萬', 'GROW', 'normal'), (5e5, 'GROW', 'normal')]
ONE = [('100萬', 'GROW', 'normal')]
if MODE == 'main':
    V = {'M': (PR['main_spec'], FULL), 'R1': (PR['report']['R1'], FULL), 'R2': (PR['report']['R2'], FULL + SAT), 'BASE': (PR['base'], FULL + SAT)}
    for k, v in PR['neighbors'].items(): V['NB_' + k] = (v, ONE)
    for k, v in IT['PLAN']['variants'].items():
        if k not in ('v4_ALL400_n10_cap50', 'v3_ALL500_n10', 'v8_TP300_n10'): V['IT_' + k] = (v, ONE)
else:
    V = {'M': (PR['main_spec'], ONE), 'BASE': (PR['base'], ONE), 'R2': (PR['report']['R2'], ONE)}


def job(a):
    name, j = a
    v, cells = V[name]
    pl = L.gen_plan(v, j)
    out = {}; acc = {}
    for t, m, c in cells:
        nav, ac = L.sim(pl, L.cfg(t, m, c))
        tn = t if isinstance(t, str) else f'{int(t / 1e4)}萬'
        k = f'{name}|J{j}|{tn}|{m}|{c}'
        out[k] = nav; acc[k] = {kk: float(vv) for kk, vv in ac.items()}
    ntp = float(np.mean([L.D.MK[sg, pk].mean() for e, pk, sg in pl]))
    return out, acc, (name, j, ntp)


if __name__ == '__main__':
    t0 = time.time()
    jobs = [(k, j) for k in V for j in range(13)]
    navs = {}; accs = {}; tpshare = {}
    with Pool(4) as P:
        for out, acc, (nm, j, ntp) in P.imap_unordered(job, jobs):
            navs.update(out); accs.update(acc); tpshare[f'{nm}|J{j}'] = ntp
    pd.DataFrame(navs).sort_index(axis=1).to_parquet(HERE / 'out' / f'navs_{MODE if MODE != "main" else "oos"}.parquet')
    (HERE / 'out' / f'acc_{MODE if MODE != "main" else "oos"}.json').write_text(json.dumps(dict(acc=accs, tpex_share=tpshare), ensure_ascii=False))
    print(MODE, len(navs), 'cells', round(time.time() - t0), 's')
