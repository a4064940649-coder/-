#!/usr/bin/env python3
"""第 2 步（資料產生，不算任何統計量）：用正式 V7X 引擎（LogSim 全開＝CSim）產生
  (1) seed 0 逐筆成交日誌：REAL_{opt,p90,carry} × 13 j × 6 格（100 萬一般／壓力、30 萬一般／壓力、300 萬一般、1,000 萬一般）；
  (2) 100 萬一般 × 3 變體 × 13 j × 20 seeds 的淨值（重現正式數字用；20 seeds 平均＝x_stats.avg_nav）。
  斷言：抽 6 個模擬，LogSim(log=True) 淨值與 cz_v7x.CSim 逐位相同（|差| ≤ 1e−12）。
輸出（scratchpad）：fills_<cell>.parquet、rebs_<cell>.pkl、navs_seed0.parquet、navs_100n_avg.parquet。
python3 -I -B run_logs.py"""
import sys, json, time, pickle, hashlib
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); import tfc_lib as TL; sys.path.pop(0)
SCR = TL.SCR
CELLS = [('normal', '100萬'), ('stress', '100萬'), ('normal', '30萬'), ('stress', '30萬'), ('normal', '300萬'), ('normal', '1000萬')]
VARS = ['REAL_opt', 'REAL_p90', 'REAL_carry']
_G = {}
def init():
    E = TL.engine(); _G['E'] = E
    P = TL.load_plans(); _G['G'] = {k: g for k, g in P.groupby(['variant', 'j', 'seed'])}
def job(a):
    kind, v, j, s, cell = a
    E = _G['E']; D = E['D']; g = _G['G'][(v, j, s)]; pl = TL.to_plan(g, D)
    cfg = TL.cfg_for(E, cell[1], cell[0])
    if kind == 'log':
        sim = E['LogSim'](pl, dict(cfg), settle=E['ST'], cmode='consideration', log=True); nav = sim.run()
        chk = None
        if j in (0, 6) and cell == ('normal', '100萬') or (j == 3 and cell == ('normal', '30萬')):
            ref = E['CV'].CSim(pl, dict(cfg), settle=E['ST'], cmode='consideration').run()
            chk = float(np.nanmax(np.abs(ref.values - nav.values)))
        return kind, v, j, s, cell, nav, sim.fills, sim.rebs, dict(sim.acc), chk
    sim = E['LogSim'](pl, dict(cfg), settle=E['ST'], cmode='consideration'); nav = sim.run()
    return kind, v, j, s, cell, nav, None, None, dict(sim.acc), None
if __name__ == '__main__':
    t0 = time.time()
    jobs = [('log', v, j, 0, c) for c in CELLS for v in VARS for j in range(13)]
    jobs += [('nav', v, j, s, ('normal', '100萬')) for v in VARS for j in range(13) for s in range(1, 20)]
    FL = {c: [] for c in CELLS}; RB = {c: {} for c in CELLS}; NV0 = {}; NV20 = {}; ACC = {}; CHK = []
    with Pool(4, initializer=init) as P:
        for n, (kind, v, j, s, cell, nav, fills, rebs, acc, chk) in enumerate(P.imap_unordered(job, jobs, chunksize=2)):
            if kind == 'log':
                f = pd.DataFrame(fills, columns=TL.FILL_COLS); f.insert(0, 'j_', j); f.insert(0, 'variant', v); FL[cell].append(f)
                RB[cell][(v, j)] = rebs; NV0['|'.join((v, str(j)) + cell)] = nav; ACC['|'.join((v, str(j)) + cell)] = acc
                if chk is not None: CHK.append(((v, j) + cell, chk))
            if cell == ('normal', '100萬'): NV20[(v, j, s)] = nav
            if n % 100 == 0: print(n, len(jobs), round(time.time() - t0), flush=True)
    assert len(CHK) >= 6 and max(c for _, c in CHK) <= 1e-12, CHK
    E = TL.engine(); S = E['S']
    for c in CELLS:
        F = pd.concat(FL[c], ignore_index=True); F.to_parquet(SCR / f'fills_{c[0]}_{c[1]}.parquet')
        pickle.dump(RB[c], open(SCR / f'rebs_{c[0]}_{c[1]}.pkl', 'wb'))
    pd.DataFrame(NV0).to_parquet(SCR / 'navs_seed0.parquet')
    AV = {f'{v}|{j}': S.avg_nav([NV20[(v, j, s)] for s in range(20)]) for v in VARS for j in range(13)}
    pd.DataFrame(AV).to_parquet(SCR / 'navs_100n_avg.parquet')
    out = dict(utc=pd.Timestamp.now('UTC').isoformat(), runtime_s=round(time.time() - t0, 1), n_jobs=len(jobs),
               identity_check_maxabs=[(list(k), c) for k, c in CHK], acc_seed0=ACC)
    (HERE / 'run_logs_meta.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    print('done', round(time.time() - t0), CHK)
