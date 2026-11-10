# -*- coding: utf-8 -*-
"""P3：30 萬整股主欄（W30）投入比率與 2021 起 W30−N100（orig 名單）。python3 -I -B rp3.py"""
import sys, os, json
import numpy as np, pandas as pd
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rengine as RE
SCR = '/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/rep_o19'
_D = None
def init():
    global _D; _D = RE.Data(f'{SCR}/rdata_strict.npz')
def job(a):
    cell, j, s, plan = a
    cfg = {'W30': RE.Cfg(30e4, False, True), 'N100': RE.Cfg(100e4), 'O30': RE.Cfg(30e4)}[cell]
    pl = [(_D.dpos[np.datetime64(x)], _D.dpos[np.datetime64(y)], [_D.col[c] for c in z.split(',')]) for x, y, z in plan]
    inv = np.full(_D.T, np.nan)
    s0, eq = RE.run_sim(_D, pl, cfg, inv=inv)
    d = _D.dates
    m1 = (d < np.datetime64('2020-10-26')); m2 = (d >= np.datetime64('2021-01-01'))
    m1[:s0 + 1] = False
    return cell, j, s, float(np.nanmean(inv[m1])), float(np.nanmean(inv[m2]))
if __name__ == '__main__':
    p = pd.read_csv(RE.AG + '/o_v7_listing_exec_engine/data/plans_j7.csv', dtype={'picks': str})
    p = p[p.strat == 'AB_V7_B10']
    tasks = []
    for cell in ['W30', 'O30', 'N100']:
        for j in range(13):
            for s in [0, 1, 2, 3]:
                g = p[(p.j == j) & (p.seed == s)].sort_values('signal')
                tasks.append((cell, j, s, list(zip(g.signal, g['exec'], g.picks))))
    with Pool(4, initializer=init) as pool:
        res = pool.map(job, tasks)
    out = {}
    for cell in ['W30', 'O30', 'N100']:
        a = [r[3] for r in res if r[0] == cell]; b = [r[4] for r in res if r[0] == cell]
        out[cell] = dict(inv_2006_2020_10=float(np.mean(a)), inv_2021on=float(np.mean(b)), n_sims=len(a))
    # 2021 起年化差（20 seeds 淨值）
    z = np.load(f'{SCR}/rdata_strict.npz'); dates = z['dates'].astype('datetime64[D]')
    ends = json.load(open(RE.AG + '/o_v7_listing_exec_engine/data/jscan_end_days.json'))
    def navj(cell, j):
        q = np.load(f'{SCR}/navs_{cell}_strict.npz'); eqs = q[f'eq_j{j}']; s0 = int(q[f's0_j{j}'])
        r = (eqs[:, s0 + 1:] / eqs[:, s0:-1] - 1).mean(0); nav = np.full(len(dates), np.nan); nav[s0] = 1; nav[s0 + 1:] = np.cumprod(1 + r); return nav
    a0 = int(np.searchsorted(dates, np.datetime64('2021-01-01'))) - 1
    diffs = []
    for j in range(13):
        e = int(np.searchsorted(dates, np.datetime64(ends[f'AB_V7_B10|J{j}'])))
        yrs = (dates[e] - dates[a0]).astype(int) / 365.25
        aw = navj('W30', j); an = navj('N100', j)
        diffs.append(((aw[e] / aw[a0]) ** (1 / yrs) - (an[e] / an[a0]) ** (1 / yrs)) * 100)
    out['diff_2021on_W30_minus_N100_med'] = float(np.median(diffs)); out['per_j'] = diffs
    print(json.dumps(out, indent=1))
    json.dump(out, open(f'{SCR}/p3.json', 'w'), indent=1)
