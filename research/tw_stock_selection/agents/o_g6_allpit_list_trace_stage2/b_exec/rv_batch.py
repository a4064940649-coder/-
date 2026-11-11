# B 對齊實驗：複製員引擎修改副本（rengine_v）× orig 名單；RVAR 控制規則對齊，ALIGN=1 時以原引擎 VRp/V60p 取代 VR/V60（資料對齊）。
import sys, os, time, json
import numpy as np, pandas as pd
from multiprocessing import Pool
sys.path.insert(0, '/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/o_g6s2')
import rengine_v as RE
SCR = '/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad'
CELLS = {'N100': (100e4, False, False), 'N300': (300e4, False, False), 'N1000': (1000e4, False, False), 'S100': (100e4, True, False), 'O30': (30e4, False, False)}
_D = None
def init():
    global _D
    _D = RE.Data(SCR + '/rep_o19/rdata_strict.npz')
    if os.environ.get('ALIGN') == '1':
        X = np.load('/home/user/-/research/tw_stock_selection/agents/o_exec_engine_rewrite_v7x/data/x_cache.npz')
        VR = np.full_like(_D.VR, np.nan); VR[:-1] = X['VRp'][1:]; _D.VR = VR
        V6 = np.full_like(_D.V60, np.nan); V6[:-1] = X['V60p'][1:]; _D.V60 = V6
def job(a):
    cell, j, seed, plan = a
    tier, stress, w30 = CELLS[cell]
    pl = [(_D.dpos[np.datetime64(s)], _D.dpos[np.datetime64(e)], [_D.col[c] for c in pk.split(',')]) for s, e, pk in plan]
    s0, eq = RE.run_sim(_D, pl, RE.Cfg(tier, stress, w30, os.environ.get('RVAR', '')))
    return j, seed, s0, eq
if __name__ == '__main__':
    cell, tag = sys.argv[1], sys.argv[2]
    p = pd.read_csv(RE.AG + '/o_v7_listing_exec_engine/data/plans_j7.csv', dtype={'picks': str}); p = p[p.strat == 'AB_V7_B10']
    tasks = [(cell, j, s, list(zip(g.signal, g['exec'], g.picks))) for (j, s), g in p.groupby(['j', 'seed'])]
    t0 = time.time(); res = {}
    with Pool(4, initializer=init) as P:
        for j, s, s0, eq in P.imap_unordered(job, tasks, chunksize=4): res[(j, s)] = (s0, eq)
    out = {}
    for j in range(13):
        out[f'eq_j{j}'] = np.array([res[(j, s)][1] for s in range(20)]).astype(np.float32 if False else np.float64); out[f's0_j{j}'] = np.array(res[(j, 0)][0])
    np.savez_compressed(f'{SCR}/o_g6s2/navs_{cell}_{tag}.npz', **out)
    print('done', cell, tag, round(time.time() - t0), 's')
