# K（保管員，凍結後）：以 o18 原程式（只讀 import，不寫入 o18 資料夾）產生原名單，輸出成本題介面 pickle
import sys, json, time, pickle
from pathlib import Path
from multiprocessing import Pool
O18 = '/home/user/-/research/tw_stock_selection/agents/o_delist_rev_backfill_allpit'
OUT = '/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/g6K_lists/'
def job(a):
    sys.path.insert(0, O18); import v7x_lib as X
    v, j, s = a
    rows, de = X.make(v, j, s, 'B1')
    return v, j, s, rows, de
if __name__ == '__main__':
    t0 = time.time()
    VARS = ['orig', 'REAL_opt', 'REAL_p90', 'REAL_carry']
    jobs = [(v, j, s) for v in VARS for j in range(13) for s in range(20)]
    KD = {'variants': {v: {} for v in VARS}, 'ends': {}}
    with Pool(4) as P:
        for n, (v, j, s, rows, de) in enumerate(P.imap_unordered(job, jobs, chunksize=4)):
            KD['variants'][v][(j, s)] = [(r[3], r[4], [c for c in r[6].split(',') if c]) for r in rows]
            KD['ends'][(v, j)] = de
            if n % 200 == 0: print(n, round(time.time() - t0), flush=True)
    pickle.dump(KD, open(OUT + 'k_v7x_plans.pkl', 'wb'))
    print('v7x plans done', round(time.time() - t0))
