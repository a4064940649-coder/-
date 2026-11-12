#!/usr/bin/env python3
"""seeds 階段的可續跑版本（凍結後的實作偏差紀錄：run_engine.py seeds 階段一次跑完 1,170 個工作才寫檔，第一次在 750/1,170 時被背景工作 2 小時上限終止、結果全失；
本檔不改 run_engine.py（sha 已凍結），直接 import 它的 init/job_seeds，逐工作寫入 scratchpad/csr/seeds_ckpt/*.npy，全部完成後組成與原本相同格式的 navs_seeds20.parquet）。
順序：中心格（正式表重現）先跑，再四角。python3 -I -B run_seeds.py"""
import sys, time, json
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); import run_engine as RE; import csr_lib as L; sys.path.pop(0)
CK = L.SCR / 'seeds_ckpt'; CK.mkdir(exist_ok=True)
def kname(a):
    v, j, c, mo, mc = a; return f'{mo}|{mc}|{v}|{j}|{L.CELLKEY[c]}'
def fname(a): return CK / (kname(a).replace('|', '__') + '.npy')
def job(a):
    a2, nav = RE.job_seeds(a); np.save(fname(a), nav); return kname(a)
if __name__ == '__main__':
    t0 = time.time()
    pts = [(1.0, 1.0)] + [p for p in L.CORNERS if p != (1.0, 1.0)]
    jobs = [(v, j, c, mo, mc) for (mo, mc) in pts for c in L.CELLS for v in L.VARS for j in range(13)]
    todo = [a for a in jobs if not fname(a).exists()]
    print('todo', len(todo), 'of', len(jobs), flush=True)
    with Pool(3, initializer=RE.init, initargs=(False, 'all')) as P:
        for i, k in enumerate(P.imap_unordered(job, todo, chunksize=1)):
            if i % 25 == 0: print(i, len(todo), round(time.time() - t0), flush=True)
    assert all(fname(a).exists() for a in jobs)
    NV = {kname(a): np.load(fname(a)) for a in jobs}
    pd.DataFrame(NV).to_parquet(L.SCR / 'navs_seeds20.parquet')
    mf = HERE / 'run_engine_meta.json'; M = json.loads(mf.read_text(encoding='utf-8'))
    M['seeds'] = dict(stage='seeds（run_seeds.py 續跑版）', utc=pd.Timestamp.now('UTC').isoformat(), n=len(NV), sha256=L.sha256f(L.SCR / 'navs_seeds20.parquet'),
                      note='第一次 run_engine.py seeds 在 750/1170 被背景時限終止；本次以同一 job_seeds 函式重跑並逐工作存檔')
    mf.write_text(json.dumps(M, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    print('done seeds', len(NV), round(time.time() - t0))
