#!/usr/bin/env python3
"""V7X 名單（正式；只跑一次）：orig、REAL_opt、REAL_p90、REAL_carry × 13 j × 20 seeds（AB_V7_B10 規格；v7x_lib）。
斷言 PREREG；orig 必須與 o_v7_listing_exec_engine/data/plans_j7.csv 的 AB_V7_B10 逐筆相同。輸出 scratch/plans_v7x.csv。
python3 -I -B v7x_plans.py"""
import sys, json, hashlib, time
from pathlib import Path
from multiprocessing import Pool
HERE = Path(__file__).resolve().parent; AG = HERE.parent
def sha(f): return hashlib.sha256(Path(f).read_bytes()).hexdigest()
R = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))
assert 'PREREG' in R and hashlib.sha256(json.dumps(R['PREREG'], ensure_ascii=False, sort_keys=True).encode()).hexdigest() == R['PREREG_sha256']
PR = R['PREREG']
for f, h in PR['file_sha256'].items(): assert sha(HERE / f) == h, f
assert 'RESULTS_V7X_PLANS' not in R
VARS = PR['v7x']['plan_variants']
def job(a):
    sys.path.insert(0, str(HERE)); import v7x_lib as X
    v, j, s = a
    rows, de = X.make(v, j, s, PR['backfill_version'])
    info = X.make.__defaults__[-1].get('tab', (None, None, None))[2] if v != 'orig' else None
    return v, j, s, rows, de, info
if __name__ == '__main__':
    import pandas as pd
    t0 = time.time()
    jobs = [(v, j, s) for v in VARS for j in range(13) for s in range(20)]
    out = []
    with Pool(4) as P:
        for n, x in enumerate(P.imap_unordered(job, jobs, chunksize=4)):
            out.append(x)
            if n % 100 == 0: print(n, round(time.time() - t0), flush=True)
    rows = [r for x in out for r in x[3]]
    D = pd.DataFrame(rows, columns=['variant', 'j', 'seed', 'signal', 'exec', 'exec_idx', 'picks', 'inserted'])
    f = Path(PR['scratch']) / 'plans_v7x.csv'; D.to_csv(f, index=False)
    ref = pd.read_csv(AG / 'o_v7_listing_exec_engine/data/plans_j7.csv', dtype={'picks': str})
    ref = ref[ref.strat == 'AB_V7_B10'].sort_values(['j', 'seed', 'signal'])
    mine = D[D.variant == 'orig'].sort_values(['j', 'seed', 'signal'])
    same = (list(zip(mine.j, mine.seed, mine.signal, mine.picks.astype(str))) == list(zip(ref.j, ref.seed, ref.signal, ref.picks.astype(str))))
    ends = {f'{x[0]}|J{x[1]}': x[4] for x in out if x[2] == 0}
    info = next((x[5] for x in out if x[5]), None)
    cnt = {v: dict(n_rows=int((D.variant == v).sum()), inserted_slots_mean_per_plan=float(D[D.variant == v].groupby(['j', 'seed']).inserted.apply(lambda s: s.fillna('').str.split(',').apply(lambda z: len([q for q in z if q])).sum()).mean()))
           for v in VARS}
    R = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))
    R['RESULTS_V7X_PLANS'] = dict(utc=pd.Timestamp.now('UTC').isoformat(), runtime_s=round(time.time() - t0, 1), orig_reproduces_plans_j7=bool(same),
                                  file=str(f), sha256=sha(f), ends=ends, real_table_info=info, counts=cnt)
    (HERE / 'results.json').write_text(json.dumps(R, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    print('orig reproduces', same, cnt, info, round(time.time() - t0))
    assert same, 'orig 名單未重現 plans_j7'
