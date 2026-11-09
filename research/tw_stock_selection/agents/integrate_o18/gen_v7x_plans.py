#!/usr/bin/env python3
"""V7X 名單重建（登記前；只產名單，不算報酬）。題 1 的名單檔已隨 scratchpad 清除，這裡用題 1 的 v7x_lib.make（只讀 import）
重建 orig、REAL_opt、REAL_p90、REAL_carry × 13 j × 20 seeds，寫入 scratch/plans_v7x_i18.csv（依 variant, j, seed, signal 排序）。
核對：orig 與 o_v7_listing_exec_engine/data/plans_j7.csv（AB_V7_B10、seed < 20；signal, picks, exec_idx）逐筆相同；
各變體 n_rows 與 inserted_slots_mean_per_plan 等於題 1 RESULTS_V7X_PLANS.counts（精確相等）；ends 等於題 1 RESULTS_V7X_PLANS.ends。
python3 -I -B gen_v7x_plans.py"""
import sys, json, hashlib, time
from pathlib import Path
from multiprocessing import Pool
HERE = Path(__file__).resolve().parent; AG = HERE.parent
T1 = AG / 'o_delist_rev_backfill_allpit'
SCR = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/integrate_o18')
VARS = ['orig', 'REAL_opt', 'REAL_p90', 'REAL_carry']
def job(a):
    sys.path.insert(0, str(T1)); import v7x_lib as X
    v, j, s = a
    rows, de = X.make(v, j, s, 'B1')
    return v, j, s, rows, de
if __name__ == '__main__':
    import pandas as pd
    t0 = time.time()
    jobs = [(v, j, s) for v in VARS for j in range(13) for s in range(20)]
    out = []
    with Pool(4) as P:
        for n, x in enumerate(P.imap_unordered(job, jobs, chunksize=4)):
            out.append(x)
            if n % 200 == 0: print(n, round(time.time() - t0), flush=True)
    rows = [r for x in out for r in x[3]]
    D = pd.DataFrame(rows, columns=['variant', 'j', 'seed', 'signal', 'exec', 'exec_idx', 'picks', 'inserted'])
    D['vo'] = D.variant.map({v: i for i, v in enumerate(VARS)})
    D = D.sort_values(['vo', 'j', 'seed', 'signal']).drop(columns='vo').reset_index(drop=True)
    f = SCR / 'plans_v7x_i18.csv'; D.to_csv(f, index=False)
    ref = pd.read_csv(AG / 'o_v7_listing_exec_engine/data/plans_j7.csv', dtype={'picks': str})
    ref = ref[(ref.strat == 'AB_V7_B10') & (ref.seed < 20)].sort_values(['j', 'seed', 'signal'])
    mine = D[D.variant == 'orig'].sort_values(['j', 'seed', 'signal'])
    same = list(zip(mine.j, mine.seed, mine.signal, mine.picks.astype(str), mine.exec_idx)) == list(zip(ref.j, ref.seed, ref.signal, ref.picks.astype(str), ref.exec_idx))
    R1 = json.loads((T1 / 'results.json').read_text(encoding='utf-8'))['RESULTS_V7X_PLANS']
    cnt = {v: dict(n_rows=int((D.variant == v).sum()), inserted_slots_mean_per_plan=float(D[D.variant == v].groupby(['j', 'seed']).inserted.apply(lambda s: s.fillna('').str.split(',').apply(lambda z: len([q for q in z if q])).sum()).mean()))
           for v in VARS}
    ends = {f'{x[0]}|J{x[1]}': x[4] for x in out if x[2] == 0}
    chk = dict(orig_eq_plans_j7=bool(same), counts_eq_T1={v: cnt[v] == R1['counts'][v] for v in VARS}, ends_eq_T1=(ends == R1['ends']))
    res = dict(utc=pd.Timestamp.now('UTC').isoformat(), runtime_s=round(time.time() - t0, 1), file=str(f), sha256=hashlib.sha256(f.read_bytes()).hexdigest(),
               counts=cnt, checks=chk)
    (HERE / 'v7x_plans_check.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(res, ensure_ascii=False))
    assert same and all(chk['counts_eq_T1'].values()) and chk['ends_eq_T1'], chk
