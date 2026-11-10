#!/usr/bin/env python3
"""integrate_o19 V7X 名單重建（只產名單）：v7x_lib.make（只讀 import）× REAL_{opt,p90,carry} × 13 j × 20 seeds（B1），
處置官方清單＝strict（DISP_OFFICIAL_v2；斷言覆蓋＝現行）、posthoc、I3a、I3b、I3c。寫 scratchpad/integrate_o19/plans_{dv}.csv。
核對：strict 計數與 ends 等於題 1 RESULTS_V7X_PLANS；其他版本對 strict 的變動列。
python3 -I -B gen_v7x_plans.py"""
import sys, json, hashlib, time
from pathlib import Path
from multiprocessing import Pool
import pandas as pd
HERE = Path(__file__).resolve().parent; AG = HERE.parent
T1 = AG / 'o_delist_rev_backfill_allpit'; AU = AG / 'o_disposal_list_pit_audit_pkl18'
SCR = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/integrate_o19')
VARS = ['REAL_opt', 'REAL_p90', 'REAL_carry']
DUF = {'strict': AU / 'data/DISP_OFFICIAL_v2.csv', 'posthoc': AU / 'data/DISP_OFFICIAL_v2_posthoc.csv',
       'I3a': HERE / 'data/DU_I3a.csv', 'I3b': HERE / 'data/DU_I3b.csv', 'I3c': HERE / 'data/DU_I3c.csv'}
_X = {}
def init(dv):
    sys.path.insert(0, str(T1)); import v7x_lib as X; sys.path.pop(0)
    d = pd.read_csv(DUF[dv], dtype={'code': str}, parse_dates=['start', 'end'])[['code', 'start', 'end']]
    if dv == 'strict':
        assert set(map(tuple, d.astype(str).values)) == set(map(tuple, X.DU[['code', 'start', 'end']].astype(str).values))
    X.DU = d; X._dc.clear(); _X['X'] = X
def job(a):
    v, j, s = a
    rows, de = _X['X'].make(v, j, s, 'B1')
    return v, j, s, rows, de
if __name__ == '__main__':
    R = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))
    assert hashlib.sha256(json.dumps(R['PREREG'], ensure_ascii=False, sort_keys=True).encode()).hexdigest() == R['PREREG_sha256']
    for f, h in R['PREREG']['file_sha256'].items(): assert hashlib.sha256((AG / f).read_bytes()).hexdigest() == h, f
    t0 = time.time(); out = {}
    for dv in DUF:
        jobs = [(v, j, s) for v in VARS for j in range(13) for s in range(20)]
        res = []
        with Pool(4, initializer=init, initargs=(dv,)) as P:
            for n, x in enumerate(P.imap_unordered(job, jobs, chunksize=4)):
                res.append(x)
                if n % 260 == 0: print(dv, n, round(time.time() - t0), flush=True)
        rows = [r for x in res for r in x[3]]
        D = pd.DataFrame(rows, columns=['variant', 'j', 'seed', 'signal', 'exec', 'exec_idx', 'picks', 'inserted'])
        D['vo'] = D.variant.map({v: i for i, v in enumerate(VARS)}); D = D.sort_values(['vo', 'j', 'seed', 'signal']).drop(columns='vo').reset_index(drop=True)
        f = SCR / f'plans_{dv}.csv'; D.to_csv(f, index=False)
        ends = {f'{x[0]}|J{x[1]}': x[4] for x in res if x[2] == 0}
        cnt = {v: dict(n_rows=int((D.variant == v).sum()), inserted_slots_mean_per_plan=float(D[D.variant == v].groupby(['j', 'seed']).inserted.apply(lambda s: s.fillna('').str.split(',').apply(lambda z: len([q for q in z if q])).sum()).mean())) for v in VARS}
        out[dv] = dict(file=str(f), sha256=hashlib.sha256(f.read_bytes()).hexdigest(), counts=cnt, ends=ends)
    R1 = json.loads((T1 / 'results.json').read_text(encoding='utf-8'))['RESULTS_V7X_PLANS']
    chk = dict(strict_counts_eq_T1={v: out['strict']['counts'][v] == R1['counts'][v] for v in VARS},
               strict_ends_eq_T1=all(R1['ends'].get(k) == e for k, e in out['strict']['ends'].items()))
    A = pd.read_csv(out['strict']['file'], dtype={'picks': str})
    for dv in DUF:
        if dv == 'strict': continue
        B = pd.read_csv(out[dv]['file'], dtype={'picks': str})
        m = A.merge(B, on=['variant', 'j', 'seed', 'signal'], suffixes=('_s', '_p'), how='outer', indicator=True)
        dd = m[(m._merge != 'both') | (m.picks_s != m.picks_p)]
        chk[dv] = dict(rows_changed=int(len(dd)), plans_changed=int(dd.groupby(['variant', 'j', 'seed']).ngroups) if len(dd) else 0,
                       signals_changed=sorted(dd.signal.unique().tolist()), ends_eq_strict=out[dv]['ends'] == out['strict']['ends'])
    res = dict(utc=pd.Timestamp.now('UTC').isoformat(), runtime_s=round(time.time() - t0, 1), out=out, checks=chk)
    (HERE / 'v7x_plans_check.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(chk, ensure_ascii=False))
    assert all(chk['strict_counts_eq_T1'].values()) and chk['strict_ends_eq_T1'], chk
