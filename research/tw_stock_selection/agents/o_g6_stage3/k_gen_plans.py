#!/usr/bin/env python3
"""integrate_o20 V7X 名單重建（只產名單，不算報酬）：v7x_lib.make（只讀 import）× REAL_{opt,p90,carry} × 13 j × 20 seeds（B1），
遮罩＝base（v7x_lib.dm_all 原樣）／M2（全期統一：官方 universe_announce「公告 ≤ d 且涵蓋 e」，逐列斷言）。
寫 scratchpad/integrate_o20/plans_{dv}.csv；變動列另存 data/v7x_plan_diff_M2.csv.gz。
核對：base 計數與 ends 等於題 1 RESULTS_V7X_PLANS。python3 -I -B gen_v7x_plans.py
【K 保管員複本（o_g6_stage3/k_gen_plans.py）】與 integrate_o20/gen_v7x_plans.py 的差別只在編排，產名單的 init()／job()（＝原程式 v7x_lib.make）逐字未動：
  (1) 輸出路徑：名單寫 scratchpad/k_stage3/plans_{dv}.csv、差異檔寫同處、核對檔寫 k_v7x_plans_check.json；results.json 改讀 integrate_o20/ 的；
  (2) 行程數：python3 -I -B k_gen_plans.py [行程數]（預設 2；原檔 4）——本機記憶體與其他工作共用上限，2026-11-14 實際用 2（base）與 1（M2）；
  (3) 可續跑：scratchpad 已有 plans_{dv}.csv 時不重產該遮罩（沿用原位元組），其 ends 由同一支原程式的 v7x_lib.days_for(j)[1]（make() 回傳的 de 即此值）重算。"""
import sys, json, hashlib, time
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; AG = HERE.parent
T1 = AG / 'o_delist_rev_backfill_allpit'
UF = AG / 'o_disposal_list_pit_audit_pkl18/data/universe_announce.csv'
SCR = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/k_stage3')
VARS = ['REAL_opt', 'REAL_p90', 'REAL_carry']; DVS = ['base', 'M2']
_X = {}
def init(dv):
    sys.path.insert(0, str(T1)); import v7x_lib as X; sys.path.pop(0)
    if dv == 'M2':
        U = pd.read_csv(UF, dtype={'code': str}, parse_dates=['announce', 'start', 'end'])
        ann, st, en = (U[c].to_numpy('datetime64[ns]') for c in ('announce', 'start', 'end'))
        jj = np.array([X.CI.get(c, -1) for c in U.code]); cache = {}
        def m2(d):
            d = pd.Timestamp(d)
            if d in cache: return cache[d]
            t = X.IDX.get_loc(d); e = X.IDX[t + 1] if t + 1 < len(X.IDX) else d
            dd, de = np.datetime64(d), np.datetime64(e)
            sel = (ann <= dd) & (st <= de) & (en >= de)
            assert all((a <= dd) and (s <= de <= z) for a, s, z in zip(ann[sel], st[sel], en[sel]))
            m = np.zeros(X.N, bool); m[jj[sel & (jj >= 0)]] = True
            cache[d] = m; return m
        X.dm_all = m2
    _X['X'] = X
def job(a):
    v, j, s = a
    rows, de = _X['X'].make(v, j, s, 'B1')
    return v, j, s, rows, de
def ends_job(j):                                   # 只在續跑時用：make() 內 `days, de = days_for(j)`、回傳 str(de.date())
    return j, str(_X['X'].days_for(j)[1].date())
if __name__ == '__main__':
    R = json.loads((AG / 'integrate_o20' / 'results.json').read_text(encoding='utf-8'))
    assert hashlib.sha256(json.dumps(R['PREREG'], ensure_ascii=False, sort_keys=True).encode()).hexdigest() == R['PREREG_sha256']
    for f, h in R['PREREG']['file_sha256'].items(): assert hashlib.sha256((AG / f).read_bytes()).hexdigest() == h, f
    SCR.mkdir(parents=True, exist_ok=True)
    NP = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    t0 = time.time(); out = {}; resumed = {}
    for dv in DVS:
        f = SCR / f'plans_{dv}.csv'; resumed[dv] = f.exists()
        if resumed[dv]:
            print(dv, 'resume: reuse', f.name, flush=True)
            D = pd.read_csv(f, dtype={'variant': str, 'signal': str, 'exec': str, 'picks': str, 'inserted': str})
            assert len(D) == D[['variant', 'j', 'seed', 'signal']].drop_duplicates().shape[0] and D.groupby(['variant', 'j', 'seed']).ngroups == 780
            with Pool(1, initializer=init, initargs=(dv,)) as P: ej = dict(P.map(ends_job, range(13)))
            ends = {f'{v}|J{j}': ej[j] for v in VARS for j in range(13)}
        else:
            jobs = [(v, j, s) for v in VARS for j in range(13) for s in range(20)]
            res = []
            with Pool(NP, initializer=init, initargs=(dv,)) as P:
                for n, x in enumerate(P.imap_unordered(job, jobs, chunksize=4)):
                    res.append(x)
                    if n % 260 == 0: print(dv, n, round(time.time() - t0), flush=True)
            rows = [r for x in res for r in x[3]]
            D = pd.DataFrame(rows, columns=['variant', 'j', 'seed', 'signal', 'exec', 'exec_idx', 'picks', 'inserted'])
            D['vo'] = D.variant.map({v: i for i, v in enumerate(VARS)}); D = D.sort_values(['vo', 'j', 'seed', 'signal']).drop(columns='vo').reset_index(drop=True)
            ftmp = SCR / f'plans_{dv}.csv.tmp'; D.to_csv(ftmp, index=False); ftmp.rename(f)
            ends = {f'{x[0]}|J{x[1]}': x[4] for x in res if x[2] == 0}
        cnt = {v: dict(n_rows=int((D.variant == v).sum()), inserted_slots_mean_per_plan=float(D[D.variant == v].groupby(['j', 'seed']).inserted.apply(lambda s: s.fillna('').str.split(',').apply(lambda z: len([q for q in z if q])).sum()).mean())) for v in VARS}
        out[dv] = dict(file=str(f), sha256=hashlib.sha256(f.read_bytes()).hexdigest(), counts=cnt, ends=ends)
    R1 = json.loads((T1 / 'results.json').read_text(encoding='utf-8'))['RESULTS_V7X_PLANS']
    chk = dict(base_counts_eq_T1={v: out['base']['counts'][v] == R1['counts'][v] for v in VARS},
               base_ends_eq_T1=all(R1['ends'].get(k) == e for k, e in out['base']['ends'].items()))
    A = pd.read_csv(out['base']['file'], dtype={'picks': str}); B = pd.read_csv(out['M2']['file'], dtype={'picks': str})
    m = A.merge(B, on=['variant', 'j', 'seed', 'signal'], suffixes=('_b', '_m'), how='outer', indicator=True)
    dd = m[(m._merge != 'both') | (m.picks_b != m.picks_m)].copy()
    def sd(a, b): return ','.join(sorted(set(str(a).split(',')) - set(str(b).split(','))))
    dd['out'] = [sd(a, b) for a, b in zip(dd.picks_b, dd.picks_m)]; dd['into'] = [sd(b, a) for a, b in zip(dd.picks_b, dd.picks_m)]
    dd[['variant', 'j', 'seed', 'signal', 'exec_b', 'exec_m', 'picks_b', 'picks_m', 'out', 'into', '_merge']].to_csv(SCR / 'v7x_plan_diff_M2.csv.gz', index=False)
    ev = dd.groupby(['signal', 'out', 'into']).size().reset_index(name='n_plans')
    chk['M2'] = dict(rows_changed=int(len(dd)), plans_changed=int(dd.groupby(['variant', 'j', 'seed']).ngroups) if len(dd) else 0,
                     plans_changed_by_variant={v: int(dd[dd.variant == v].groupby(['j', 'seed']).ngroups) for v in VARS},
                     n_signal_days=int(dd.signal.nunique()), n_distinct_swaps=int(len(ev)), ends_eq_base=out['M2']['ends'] == out['base']['ends'],
                     swaps_pre2021_09_24=int((ev.signal < '2021-09-24').sum()))
    res = dict(utc=pd.Timestamp.now('UTC').isoformat(), runtime_s=round(time.time() - t0, 1), n_proc=NP, resumed=resumed, out=out, checks=chk)
    (HERE / 'k_v7x_plans_check.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(chk, ensure_ascii=False))
    assert all(chk['base_counts_eq_T1'].values()) and chk['base_ends_eq_T1'], chk
