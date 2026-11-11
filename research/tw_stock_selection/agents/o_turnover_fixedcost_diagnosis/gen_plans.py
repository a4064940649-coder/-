#!/usr/bin/env python3
"""o_turnover_fixedcost_diagnosis 第 1 步：重建主策略官方 V7X 名單（ALLPIT_REAL_{opt,p90,carry} × 13 j × 20 seeds，B1），
並在 seed 0 額外輸出每個訊號日的排名表（診斷用：被換出股排名、新進股來源、分數差）。
只讀 import o_delist_rev_backfill_allpit/v7x_lib（make 的邏輯原樣複製，只多讀 S._cs 的排名）；不修改任何其他資料夾。
斷言：各變體列數與 ends 等於 o18 RESULTS_V7X_PLANS（integrate_o19 同法）。
輸出：data/plans_real_allseeds.csv.gz（研究資料夾，關鍵中間檔）、data/ranks_seed0.csv.gz。
python3 -I -B gen_plans.py"""
import sys, json, hashlib, time
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; AG = HERE.parent
T1 = AG / 'o_delist_rev_backfill_allpit'
VARS = ['REAL_opt', 'REAL_p90', 'REAL_carry']
_X = {}
def init():
    sys.path.insert(0, str(T1)); import v7x_lib as X; sys.path.pop(0); _X['X'] = X
def make_with_ranks(variant, j, seed, bver='B1'):
    X = _X['X']; K = X.K
    days, de = X.days_for(j)
    fill = variant.split('_', 1)[1]
    if 'tab' not in _X: _X['tab'] = X.real_tables(bver)
    Y, Y5, _ = _X['tab']
    S = X.JV7R(Y, Y5, X.dm_all, putback='UNION', fill=fill, unavail=None, gap=None, pfix=None)
    def _u(d, _s=seed):
        u = X.STC.unavail_for(S.anchor[d], d, _s); ub = X.a_ub(d)
        if ub.any() and np.isfinite(S.y3(d, 1)).sum() < 100: return u
        return u | ub
    S.unavail = _u
    S.set_days(days, de)
    pl = S.plan(10, 'official')
    COLS = X.COLS
    rows = [(variant, j, seed, str(p[3].date()), str(K.IDX[p[0]].date()), int(p[0]), ','.join(COLS[jj] for jj in p[1]),
             ','.join(COLS[jj] for jj in p[1] if jj in set(S.ins.get(p[3], [])))) for p in pl]
    rk = []
    if seed == 0:
        prev = set()
        for p in pl:
            d = p[3]; cs = S.cross_section(d); el = cs['el']; e = cs['e']
            order = S.order(cs, 'official')[:int(el.sum())]
            rank = np.full(len(COLS), np.nan); rank[order] = np.arange(1, len(order) + 1)
            cur = set(int(x) for x in p[1])
            want = set(int(x) for x in order[:60]) | prev | cur
            dm = X.dm_all(d); top = cs['top']; okx = cs['ok_other']
            trd = np.isfinite(K.On[e]) & K.TRADED[e]
            for jj in sorted(want):
                if np.isfinite(rank[jj]): why = 'el'
                elif not top[jj]: why = 'pool'
                elif dm[jj]: why = 'disp'
                elif not trd[jj]: why = 'notrade'
                elif okx[jj]: why = 'rev'
                else: why = 'other'
                rk.append((variant, j, str(d.date()), COLS[jj], rank[jj], float(cs['sc'][jj]) if np.isfinite(cs['sc'][jj]) else np.nan,
                           why, int(jj in prev), int(jj in cur), int(len(order))))
            prev = cur
    return rows, str(de.date()), rk
def job(a):
    v, j, s = a
    rows, de, rk = make_with_ranks(v, j, s)
    return v, j, s, rows, de, rk
if __name__ == '__main__':
    t0 = time.time()
    jobs = [(v, j, s) for v in VARS for j in range(13) for s in range(20)]
    res = []
    with Pool(4, initializer=init) as P:
        for n, x in enumerate(P.imap_unordered(job, jobs, chunksize=4)):
            res.append(x)
            if n % 100 == 0: print(n, round(time.time() - t0), flush=True)
    rows = [r for x in res for r in x[3]]
    D = pd.DataFrame(rows, columns=['variant', 'j', 'seed', 'signal', 'exec', 'exec_idx', 'picks', 'inserted'])
    D['vo'] = D.variant.map({v: i for i, v in enumerate(VARS)}); D = D.sort_values(['vo', 'j', 'seed', 'signal']).drop(columns='vo').reset_index(drop=True)
    R1 = json.loads((T1 / 'results.json').read_text(encoding='utf-8'))['RESULTS_V7X_PLANS']
    cnt = {v: int((D.variant == v).sum()) for v in VARS}
    ends = {f'{x[0]}|J{x[1]}': x[4] for x in res if x[2] == 0}
    chk = dict(counts_eq={v: cnt[v] == R1['counts'][v]['n_rows'] for v in VARS}, ends_eq=all(R1['ends'].get(k) == e for k, e in ends.items()))
    print(chk)
    assert all(chk['counts_eq'].values()) and chk['ends_eq'], chk
    f = HERE / 'data/plans_real_allseeds.csv.gz'; D.to_csv(f, index=False, compression='gzip')
    RK = pd.DataFrame([r for x in res for r in x[5]], columns=['variant', 'j', 'signal', 'code', 'rank', 'score', 'why', 'prev_held', 'cur_pick', 'n_el'])
    RK = RK.sort_values(['variant', 'j', 'signal', 'rank', 'code']).reset_index(drop=True)
    f2 = HERE / 'data/ranks_seed0.csv.gz'; RK.to_csv(f2, index=False, compression='gzip', float_format='%.6g')
    out = dict(utc=pd.Timestamp.now('UTC').isoformat(), runtime_s=round(time.time() - t0, 1), checks=chk, counts=cnt, ends=ends,
               plans_file=str(f.relative_to(HERE)), plans_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),
               ranks_file=str(f2.relative_to(HERE)), ranks_sha256=hashlib.sha256(f2.read_bytes()).hexdigest(), n_rank_rows=int(len(RK)))
    (HERE / 'plans_check.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps({k: v for k, v in out.items() if k != 'ends'}, ensure_ascii=False))
