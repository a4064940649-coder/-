#!/usr/bin/env python3
"""o_cost_aware_swap_exec 第 1 步：重建主策略官方 V7X 名單的「候選序列」（ALLPIT_REAL_{opt,p90,carry} × 13 j × 20 seeds，B1）。
只讀 import o_delist_rev_backfill_allpit/v7x_lib（make 邏輯原樣複製；只多讀官方同分序的完整排序 order 與成交日開盤漲停旗標 gap_up_open）。
每個訊號日存：官方排序前 40 名（合格股，同分序＝official）與其中成交日開盤漲停者。
官方 plan 的選法：依序走過候選，「開盤漲停且不在前月名單」者跳過，取滿 10 檔。包裝器用同法重建，並在本程式斷言
（a）重建名單＝plan 名單（逐列）；（b）＝o_turnover_fixedcost_diagnosis/data/plans_real_allseeds.csv.gz（逐列 picks 相同）。
輸出：data/cands_allseeds.csv.gz（研究資料夾，關鍵中間檔）、cands_check.json。
python3 -I -B gen_cands.py"""
import sys, json, hashlib, time
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; AG = HERE.parent
T1 = AG / 'o_delist_rev_backfill_allpit'
VARS = ['REAL_opt', 'REAL_p90', 'REAL_carry']
NC = 40
_X = {}
def init():
    sys.path.insert(0, str(T1)); import v7x_lib as X; sys.path.pop(0); _X['X'] = X
def make_cands(variant, j, seed, bver='B1'):
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
    rows = []; held = set()
    for p in pl:
        d = p[3]; cs = S.cross_section(d); el = cs['el']; e = cs['e']
        cand = S.order(cs, 'official')[:int(el.sum())]
        gu = K.gap_up_open(e)
        top = [int(x) for x in cand[:NC]]
        gl = [x for x in top if gu[x]]
        # 重建
        pick = []
        for x in top:
            if gu[x] and x not in held: continue
            pick.append(x)
            if len(pick) == 10: break
        assert pick == [int(x) for x in p[1]], (variant, j, seed, str(d.date()))
        held = set(pick)
        rows.append((variant, j, seed, str(d.date()), int(p[0]), ','.join(COLS[x] for x in top), ','.join(COLS[x] for x in gl),
                     ','.join(COLS[x] for x in p[1]), int(len(cand))))
    return rows
def job(a):
    v, j, s = a
    return v, j, s, make_cands(v, j, s)
if __name__ == '__main__':
    t0 = time.time()
    jobs = [(v, j, s) for v in VARS for j in range(13) for s in range(20)]
    res = []
    with Pool(4, initializer=init) as P:
        for n, x in enumerate(P.imap_unordered(job, jobs, chunksize=4)):
            res.append(x)
            if n % 100 == 0: print(n, round(time.time() - t0), flush=True)
    rows = [r for x in res for r in x[3]]
    D = pd.DataFrame(rows, columns=['variant', 'j', 'seed', 'signal', 'exec_idx', 'cand40', 'gapup', 'picks', 'n_el'])
    D['vo'] = D.variant.map({v: i for i, v in enumerate(VARS)}); D = D.sort_values(['vo', 'j', 'seed', 'signal']).drop(columns='vo').reset_index(drop=True)
    REF = pd.read_csv(AG / 'o_turnover_fixedcost_diagnosis/data/plans_real_allseeds.csv.gz', dtype={'picks': str, 'inserted': str})
    assert len(REF) == len(D)
    m = (REF[['variant', 'j', 'seed', 'signal', 'exec_idx', 'picks']].reset_index(drop=True) == D[['variant', 'j', 'seed', 'signal', 'exec_idx', 'picks']]).all(axis=1)
    chk = dict(n_rows=int(len(D)), rows_equal_ref=int(m.sum()), ref_sha256=hashlib.sha256((AG / 'o_turnover_fixedcost_diagnosis/data/plans_real_allseeds.csv.gz').read_bytes()).hexdigest())
    print(chk); assert m.all()
    f = HERE / 'data/cands_allseeds.csv.gz'; D.to_csv(f, index=False, compression='gzip')
    out = dict(utc=pd.Timestamp.now('UTC').isoformat(), runtime_s=round(time.time() - t0, 1), checks=chk,
               file=str(f.relative_to(HERE)), sha256=hashlib.sha256(f.read_bytes()).hexdigest())
    (HERE / 'cands_check.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(out, ensure_ascii=False))
