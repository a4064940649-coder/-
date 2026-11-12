#!/usr/bin/env python3
"""模擬批次共用：名單快取（scratchpad pkl，numpy 陣列）、平行執行、彙整（20 seeds 平均淨值 → 各 j 統計）。"""
import sys, json, time, pickle, math
from pathlib import Path
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); import cas_lib as CL; sys.path.pop(0)
SCR = CL.SCR
VARS = ['REAL_opt', 'REAL_p90', 'REAL_carry']
PCACHE = SCR / 'plans_cache.pkl'

def build_cache():
    if PCACHE.exists(): return
    E = CL.engine(); D = E['D']
    out = {}
    for v in VARS:
        C = CL.load_cands(v)
        for (vv, j, s), g in C.groupby(['variant', 'j', 'seed']):
            pl, cands = CL.to_plan(g, D)
            out[(vv, int(j), int(s))] = (pl, {e: (np.array(c, np.int32), frozenset(gu)) for e, (c, gu) in cands.items()})
        del C
    pickle.dump(out, open(PCACHE, 'wb'), protocol=4)

_G = {}
def init():
    E = CL.engine(); _G['E'] = E
    _G['P'] = pickle.load(open(PCACHE, 'rb'))
def job(a):
    """a＝(tag, variant, j, seed, tier, cost, opt, sw, t1, track)"""
    tag, v, j, s, tier, cost, opt, sw, t1, track = a
    E = _G['E']; pl, cands = _G['P'][(v, j, s)]
    cands = {e: ([int(x) for x in c], gu) for e, (c, gu) in cands.items()}
    cfg = CL.cfg_for(E, tier, cost)
    sim = E['CASim'](pl, dict(cfg), settle=E['ST'], cands=cands, opt=opt, sw=sw, rng_seed=1000 * s + j, track=track)
    nav = sim.run(t1=t1)
    tr = None
    if track:
        idx = np.arange(sim.t0, sim.t0 + len(nav))
        tr = {k: arr[idx].astype(np.float64) for k, arr in sim.dc_.items()}
    return tag, v, j, s, nav.astype(np.float64), dict(sim.st), tr

def _iter_results(jobs, nproc, initf):
    """ProcessPoolExecutor：工作行程被 OOM 砍掉時重開並只補跑缺的工作（結果與順序無關）。"""
    todo = list(range(len(jobs)))
    while todo:
        done = []
        try:
            with ProcessPoolExecutor(nproc, initializer=initf) as ex:
                futs = {ex.submit(job, jobs[i]): i for i in todo}
                from concurrent.futures import as_completed
                for fu in as_completed(futs):
                    i = futs[fu]
                    yield fu.result(); done.append(i)
        except BrokenProcessPool:
            print('BrokenProcessPool：重開，剩', len(todo) - len(done), flush=True)
        dset = set(done); todo = [i for i in todo if i not in dset]

def t_end_idx(date):
    """不載入引擎：只讀 x_cache 的 IDX。"""
    Z = np.load(AGX / 'x_cache.npz')
    idx = Z['IDX']
    return int(np.searchsorted(idx, np.datetime64(pd.Timestamp(date)), side='right') - 1)
AGX = HERE.parent / 'o_exec_engine_rewrite_v7x' / 'data'

def run_batch(jobs, nproc=2, log=None, ckpt=None):
    """回傳 dict (tag, v, j) → {'nav': 20 seeds 平均淨值, 'st': 加總, 'tr': 平均追蹤（若有）}。
    母行程不載入引擎；以報酬加總累積（不存逐 seed 序列）；ckpt＝scratchpad 檔名（存在就直接讀）。"""
    if ckpt is not None and (SCR / ckpt).exists():
        return pickle.load(open(SCR / ckpt, 'rb'))
    assert PCACHE.exists(), '先建 plans_cache'
    t0 = time.time(); acc = {}
    for n, (tag, v, j, s, nav, st, tr) in enumerate(_iter_results(jobs, nproc, init)):
        k = (tag, v, j)
        r = nav.pct_change()
        a = acc.get(k)
        if a is None:
            a = acc[k] = dict(rs=r.fillna(0.0) * 0.0, cnt=r.notna().astype(float) * 0.0, st={}, tr=None, n=0)
        a['rs'] = a['rs'].add(r.fillna(0.0), fill_value=0.0); a['cnt'] = a['cnt'].add(r.notna().astype(float), fill_value=0.0)
        for kk, vv in st.items(): a['st'][kk] = a['st'].get(kk, 0) + vv
        if tr is not None:
            trs = pd.DataFrame(tr, index=nav.index)
            a['tr'] = trs if a['tr'] is None else a['tr'].add(trs, fill_value=0.0)
        a['n'] += 1
        if log and n % 500 == 0: print(log, n, len(jobs), round(time.time() - t0), flush=True)
    out = {}
    for k, a in acc.items():
        R = (a['rs'] / a['cnt'].where(a['cnt'] > 0)).fillna(0.0)
        nav = (1 + R).cumprod()
        tr = a['tr'] / a['n'] if a['tr'] is not None else None
        out[k] = dict(nav=nav, st=a['st'], tr=tr, n=a['n'])
    if ckpt is not None: pickle.dump(out, open(SCR / ckpt, 'wb'), protocol=4)
    return out

def jobs_for(tag, tier, cost, opt, sw=None, t1=None, seeds=range(20), vars_=VARS, js=range(13), track=False):
    return [(tag, v, j, s, tier, cost, opt, sw, t1, track) for v in vars_ for j in js for s in seeds]

def paired(res, tag, base, a, b, vars_=VARS, js=range(13), ia=False):
    """各放回版本：逐 j 日曆年化配對差 → 中位、平均、最大、p10；回傳 dict 與最差放回版本。"""
    f = CL.cagr_td if ia else CL.cagr
    o = {}
    for v in vars_:
        d = []
        for j in js:
            n1 = res[(tag, v, j)]['nav']; n0 = res[(base, v, j)]['nav']
            aa = max(pd.Timestamp(a), n1.index[0])
            d.append(f(n1, aa, b) - f(n0, aa, b))
        d = np.array(d)
        o[v] = dict(by_j=d.tolist(), med=float(np.median(d)), mean=float(d.mean()), max=float(d.max()), min=float(d.min()),
                    p10=float(np.percentile(d, 10)), med_j=int(np.argsort(d)[len(d) // 2]))
    worst = min(vars_, key=lambda v: o[v]['med'])
    o['worst'] = dict(variant=worst, med=o[worst]['med'], mean=min(o[v]['mean'] for v in vars_), max_abs=max(max(abs(x) for x in o[v]['by_j']) for v in vars_))
    return o

def monthly_diff(res, tag, base, a, b, v, js=range(13)):
    ds = []
    for j in js:
        m1 = CL.monthly(res[(tag, v, j)]['nav']); m0 = CL.monthly(res[(base, v, j)]['nav'])
        ds.append(m1 - m0)
    dm = pd.concat(ds, axis=1)
    dm = dm[(dm.index >= pd.Period(pd.Timestamp(a), 'M')) & (dm.index <= pd.Period(pd.Timestamp(b), 'M'))]
    return dm.mean(axis=1).dropna()

def boot_ci(x, n=2000, seed=20261112):
    x = np.asarray(x, float); rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), (n, len(x)))
    m = x[idx].mean(1)
    return dict(ci90=[float(np.percentile(m, 5)), float(np.percentile(m, 95))], ci95=[float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))],
                p_le0=float((m <= 0).mean()))
