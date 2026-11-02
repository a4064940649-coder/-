#!/usr/bin/env python3
"""主跑（PREREG 後）：一個設定 × 一個期間 → 14 個日期集合（官方日＋J0～J12）× 實際分數＋1,000 次隨機分數。
python3 -I -B s3_run.py <config> <period> [--exclude CODE] [--nseed 1000]
輸出 out/run_<config>_<period>[_ex<code>].npz（每個日期集合：實際窗報酬、隨機窗報酬矩陣、0050 窗報酬、
實際權重與隨機平均權重、訊號日市值名次、窗因子；供 s4_stats 歸因用）。"""
import sys, json, time, argparse
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
from multiprocessing import Pool
import pcore as P

PRE = json.loads((P.DATA / 'PREREG.json').read_text(encoding='utf-8')) if (P.DATA / 'PREREG.json').exists() else None
assert PRE is not None, 'PREREG 不存在：不得計算任何報酬'
for f, h in PRE['code_sha256'].items():
    if f.startswith('s3_') or f == 'pcore.py':
        assert P.sha(HERE / f) == h, f'程式雜湊與 PREREG 不符：{f}'

ap = argparse.ArgumentParser()
ap.add_argument('config'); ap.add_argument('period'); ap.add_argument('--exclude', default=None); ap.add_argument('--nseed', type=int, default=PRE['n_random'])
ap.add_argument('--cost', default='normal')
A = ap.parse_args()
T = P.setup(A.config)
L = P.L
DS = P.daysets(A.period)
EXC = None if A.exclude is None else np.array([L.CI[c] for c in A.exclude.split(',')], int)
SEED0 = PRE['seed0']
# 先在父行程把所有橫斷面算好（fork 後共用）
for k, (sd, de) in DS.items():
    for d in sd: T.cross_section(d)

def job(args):
    key, s0, s1 = args
    sd, de = DS[key]
    R = []; W = np.zeros((len(sd), P.N))
    for s in range(s0, s1):
        rg = np.random.default_rng(SEED0 + s)
        r, pl = P.run_scores(sd, de, P.rand_fn(rg), cost=A.cost, exclude=EXC, want_plan=True)
        R.append(r)
        for i, (d, js, ws) in enumerate(pl): W[i, js] += ws
    return key, s0, np.array(R), W

if __name__ == '__main__':
    t0 = time.time()
    CH = 125
    jobs = [(k, s, min(s + CH, A.nseed)) for k in DS for s in range(0, A.nseed, CH)]
    with Pool(4) as pool: res = pool.map(job, jobs, chunksize=1)
    out = {}
    for k, (sd, de) in DS.items():
        ra, pl = P.run_scores(sd, de, None, cost=A.cost, exclude=EXC, want_plan=True)
        parts = sorted([x for x in res if x[0] == k], key=lambda x: x[1])
        Rr = np.vstack([x[2] for x in parts]); Wr = sum(x[3] for x in parts) / A.nseed
        Wa = np.zeros((len(sd), P.N))
        for i, (d, js, ws) in enumerate(pl): Wa[i, js] = ws
        rk = np.array([np.where(T.cross_section(d)['rank'] < 10 ** 8, T.cross_section(d)['rank'], 0) for d in sd], np.int32)
        Rrep = np.array([T.cross_section(d)['R'] for d in sd], np.float32)
        # 窗因子：成交日 e_k 開盤 → e_{k+1} 開盤（最後一窗到 de 收盤），靜態歸因用
        ei = [T.cross_section(d)['e'] for d in sd] + [L.IDX.get_loc(de)]
        F = np.array([L.window_factors_all(ei[i], ei[i + 1], P.GX) if i + 1 < len(sd) else
                      np.where(L.TRADED[ei[i]] & np.isfinite(L.On[ei[i]]), L.Cn[ei[i]] / L.On[ei[i]], np.nan) * (P.CP[ei[i + 1]] / P.CP[ei[i]])
                      for i in range(len(sd))], np.float32)
        out[k] = dict(sd=np.array([str(d.date()) for d in sd]), de=str(de.date()), ra=ra, Rr=Rr.astype(np.float64), r5=P.bench_win(sd, de),
                      Wa=Wa.astype(np.float32), Wr=Wr.astype(np.float32), rk=rk, Rrep=Rrep, F=F)
    tag = f'run_{A.config}_{A.period}' + (f'_ex{A.exclude.replace(",", "-")}' if A.exclude else '') + ('' if A.cost == 'normal' else f'_{A.cost}')
    np.savez_compressed(P.OUT / f'{tag}.npz', **{f'{k}|{n}': v for k, d in out.items() for n, v in d.items()})
    summ = {}
    for k, d in out.items():
        x = d['ra'] - d['Rr'].mean(axis=0)
        summ[k] = (round(float(x.mean() * 100), 3), round(P.tstat(x), 2), len(x))
    print(tag, 'done', round(time.time() - t0, 1), 's', json.dumps(summ), flush=True)
