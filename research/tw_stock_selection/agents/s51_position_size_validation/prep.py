#!/usr/bin/env python3
"""S51 第 0 步（寫 PREREG 之前；只算現行格 S70_D10，不算任何候選格的報酬）：python3 -I -B prep.py → data_check.json
- P0 面板；正式開關全開；
- A0 現行格 100 萬位移 0 逐位重現 2019 起 48.76%／全期 34.40%／回撤 −32.12%／已平倉 381 筆（與 deadline_engine_fix ALL 逐位相同）；
- A2 現行格 20 條路徑與 deadline_engine_fix ALL 逐條相同；
- A3 引擎副本 simulate_s51（ndiv=10）在現行格 20 條路徑（連續）＋重新起跑主跑＋禁買主跑與原引擎 simulate_s10 淨值、交易逐位相同；
- 容量口徑試算（只現行格）：1,000 萬、5% 上限、位移 0～3 的 capped_rate（S37 口徑；S37 當時開關未全開，只作量級對照）；
- 重新起跑：營收月份 ≥ 2021-12 才有訊號，2022-01-01 前無交易；起始資金縮放比。"""
import sys, json, time, resource, gc
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s51lib as L
from s51lib import s10lib, s2lib, stlib, cslib

H = Path(__file__).resolve().parent
t0 = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
P = L.load()
assert (P.T, P.N, str(P.IDX[-1].date())) == (2876, 2142, '2026-10-22'), 'P0 失敗'
U, RT = L.base(P)
fix = L.s17lib.e50_fix_suspension(P)
L.slim(P); gc.collect()
print('載入完成', f'{time.time()-t0:.0f}s', rss(), 'MB', flush=True)
lastb = lambda d: P.IDX[P.IDX < pd.Timestamp(d)][-1]
D0, DE, D19, TB = P.IDX[P.t0], P.IDX[-1], lastb('2019-01-01'), lastb('2022-01-01')
dl = json.loads((L.AG / 'deadline_engine_fix' / 'out' / 'results.json').read_text())
cols = ['j', 'ent_t', 'exit_t', 'why', 'shares', 'cost', 'proceeds']
same = lambda a, b: bool(np.array_equal(a[0].to_numpy(), b[0].to_numpy(), equal_nan=True) and a[1][cols].equals(b[1][cols]))
out = dict(flags=dict(EXRIGHT_FIX=cslib.EXRIGHT_FIX, DEADLINE_FIX=cslib.DEADLINE_FIX, INS15_MASK=cslib.INS15_MASK),
           P0=dict(T=P.T, N=P.N, first=str(D0.date()), last=str(DE.date()), train_end=str(TB.date())), e50_fix=fix, src_sha256=L.SRC_SHA, paths={}, A3={}, capacity_cur={})
a3 = True
for s in L.SHIFTS:
    sg = s10lib.make_signal_s10(P, U, RT, shift=s)
    for cap in L.CAPS:
        key = f'{int(cap/1e4)}萬|+{s}'
        r0 = L.sim(P, sg, U, L.CUR, cap=cap, orig=True)
        r1 = L.sim(P, sg, U, L.CUR, cap=cap)
        e = same(r0, r1); out['A3'][key] = e; a3 &= e
        nav, lg, st = r0
        out['paths'][key] = dict(since2019=stlib.cagr(nav, D19, DE), full=stlib.cagr(nav, D0, DE), maxdd=stlib.maxdd(nav, D0, DE))
        if s == 0 and cap == 1e6:
            main = dict(out['paths'][key], n_closed=int((lg['why'] != 'open_end').sum()))
            f0 = L.sim(P, sg, U, L.CUR, cap=cap, fresh=True, orig=True); f1 = L.sim(P, sg, U, L.CUR, cap=cap, fresh=True)
            out['A3']['fresh|100萬|+0'] = same(f0, f1); a3 &= out['A3']['fresh|100萬|+0']
            out['fresh'] = dict(no_trade_before_2022=bool((f0[1]['ent_date'] >= pd.Timestamp('2022-01-01')).all()), first_entry=str(f0[1]['ent_date'].min().date()),
                                ratio_nav_TB=float(f0[0].loc[TB] / 1e6))
            blk = s2lib.block_from(f0[1], 5)[0]
            b0 = L.sim(P, sg, U, L.CUR, cap=cap, fresh=True, block=blk, orig=True); b1 = L.sim(P, sg, U, L.CUR, cap=cap, fresh=True, block=blk)
            out['A3']['fresh_block|100萬|+0'] = same(b0, b1); a3 &= out['A3']['fresh_block|100萬|+0']
            del f0, f1, b0, b1
        del r0, r1
    _, _, stc = L.sim(P, sg, U, L.CUR, cap=1e7)
    out['capacity_cur'][f'1000萬|+{s}'] = L.capped_rate(stc)
    del sg; gc.collect()
    print('位移', s, f'{time.time()-t0:.0f}s', rss(), 'MB', flush=True)
out['A0'] = dict(main, tol_ok=bool(abs(main['since2019'] - L.ACC['since2019']) < 2.5e-4 and abs(main['full'] - L.ACC['full']) < 2.5e-4
                                   and abs(main['maxdd'] - L.ACC['maxdd']) < 2.5e-4 and main['n_closed'] == L.ACC['n_closed']),
                 bitwise_vs_deadline_ALL=bool(main['since2019'] == dl['standard']['ALL']['since2019'] and main['full'] == dl['standard']['ALL']['full']
                                              and main['maxdd'] == dl['standard']['ALL']['maxdd_full']))
d2 = [abs(out['paths'][k]['since2019'] - v['since2019']) for k, v in dl['paths']['ALL'].items()]
out['A2'] = dict(n=len(d2), max_abs_diff=float(max(d2)), ok=bool(len(d2) == 20 and max(d2) < 1e-9))
out['A3_all_ok'] = bool(a3)
out['runtime_s'] = round(time.time() - t0, 1); out['maxrss_mb'] = rss()
(H / 'data_check.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))
print(json.dumps({k: out[k] for k in ('A0', 'A2', 'A3_all_ok', 'fresh', 'capacity_cur', 'runtime_s', 'maxrss_mb')}, ensure_ascii=False, indent=1))
assert out['A0']['tol_ok'] and out['A0']['bitwise_vs_deadline_ALL'] and out['A2']['ok'] and a3 and out['fresh']['no_trade_before_2022'], '驗收失敗'
