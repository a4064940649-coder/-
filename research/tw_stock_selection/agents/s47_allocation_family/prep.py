#!/usr/bin/env python3
"""S47 第 0 步（寫 PREREG 之前；只算現行格，不算任何非現行格的報酬）：python3 -I -B prep.py → data_check.json
- P0 面板；A0 現行格 100 萬位移 0 逐位重現 48.76%／34.40%／−32.12%／381 筆，並與 deadline_engine_fix ALL 逐位相同；
- A2 現行格 20 條路徑與 deadline_engine_fix ALL 逐條相同；
- A3 引擎副本 simulate_s47（ndiv=None 與 ndiv=10，idle='e50'）在現行格 20 條路徑與 simulate_s10 淨值、交易逐位相同；
- 官方淨值：xq_scripts/site/navs_official.csv.gz 的 E0|W10_70 對現行主跑淨值（檔案存 2 位小數）；
- 描述（只現行格）：平均持股檔數、個股占淨值、閒置（現金＋0050 − 底倉）占淨值。"""
import sys, json, time, resource, gc
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s47lib as L
from s47lib import s17lib, s10lib, s4lib, stlib, cslib

H = Path(__file__).resolve().parent
t0 = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
P = L.load()
assert (P.T, P.N, str(P.IDX[-1].date())) == (2876, 2142, '2026-10-22'), 'P0 失敗'
INNOV, _ = s4lib.innov_panel(P)
U = stlib.universe(P, 'twse') & ~INNOV
del INNOV
F = s17lib.rev_family(P, defs=['STD'])
RT = dict(S=F['S']['STD'], av=F['av'])
del F
fix = s17lib.e50_fix_suspension(P)
for a in ('H', 'L', 'VOL', 'CHG', 'HF', 'VOLF', 'adjH', 'HI240', 'MA60', 'nbar', 'F_HI', 'IDXC', 'AO', 'listed', 'NEARHI', 'V60', 'liq'):
    if hasattr(P, a):
        delattr(P, a)
gc.collect()
print('載入完成', f'{time.time()-t0:.0f}s', rss(), 'MB', flush=True)

lastb = lambda d: P.IDX[P.IDX < pd.Timestamp(d)][-1]
D0, DE, D19 = P.IDX[P.t0], P.IDX[-1], lastb('2019-01-01')
dl = json.loads((L.AG / 'deadline_engine_fix' / 'out' / 'results.json').read_text())
out = dict(flags=dict(EXRIGHT_FIX=cslib.EXRIGHT_FIX, DEADLINE_FIX=cslib.DEADLINE_FIX, INS15_MASK=cslib.INS15_MASK),
           P0=dict(T=P.T, N=P.N, last=str(P.IDX[-1].date())), e50_fix=fix, src_sha256=L.SRC_SHA, paths={}, A3={})
a3_ok = True
for s in L.SHIFTS:
    sg = s10lib.make_signal_s10(P, U, RT, shift=s)
    for cap in L.CAPS:
        key = f'{int(cap/1e4)}萬|+{s}'
        nav, lg, st = L.sim(P, sg, U, cap=cap)
        nav1, lg1, _ = L.sim(P, sg, U, cap=cap, div=None, force_copy=True)
        nav2, lg2, _ = L.sim(P, sg, U, cap=cap, div=10, force_copy=True)
        cols = ['j', 'ent_t', 'exit_t', 'why', 'shares', 'cost', 'proceeds']
        e1 = bool(np.array_equal(nav.to_numpy(), nav1.to_numpy()) and lg[cols].equals(lg1[cols]))
        e2 = bool(np.array_equal(nav.to_numpy(), nav2.to_numpy()) and lg[cols].equals(lg2[cols]))
        out['A3'][key] = dict(ndiv_none=e1, ndiv_10=e2); a3_ok &= e1 and e2
        out['paths'][key] = dict(since2019=stlib.cagr(nav, D19, DE), full=stlib.cagr(nav, D0, DE), maxdd=stlib.maxdd(nav, D0, DE))
        if s == 0 and cap == 1e6:
            x = lg[lg['why'] != 'open_end']
            sf = st['stockfrac']; Bs = st['B']
            main = dict(out['paths'][key], n_closed=int(len(x)), avg_npos=float(st['avg_npos']), avg_idle_frac=float(st['avg_idle_frac']),
                        stockfrac_mean=float(sf.loc[sf.index >= lg['ent_date'].min()].mean()),
                        sat_idle_frac_mean=float(((1 - sf) - Bs / nav).loc[sf.index >= lg['ent_date'].min()].mean()))
            navc = nav
        del nav1, nav2, lg1, lg2
    del sg; gc.collect()
    print('位移', s, f'{time.time()-t0:.0f}s', rss(), 'MB', flush=True)

out['A0'] = dict(main, tol_ok=bool(abs(main['since2019'] - L.ACC['since2019']) < 2.5e-4 and abs(main['full'] - L.ACC['full']) < 2.5e-4
                                   and abs(main['maxdd'] - L.ACC['maxdd']) < 2.5e-4 and main['n_closed'] == L.ACC['n_closed']),
                 bitwise_vs_deadline_ALL=bool(main['since2019'] == dl['standard']['ALL']['since2019'] and main['full'] == dl['standard']['ALL']['full']
                                              and main['maxdd'] == dl['standard']['ALL']['maxdd_full']))
d2 = [abs(out['paths'][k]['since2019'] - v['since2019']) for k, v in dl['paths']['ALL'].items()]
d2dd = [abs(out['paths'][k]['maxdd'] - v['maxdd_full']) for k, v in dl['paths']['ALL'].items()]
out['A2'] = dict(n=len(d2), max_abs_diff=float(max(d2)), max_abs_diff_maxdd=float(max(d2dd)), ok=bool(len(d2) == 20 and max(d2) < 1e-9))
out['A3_all_ok'] = bool(a3_ok)
off = Path('/home/user/-/xq_scripts/site/navs_official.csv.gz')
if off.exists():
    o = pd.read_csv(off, parse_dates=['date']).set_index('date')['E0|W10_70']
    m = navc.reindex(o.index)
    d = (m - o).abs()
    out['official_nav'] = dict(file=str(off), n=int(len(o)), n_missing=int(m.isna().sum()), max_abs_diff_yuan=float(d.max()),
                               ok=bool(m.notna().all() and d.max() <= 0.0051 and len(o) == len(navc)))
else:
    out['official_nav'] = dict(file=str(off), exists=False)
out['runtime_s'] = round(time.time() - t0, 1); out['maxrss_mb'] = rss()
(H / 'data_check.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))
print(json.dumps({k: out[k] for k in ('A0', 'A2', 'A3_all_ok', 'official_nav', 'runtime_s', 'maxrss_mb')}, ensure_ascii=False, indent=1))
assert out['A0']['tol_ok'] and out['A0']['bitwise_vs_deadline_ALL'] and out['A2']['ok'] and a3_ok, '驗收失敗'
