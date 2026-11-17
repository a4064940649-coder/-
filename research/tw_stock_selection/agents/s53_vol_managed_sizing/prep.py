#!/usr/bin/env python3
"""S53 第 0 步（寫 SPEC／PREREG 之前；只算現行格，不算任何非現行格的報酬）：python3 -I -B prep.py → data_check.json
- P0 面板；A0 現行格 100 萬位移 0 逐位重現 48.76%／34.40%／−32.12%／381 筆，且與 deadline_engine_fix ALL 逐位相同；
- A2 現行格 20 條路徑與 deadline_engine_fix ALL 逐條相同；
- A3 引擎副本 simulate_s53(ndiv=10, wmult=None, vt=None) 在現行格 20 條路徑與 simulate_s10 淨值、交易逐位相同；
- 官方淨值：xq_scripts/site/navs_official.csv.gz 的 E0|W10_70 對現行主跑；
- B 目標波動（事前常數，依題目指定用「現行全期衛星波動中位數」）：現行主跑 sret 套 B 的估計式（60／120 日），全期每日 σ̂ 的中位數；
- 描述（只用現行格與訊號，不算非現行格報酬）：現行 sret 的有效天數；A 倍數 m 在現行訊號上的分布；B 規則套在現行自己 sret 上會給的衛星目標分布（不是 B 格的實際路徑）。"""
import sys, json, time, resource, gc
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s53lib as L
from s53lib import s17lib, s10lib, stlib, cslib

H = Path(__file__).resolve().parent
t0 = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
P = L.load()
assert (P.T, P.N, str(P.IDX[-1].date())) == (2876, 2142, '2026-10-22'), 'P0 失敗'
U, RT = L.base(P)
fix = s17lib.e50_fix_suspension(P)
SIG = L.sigma60(P)
L.slim(P); gc.collect()
print('載入完成', f'{time.time()-t0:.0f}s', rss(), 'MB', flush=True)
lastb = lambda d: P.IDX[P.IDX < pd.Timestamp(d)][-1]
D0, DE, D19 = P.IDX[P.t0], P.IDX[-1], lastb('2019-01-01')
dl = json.loads((L.AG / 'deadline_engine_fix' / 'out' / 'results.json').read_text())
out = dict(flags=dict(EXRIGHT_FIX=cslib.EXRIGHT_FIX, DEADLINE_FIX=cslib.DEADLINE_FIX, INS15_MASK=cslib.INS15_MASK),
           P0=dict(T=P.T, N=P.N, last=str(P.IDX[-1].date())), e50_fix=fix, src_sha256=L.SRC_SHA, paths={}, A3={})
a3 = True; mdist = {6: [], 8: []}
cols = ['j', 'ent_t', 'exit_t', 'why', 'shares', 'cost', 'proceeds']
for s in L.SHIFTS:
    sg = s10lib.make_signal_s10(P, U, RT, shift=s)
    if s == 0:
        S2, _ = s10lib.variant_S(sg, 'E0')
        for cd in (6, 8):
            W = L.make_wmult(S2, U, SIG, cd)
            mdist[cd] = [m for d in W.values() for m in d.values()]
        del S2
    for cap in L.CAPS:
        key = f'{int(cap/1e4)}萬|+{s}'
        nav, lg, st0 = L.sim(P, sg, U, 'CUR', cap=cap, orig=True)
        nav1, lg1, st = L.sim(P, sg, U, 'CUR', cap=cap)
        e = bool(np.array_equal(nav.to_numpy(), nav1.to_numpy()) and lg[cols].equals(lg1[cols]))
        out['A3'][key] = e; a3 &= e
        out['paths'][key] = dict(since2019=stlib.cagr(nav, D19, DE), full=stlib.cagr(nav, D0, DE), maxdd=stlib.maxdd(nav, D0, DE))
        if s == 0 and cap == 1e6:
            x = lg[lg['why'] != 'open_end']
            main = dict(out['paths'][key], n_closed=int(len(x)))
            navc, sretc = nav, st['sret']
        del nav1, lg1
    del sg; gc.collect()
    print('位移', s, f'{time.time()-t0:.0f}s', rss(), 'MB', flush=True)

out['A0'] = dict(main, tol_ok=bool(abs(main['since2019'] - L.ACC['since2019']) < 2.5e-4 and abs(main['full'] - L.ACC['full']) < 2.5e-4
                                   and abs(main['maxdd'] - L.ACC['maxdd']) < 2.5e-4 and main['n_closed'] == L.ACC['n_closed']),
                 bitwise_vs_deadline_ALL=bool(main['since2019'] == dl['standard']['ALL']['since2019'] and main['full'] == dl['standard']['ALL']['full']
                                              and main['maxdd'] == dl['standard']['ALL']['maxdd_full']))
d2 = [abs(out['paths'][k]['since2019'] - v['since2019']) for k, v in dl['paths']['ALL'].items()]
d2dd = [abs(out['paths'][k]['maxdd'] - v['maxdd_full']) for k, v in dl['paths']['ALL'].items()]
out['A2'] = dict(n=len(d2), max_abs_diff=float(max(d2)), max_abs_diff_maxdd=float(max(d2dd)), ok=bool(len(d2) == 20 and max(d2) < 1e-9 and max(d2dd) < 1e-9))
out['A3_all_ok'] = bool(a3)
off = Path('/home/user/-/xq_scripts/site/navs_official.csv.gz')
o = pd.read_csv(off, parse_dates=['date']).set_index('date')['E0|W10_70']
d = (navc.reindex(o.index) - o).abs()
out['official_nav'] = dict(n=int(len(o)), max_abs_diff_yuan=float(d.max()), ok=bool(d.notna().all() and d.max() <= 0.0051))

# ---- B 目標波動（現行主跑；全期每日 σ̂ 中位數）與描述
tg, bdesc = {}, {}
S0 = s10lib.make_signal_s10(P, U, RT, shift=0)
exec_days = sorted(set(int(x['k']) + 1 for x in S0['info']))
del S0
for w in (60, 120):
    rv = L.roll_vol(sretc, w)
    tg[str(w)] = float(rv.median())
    ex = rv.reindex(P.IDX[[t for t in exec_days if t < P.T]]).dropna()
    imp = np.clip(L.VT_BASE * tg[str(w)] / ex, L.VT_LO, L.VT_HI)
    bdesc[str(w)] = dict(n_exec=int(len(ex)), sigma_p10=float(rv.quantile(0.1)), sigma_p90=float(rv.quantile(0.9)),
                         implied_sat_median=float(np.median(imp)), implied_sat_mean=float(np.mean(imp)),
                         frac_at_lo=float((imp <= L.VT_LO + 1e-9).mean()), frac_at_hi=float((imp >= L.VT_HI - 1e-9).mean()),
                         frac_outside_band10=float((np.abs(imp - 0.7) > 0.10).mean()))
out['B_target_vol'] = tg
out['B_on_current_sret_desc'] = bdesc
out['sret_desc'] = dict(n_days=int(len(sretc) - 1), n_valid=int(sretc.notna().sum()), ann_vol_full=float(sretc.std() * np.sqrt(252)))
out['A_mult_desc'] = {str(cd): dict(n=len(v), median=float(np.median(v)), mean=float(np.mean(v)), frac_at_cap=float(np.mean(np.array(v) >= 10 / cd - 1e-9)),
                                    frac_at_floor=float(np.mean(np.array(v) <= L.W_FLOOR + 1e-9))) for cd, v in mdist.items()}
out['runtime_s'] = round(time.time() - t0, 1); out['maxrss_mb'] = rss()
(H / 'data_check.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))
print(json.dumps({k: out[k] for k in ('A0', 'A2', 'A3_all_ok', 'official_nav', 'B_target_vol', 'B_on_current_sret_desc', 'sret_desc', 'A_mult_desc', 'runtime_s', 'maxrss_mb')}, ensure_ascii=False, indent=1))
assert out['A0']['tol_ok'] and out['A0']['bitwise_vs_deadline_ALL'] and out['A2']['ok'] and a3 and out['official_nav']['ok'], '驗收失敗'
