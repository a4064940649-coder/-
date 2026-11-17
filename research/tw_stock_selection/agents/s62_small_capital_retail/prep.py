#!/usr/bin/env python3
"""S62 第 0 步（寫 PREREG 之前；只算現行 100 萬正式口徑，不算任何小本金或散戶情境的報酬）：python3 -I -B prep.py → data_check.json
A0：原引擎 100 萬位移 0 逐位重現 2019 起 48.76%／全期 34.40%／最大回撤 −32.12%／381 筆（＝deadline_engine_fix ALL）。
A1：引擎副本 simulate_s62 全預設，100 萬位移 0～3 與 simulate_s10 淨值、交易逐位相同。
另記：單次模擬秒數（估工時用）、0050 2019 起年化。"""
import sys, json, time, resource
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s62lib as L
from s62lib import s10lib, stlib, cslib

H = Path(__file__).resolve().parent
t0 = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
P, U, RT = L.setup()
print('載入完成', f'{time.time()-t0:.0f}s', rss(), 'MB', flush=True)
lastb = lambda d: P.IDX[P.IDX < pd.Timestamp(d)][-1]
D0, DE, D19 = P.IDX[P.t0], P.IDX[-1], lastb('2019-01-01')
dl = json.loads((L.AG / 'deadline_engine_fix' / 'out' / 'results.json').read_text())
out = dict(flags=dict(EXRIGHT_FIX=cslib.EXRIGHT_FIX, DEADLINE_FIX=cslib.DEADLINE_FIX, INS15_MASK=cslib.INS15_MASK),
           P0=dict(T=P.T, N=P.N, last=str(P.IDX[-1].date())), src_sha256=L.SRC_SHA, A1={})
cols = ['j', 'ent_t', 'exit_t', 'why', 'shares', 'cost', 'proceeds']
for s in L.SHIFTS:
    sg = s10lib.make_signal_s10(P, U, RT, shift=s)
    t1 = time.time(); nav, lg, st = L.sim_official(P, sg, U); dt = time.time() - t1
    nav1, lg1, _ = L.sim(P, sg, U, 1e6, None, None, defaults=True)
    out['A1'][f'+{s}'] = bool(np.array_equal(nav.to_numpy(), nav1.to_numpy()) and lg[cols].equals(lg1[cols]))
    if s == 0:
        a0 = dict(since2019=stlib.cagr(nav, D19, DE), full=stlib.cagr(nav, D0, DE), maxdd=stlib.maxdd(nav, D0, DE), n_closed=int((lg['why'] != 'open_end').sum()))
        a0['tol_ok'] = bool(abs(a0['since2019'] - L.ACC['since2019']) < 1e-12 and abs(a0['full'] - L.ACC['full']) < 1e-12
                            and abs(a0['maxdd'] - L.ACC['maxdd']) < 1e-12 and a0['n_closed'] == L.ACC['n_closed'])
        a0['bitwise_vs_deadline_ALL'] = bool(a0['since2019'] == dl['standard']['ALL']['since2019'] and a0['full'] == dl['standard']['ALL']['full'])
        out['A0'] = a0; out['sim_secs_full'] = dt
b = pd.Series(P.E50[P.t0:], index=P.IDX[P.t0:])
out['e50_since2019'] = stlib.cagr(b, D19, DE)
out['start_row_2019'] = int(np.searchsorted(P.IDX.values, np.datetime64(D19))); out['D19'] = str(D19.date()); out['odd_date'] = str(cslib.ODD_DATE.date())
out['A1_all_ok'] = bool(all(out['A1'].values())); out['maxrss_mb'] = rss(); out['secs'] = round(time.time() - t0)
json.dump(out, open(H / 'data_check.json', 'w'), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
assert out['A0']['tol_ok'] and out['A0']['bitwise_vs_deadline_ALL'] and out['A1_all_ok'], '驗收失敗'
