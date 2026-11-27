#!/usr/bin/env python3
"""R108：U2|any 放進 S75 v3 連續帳戶（2008-01-02～2026-10-22；7% 漲跌幅＋2008-10 跌幅 3.5%、0050 銜接檔、REDUCE_FIX），標準路徑（100 萬、位移 0）。
載入與漲跌停覆寫照 agents/s75_continuous_2008_2026/v3_2026-11-25_limit7/run_v3.py（只在記憶體）。先驗：現行與 nav_2008_2026_v3.csv.gz 逐日差 0、V_BRIDGE 627 筆；
自寫引擎 rebuy=None 與原引擎逐筆相同。輸出 cont2008.json。用法：python3 -I -B cont2008.py"""
import sys, json, time, resource, gc
from pathlib import Path
HERE = Path(__file__).resolve().parent
AG = HERE.parent
S75 = AG / 's75_continuous_2008_2026'; V3 = S75 / 'v3_2026-11-25_limit7'; V2 = S75 / 'v2_2026-11-21'
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(S75))
import numpy as np, pandas as pd
import s75lib as K
from s75lib import R, cslib, s10lib
import r108lib as L
assert sys.modules['s10lib'] is s10lib and sys.modules['cslib'] is cslib
t0 = time.time(); rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
assert cslib.REDUCE_FIX is False and cslib.E50_SOURCE is None
old = pd.read_csv(cslib.E50_CSV, parse_dates=['date']).set_index('date')['adj'].sort_index()
new = pd.read_csv(cslib.E50_OFFICIAL_CSV, parse_dates=['date']).set_index('date')['adj'].sort_index()
seam = new.index[0]
sp = pd.concat([old.loc[:seam].iloc[:-1] * float(new.loc[seam] / old.loc[seam]), new])
ex = pd.read_csv(V2 / 'e50_spliced.csv', parse_dates=['date']).set_index('date')['adj']
assert sp.index.equals(ex.index) and float((sp - ex).abs().max()) < 1e-9
cslib.E50_SOURCE = V2 / 'e50_spliced.csv'; cslib.REDUCE_FIX = True
P, U = K.load_cont()
D0, DE = P.IDX[P.t0], P.IDX[-1]
assert (str(D0.date()), str(DE.date())) == ('2008-01-02', '2026-10-22')
D19 = K.last_before(P.IDX, '2019-01-01'); DH1 = K.last_before(P.IDX, '2015-07-01')


def limits(ref, up_pct, dn_pct):
    u = ref * (1 + up_pct); t = cslib.tick(u); u = np.floor(u / t + 1e-9) * t
    d = ref * (1 - dn_pct); t = cslib.tick(d); d = np.ceil(d / t - 1e-9) * t
    return u, d


iv = P.IDX.values
cut = int(np.searchsorted(iv, np.datetime64('2015-06-01'), side='left'))
h0 = int(np.searchsorted(iv, np.datetime64('2008-10-13'), side='left')); h1 = int(np.searchsorted(iv, np.datetime64('2008-10-24'), side='right'))
u7, d7 = limits(P.REF[:cut], 0.07, 0.07)
LU = P.LIMUP.copy(); LU[:cut] = u7; LD = P.LIMDN.copy(); LD[:cut] = d7
_, d35 = limits(P.REF[h0:h1], 0.07, 0.035); LD[h0:h1] = d35
P.LIMUP, P.LIMDN = LU, LD
del u7, d7, d35; gc.collect()
ref_nav = pd.read_csv(V3 / 'nav_2008_2026_v3.csv.gz', index_col=0, parse_dates=True)
WINS = dict(full=(D0, DE), y2008_2015H1=(D0, DH1), y2015H2_2018=(DH1, D19), since2019=(D19, DE))
res = {}
for v, rule in K.VERSIONS.items():
    F, RT = K.surge(P, rule)
    sg = K.signal(P, U, RT, 0)
    ROWS = L.row_scores(P, U, RT, sg, 0)
    navC, lgC, _ = K.sim(P, sg, U, 1e6)
    n2, l2, _ = L.simulate_u2(P, sg, U, ROWS, rebuy=None, capital=1e6, cap_pct=0.05)
    cols = ['j', 'sig_t', 'ent_t', 'exit_t', 'cost', 'proceeds', 'why', 'shares']
    a1 = bool(np.array_equal(navC.to_numpy(), n2.to_numpy(), equal_nan=True) and lgC[cols].equals(l2[cols]))
    navF, lgF, stF = L.simulate_u2(P, sg, U, ROWS, rebuy=dict(r5=False), capital=1e6, cap_pct=0.05)
    rb = lgF[lgF['tag'] == 'rebuy']
    o = dict(baseline=dict(nav_max_abs_diff=float((navC.round(2) - ref_nav[v]).abs().max()), n_all=int(len(lgC)), A1_engine_identical=a1),
             CUR={k: K.cagr(navC, a, b) for k, (a, b) in WINS.items()}, FZ={k: K.cagr(navF, a, b) for k, (a, b) in WINS.items()},
             maxdd=dict(CUR=R.maxdd(navC), FZ=R.maxdd(navF)), n_all=dict(CUR=int(len(lgC)), FZ=int(len(lgF))),
             rebuys=dict(n=int(len(rb)), trig=stF['rb_trig'], fail=stF['rb_fail'], flag=stF['rb_flag'], mean_ret=float(rb['ret'].mean()) if len(rb) else None,
                         pnl_sum=float(rb['pnl'].sum()), win=float((rb['ret'] > 0).mean()) if len(rb) else None))
    o['diff_pp'] = {k: (o['FZ'][k] - o['CUR'][k]) * 100 for k in WINS}
    res[v] = o
    print(v, json.dumps(o, ensure_ascii=False, default=str), f'{time.time()-t0:.0f}s', rss(), 'MB', flush=True)
    del F, RT, sg, ROWS, navC, lgC, n2, l2, navF, lgF, stF; gc.collect()
assert res['V_BRIDGE']['baseline']['nav_max_abs_diff'] == 0 and res['V_BRIDGE']['baseline']['n_all'] == 627 and res['V_BRIDGE']['baseline']['A1_engine_identical']
res['meta'] = dict(runtime_s=round(time.time() - t0, 1), maxrss_mb=rss())
(HERE / 'cont2008.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
