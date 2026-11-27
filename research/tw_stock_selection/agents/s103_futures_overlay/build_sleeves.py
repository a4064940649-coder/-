#!/usr/bin/env python3
"""S103 步驟 1：用 S75 v3 口徑（V_BRIDGE、漲跌幅 7%/3.5%/10%、0050 銜接檔、REDUCE_FIX）重跑引擎，輸出每日 NAV 與 stockfrac（股票市值 ÷ 淨值）。
衛星權重 sat_w ∈ {0.70（＝v3 正式）, 0.85, 1.00} × 位移 0～3 × 本金 5 種 ＝ 每個 sat_w 20 條路徑。
不改任何引擎或既有檔；只在記憶體覆寫 P.LIMUP／P.LIMDN（照 run_v3.py 同一寫法）。
用法：python3 -I -B build_sleeves.py   輸出：sleeves.npz（本資料夾）＋ sleeves_check.json
驗收：sat_w=0.70、100萬、位移 0 的 NAV 必須與 v3 nav_2008_2026_v3.csv.gz 的 V_BRIDGE 逐日相同（差 < 0.01 元），否則中止。"""
import sys, json, time, resource, gc
from pathlib import Path
HERE = Path(__file__).resolve().parent
AG = HERE.parent
S75 = AG / 's75_continuous_2008_2026'
V3 = S75 / 'v3_2026-11-25_limit7'; V2 = S75 / 'v2_2026-11-21'
sys.path.insert(0, str(S75))
import numpy as np, pandas as pd
import s75lib as K
from s75lib import cslib, L

t0 = time.time(); rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
cslib.E50_SOURCE = V2 / 'e50_spliced.csv'; cslib.REDUCE_FIX = True
P, U = K.load_cont()
D0, DE = P.IDX[P.t0], P.IDX[-1]
assert (str(D0.date()), str(DE.date())) == ('2008-01-02', '2026-10-22')

def limits(ref, up_pct, dn_pct):
    u = ref * (1 + up_pct); t = cslib.tick(u); u = np.floor(u / t + 1e-9) * t
    d = ref * (1 - dn_pct); t = cslib.tick(d); d = np.ceil(d / t - 1e-9) * t
    return u, d

iv = P.IDX.values
cut = int(np.searchsorted(iv, np.datetime64('2015-06-01'), side='left'))
h0 = int(np.searchsorted(iv, np.datetime64('2008-10-13'), side='left')); h1 = int(np.searchsorted(iv, np.datetime64('2008-10-24'), side='right'))
LU, LD = P.LIMUP.copy(), P.LIMDN.copy()
u7, d7 = limits(P.REF[:cut], 0.07, 0.07); LU[:cut] = u7; LD[:cut] = d7
_, d35 = limits(P.REF[h0:h1], 0.07, 0.035); LD[h0:h1] = d35
P.LIMUP, P.LIMDN = LU, LD
del u7, d7, d35

F, RT = K.surge(P, K.VERSIONS['V_BRIDGE'])
SATS = (0.70, 0.85, 1.00)
idx = P.IDX[P.t0:]
out = {'dates': np.array([str(d.date()) for d in idx]), 'e50': np.asarray(P.E50[P.t0:], float)}
keys = []
for s in K.SHIFTS:
    sg = K.signal(P, U, RT, s)
    for sw in SATS:
        for cap in K.CAPS:
            nav, lg, st = L.sim(P, sg, U, sat_w=sw, cap=cap)
            k = f's{int(round(sw * 100))}|{int(cap / 1e4)}|{s}'
            out['nav|' + k] = nav.to_numpy(float); out['sf|' + k] = st['stockfrac'].to_numpy(float)
            out['n|' + k] = np.array([len(lg)])
            keys.append(k)
            del nav, lg, st
    del sg; gc.collect()
    print('shift', s, f'{time.time() - t0:.0f}s', rss(), 'MB', flush=True)
ref = pd.read_csv(V3 / 'nav_2008_2026_v3.csv.gz', index_col=0, parse_dates=True)
dmax = float(np.abs(out['nav|s70|100|0'] - ref['V_BRIDGE'].to_numpy()).max())
e50chk = float(np.abs(out['e50'] / out['e50'][0] * 1e6 - ref['0050'].to_numpy()).max())
chk = dict(v3_std_nav_max_abs_diff=dmax, e50_vs_v3_max_abs_diff=e50chk, n_keys=len(keys), runtime_s=round(time.time() - t0), maxrss_mb=rss(),
           cagr_std=K.cagr(pd.Series(out['nav|s70|100|0'], index=idx), D0, DE))
print(chk)
assert dmax < 0.01, 'sat 0.70 標準路徑與 v3 不同'
np.savez_compressed(HERE / 'sleeves.npz', **out)
(HERE / 'sleeves_check.json').write_text(json.dumps(chk, ensure_ascii=False, indent=1))
