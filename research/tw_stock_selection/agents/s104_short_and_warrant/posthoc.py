#!/usr/bin/env python3
"""S104 事後描述（驗證後才寫，不參與判定、不選格）：
1) K1 放空腿 vs 直接放空 0050（同 30%，年化成本 1%）的合併效果——個股選空有沒有比放空大盤好；
2) K1、K2 零成本（毛）版本——成本吃掉多少；
3) K1 合併 vs「現行 × λ＋現金」同回撤減碼對照。位移 0。用法：python3 -I -B posthoc.py → results_posthoc.json"""
import sys, json
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import s104lib as L, s104cells as C, s104eval as E
d = L.load_cache(); L.set_idx(d)
FZ = json.loads((HERE / 'FROZEN.json').read_text())
offi = E.off_array(d)
W = {'valid': (d.TB, d.TE), 'full': (d.TA, d.TE), 'train': (d.TA, d.TB)}
out = {}
nav, tr, _ = C.k1_run(d, FZ['selection']['K1']['frozen'], 0); nav = E._flat(nav, d.TA)
navz, trz, _ = C.k1_run(d, FZ['selection']['K1']['frozen'], 0, borrow=0.0, cost='zero'); navz = E._flat(navz, d.TA)
r50 = d.r50.copy(); short50 = np.cumprod(1 - r50 - 0.01 / 252)
combo = L.overlay(offi, nav, 0.3); combo50 = L.overlay(offi, short50, 0.3)
o = {}
for k, (a, b) in W.items():
    mdd_c = L.maxdd(combo, a, b)
    lam = None
    for x in np.arange(1.0, 0.3, -0.01):           # 找同回撤的減碼比例
        sc = np.cumprod(1 + x * L.rets(offi))
        if L.maxdd(sc, a, b) >= mdd_c:
            lam = float(x); break
    sc = np.cumprod(1 + lam * L.rets(offi))
    o[k] = dict(short_sleeve=L.metrics(nav, d, a, b), short_sleeve_gross_zero_cost=L.metrics(navz, d, a, b), short_0050=L.metrics(short50, d, a, b),
                combo_stock_short_w03=L.metrics(combo, d, a, b), combo_0050_short_w03=L.metrics(combo50, d, a, b),
                official_scaled_same_mdd=dict(lam=lam, **L.metrics(sc, d, a, b)), official=L.metrics(offi, d, a, b))
out['K1'] = o
pre = C.K2Pre(d)
nL, tL, nS, tS = C.k2_run(d, pre, FZ['selection']['K2']['frozen'], 0)
zL, _, zS, _ = C.k2_run(d, pre, FZ['selection']['K2']['frozen'], 0, borrow=0.0, cost='zero')
nL, nS, zL, zS = (E._flat(x, d.TA) for x in (nL, nS, zL, zS))
out['K2'] = {k: dict(long=L.metrics(nL, d, a, b), short=L.metrics(nS, d, a, b), long_gross=L.metrics(zL, d, a, b), short_gross=L.metrics(zS, d, a, b),
                     dual_gross=L.metrics(L.blend(zL, zS, 0.5), d, a, b)) for k, (a, b) in W.items()}
(HERE / 'results_posthoc.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
for K in out:
    for k in out[K]:
        print(K, k, {n: {m: round(v, 4) for m, v in x.items()} for n, x in out[K][k].items()})
