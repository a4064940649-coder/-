#!/usr/bin/env python3
"""冒煙測試（PREREG 前，只重現已發布的 V4 數字，不碰 V5 報酬）：fv=None 的 V5 類別＝V4；AB 40 seed 平均淨值＝navs_v4.parquet 的 AB_V4|normal。"""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import v5lib as L
g = L.setup(); K, V = L.K, L.V
NV = pd.read_parquet(L.V4DIR / 'navs_v4.parquet')
S = L.make(); t0, t1, sd = S.bounds(); pl = S.plan(10, 'official')
nav = L.sim(pl, 'normal', t0, t1)
d1 = float((nav / NV['V4|normal'].reindex(nav.index) - 1).abs().max())
rets = []
for s in range(40):
    Sx = L.make(unavail=V.make_unavail('mc', s)); rets.append(L.sim(Sx.plan(10, 'official'), 'normal', t0, t1).pct_change())
ab = L.mean_nav(rets)
d2 = float((ab / NV['AB_V4|normal'].reindex(ab.index) - 1).abs().max())
out = {'V4_maxreldiff': d1, 'AB_V4_maxreldiff': d2, 'pass': d1 < 1e-9 and d2 < 1e-9}
json.dump(out, open(HERE / 'smoke.json', 'w'), indent=1); print(out)
