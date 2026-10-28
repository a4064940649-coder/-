#!/usr/bin/env python3
"""引擎驗證（只重現已公開的 V4 數字）：V4J 用官方訊號日 → 與 o_v4 的 V4／AB_V4 淨值逐日相同。"""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import fx_core as F
L, K, V = F.L, F.K, F.V
st = F.setup()
NV = pd.read_parquet(F.V4DIR / 'navs_v4.parquet')
off, end = F.official_days()
S, pl, t0, t1, sd = F.build(off, end, None)
nav, lg = F.sim(pl, t0, t1, 'normal')
ref = NV['V4|normal']
d1 = float((nav / ref.reindex(nav.index) - 1).abs().max())
P = pd.read_csv(F.V4DIR / 'plan_V4.csv', dtype=str)
mine = [','.join(K.COLS[j] for j in p[1]) for p in pl]
same_plan = mine == P['picks'].tolist()
rets = []
for s in range(40):
    S2, pl2, a, b, _ = F.build(off, end, V.make_unavail('mc', s))
    n2, _ = F.sim(pl2, a, b, 'normal'); rets.append(n2.pct_change())
ab = (1 + pd.concat(rets, axis=1).mean(axis=1).fillna(0)).cumprod()
d2 = float((ab / NV['AB_V4|normal'].reindex(ab.index) - 1).abs().max())
W = F.windows(S.sig_days, S.end_day)
b = F.block(ab, W, ('ALL',))
out = {'V4_maxreldiff': d1, 'V4_plan_identical': bool(same_plan), 'AB_V4_maxreldiff': d2, 'AB_V4_ALL_CAGR_mine': b['ALL']['CAGR'], 'AB_V4_ALL_t_mine': b['ALL']['t_0050'],
       'AB_V4_ALL_t_calendar': b['ALL']['t_0050_cal'], 'turnover_V4': lg['turnover']}
assert d1 < 1e-9 and same_plan and d2 < 1e-9, out
(HERE / 'validate.json').write_text(json.dumps(F.jsonable(out), ensure_ascii=False, indent=1), encoding='utf-8')
print(out)
