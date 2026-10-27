import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
import json, time
import numpy as np, pandas as pd
import rlib as R
T0 = time.time()
S = R.main_strategy(); pl = R.plan(10)
t0, t1, sd = S.bounds()
nav = R.sim(pl, t0, t1); ew = R.ew_nav(pl, t0, t1)
print('built', time.time() - T0)
ref = pd.read_parquet(R.MV2 / 'navs_gapfix.parquet')['EXFIN|n10|normal']
ref = ref.reindex(nav.index)
print('max rel diff vs main_v2 nav', float((nav / ref - 1).abs().max()))
out = {}
for k, (a, b) in {'ALL': ('2006-04-01', '2026-10-12'), 'P0620': ('2006-04-01', '2020-12-31'), 'S3': ('2021-03-12', '2026-10-12'),
                  'S4': ('2024-01-01', '2026-10-12'), 'VAL': ('2019-05-14', '2026-10-12'), 'SEL': ('2006-04-01', '2019-05-13')}.items():
    wa, wb = a, b
    if k == 'S3': wa = '2021-03-01'
    if k == 'VAL': wa = '2019-05-01'
    if k == 'SEL': wb = '2019-04-30'
    o = R.seg(nav, sd, a, b, ew, wa, wb); o.pop('yearly'); out[k] = o; print(k, o)
pd.DataFrame({'nav': nav, 'ew': ew}).to_parquet(R.HERE / 'main_nav.parquet')
json.dump(out, open(R.HERE / 'main_rebuild.json', 'w'), ensure_ascii=False, indent=1)
