#!/usr/bin/env python3
"""引擎驗證：(1) simulate2 純量成本＝ia_core.simulate；(2) 正式主策略（disposal_5y 遮罩）＝main_v2 navs_gapfix EXFIN|n10|normal。
不輸出任何挑選期以外的績效數字（只比對 nav 是否相同）。"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import drj_core as R
K, V = R.K, R.V
YL, Y5 = R.revenue()
S = V.V2('gapfix', YL, Y5, exfin=True)
t0, t1, sd = S.bounds()
pl = S.plan(10, 'official')
b, s = K.rates(0.001)
n1, _ = K.simulate([(p[0], p[1]) for p in pl], b, s, t0, t1)
n2, _ = R.simulate2(pl, b, s, t0, t1)
ref = pd.read_parquet(R.MV2 / 'navs_gapfix.parquet')['EXFIN|n10|normal'].dropna()
d12 = float((n1 / n2 - 1).abs().max()); dref = float((n1.reindex(ref.index) / ref - 1).abs().max())
out = {'max_rel_diff_simulate2_vs_ia': d12, 'max_rel_diff_official_vs_main_v2_nav': dref, 'n_sig': len(sd)}
print(out)
assert d12 < 1e-9 and dref < 1e-9
(Path(__file__).resolve().parent / 'validate.json').write_text(json.dumps(out), encoding='utf-8')
