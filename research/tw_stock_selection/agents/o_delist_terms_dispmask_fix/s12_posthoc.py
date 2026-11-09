#!/usr/bin/env python3
"""事後描述（不得推翻判定）：V7X union 對 orig 的逐 j 差（中位數的差 ≠ 差的中位數）。python3 -I -B s12_posthoc.py"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'o_exec_engine_rewrite_v7x')); import x_stats as S; sys.path.pop(0)
R = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))
N = pd.read_parquet(R['RESULTS_V7X']['navs_file'])
out = {}
for pair in (('union', 'orig'), ('orig', 'v3_main'), ('orig', 'v3_lo')):
    for c in ('normal|100萬', 'stress|100萬', 'normal|300萬', 'normal|1000萬'):
        d = {'exc': [], 'exc_Y19on': [], 't_0050': []}
        for j in range(13):
            a = S.stats(N[f'{j}|{pair[0]}|{pair[1]}|{c}'], 'AB_V7_B10', j); b = S.stats(N[f'{j}|orig|orig|{c}'], 'AB_V7_B10', j)
            for k in d: d[k].append(a[k] - b[k])
        out[f'{pair[0]}|{pair[1]}|{c}'] = {k: dict(per_j=[round(x, 4) for x in v], median=float(np.median(v)), mean=float(np.mean(v)), min=float(min(v)), max=float(max(v))) for k, v in d.items()}
R['POSTHOC_DESC'] = dict(note='事後描述，不推翻判定：V7X 正式欄是「13 個 j 的中位數」；中位數的差可能來自哪個 j 落在中位，不代表每個 j 都同向。', per_j_delta=out)
(HERE / 'results.json').write_text(json.dumps(R, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
for k, v in out.items(): print(k, {kk: (round(vv['median'], 3), round(vv['mean'], 3)) for kk, vv in v.items()})
