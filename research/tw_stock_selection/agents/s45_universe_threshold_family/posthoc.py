#!/usr/bin/env python3
"""S45 事後描述（不影響判定）：只讀 results.json、trades.csv.gz → results_posthoc.json。python3 -I -B posthoc.py
- 現行在每條路徑（同資金、同位移）15 格中的名次分布；主跑名次；2024 起與 2019～2023 路徑中位數名次。
- 依成交值門檻、依股價門檻的邊際平均（路徑中位數）。
- 逐年報酬（主跑）。"""
import json
from pathlib import Path
import numpy as np, pandas as pd
H = Path(__file__).resolve().parent
r = json.loads((H / 'results.json').read_text())
C = r['cells']; ids = list(C); CUR = 'A1000_P10'
keys = list(C[CUR]['paths'])
rk = lambda vals, k: 1 + sum(v > vals[k] for v in vals.values())
per_path = {q: rk({k: C[k]['paths'][q]['since2019'] for k in ids}, CUR) for q in keys}
out = dict(
    cur_rank_per_path=per_path, cur_rank_per_path_median=float(np.median(list(per_path.values()))),
    cur_rank_per_path_n_top3=int(sum(v <= 3 for v in per_path.values())),
    cur_rank_main_100w=rk({k: C[k]['main']['since2019'] for k in ids}, CUR),
    cur_rank_since2024_median=rk({k: C[k]['path_summary']['since2024_median'] for k in ids}, CUR),
    cur_rank_y2019_2023_median=rk({k: C[k]['path_summary']['y2019_2023_median'] for k in ids}, CUR),
    margin_over_2nd_pp=float((C[CUR]['path_summary']['since2019_median'] - sorted(C[k]['path_summary']['since2019_median'] for k in ids)[-2]) * 100),
    by_adv={a: float(np.mean([C[k]['path_summary']['since2019_median'] for k in ids if C[k]['adv'] == a])) for a in sorted({C[k]['adv'] for k in ids})},
    by_px={p: float(np.mean([C[k]['path_summary']['since2019_median'] for k in ids if C[k]['px'] == p])) for p in sorted({C[k]['px'] for k in ids})},
    yearly_main={k: C[k]['main']['yearly'] for k in ids}, yearly_0050=r['bench_0050']['yearly'])
tr = pd.read_csv(H / 'trades.csv.gz', dtype={'code': str})
cur = tr[tr['cell'] == CUR]
diff = {}
for k in ids:
    if k == CUR:
        continue
    t = tr[tr['cell'] == k]
    a = t.groupby('code')['pnl'].sum(); b = cur.groupby('code')['pnl'].sum()
    d = a.sub(b, fill_value=0).sort_values()
    diff[k] = dict(worst3={c: float(v) for c, v in d.head(3).items()}, best3={c: float(v) for c, v in d.tail(3).items()})
out['pnl_diff_by_code_main'] = diff
(H / 'results_posthoc.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
print(json.dumps({k: v for k, v in out.items() if k not in ('cur_rank_per_path', 'yearly_main', 'pnl_diff_by_code_main')}, ensure_ascii=False, indent=1))
print(json.dumps(diff, ensure_ascii=False)[:1500])
